import json

import pytest

from src import config, promote, train
from src.monitor import psi


def test_select_needs_gain_beyond_std():
    results = {"logreg": {"cv_mean": 0.80, "cv_std": 0.05},
               "random_forest": {"cv_mean": 0.84, "cv_std": 0.05},   # +0.04 < std: keep logreg
               "lightgbm": {"cv_mean": 0.86, "cv_std": 0.04}}        # +0.06 > std: switch
    assert train.select(results) == "lightgbm"
    assert train.select({"logreg": results["logreg"], "random_forest": results["random_forest"]}) == "logreg"


def test_split_is_stratified_and_deterministic(synthetic_df):
    a = train.split(synthetic_df, 0.2, 42)
    b = train.split(synthetic_df, 0.2, 42)
    assert list(a[1]) == list(b[1])
    test = synthetic_df.loc[a[1]]
    assert set(test["ROOFTYPE"]) == set(synthetic_df["ROOFTYPE"])


def test_quick_training_end_to_end(synthetic_df, tmp_path, monkeypatch):
    processed, reports = tmp_path / "processed", tmp_path / "reports"
    processed.mkdir()
    reports.mkdir()
    synthetic_df.to_csv(processed / config.FEATURES_CSV, index=False)
    (reports / "validation.json").write_text(json.dumps({"passed": True}))
    monkeypatch.setattr(config, "processed_dir", lambda: processed)
    monkeypatch.setattr(config, "REPORTS_DIR", reports)
    monkeypatch.setattr(train, "CANDIDATE_DIR", tmp_path / "candidate")
    monkeypatch.setattr(train, "TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")

    metrics = train.main(quick=True)
    assert metrics["test_macro_f1"] > 0.5
    assert set(metrics["grouped_cv_by_source"]) == set(config.SOURCES)
    meta = json.loads((tmp_path / "candidate" / "metadata.json").read_text())
    assert meta["features"] and meta["reference_stats"] and meta["trained_uids"]
    for f in ("metrics.json", "confusion_matrix.png", "cv_results.csv", "feature_importance.csv"):
        assert (reports / f).exists()


def test_training_refuses_unvalidated_data(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path)
    (tmp_path / "validation.json").write_text(json.dumps({"passed": False}))
    with pytest.raises(SystemExit):
        train.load_data()


@pytest.mark.parametrize("prod_f1,ok", [(None, True), (0.85, True), (0.95, False)])
def test_quality_gate(prod_f1, ok):
    cand = {"test_macro_f1": 0.9, "model_version": "m:2"}
    prod = {"test_macro_f1": prod_f1, "model_version": "m:1"} if prod_f1 else None
    assert promote.gate(cand, prod, {"passed": True}, 0.75)[0] is ok


def test_gate_blocks_on_floor_and_validation():
    cand = {"test_macro_f1": 0.7, "model_version": "m:2"}
    ok, reasons = promote.gate(cand, None, {"passed": False}, 0.75)
    assert not ok and len(reasons) == 2


def test_promote_then_rollback(model_dir, tmp_path, monkeypatch):
    import shutil
    models_dir = tmp_path / "models"
    cand = models_dir / "candidate"
    shutil.copytree(model_dir, cand)
    monkeypatch.setattr(config, "PRODUCTION_DIR", models_dir / "production")
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(promote, "CANDIDATE_DIR", cand)
    monkeypatch.setattr(promote, "ARCHIVE_DIR", models_dir / "archive")
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "validation.json").write_text(json.dumps({"passed": True}))

    assert promote.promote()["promoted"]
    # second, better model
    meta = json.loads((cand / "metadata.json").read_text())
    meta.update(model_version="roof-classifier:v2", test_macro_f1=0.95, trained_at="2026-02-01T00:00:00+00:00")
    (cand / "metadata.json").write_text(json.dumps(meta))
    assert promote.promote()["promoted"]
    assert json.loads((models_dir / "production" / "metadata.json").read_text())["model_version"] == "roof-classifier:v2"

    back = promote.rollback()
    assert back["rollback_to"] == "roof-classifier:test"


def test_psi_detects_shift():
    import numpy as np
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 2000)
    q = np.quantile(ref, np.linspace(0, 1, 11)).tolist()
    assert psi(q, rng.normal(0, 1, 500)) < 0.1
    assert psi(q, rng.normal(1.5, 1, 500)) > 0.2
    # constant (tied) reference values do not crash
    assert psi([0.0] * 11, np.zeros(10)) == 0.0
