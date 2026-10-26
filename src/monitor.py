"""Phase 6: drift and retraining check over the prediction log.

    python -m src.monitor            # writes reports/monitoring.json and .md

Compares logged prediction inputs with the training distribution stored in
the production model's metadata (Population Stability Index per feature,
missing-rate shift, predicted class mix) and checks how many labelled
buildings are new since training. Recommends retraining when:
  * any feature drifts beyond the PSI threshold, or
  * new labelled buildings reach the configured fraction of the training set, or
  * accuracy on buildings labelled after training falls below the floor.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src import config
from src.serving import RoofModel, prediction_log_path

log = logging.getLogger("monitor")


def psi(reference_quantiles: list[float], values: np.ndarray) -> float:
    """PSI of `values` against the decile bins of the training data."""
    values = values[~np.isnan(values)]
    q = np.asarray(reference_quantiles, dtype=float)
    edges = np.unique(q)
    if values.size == 0 or edges.size < 3:
        return 0.0
    inner = edges[1:-1]
    # Each decile holds 10% of training data; tied quantiles merge deciles into one bin.
    idx = np.searchsorted(q, edges, side="left")
    idx[-1] = len(q) - 1
    expected = np.diff(idx) / (len(q) - 1)
    actual = np.bincount(np.searchsorted(inner, values, side="right"), minlength=len(expected)) / values.size
    eps = 1e-4
    e, a = np.clip(expected, eps, None), np.clip(actual, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def load_log(model_version: str | None = None) -> pd.DataFrame:
    path = prediction_log_path()
    if path is None or not path.exists():
        return pd.DataFrame()
    recs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    df = pd.DataFrame(recs)
    if model_version and len(df):
        df = df[df["model_version"] == model_version]
    return df


def run(min_rows: int = 20) -> dict:
    p = config.params()["monitor"]
    model = RoofModel()
    meta = model.metadata
    logged = load_log(model.version)
    # Batch scoring of the training layers would trivially match the reference; watch API traffic.
    live = logged[logged["source"] != "batch"] if len(logged) else logged
    report: dict = {"checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "model_version": model.version, "logged_predictions": int(len(logged)),
                    "live_predictions": int(len(live)), "drift": {}, "reasons_to_retrain": []}

    if len(live) >= min_rows:
        feats = pd.DataFrame(list(live["features"])).astype(float)
        for col, ref in meta["reference_stats"].items():
            if col not in feats:
                continue
            value = psi(ref["quantiles"], feats[col].to_numpy())
            miss = float(feats[col].isna().mean())
            report["drift"][col] = {"psi": round(value, 4), "missing_rate": round(miss, 4),
                                    "train_missing_rate": ref["missing_rate"]}
        drifted = [c for c, d in report["drift"].items() if d["psi"] > p["drift_psi_threshold"]]
        report["drifted_features"] = drifted
        if drifted:
            report["reasons_to_retrain"].append(f"feature drift (PSI > {p['drift_psi_threshold']}): {drifted}")
        mix = live["pred_class"].value_counts(normalize=True).round(4).to_dict()
        report["predicted_class_mix"] = mix
        report["training_class_mix"] = meta["class_distribution"]
    else:
        report["note"] = f"fewer than {min_rows} live predictions; drift not assessed"

    # New field labels since this model was trained.
    features_csv = config.processed_dir() / config.FEATURES_CSV
    if features_csv.exists():
        labelled = pd.read_csv(features_csv)
        known = set(meta.get("trained_uids", [])) | set(meta.get("test_uids", []))
        new = labelled[~labelled["UID"].isin(known)] if known else labelled.iloc[0:0]
        frac = len(new) / max(1, len(known))
        report["new_labelled_buildings"] = int(len(new))
        report["new_label_fraction"] = round(frac, 4)
        if frac >= p["retrain_new_label_fraction"]:
            report["reasons_to_retrain"].append(
                f"{len(new)} new labelled buildings ({frac:.0%} of the training set)")
        fresh = new
        if len(fresh) >= 10:
            preds = model.predict_frame(fresh)
            from sklearn.metrics import f1_score
            f1 = float(f1_score(fresh["ROOFTYPE"], preds["pred_class"], average="macro"))
            report["macro_f1_on_new_labels"] = round(f1, 4)
            floor = config.params()["promote"]["min_macro_f1"]
            if f1 < floor:
                report["reasons_to_retrain"].append(f"macro-F1 on new labels {f1:.3f} < {floor}")

    report["retrain_recommended"] = bool(report["reasons_to_retrain"])
    config.REPORTS_DIR.mkdir(exist_ok=True)
    (config.REPORTS_DIR / "monitoring.json").write_text(json.dumps(report, indent=2))
    (config.REPORTS_DIR / "monitoring.md").write_text(to_markdown(report), encoding="utf-8")
    return report


def to_markdown(r: dict) -> str:
    lines = [f"# Monitoring report — {r['model_version']}", "",
             f"Checked {r['checked_at']}. Live predictions: {r['live_predictions']} "
             f"(all logged: {r['logged_predictions']}).", "",
             f"**Retrain recommended: {'YES' if r['retrain_recommended'] else 'no'}**", ""]
    lines += [f"- {x}" for x in r["reasons_to_retrain"]] or ["- no trigger fired"]
    if r.get("note"):
        lines += ["", f"_{r['note']}_"]
    if r["drift"]:
        lines += ["", "| Feature | PSI | Missing (live) | Missing (train) |", "|---|---|---|---|"]
        for c, d in sorted(r["drift"].items(), key=lambda kv: -kv[1]["psi"]):
            lines.append(f"| {c} | {d['psi']:.3f} | {d['missing_rate']:.0%} | {d['train_missing_rate']:.0%} |")
    if "new_labelled_buildings" in r:
        lines += ["", f"New labelled buildings since training: {r['new_labelled_buildings']} "
                      f"({r['new_label_fraction']:.0%})."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    out = run()
    log.info("retrain recommended: %s %s", out["retrain_recommended"], out["reasons_to_retrain"])
