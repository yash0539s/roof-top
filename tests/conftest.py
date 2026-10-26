from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import config, models  # noqa: E402
from src.features import MODEL_FEATURES, TEXTURE_FEATURES, colour_derived, geometry_features  # noqa: E402

# Rough class colour centres so the synthetic task is learnable.
CENTRES = {"ASBESTOS": (150, 150, 145), "METAL": (110, 130, 170), "RCC": (200, 195, 190), "TILED": (170, 90, 70)}
PREFIX = {"guesthouse": "GH", "commercial": "COM", "industrial": "IND", "residential": "RES"}


def synthetic_features(n_per_class: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    i = 0
    for cls, (r, g, b) in CENTRES.items():
        for _ in range(n_per_class):
            i += 1
            source = config.SOURCES[i % 4]
            area = float(rng.uniform(40, 800))
            side = np.sqrt(area)
            perim = 4 * side * rng.uniform(1.0, 1.3)
            row = {"UID": f"{PREFIX[source]}_B{i}", "SOURCE": source, "ROOFTYPE": cls,
                   **geometry_features(area, perim)}
            mean = np.clip(np.array([r, g, b]) + rng.normal(0, 12, 3), 5, 250)
            row.update(dict(zip(["mean_r", "mean_g", "mean_b"], mean)))
            row.update(dict(zip(["std_r", "std_g", "std_b"], rng.uniform(5, 40, 3))))
            row.update(colour_derived(*mean))
            for t in TEXTURE_FEATURES:
                row[t] = np.nan if source == "guesthouse" else float(rng.uniform(0, 1))
            row["colour_source"] = "attribute" if source == "guesthouse" else "raster"
            row["photo"] = None
            row["chip"] = None
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def synthetic_df() -> pd.DataFrame:
    return synthetic_features()


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory, synthetic_df) -> Path:
    """A small packaged model with the same layout as models/production."""
    d = tmp_path_factory.mktemp("model")
    X, y = synthetic_df[MODEL_FEATURES].astype(float), synthetic_df["ROOFTYPE"]
    est = models.build("logreg", seed=0).fit(X, y)
    joblib.dump(est, d / "model.joblib")
    meta = {
        "model_version": "roof-classifier:test", "registered_version": None, "algorithm": "logreg",
        "classes": list(est.classes_), "features": MODEL_FEATURES, "trained_at": "2026-01-01T00:00:00+00:00",
        "test_macro_f1": 0.9, "test_accuracy": 0.9, "cv_macro_f1_mean": 0.9, "cv_macro_f1_std": 0.02,
        "data_md5": "x", "git_commit": None, "mlflow_run_id": None,
        "reference_stats": {c: {"quantiles": X[c].quantile(np.linspace(0, 1, 11)).tolist(),
                                "missing_rate": float(X[c].isna().mean())} for c in MODEL_FEATURES},
        "class_distribution": y.value_counts(normalize=True).to_dict(),
        "trained_uids": synthetic_df["UID"].tolist(), "test_uids": [],
    }
    (d / "metadata.json").write_text(json.dumps(meta))
    return d
