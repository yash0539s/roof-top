"""Load a packaged model (model.joblib + metadata.json) and score feature rows.

Shared by the batch predictor and the API so both produce identical outputs.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import config
from src.features import MODEL_FEATURES


def default_model_dir() -> Path:
    return Path(os.environ.get("MODEL_DIR", config.PRODUCTION_DIR))


class RoofModel:
    def __init__(self, model_dir: Path | str | None = None):
        self.model_dir = Path(model_dir or default_model_dir())
        model_file = self.model_dir / "model.joblib"
        if not model_file.is_file():
            raise FileNotFoundError(f"no model at {model_file}; train and promote one first")
        self.estimator = joblib.load(model_file)
        self.metadata: dict = json.loads((self.model_dir / "metadata.json").read_text())
        self.features: list[str] = self.metadata["features"]
        if self.features != MODEL_FEATURES:
            raise ValueError("model was trained on a different feature list than src.features defines")
        self.classes: list[str] = [str(c) for c in self.estimator.classes_]

    @property
    def version(self) -> str:
        return self.metadata["model_version"]

    def predict_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        proba = self.estimator.predict_proba(X[self.features].astype(float))
        best = proba.argmax(axis=1)
        out = pd.DataFrame(proba, columns=[f"prob_{c}" for c in self.classes], index=X.index)
        out.insert(0, "pred_class", np.asarray(self.classes)[best])
        out.insert(1, "confidence", proba.max(axis=1))
        out["model_version"] = self.version
        return out


class PredictionLogger:
    """Append-only JSON-lines log of inputs and outputs, read by src.monitor."""

    def __init__(self, path: Path | str | None):
        self.path = Path(path) if path else None
        self._lock = threading.Lock()

    def log(self, features: pd.DataFrame, preds: pd.DataFrame, source: str, uids=None) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        uids = list(uids) if uids is not None else [None] * len(features)
        lines = []
        for uid, (_, f), (_, p) in zip(uids, features.iterrows(), preds.iterrows()):
            rec = {"ts": ts, "source": source, "uid": uid, "model_version": p["model_version"],
                   "pred_class": p["pred_class"], "confidence": round(float(p["confidence"]), 4),
                   "features": {k: (None if pd.isna(v) else round(float(v), 6)) for k, v in f.items()}}
            lines.append(json.dumps(rec))
        with self._lock, open(self.path, "a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")


def prediction_log_path() -> Path | None:
    value = os.environ.get("PREDICTION_LOG", str(config.ROOT / "logs" / "predictions.jsonl"))
    return Path(value) if value else None
