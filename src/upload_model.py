"""Scale-free roof model for user-uploaded images (screenshots, drone photos).

A screenshot has no reliable pixel size, so this model never sees area or
perimeter. It uses shape ratios, colour and texture measured at the image's own
resolution, and is trained on the orthophoto roofs resampled to several ground
resolutions with blur/JPEG/colour augmentation so that it tolerates imagery
from other sources.

    python -m src.upload_model          # train -> models/upload/
"""
from __future__ import annotations

import io
import json
import logging
import warnings
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import rasterio
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold

from src import config, models
from src.features import COLOUR_BASE, COLOUR_DERIVED, TEXTURE_FEATURES, colour_stats, texture_features
from src.raster import read_roof

log = logging.getLogger("upload_model")
warnings.filterwarnings("ignore", message=".*X does not have valid feature names.*")

UPLOAD_FEATURES = ["compactness", "elongation", *COLOUR_BASE, *COLOUR_DERIVED, *TEXTURE_FEATURES]
UPLOAD_DIR = config.MODELS_DIR / "upload"
TRAIN_GSDS = [0.1, 0.15, 0.2, 0.3, 0.4]
AUGMENTS_PER_GSD = 2


def shape_ratios(area: float, perimeter: float) -> dict[str, float]:
    """Compactness and elongation do not depend on the pixel size."""
    return {"compactness": 4 * np.pi * area / perimeter**2, "elongation": perimeter / np.sqrt(area)}


def roof_features(rgb: np.ndarray, mask: np.ndarray, area: float, perimeter: float) -> dict[str, float]:
    """rgb (3,H,W) uint8 at the image's native resolution; mask True inside the roof."""
    out = shape_ratios(area, perimeter)
    out.update(colour_stats(rgb[:, mask]))
    out.update(texture_features(rgb, mask))
    return {k: out[k] for k in UPLOAD_FEATURES}


def degrade(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Screenshot-like degradation: blur, JPEG compression and a mild colour shift."""
    img = Image.fromarray(np.moveaxis(rgb, 0, -1))
    if rng.random() < 0.7:
        img = img.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.3, 1.2))))
    img = ImageEnhance.Brightness(img).enhance(float(rng.uniform(0.88, 1.12)))
    img = ImageEnhance.Contrast(img).enhance(float(rng.uniform(0.88, 1.12)))
    img = ImageEnhance.Color(img).enhance(float(rng.uniform(0.85, 1.15)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(rng.integers(55, 92)))
    return np.moveaxis(np.asarray(Image.open(buf).convert("RGB")), -1, 0)


def build_training_rows(seed: int) -> pd.DataFrame:
    import geopandas as gpd  # training only; the API image does not ship geopandas

    rng = np.random.default_rng(seed)
    gdf = gpd.read_file(config.processed_dir() / config.BUILDINGS_GPKG, layer="buildings")
    layers = config.params()["data"]["layers"]
    rows = []
    for source, cfg in layers.items():
        if not cfg.get("raster"):
            continue
        sub = gdf[gdf["SOURCE"] == source]
        with rasterio.open(config.raw_dir() / cfg["raster"]) as src:
            native = abs(src.transform.a)
            for _, b in sub.iterrows():
                for gsd in TRAIN_GSDS:
                    if gsd < native * 0.9:
                        continue
                    roof = read_roof(src, b.geometry, gsd=gsd)
                    if roof.mask.sum() < 25:
                        continue
                    for k in range(AUGMENTS_PER_GSD + 1):
                        rgb = roof.rgb if k == 0 else degrade(roof.rgb, rng)
                        mask = roof.mask & rgb.any(axis=0)
                        feats = roof_features(rgb, mask, b.geometry.area, b.geometry.length)
                        rows.append({"UID": b.UID, "SOURCE": source, "ROOFTYPE": b.ROOFTYPE,
                                     "gsd": gsd, "augmented": k > 0, **feats})
    return pd.DataFrame(rows)


def evaluate(df: pd.DataFrame, seed: int) -> dict:
    X, y, uid = df[UPLOAD_FEATURES].astype(float), df["ROOFTYPE"], df["UID"]
    preds = pd.Series(index=df.index, dtype=object)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in cv.split(X, y, uid):
        m = models.build("catboost", seed).fit(X.iloc[tr], y.iloc[tr])
        preds.iloc[te] = m.predict(X.iloc[te]).ravel()
    clean = ~df["augmented"]
    by_gsd = {str(g): round(float(f1_score(y[clean & (df.gsd == g)], preds[clean & (df.gsd == g)], average="macro")), 4)
              for g in sorted(df.loc[clean, "gsd"].unique())}
    grouped = {}
    for tr, te in LeaveOneGroupOut().split(X, y, df["SOURCE"]):
        m = models.build("catboost", seed).fit(X.iloc[tr], y.iloc[tr])
        p = m.predict(X.iloc[te]).ravel()
        labels = sorted(set(y.iloc[te]))
        grouped[df["SOURCE"].iloc[te[0]]] = round(float(f1_score(y.iloc[te], p, labels=labels, average="macro", zero_division=0)), 4)
    return {
        "cv": "5-fold StratifiedGroupKFold by building (all resolutions of a roof in one fold)",
        "macro_f1_all": round(float(f1_score(y, preds, average="macro")), 4),
        "accuracy_all": round(float(accuracy_score(y, preds)), 4),
        "macro_f1_degraded": round(float(f1_score(y[~clean], preds[~clean], average="macro")), 4),
        "macro_f1_by_gsd_clean": by_gsd,
        "grouped_by_source_macro_f1": grouped,
    }


def main() -> dict:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    seed = config.params()["seed"]
    df = build_training_rows(seed)
    log.info("%d training rows from %d roofs", len(df), df["UID"].nunique())
    metrics = evaluate(df, seed)
    log.info("CV %s", metrics)

    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.model_selection import GroupKFold

    X, y = df[UPLOAD_FEATURES].astype(float), df["ROOFTYPE"]
    # Calibrate with folds grouped by building so augmented copies never leak.
    folds = list(GroupKFold(n_splits=5).split(X, y, df["UID"]))
    model = CalibratedClassifierCV(models.build("catboost", seed), method="sigmoid", cv=folds).fit(X, y)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, UPLOAD_DIR / "model.joblib")
    meta = {
        "model_version": "roof-upload:1",
        "algorithm": "catboost (sigmoid-calibrated)",
        "features": UPLOAD_FEATURES,
        "classes": [str(c) for c in model.classes_],
        "train_gsds_m": TRAIN_GSDS,
        "training_rows": int(len(df)),
        "training_roofs": int(df["UID"].nunique()),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metrics": metrics,
    }
    (UPLOAD_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    config.REPORTS_DIR.mkdir(exist_ok=True)
    (config.REPORTS_DIR / "upload_model_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    main()
