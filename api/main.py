"""Roof material classification API.

    uvicorn api.main:app --port 8000

Env: MODEL_DIR (default models/production), PREDICTION_LOG (JSON-lines path,
empty to disable), RASTER_DIR (optional folder of EPSG:32643 orthophotos used
by /predict/geometry to compute colour and texture from the polygon).
"""
from __future__ import annotations

import logging
import math
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.features import MODEL_FEATURES, TEXTURE_FEATURES, complete_features, polygon_features
from src.serving import PredictionLogger, RoofModel, prediction_log_path

log = logging.getLogger("api")
MIN_ELONGATION = 2 * math.sqrt(math.pi)

Positive = Annotated[float, Field(gt=0)]
RgbMean = Annotated[float, Field(gt=0, le=255)]
RgbStd = Annotated[float, Field(ge=0, le=255)]


class ColourFields(BaseModel):
    mean_r: RgbMean
    mean_g: RgbMean
    mean_b: RgbMean
    std_r: RgbStd
    std_g: RgbStd
    std_b: RgbStd


class TextureFields(BaseModel):
    """Optional texture features; the model handles them being absent."""
    hue_sin: float | None = Field(None, ge=-1, le=1)
    hue_cos: float | None = Field(None, ge=-1, le=1)
    sat_mean: float | None = Field(None, ge=0, le=1)
    val_std: float | None = Field(None, ge=0, le=1)
    glcm_contrast: float | None = Field(None, ge=0)
    glcm_homogeneity: float | None = Field(None, ge=0, le=1)
    glcm_energy: float | None = Field(None, ge=0, le=1)
    glcm_entropy: float | None = Field(None, ge=0)
    glcm_correlation: float | None = Field(None, ge=-1, le=1)
    lbp_entropy: float | None = Field(None, ge=0)
    lbp_flat_frac: float | None = Field(None, ge=0, le=1)
    edge_density: float | None = Field(None, ge=0, le=1)
    edge_coherence: float | None = Field(None, ge=0, le=1)


class BuildingFeatures(ColourFields, TextureFields):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": {
        "uid": "IND_B12", "area_m2": 451.66, "perimeter_m": 102.05,
        "mean_r": 85.06, "mean_g": 105.91, "mean_b": 156.42,
        "std_r": 12.03, "std_g": 10.59, "std_b": 9.26}})
    uid: str | None = Field(None, max_length=64)
    area_m2: Positive
    perimeter_m: Positive

    @model_validator(mode="after")
    def _plausible_shape(self):
        # No polygon has a smaller perimeter than the circle of the same area.
        if self.perimeter_m / math.sqrt(self.area_m2) < MIN_ELONGATION - 1e-6:
            raise ValueError("perimeter_m is smaller than any polygon with this area_m2 allows")
        return self


class Prediction(BaseModel):
    uid: str | None
    roof_class: str
    confidence: float
    probabilities: dict[str, float]
    model_version: str


class BatchRequest(BaseModel):
    buildings: list[BuildingFeatures] = Field(min_length=1, max_length=5000)


class GeoJSONPolygon(BaseModel):
    type: Literal["Polygon", "MultiPolygon"]
    coordinates: list[Any]


class GeometryRequest(BaseModel):
    """A GeoJSON Feature in EPSG:32643. Colour fields go in properties unless RASTER_DIR covers it."""
    type: Literal["Feature"] = "Feature"
    geometry: GeoJSONPolygon
    properties: dict[str, Any] = Field(default_factory=dict)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = RoofModel()
    app.state.logger = PredictionLogger(prediction_log_path())
    raster_dir = os.environ.get("RASTER_DIR")
    app.state.rasters = sorted(Path(raster_dir).rglob("*.tif")) if raster_dir else []
    log.info("loaded %s; %d rasters available", app.state.model.version, len(app.state.rasters))
    yield


app = FastAPI(title="Kanjikode roof material classifier", version="1.0.0", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    errors = [{"field": ".".join(str(p) for p in e["loc"] if p != "body"), "message": e["msg"]}
              for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.middleware("http")
async def timing(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - t0) * 1000:.1f}"
    return response


def _score(rows: list[dict], uids: list[str | None], source: str) -> list[Prediction]:
    model: RoofModel = app.state.model
    X = pd.DataFrame([complete_features(r) for r in rows], columns=MODEL_FEATURES)
    preds = model.predict_frame(X)
    app.state.logger.log(X, preds, source=source, uids=uids)
    prob_cols = [f"prob_{c}" for c in model.classes]
    return [Prediction(uid=uid, roof_class=p["pred_class"], confidence=round(float(p["confidence"]), 4),
                       probabilities={c: round(float(p[col]), 4) for c, col in zip(model.classes, prob_cols)},
                       model_version=p["model_version"])
            for uid, (_, p) in zip(uids, preds.iterrows())]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": hasattr(app.state, "model")}


@app.get("/model-info")
def model_info():
    m = app.state.model.metadata
    return {k: m.get(k) for k in ("model_version", "algorithm", "trained_at", "test_macro_f1", "test_accuracy",
                                  "cv_macro_f1_mean", "cv_macro_f1_std", "classes", "features",
                                  "data_md5", "git_commit", "mlflow_run_id")}


@app.post("/predict", response_model=Prediction)
def predict(b: BuildingFeatures):
    return _score([b.model_dump()], [b.uid], "api")[0]


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(req: BatchRequest):
    return _score([b.model_dump() for b in req.buildings], [b.uid for b in req.buildings], "api_batch")


@app.post("/predict/geometry", response_model=Prediction)
def predict_geometry(req: GeometryRequest):
    from shapely.geometry import shape

    try:
        geom = shape(req.geometry.model_dump())
    except Exception as e:
        raise HTTPException(422, detail=[{"field": "geometry", "message": f"invalid polygon: {e}"}])
    if not geom.is_valid or geom.area <= 0:
        raise HTTPException(422, detail=[{"field": "geometry", "message": "polygon is invalid or has zero area"}])
    rec: dict[str, Any] = polygon_features(geom)
    rec.update(_raster_features(geom))
    for k, v in req.properties.items():
        if k in ColourFields.model_fields or k in TextureFields.model_fields:
            rec.setdefault(k, v)
    uid = req.properties.get("uid")
    try:
        b = BuildingFeatures(uid=uid, **{k: v for k, v in rec.items() if k in BuildingFeatures.model_fields})
    except Exception as e:
        missing = [f for f in ColourFields.model_fields if f not in rec]
        msg = (f"no raster covers this polygon; supply {missing} in properties" if missing else str(e))
        raise HTTPException(422, detail=[{"field": "properties", "message": msg}])
    return _score([b.model_dump()], [uid], "api_geometry")[0]


def _raster_features(geom) -> dict:
    if not app.state.rasters:
        return {}
    import rasterio

    from src.features import TEXTURE_GSD_M, colour_stats, texture_features
    from src.raster import covers, read_roof

    for path in app.state.rasters:
        with rasterio.open(path) as src:
            if src.crs is None or src.crs.to_epsg() != 32643 or src.count < 3 or not covers(src, geom):
                continue
            native = read_roof(src, geom)
            if native.mask.sum() == 0:
                continue
            out = colour_stats(native.pixels())
            coarse = read_roof(src, geom, gsd=TEXTURE_GSD_M)
            out.update(texture_features(coarse.rgb, coarse.mask))
            return {k: v for k, v in out.items() if k not in TEXTURE_FEATURES or not math.isnan(v)}
    return {}
