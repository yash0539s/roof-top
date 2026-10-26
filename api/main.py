"""Roof material classification API.

    uvicorn api.main:app --port 8000

Env: MODEL_DIR (default models/production), PREDICTION_LOG (JSON-lines path,
empty to disable), RASTER_DIR (optional folder of EPSG:32643 orthophotos used
by /predict/geometry to compute colour and texture from the polygon),
PREDICTIONS_GEOJSON and METRICS_PATH (data for the web frontend at /).
"""
from __future__ import annotations

import json
import logging
import math
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src import config
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
    app.state.upload_model = _load_upload_model()
    app.state.logger = PredictionLogger(prediction_log_path())
    raster_dir = os.environ.get("RASTER_DIR")
    app.state.rasters = sorted(Path(raster_dir).rglob("*.tif")) if raster_dir else []
    app.state.media = _media_index()
    log.info("loaded %s; %d rasters available", app.state.model.version, len(app.state.rasters))
    yield


def _load_upload_model() -> RoofModel | None:
    """The size-free model for uploaded images without a known scale (models/upload)."""
    from src.upload_model import UPLOAD_DIR, UPLOAD_FEATURES

    path = Path(os.environ.get("UPLOAD_MODEL_DIR", UPLOAD_DIR))
    try:
        return RoofModel(path, expected_features=UPLOAD_FEATURES)
    except FileNotFoundError:
        log.warning("no upload model at %s; image uploads without a GeoTIFF scale are disabled", path)
        return None


def _predictions_path() -> Path:
    return Path(os.environ.get("PREDICTIONS_GEOJSON", config.processed_dir() / "predictions.geojson"))


def _metrics_path() -> Path:
    return Path(os.environ.get("METRICS_PATH", config.REPORTS_DIR / "metrics.json"))


def _media_index() -> dict[str, dict[str, Path]]:
    """UID -> {"photo": path, "chip": path} for files that exist, from features.csv when present."""
    csv = config.processed_dir() / config.FEATURES_CSV
    if not csv.exists():
        return {}
    df = pd.read_csv(csv, usecols=["UID", "photo", "chip"])
    index: dict[str, dict[str, Path]] = {}
    for row in df.itertuples():
        entry = {}
        for kind in ("photo", "chip"):
            rel = getattr(row, kind)
            if isinstance(rel, str) and (config.ROOT / rel).is_file():
                entry[kind] = (config.ROOT / rel).resolve()
        index[row.UID] = entry
    return index


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


@app.get("/api/config")
def frontend_config():
    return {"raster_lookup": bool(app.state.rasters), "buildings_available": _predictions_path().is_file(),
            "metrics_available": _metrics_path().is_file(), "classes": app.state.model.classes,
            "texture_features": TEXTURE_FEATURES}


@app.get("/api/buildings")
def buildings():
    path = _predictions_path()
    if not path.is_file():
        raise HTTPException(404, "predictions not generated; run `python -m src.predict`")
    return FileResponse(path, media_type="application/geo+json")


@app.get("/api/metrics")
def metrics():
    path = _metrics_path()
    if not path.is_file():
        raise HTTPException(404, "metrics not available; run `python -m src.train`")
    return json.loads(path.read_text())


@app.get("/api/buildings/{uid}/{kind}")
def building_media(uid: str, kind: Literal["photo", "chip"]):
    # Only files listed in features.csv are served, never arbitrary paths.
    path = app.state.media.get(uid, {}).get(kind)
    if path is None:
        raise HTTPException(404, f"no {kind} for {uid}")
    return FileResponse(path, headers={"Cache-Control": "public, max-age=86400"})


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


MAX_UPLOAD_BYTES = int(float(os.environ.get("MAX_UPLOAD_MB", "500")) * 1024 * 1024)


class RoofPrediction(Prediction):
    polygon: list[list[float]] = Field(description="roof outline [[col, row], ...] in image pixels")
    latitude: float
    longitude: float
    area_m2: float | None = Field(None, description="only when the image scale is known (GeoTIFF)")
    perimeter_m: float | None = None
    mean_rgb: list[float]
    size_known: bool = Field(description="true: main model with roof size; false: size-free model")
    detection_score: float
    roof_image: str = Field(description="PNG data URL of the roof, background masked out")


class ImagePrediction(RoofPrediction):
    m_per_px: float | None
    warnings: list[str]


class AutoImagePrediction(BaseModel):
    roofs: list[RoofPrediction]
    counts: dict[str, int]
    latitude: float
    longitude: float
    image_width: int
    image_height: int
    georeferenced: bool
    m_per_px: float | None
    detector: str
    labelled_image: str = Field(description="PNG data URL: the image with every roof outlined and labelled")
    warnings: list[str]
    seconds: float


CLASS_COLOURS = {"ASBESTOS": "#8e5cc0", "METAL": "#1f77b4", "RCC": "#8a8f98", "TILED": "#d6604d"}


def _score_roofs(roofs, uid_prefix: str | None, source: str) -> list[RoofPrediction]:
    """Classify roofs from src.image_upload with the main or the size-free model, as each requires."""
    out: list[RoofPrediction | None] = [None] * len(roofs)
    for kind in ("main", "upload"):
        idx = [i for i, r in enumerate(roofs) if r.model == kind]
        if not idx:
            continue
        model: RoofModel | None = app.state.model if kind == "main" else app.state.upload_model
        if model is None:
            raise HTTPException(503, "the size-free upload model is not installed; run `python -m src.upload_model`")
        rows = [complete_features(roofs[i].features) if kind == "main" else roofs[i].features for i in idx]
        X = pd.DataFrame(rows, columns=model.features).astype(float)
        uids = [f"{uid_prefix}_{i + 1}" if uid_prefix and len(roofs) > 1 else uid_prefix for i in idx]
        preds = model.predict_frame(X)
        app.state.logger.log(X, preds, source=f"{source}_{kind}", uids=uids)
        for i, uid, (_, p) in zip(idx, uids, preds.iterrows()):
            r = roofs[i]
            f = r.features
            out[i] = RoofPrediction(
                uid=uid, roof_class=p["pred_class"], confidence=round(float(p["confidence"]), 4),
                probabilities={c: round(float(p[f"prob_{c}"]), 4) for c in model.classes},
                model_version=p["model_version"], polygon=r.polygon,
                latitude=round(r.latitude, 7), longitude=round(r.longitude, 7),
                area_m2=None if r.area_m2 is None else round(r.area_m2, 2),
                perimeter_m=None if r.perimeter_m is None else round(r.perimeter_m, 2),
                mean_rgb=[round(f[k], 1) for k in ("mean_r", "mean_g", "mean_b")],
                size_known=kind == "main", detection_score=round(r.score, 3), roof_image=r.roof_image)
    return out


def _err(field: str, message: str, status: int = 422) -> HTTPException:
    return HTTPException(status, detail=[{"field": field, "message": message}])


async def _read_upload(file: UploadFile) -> bytes:
    data = bytearray()
    while chunk := await file.read(1 << 20):
        data += chunk
        if len(data) > MAX_UPLOAD_BYTES:
            raise _err("file", f"file is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB", 413)
    return bytes(data)


@app.post("/api/image/inspect")
async def image_inspect(file: UploadFile = File(...)):
    """Preview and metadata (georeferencing, pixel size, centre) of an uploaded image."""
    from src.image_upload import ImageError, inspect

    data = await _read_upload(file)
    try:
        info, preview = inspect(data)
    except ImageError as e:
        raise _err(e.field, str(e))
    return {**info.__dict__, "filename": file.filename, "preview": preview}


@app.post("/predict/image", response_model=ImagePrediction)
async def predict_image(
    file: UploadFile = File(...),
    outline: str = Form(..., description="JSON [[col, row], ...] in original image pixels"),
    latitude: float | None = Form(None),
    longitude: float | None = Form(None),
    m_per_px: float | None = Form(None, gt=0),
    roof_area_m2: float | None = Form(None, gt=0),
    uid: str | None = Form(None, max_length=64),
):
    """Classify the roof outlined on an uploaded satellite/drone image."""
    from src.image_upload import ImageError, outline_features

    try:
        points = json.loads(outline)
    except json.JSONDecodeError:
        raise _err("outline", "outline must be JSON")
    data = await _read_upload(file)
    try:
        res = outline_features(data, points, latitude, longitude, m_per_px, roof_area_m2)
    except ImageError as e:
        raise _err(e.field, str(e))
    pred = _score_roofs([res], uid, "api_image")[0]
    return ImagePrediction(**pred.model_dump(), m_per_px=None if res.m_per_px is None else round(res.m_per_px, 5),
                           warnings=res.warnings)


@app.post("/predict/image/auto", response_model=AutoImagePrediction)
async def predict_image_auto(
    file: UploadFile = File(..., description="satellite/drone image: JPG, PNG, WebP or GeoTIFF"),
    latitude: float | None = Form(None, description="required unless the image is a GeoTIFF"),
    longitude: float | None = Form(None, description="required unless the image is a GeoTIFF"),
    uid: str | None = Form(None, max_length=48, description="name prefix for the roofs found"),
):
    """Find every roof in the image and classify each one. Only the image and its location are needed."""
    from starlette.concurrency import run_in_threadpool

    from src.image_upload import ImageError, auto_detect, labelled_image

    t0 = time.perf_counter()
    data = await _read_upload(file)
    try:
        res = await run_in_threadpool(auto_detect, data, latitude, longitude)
    except ImageError as e:
        raise _err(e.field, str(e))
    preds = _score_roofs(res.roofs, uid or "ROOF", "api_image_auto")
    counts = {c: 0 for c in CLASS_COLOURS}
    for p in preds:
        counts[p.roof_class] = counts.get(p.roof_class, 0) + 1
    labelled = labelled_image(res.preview, res.preview_scale,
                              [{"polygon": p.polygon, "roof_class": p.roof_class, "confidence": p.confidence}
                               for p in preds], CLASS_COLOURS)
    return AutoImagePrediction(
        roofs=preds, counts=counts, latitude=round(res.latitude, 7), longitude=round(res.longitude, 7),
        image_width=res.width, image_height=res.height, georeferenced=res.georeferenced,
        m_per_px=None if res.m_per_px is None else round(res.m_per_px, 5), detector=res.detector,
        labelled_image=labelled, warnings=res.warnings, seconds=round(time.perf_counter() - t0, 2))


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


# The web frontend; mounted last so the API routes above take precedence.
WEB_DIR = Path(__file__).resolve().parents[1] / "web"
if WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
