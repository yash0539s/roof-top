# Kanjikode rooftop material classification

Classifies building roofs in the Kanjikode study area as **METAL, RCC, TILED or ASBESTOS** from digitised
footprints and orthophotos, and serves the model as an API. GIS + MLOps, OELP. The full plan is in
[Kanjikode Rooftop Plan.md](Kanjikode%20Rooftop%20Plan.md).

## Results (model `roof-classifier:1`, CatBoost, calibrated)

| Metric | Value |
|---|---|
| Hold-out test macro-F1 (44 buildings, used once) | **0.939** |
| Hold-out test accuracy | 0.932 |
| Repeated stratified 5×5 CV macro-F1 (train set) | 0.866 ± 0.056 |
| Midterm Random Forest baseline (accuracy) | 0.796 |

Per-class test recall: ASBESTOS 1.00 · METAL 0.94 · RCC 0.93 · TILED 0.83. The confusion matrix is in
`reports/confusion_matrix.png`, the CV comparison of all models in `reports/cv_results.csv`.

**Cross-area check (train on three areas, test on the fourth):**

| Held-out area | Macro-F1 | Note |
|---|---|---|
| commercial | 0.85 | |
| residential | 0.64 | |
| industrial | 0.41 | ASBESTOS only exists here, so it can't be learnt when this area is held out |
| guesthouse | 0.38 | no orthophoto in the repo, so no texture features; colour comes from the ArcMap table |

The headline numbers apply to these four areas. The model does not yet transfer reliably to an unseen area;
the fixes are more asbestos and tiled labels from other areas, and the guesthouse orthophoto.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

dvc repro                          # build → validate → train → promote → predict → maps (~2 min)
pytest                             # 60 tests, synthetic data, no rasters needed
uvicorn api.main:app --port 8000   # http://localhost:8000/docs
```

`dvc repro` needs the raw data in `data/raw/` (`dvc pull` once a remote is configured, see below).

## Pipeline

| Stage | Command | Output |
|---|---|---|
| build | `python -m src.build_dataset` | `data/processed/buildings.gpkg`, `features.csv`, `chips/<UID>.png`, `reports/data_fixes.json` |
| validate | `python -m src.validate` | `reports/validation.json`; exits 1 and stops the pipeline on bad data |
| train | `python -m src.train` | `models/candidate/`, `reports/metrics.json`, MLflow run + registered version (alias `staging`) |
| promote | `python -m src.promote` | copies candidate → `models/production/` if it passes the quality gate |
| predict | `python -m src.predict` | `data/processed/predictions.gpkg` (QGIS style embedded) |
| maps | `python -m src.maps` | `reports/maps/<area>.pdf/.png` |
| monitor | `python -m src.monitor` | `reports/monitoring.md/.json` (run on demand) |

All parameters live in `params.yaml`. `dvc metrics show` prints the current metrics, and `dvc metrics diff`
compares them with the last commit.

### What the build fixes

- One formula for every layer, computed from the polygons: compactness `4πA/P²`, elongation `P/√A`.
  The old industrial ELONGATION was a copy of BRIGHTNESS, and map2 used `P/(2√(πA))`.
- RGB mean/std, brightness and ratios recomputed from the orthophotos for commercial, industrial and residential.
  Before, residential and 52 industrial rows were 0 and commercial had no RGB fields.
- Texture features at a common 0.2 m resolution: GLCM, LBP, edge density and orientation coherence, HSV.
- Unique `UID` (`GH_`, `COM_`, `IND_`, `RES_` + BUILD_ID), cleaned `ROOFTYPE` and `BUILD_TYPE`, relative photo paths.
- 128×128 masked roof chips for every building with imagery.

### Data contract (`src/validate.py`)

Unique UID whose prefix matches SOURCE; valid labels; area and perimeter > 0; compactness in (0, 1];
elongation ≥ 2√π; RGB mean in (0, 255]; ratios sum to 1; no infinite values; valid polygons in EPSG:32643;
photo and chip paths relative and existing. Missing photos (59 of 220, mostly industrial) are a warning;
set `validate.require_photo: true` to make them fatal.

### Model selection

Every model uses the same seed, the same 80/20 split (stratified by class × area) and the same 5×5 CV.
Candidates: logistic regression, Random Forest, LightGBM, CatBoost and Optuna-tuned LightGBM (40 trials),
all with class weights. Models are compared from simplest to most complex, and a more complex one is chosen
only if its mean CV macro-F1 beats the current choice by more than one standard deviation. The chosen model
is sigmoid-calibrated and evaluated once on the hold-out set.

## Model lifecycle

```
train → MLflow (alias staging) → quality gate → models/production + alias production → API loads it
```

- **Gate:** validation passed, test macro-F1 ≥ 0.75 (`promote.min_macro_f1`) and ≥ the current production model.
- **Rollback:** `python -m src.promote --rollback`, then restart the API. The previous model is kept in `models/archive/`.
- **MLflow UI:** `mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000`.
- **Traceability:** every model's `metadata.json` records the data md5, git commit, MLflow run and its
  training/test UIDs. Every prediction records the model version.

## API

| Route | Purpose |
|---|---|
| `POST /predict` | features → class, confidence, probabilities, model version |
| `POST /predict/batch` | up to 5000 buildings |
| `POST /predict/geometry` | GeoJSON polygon (EPSG:32643); colour and texture are read from `RASTER_DIR` if set, otherwise pass colour fields in `properties` |
| `GET /health`, `GET /model-info` | status; version, training date, test macro-F1 |

```bash
curl -X POST localhost:8000/predict -H "content-type: application/json" -d '{"uid":"IND_B12","area_m2":451.66,"perimeter_m":102.05,"mean_r":85.06,"mean_g":105.91,"mean_b":156.42,"std_r":12.03,"std_g":10.59,"std_b":9.26}'
```

Derived features (compactness, elongation, brightness, ratios) are computed by the API. The 13 texture fields
are optional: the model is more accurate with them, and they are computed automatically by `/predict/geometry`
when rasters are available. Invalid input returns 422 with the field name.

Measured locally: p50 45 ms, p95 55 ms per request; 220 buildings in under 0.1 s in batch.

Environment variables: `MODEL_DIR` (default `models/production`), `PREDICTION_LOG` (default
`logs/predictions.jsonl`, empty to disable), `RASTER_DIR`.

### Docker

```bash
docker build -t roof-api .
docker run --rm -p 8000:8000 roof-api
```

The production model is baked into the image. The Dockerfile has not been built or tested yet; CI builds
and smoke-tests it on every push.

## Monitoring and retraining runbook

`python -m src.monitor` compares API inputs with the training distribution (PSI per feature, threshold 0.2),
compares the predicted class mix with training, and counts labelled buildings added since training.
It recommends retraining when any feature drifts, when new labels reach 20% of the known set, or when
macro-F1 on newly labelled buildings drops below 0.75.

To add field-verified labels or corrections:

1. Edit or add the shapefiles under `data/raw/` (keep `BUILD_ID` and `ROOFTYPE`).
2. `dvc add data/raw && git commit -am "new labels"`.
3. `dvc repro`: rebuilds, validates, retrains, and promotes only if the gate passes.
4. `dvc push && git push`, then rebuild and restart the API container.

## QGIS

Open `data/processed/predictions.gpkg`. The embedded default style fills roofs by predicted class and outlines
misclassified buildings in red; it is also in `qgis/predictions.qml`. The fields are `pred_class`, `confidence`,
`prob_*`, `correct`, `split` (train/test) and `model_version`.

## Data versioning (DVC)

`data/raw` (~1.1 GB of shapefiles, TIFs and photos) is tracked by `data/raw.dvc`, not Git. To share it:

```bash
dvc remote add -d gdrive gdrive://<google-drive-folder-id>
dvc push
```

For CI, add the service-account JSON as the repo secret `GDRIVE_CREDENTIALS_DATA`; the `data` job then runs
`dvc pull`, validation and a quick training run.

## Layout

```
src/        build_dataset, features (all formulas), raster, validate, models, train,
            promote, serving, predict, maps, monitor
api/        FastAPI app
tests/      features, data contract, training/gate/monitor, API
qgis/       layer style
models/     production/ (in Git), candidate/ (DVC), archive/ (local)
reports/    metrics, validation, data fixes, confusion matrix, maps, monitoring
data/       raw/ (DVC), processed/ (DVC pipeline outputs)
```

## Known limitations / next steps

- Cross-area generalisation is weak (see the table above). Collect asbestos roofs outside the industrial area and more tiled roofs.
- The guesthouse orthophoto is not in the repo. Adding it gives those 97 buildings texture features and raster-based colour.
- The DINOv2 image model and the stacked ensemble from the plan are not implemented. The chips in
  `data/processed/chips/` are ready for it.
- The Docker image has not been built locally yet.
