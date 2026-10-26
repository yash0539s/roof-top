GIS + MLOps · OELP · Kanjikode study area

# Rooftop material classification: end-to-end plan

Built from a review of `D:\GIS_OELP_PROJECT`: the four shapefiles and their attribute tables, the ArcGIS processing history, the photos, the rasters and the midterm PPT.

Where things stand

DoneThe plan below is complete, from setting up the repo to monitoring the deployed model, and includes a full system design. DoneGIS work: 220 digitised buildings across 4 layers, roof labels checked in the field, photos, orthophotos. Needs fixingAttribute data: wrong elongation values, missing RGB values and IDs that repeat across layers. See the data problems below. Not startedCode. The folder has no scripts, notebooks, CSV, saved model, API or Docker setup. The ML results in the PPT came from work that isn't in this folder.

What is in the folder

## Data inventory

| Layer | Buildings | Roof labels | RGB features | Raster |
| --- | --- | --- | --- | --- |
| `map1/FINAL_UTM` guest house | 97 | 45 RCC · 44 Metal · 8 Tiled | Present | Orthophoto JPG |
| `map2/commercial` | 32 | 20 RCC · 8 Metal · 4 Tiled | No fields | 129 MB TIF |
| `map2/industrial` | 73 | 29 Asbestos · 23 Metal · 21 Tiled | 52 rows are 0 | \~500 MB TIFs |
| `map2/residential` | 18 | 7 RCC · 9 Metal · 2 Tiled | All 18 rows are 0 | 374 MB TIF |
| **Total** | **220** | CRS WGS 84 / UTM 43N · ArcMap 10.8 `.mxd` projects · total size about 1.1 GB |  |  |

Metal

84 · 38.2%

RCC

72 · 32.7%

Tiled

35 · 15.9%

Asbestos

29 · 13.2%

Fix before training

## Data problems

Gives wrong model resultsData hygiene

- ### Industrial ELONGATION is wrong in all 73 rows

  A Calculate Field run put the brightness formula into ELONGATION, so ELONGATION equals BRIGHTNESS in every row (B1: 200.128).

  Fix: recalculate the field in code (Phase 1).
- ### The two maps use different elongation formulas

  map1 uses P/√A (values around 4.3) and map2 uses P/(2√(πA)) (around 1.15). Compactness uses π = 3.1416 in one map and 3.14 in the other.

  Fix: use one formula for every layer.
- ### About 100 of 220 buildings have no real RGB values

  All residential rows and 52 industrial rows are 0. Commercial has no RGB fields. A 0–255 range check doesn't catch zeros, so the midterm validation passed.

  Fix: run zonal statistics on the TIFs and add a test that RGB values are greater than 0.
- ### Asbestos only appears in the industrial layer

  The model may learn the area instead of the material.

  Fix: validate with groups by source and collect asbestos roofs from other areas.
- ### BUILD_IDs repeat across layers

  B1 exists in all four layers.

  Fix: add a prefix: `GH_`, `COM_`, `IND_`, `RES_`.
- ### Broken photo links and messy labels

  Photo paths point to `C:\Users\gouth\…`. map2 photos have names like `B10 (2).png`. BUILD_TYPE contains `R?`, `DOUBT` and `R*MULTI)`, and 52 industrial rows are blank.

  Fix: use relative paths named after the UID and a fixed list of BUILD_TYPE values.

The pipeline

## From shapefile to deployed model

1. **raw**4 shapefiles + TIFs
2. **build**Merge, clean and recalculate features
3. **validate**Data tests
4. **train**Models logged in MLflow
5. **evaluate**Macro-F1, confusion matrix, grouped CV
6. **predict**predictions.gpkg
7. **map**QGIS layouts
8. **serve**FastAPI in Docker
9. **monitor**Drift checks and retraining

System design

## Architecture

The system has six parts. Data is versioned with DVC. A Python pipeline builds the features. Training runs are tracked in MLflow, and the approved model is served by an API in Docker. Predictions go back to QGIS, and a monitoring step decides when to retrain. GitHub Actions tests everything on every push.

### Components

| Component | Tool | Job | Input → output |
| --- | --- | --- | --- |
| Data store | DVC + Google Drive remote | Version the shapefiles, TIFs and photos | raw files → hashed versions |
| Feature pipeline | geopandas, rasterio, rasterstats | Merge layers, clean labels, compute features | shapefiles + TIFs → `buildings.gpkg` |
| Validation | pandera, pytest | Stop the pipeline on bad data | gpkg → pass / fail report |
| Training | scikit-learn, CatBoost, PyTorch (DINOv2), Optuna | Train and compare models | features → model + metrics |
| Tracking and registry | MLflow | Record runs and promote models | runs → versioned model |
| Serving | FastAPI, Docker | Predict roof class on request | JSON features → class + probabilities |
| Batch prediction | predict.py | Score every building in a layer | gpkg → `predictions.gpkg` |
| Maps | QGIS | Show and check predictions on the map | predictions → map layouts |
| Monitoring | Evidently, logs | Detect drift and decide when to retrain | logs → drift report |
| CI | GitHub Actions | Test every change automatically | push → green or red build |

### Data contract

Every building in `buildings.gpkg` has these fields. The validation step checks the rules on every run.

| Field | Type | Rule |
| --- | --- | --- |
| `UID` | text | Unique, for example `IND_B12` |
| `SOURCE` | text | guesthouse, commercial, industrial or residential |
| `ROOFTYPE` | text (label) | METAL, RCC, TILED or ASBESTOS |
| `area_m2`, `perimeter_m` | float | > 0 |
| `compactness` | float | 4πA/P², between 0 and 1 |
| `elongation` | float | P/√A, ≥ 3.54 (the value for a circle) |
| `mean_r/g/b`, `std_r/g/b` | float | mean 0 \< x ≤ 255, std ≥ 0 |
| `brightness`, `r/g/b_ratio` | float | ratios between 0 and 1 and sum to 1 |
| `photo` | path | Relative path, file exists |
| `geometry` | polygon | Valid, EPSG:32643 |

### API specification

Request · POST /predict

```
{
  "uid": "IND_B12",
  "area_m2": 451.66,
  "perimeter_m": 102.05,
  "mean_r": 85.06, "mean_g": 105.91,
  "mean_b": 156.42,
  "std_r": 12.03, "std_g": 10.59,
  "std_b": 9.26
}
# the API computes the derived features
```

Response · 200

```
{
  "uid": "IND_B12",
  "roof_class": "METAL",
  "confidence": 0.81,
  "probabilities": {
    "METAL": 0.81, "ASBESTOS": 0.12,
    "RCC": 0.04, "TILED": 0.03
  },
  "model_version": "roof-rf:3"
}
```

Other routes: `GET /health` returns status. `GET /model-info` returns the model version, training date and test macro-F1. Bad input returns 422 with the name of the invalid field.

### Model lifecycle

Train*→*Log in MLflow*→*Register as Staging*→*Quality gate*→*Production*→*API loads it

The quality gate promotes a model only if all data tests pass and its macro-F1 is at least the current Production model's. To roll back, switch the Production tag to the previous version and restart the container.

### Requirements

| Area | Target |
| --- | --- |
| Model quality | Macro-F1 ≥ 0.75 on the test set, per-class recall reported, grouped CV results shown |
| Reproducibility | `dvc repro` on a clean machine gives the same metrics |
| Speed | Under 200 ms per building on the API; a full layer in under a minute on a laptop CPU |
| Scale | 220 buildings now; the design works unchanged for a few thousand |
| Hardware | One laptop, no GPU, no paid cloud services |
| Traceability | Every prediction records the model version and every model records its data version |

### Deployment view

| Where | What runs there |
| --- | --- |
| Laptop | DVC pipeline, MLflow server on `localhost:5000`, QGIS |
| Docker container | FastAPI on port 8000 with the Production model inside |
| GitHub | Code, CI workflow, issue tracking |
| Google Drive | DVC remote for the 1.1 GB of rasters and photos |

### Design decisions

| Decision | Reason |
| --- | --- |
| Compute features in Python, not ArcMap | ArcMap edits caused the elongation bug and can't be rerun automatically |
| One object per building | Labels come from field checks per building, and the maps join on UID |
| Tree models such as Random Forest and XGBoost | Work well on small tabular data, need no GPU, and show feature importance |
| Cross-validation grouped by SOURCE | Shows whether the model learned the area instead of the material, which matters for asbestos |
| DVC for data, Git for code | The rasters are too large for Git, and DVC links each model to its data version |
| MLflow running locally | Free, standard tool, and enough for one team |

Getting the best accuracy

## Best model to use

Your current best is Random Forest at 79.55% accuracy and 79.40% F1, trained on geometry and mean colour only. Colour alone can't separate grey RCC from grey asbestos, or tiled roofs from rusty metal. What separates them is **texture**: corrugation lines on metal and asbestos sheets, the repeating pattern of tiles, flat smooth concrete. The biggest gains come from giving the model texture, either as extra features or by letting it look at the roof image itself.

### Recommended final model

A **stacked ensemble** of two models whose predictions are combined:

1. **Tabular model:** a tuned gradient-boosted model (CatBoost or LightGBM) on geometry, colour and the new texture features.
2. **Image model:** a pretrained vision model (DINOv2 or EfficientNet-B0) that looks at a cropped image chip of each roof.
3. A simple logistic regression combines the two sets of class probabilities into the final answer.

This should beat Random Forest. Measure the gain with the evaluation rules below; don't promise a number before you have one.

### All model options compared

| Model | Input | Why use it | Fit for 220 buildings | Effort |
| --- | --- | --- | --- | --- |
| Logistic Regression | Tabular | Simple baseline; midterm result 75.00% | Good | Low |
| KNN, SVM | Tabular | Midterm results 61.36% and 68.18%; drop or keep as reference | Weak | Low |
| Random Forest | Tabular | Current best, 79.55%; robust, shows feature importance | Good | Low |
| CatBoost / LightGBM / XGBoost | Tabular + texture | Usually the most accurate on tabular data; CatBoost handles small data well with little tuning | Good | Low |
| DINOv2 embeddings + logistic regression | Roof image chip | Pretrained features capture texture; only a small classifier is trained, so few labels are enough | Good | Medium |
| Fine-tuned CNN (EfficientNet-B0, ResNet-18) | Roof image chip | Learns roof texture directly; needs heavy augmentation with this little data | Risky | Medium |
| Stacked ensemble (tabular + image) | Both | Combines shape and colour with texture; the two models make different mistakes | Good | Medium |
| Training a CNN from scratch | Roof image chip | Needs thousands of labelled roofs | No | High |
| General chat-model visual classification | Image | Tried at midterm; roof material answers were unreliable | No | Low |

### Ways to raise accuracy, in order of impact

| Step | What to do | Why it helps |
| --- | --- | --- |
| Fix the data | Phase 1: correct elongation, fill the \~100 missing RGB rows, use one formula per feature | The current model was trained partly on wrong and zero values |
| Add texture features | GLCM contrast, homogeneity, energy and entropy; LBP histogram; edge density and main edge direction; HSV hue and saturation | Corrugated sheets give strong parallel edges, tiles give a repeating pattern, RCC is smooth |
| Add image chips | Crop each roof from the orthophoto with its polygon as a mask, resize to 128×128 | Lets the image model see what the numbers miss |
| Augment images | Rotate 90°, flip, small brightness changes | Roofs look the same from any direction, so this multiplies the training data safely |
| Handle class imbalance | Class weights; try SMOTE only inside the training folds | Stops the model ignoring Tiled and Asbestos |
| Tune hyperparameters | Optuna, 50–100 trials, optimising macro-F1 in cross-validation | Default settings are rarely best |
| Select features | Drop highly correlated ones (brightness vs mean RGB, area vs perimeter); check permutation importance | Less noise with small data |
| Calibrate probabilities | Isotonic or Platt calibration on the final model | Confidence values become trustworthy for the map and the API |
| Collect more labels | Especially Asbestos outside industrial areas, and Tiled | More data helps more than any model change |

### How to compare models fairly

| Rule | Detail |
| --- | --- |
| Main metric | Macro-F1, with per-class recall and a confusion matrix. Accuracy alone hides poor Asbestos and Tiled results. |
| Validation | Repeated stratified 5-fold CV (5 repeats) for model choice; report mean ± standard deviation |
| Spatial check | Cross-validation grouped by SOURCE, reported separately |
| Test set | Keep a 20% hold-out untouched until the final model is chosen; use it once |
| No leakage | Scaling, SMOTE, feature selection and tuning all happen inside the training folds |
| Same protocol | Every model in MLflow uses the same splits and seed, so results are comparable |

### Experiment order

RF on fixed data*→*+ texture features*→*CatBoost / LightGBM tuned*→*DINOv2 image model*→*Stacked ensemble

Log each step in MLflow. Keep a step only if macro-F1 improves by more than the CV standard deviation. If the ensemble doesn't beat the tuned tabular model, ship the simpler one.

Step by step

## Phases

Core: do these first, in orderMLOps layer

### Phase 0 · Project setup

\~1 day

- Create a Git repo using the folder layout below. Keep the current folders unchanged in `data/raw`.
- Track the 1.1 GB of TIFs with DVC, not Git.
- Write `requirements.txt`: geopandas, rasterio, rasterstats, scikit-learn, xgboost, mlflow, pandera, pytest, fastapi, uvicorn.

Output: repo, environment, `dvc.yaml`

### Phase 1 · Rebuild the dataset in Python

2–3 days

- `src/build_dataset.py` loads the 4 layers and adds `SOURCE` and a unique `UID`. It cleans ROOFTYPE to METAL, RCC, TILED or ASBESTOS.
- Recalculate geometry from the polygons: area, perimeter, compactness 4πA/P² and elongation P/√A, with the same formula for every layer.
- Recalculate RGB from the TIFs with `rasterstats.zonal_stats`: mean and std for R, G and B, plus brightness and the three ratios. First check each TIF's CRS and band order.
- Stop using ArcMap Calculate Field for features.

Output: `data/processed/buildings.gpkg`, `features.csv`

### Phase 2 · Automated data validation

\~1 day

- Checks: unique UIDs, valid labels, 0 \< RGB ≤ 255, area > 0, compactness between 0 and 1, no NaN or infinite values, and a photo for every row.
- Checks run as a step in the pipeline. The pipeline stops if a check fails.

Output: `tests/test_data.py`

### Phase 3 · Training and evaluation

2–3 days

- Stratified 80/20 split, plus cross-validation grouped by `SOURCE` as the spatial check.
- Add texture features (GLCM, LBP, edges, HSV) and 128×128 roof image chips.
- Follow the experiment order above: Random Forest, then tuned CatBoost/LightGBM, then the DINOv2 image model, then the stacked ensemble. Use class weights throughout.
- Report per-class precision, recall and F1, macro-F1 and a confusion matrix. This was the missing item in the midterm.
- Log every run in MLflow and register the best model.
- Compare with the 79.55% Random Forest baseline. Expect the numbers to change because some features were wrong.

Output: `src/train.py`, MLflow runs, `models/`

### Phase 4 · Predict and map back

1–2 days

- `src/predict.py` writes the predicted class and confidence for each building.
- In QGIS, style by predicted class and highlight misclassified buildings. Export a print layout for each area.

Output: `predictions.gpkg`, map PDFs

### Phase 5 · Package and deploy

\~2 days

- FastAPI `POST /predict` takes features or a GeoJSON polygon and returns the class and probabilities.
- Write a `Dockerfile`. One `docker run` serves the API.
- GitHub Actions runs the data tests, a small training run and the API test on every push.

Output: `api/`, Docker image, CI workflow

### Phase 6 · Monitoring and retraining

\~1 day

- Log predictions and track class distribution and feature drift, for example with Evidently.
- Add field-verified corrections to `data/raw`, then run `dvc repro` to rebuild, retrain and re-evaluate.
- Retrain when macro-F1 drops or about 20% new labelled buildings have been added.

Output: monitoring report, retraining runbook

### Phase 7 · Final deliverables

2–3 days

- Report covering method, data fixes, metrics, maps, limitations and civil-engineering use.
- Final slides.
- Live demo: `dvc repro`, then MLflow, the API in Docker and the QGIS map.
- Next steps: asbestos samples outside industrial areas, higher-resolution imagery, texture features (GLCM).

Target structure

## Repository layout

```
GIS_OELP_PROJECT/
├── data/
│   ├── raw/            # map1/, map2/ unchanged (DVC-tracked)
│   └── processed/      # buildings.gpkg, features.csv
├── src/
│   ├── build_dataset.py
│   ├── features.py     # one formula for each feature
│   ├── train.py
│   └── predict.py
├── tests/
│   ├── test_data.py
│   └── test_api.py
├── api/main.py         # FastAPI
├── qgis/               # .qgz project + styles
├── models/
├── notebooks/eda.ipynb
├── dvc.yaml            # build → validate → train → evaluate
├── Dockerfile
├── requirements.txt
└── .github/workflows/ci.yml
```

Total time: about 3 weeks. Phases 1–3 are the core. Phases 4–6 add the MLOps layer. Next step: write `src/build_dataset.py` (Phase 1).