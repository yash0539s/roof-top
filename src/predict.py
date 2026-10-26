"""Phase 4: score every building in a layer and write predictions.gpkg.

    python -m src.predict                                  # the processed dataset
    python -m src.predict --input new.shp --raster ortho.tif --source newarea --output new_pred.gpkg

The output carries pred_class, confidence, per-class probabilities and the
model version; when field labels exist it also flags misclassified buildings.
The QGIS style in qgis/predictions.qml is embedded as the layer default.
"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import pandas as pd

from src import config
from src.features import MODEL_FEATURES
from src.serving import PredictionLogger, RoofModel, prediction_log_path

log = logging.getLogger("predict")
STYLE_FILE = config.ROOT / "qgis" / "predictions.qml"


def features_for_new_layer(path: Path, raster: Path, source: str) -> gpd.GeoDataFrame:
    """Compute features for an unlabelled layer with the same code as the training build."""
    from src.build_dataset import add_features

    epsg = config.params()["data"]["crs_epsg"]
    gdf = gpd.read_file(path)
    gdf = gdf.to_crs(epsg=epsg) if gdf.crs and gdf.crs.to_epsg() != epsg else gdf.set_crs(epsg, allow_override=gdf.crs is None)
    ids = gdf["BUILD_ID"].astype(str) if "BUILD_ID" in gdf else pd.Series(range(1, len(gdf) + 1)).map("B{}".format)
    base = gpd.GeoDataFrame({"UID": f"{source.upper()}_" + ids.values, "SOURCE": source},
                            geometry=gdf.geometry.make_valid().values, crs=gdf.crs)
    chips = config.processed_dir() / "chips_predict"
    chips.mkdir(parents=True, exist_ok=True)
    out = add_features(base, raster, chips, config.params()["data"]["chip_size"])
    missing = out.loc[out["colour_source"] != "raster", "UID"].tolist()
    if missing:
        raise SystemExit(f"{len(missing)} buildings are outside {raster.name}: {missing[:5]}")
    if "ROOFTYPE" in gdf:
        from src.build_dataset import clean_rooftype
        out["ROOFTYPE"] = gdf["ROOFTYPE"].map(clean_rooftype).values
    return out


def embed_qgis_style(gpkg: Path, layer: str) -> None:
    """Store the QML as the layer's default style so QGIS applies it on open."""
    qml = STYLE_FILE.read_text(encoding="utf-8")
    con = sqlite3.connect(gpkg)
    try:
        con.execute("""CREATE TABLE IF NOT EXISTS layer_styles (
            id INTEGER PRIMARY KEY AUTOINCREMENT, f_table_catalog TEXT, f_table_schema TEXT,
            f_table_name TEXT, f_geometry_column TEXT, styleName TEXT, styleQML TEXT, styleSLD TEXT,
            useAsDefault BOOLEAN, description TEXT, owner TEXT, ui TEXT, update_time DATETIME)""")
        con.execute("DELETE FROM layer_styles WHERE f_table_name = ?", (layer,))
        con.execute("""INSERT INTO layer_styles (f_table_catalog, f_table_schema, f_table_name,
            f_geometry_column, styleName, styleQML, styleSLD, useAsDefault, description, owner, update_time)
            VALUES ('', '', ?, 'geom', 'roof prediction', ?, '', 1, 'pred_class + misclassified', '', ?)""",
                    (layer, qml, datetime.now(timezone.utc).isoformat()))
        con.commit()
    finally:
        con.close()


def write_web_geojson(gdf: gpd.GeoDataFrame, path: Path) -> None:
    """WGS84 copy for the web map. Photo/chip paths stay server side; only flags are exposed."""
    web = gdf.drop(columns=[c for c in ("photo", "chip", "predicted_at") if c in gdf]).copy()
    web["has_photo"] = gdf["photo"].notna() if "photo" in gdf else False
    web["has_chip"] = gdf["chip"].notna() if "chip" in gdf else False
    for c in web.columns:
        if c.startswith("prob_") or c == "confidence":
            web[c] = web[c].round(4)
    web = web.to_crs(epsg=4326)
    web["geometry"] = web.geometry.set_precision(1e-7)
    if path.exists():
        path.unlink()
    web.to_file(path, driver="GeoJSON")


def predict(input_path: Path | None = None, raster: Path | None = None, source: str = "new",
            model_dir: Path | None = None, output: Path | None = None) -> gpd.GeoDataFrame:
    model = RoofModel(model_dir)
    if input_path is None or input_path.suffix == ".gpkg" and raster is None:
        gdf = gpd.read_file(input_path or config.processed_dir() / config.BUILDINGS_GPKG, layer="buildings")
    else:
        if raster is None:
            raise SystemExit("--raster is required for a layer that has no computed features")
        gdf = features_for_new_layer(input_path, raster, source)

    t0 = time.perf_counter()
    preds = model.predict_frame(gdf[MODEL_FEATURES])
    elapsed = time.perf_counter() - t0
    out = gdf[[c for c in ("UID", "SOURCE", "ROOFTYPE", "photo", "chip") if c in gdf]].join(preds)
    if "ROOFTYPE" in gdf:
        out["correct"] = (out["pred_class"] == out["ROOFTYPE"]).astype(int)
    split_file = config.processed_dir() / "split.json"
    if split_file.exists() and input_path is None:
        split = json.loads(split_file.read_text())
        out["split"] = out["UID"].map({**{u: "train" for u in split["train"]}, **{u: "test" for u in split["test"]}})
    out["predicted_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out = gpd.GeoDataFrame(out, geometry=gdf.geometry, crs=gdf.crs)

    output = output or config.processed_dir() / "predictions.gpkg"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    out.to_file(output, layer="predictions", driver="GPKG")
    embed_qgis_style(output, "predictions")
    write_web_geojson(out.join(gdf[MODEL_FEATURES].round(4)), output.with_suffix(".geojson"))
    PredictionLogger(prediction_log_path()).log(gdf[MODEL_FEATURES], preds, source="batch", uids=out["UID"])

    summary = {"model_version": model.version, "buildings": len(out),
               "seconds": round(elapsed, 3), "predicted": out["pred_class"].value_counts().to_dict()}
    if "correct" in out:
        summary["accuracy_all"] = round(float(out["correct"].mean()), 4)
        if "split" in out:
            summary["accuracy_test_only"] = round(float(out.loc[out["split"] == "test", "correct"].mean()), 4)
    log.info("wrote %s %s", output, summary)
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path)
    ap.add_argument("--raster", type=Path)
    ap.add_argument("--source", default="new")
    ap.add_argument("--model-dir", type=Path)
    ap.add_argument("--output", type=Path)
    a = ap.parse_args()
    predict(a.input, a.raster, a.source, a.model_dir, a.output)
