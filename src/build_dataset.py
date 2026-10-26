"""Phase 1: merge the four layers, clean labels and recompute every feature.

    python -m src.build_dataset

Output: data/processed/buildings.gpkg, features.csv, chips/<UID>.png and
reports/data_fixes.json (legacy ArcMap values vs recomputed values).
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from PIL import Image

from src import config
from src.features import (COLOUR_BASE, MODEL_FEATURES, TEXTURE_FEATURES, TEXTURE_GSD_M,
                          colour_derived, colour_stats, polygon_features, texture_features)
from src.raster import covers, make_chip, read_roof

log = logging.getLogger("build_dataset")

ROOF_LABELS = {"METAL": "METAL", "RCC": "RCC", "TILED": "TILED", "TILE": "TILED", "ASBESTOS": "ASBESTOS"}
BUILD_TYPES = {
    "R": "RESIDENTIAL", "RESIDENTIAL": "RESIDENTIAL",
    "R(MULTI)": "RESIDENTIAL_MULTI", "R*MULTI)": "RESIDENTIAL_MULTI",
    "I": "INDUSTRIAL", "INDUSTRIAL": "INDUSTRIAL",
    "COMMERCIAL": "COMMERCIAL",
    "R?": "UNCERTAIN", "DOUBT": "UNCERTAIN",
}
# Legacy RGB column names differ between layers.
LEGACY_RGB = {"Mean_R_1": "mean_r", "Mean_R": "mean_r", "Mean_G": "mean_g", "Mean_B": "mean_b",
              "STD_R": "std_r", "STD_G": "std_g", "STD_B": "std_b"}
PHOTO_EXTS = {".jpg", ".jpeg", ".png"}


def clean_rooftype(value: str) -> str:
    key = re.sub(r"\s*ROOF\s*$", "", str(value).strip().upper())
    if key not in ROOF_LABELS:
        raise ValueError(f"unknown ROOFTYPE {value!r}")
    return ROOF_LABELS[key]


def clean_build_type(value, source: str) -> str:
    key = str(value or "").replace("\x00", "").strip().upper()
    if not key:
        return source.upper()
    return BUILD_TYPES.get(key, "UNCERTAIN")


def index_photos(photo_dir: Path) -> dict[str, Path]:
    """Map BUILD_ID -> photo, tolerating names like 'B10 (2).png'. Exact names win."""
    found: dict[str, Path] = {}
    if not photo_dir.is_dir():
        return found
    for p in sorted(photo_dir.iterdir()):
        if p.suffix.lower() not in PHOTO_EXTS:
            continue
        m = re.fullmatch(r"(B\d+)(\s*\(\d+\))?", p.stem, flags=re.IGNORECASE)
        if not m:
            continue
        bid = m.group(1).upper()
        if bid not in found or not m.group(2):
            found[bid] = p
    return found


def load_layer(source: str, cfg: dict, raw: Path, crs_epsg: int) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(raw / cfg["shapefile"])
    if gdf.crs is None or gdf.crs.to_epsg() != crs_epsg:
        gdf = gdf.to_crs(epsg=crs_epsg)
    gdf["geometry"] = gdf.geometry.make_valid()
    gdf["BUILD_ID"] = gdf["BUILD_ID"].astype(str).str.strip().str.upper()
    dupes = gdf["BUILD_ID"][gdf["BUILD_ID"].duplicated()].tolist()
    if dupes:
        raise ValueError(f"{source}: duplicate BUILD_IDs {dupes}")
    out = gpd.GeoDataFrame({
        "UID": cfg["prefix"] + "_" + gdf["BUILD_ID"],
        "SOURCE": source,
        "BUILD_ID": gdf["BUILD_ID"],
        "ROOFTYPE": gdf["ROOFTYPE"].map(clean_rooftype),
        "BUILD_TYPE": [clean_build_type(v, source) for v in gdf.get("BUILD_TYPE", [None] * len(gdf))],
    }, geometry=gdf.geometry, crs=gdf.crs)
    for legacy, name in (("ELONGATION", "legacy_elongation"), ("COMPACTNES", "legacy_compactness")):
        out[name] = pd.to_numeric(gdf.get(legacy), errors="coerce")
    for legacy, name in LEGACY_RGB.items():
        if legacy in gdf:
            out[f"legacy_{name}"] = pd.to_numeric(gdf[legacy], errors="coerce")
    return out


def add_features(gdf: gpd.GeoDataFrame, raster_path: Path | None, chips_dir: Path,
                 chip_size: int) -> gpd.GeoDataFrame:
    rows = []
    src = rasterio.open(raster_path) if raster_path else None
    try:
        if src is not None and src.crs.to_epsg() != gdf.crs.to_epsg():
            raise ValueError(f"{raster_path} CRS {src.crs} != {gdf.crs}")
        for _, b in gdf.iterrows():
            feat = polygon_features(b.geometry)
            feat["chip"] = None
            if src is not None and covers(src, b.geometry):
                native = read_roof(src, b.geometry)
                if native.mask.sum() == 0:
                    raise ValueError(f"{b.UID}: no valid pixels in {raster_path.name}")
                feat.update(colour_stats(native.pixels()))
                feat.update(texture_features(**_roof_args(read_roof(src, b.geometry, gsd=TEXTURE_GSD_M))))
                chip_path = chips_dir / f"{b.UID}.png"
                Image.fromarray(make_chip(native, chip_size)).save(chip_path)
                feat["chip"] = chip_path.relative_to(config.ROOT).as_posix()
                feat["colour_source"] = "raster"
            else:
                # No orthophoto for this layer: fall back to the ArcMap zonal statistics.
                for k in COLOUR_BASE:
                    feat[k] = float(b.get(f"legacy_{k}", np.nan))
                if all(feat[k] > 0 for k in ("mean_r", "mean_g", "mean_b")):
                    feat.update(colour_derived(feat["mean_r"], feat["mean_g"], feat["mean_b"]))
                feat.update({k: np.nan for k in TEXTURE_FEATURES})
                feat["colour_source"] = "attribute"
            rows.append(feat)
    finally:
        if src is not None:
            src.close()
    feats = pd.DataFrame(rows, index=gdf.index)
    return gdf.join(feats)


def _roof_args(roof) -> dict:
    return {"rgb": roof.rgb, "mask": roof.mask}


def fixes_report(df: pd.DataFrame) -> dict:
    """Summarise how far the legacy ArcMap attributes were from the recomputed values."""
    report = {}
    for source, g in df.groupby("SOURCE"):
        r = {"buildings": int(len(g))}
        for col in ("elongation", "compactness", "mean_r", "mean_g", "mean_b"):
            legacy = g.get(f"legacy_{col}")
            if legacy is None or legacy.isna().all():
                r[col] = "no legacy field"
                continue
            r[col] = {
                "legacy_zero_rows": int((legacy == 0).sum()),
                "legacy_median": round(float(legacy.median()), 3),
                "recomputed_median": round(float(g[col].median()), 3),
            }
        report[source] = r
    return report


def build() -> gpd.GeoDataFrame:
    p = config.params()["data"]
    raw, out_dir = config.raw_dir(), config.processed_dir()
    chips_dir = out_dir / "chips"
    chips_dir.mkdir(parents=True, exist_ok=True)
    for old in chips_dir.glob("*.png"):
        old.unlink()

    layers = []
    for source, cfg in p["layers"].items():
        log.info("layer %s", source)
        gdf = load_layer(source, cfg, raw, p["crs_epsg"])
        raster = raw / cfg["raster"] if cfg.get("raster") else None
        gdf = add_features(gdf, raster, chips_dir, p["chip_size"])
        photos = index_photos(raw / cfg["photo_dir"])
        gdf["photo"] = [photos[b].relative_to(config.ROOT).as_posix() if b in photos else None
                        for b in gdf["BUILD_ID"]]
        layers.append(gdf)

    allb = gpd.GeoDataFrame(pd.concat(layers, ignore_index=True), crs=layers[0].crs)
    report = fixes_report(allb)
    allb = allb.drop(columns=[c for c in allb.columns if c.startswith("legacy_")])
    cols = ["UID", "SOURCE", "BUILD_ID", "ROOFTYPE", "BUILD_TYPE", *MODEL_FEATURES,
            "colour_source", "photo", "chip", "geometry"]
    allb = allb[cols]

    allb.to_file(out_dir / config.BUILDINGS_GPKG, layer="buildings", driver="GPKG")
    allb.drop(columns="geometry").to_csv(out_dir / config.FEATURES_CSV, index=False)
    config.REPORTS_DIR.mkdir(exist_ok=True)
    (config.REPORTS_DIR / "data_fixes.json").write_text(json.dumps(report, indent=2))
    log.info("wrote %d buildings: %s", len(allb), allb["ROOFTYPE"].value_counts().to_dict())
    return allb


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    build()
