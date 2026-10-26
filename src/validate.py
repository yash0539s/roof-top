"""Phase 2: data contract for buildings.gpkg. The pipeline stops on any failure.

    python -m src.validate            # exit code 1 if a check fails
"""
from __future__ import annotations

import json
import logging
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import pandera.pandas as pa

from src import config
from src.features import TEXTURE_FEATURES

log = logging.getLogger("validate")

# Elongation P/√A is smallest for a circle: 2√π ≈ 3.5449.
MIN_ELONGATION = 2 * np.sqrt(np.pi) - 1e-6

positive = pa.Check.gt(0)
rgb_mean = [pa.Check.gt(0, error="RGB mean is 0 (missing zonal stats)"), pa.Check.le(255)]
ratio = [pa.Check.gt(0), pa.Check.lt(1)]


def _finite(s: pd.Series) -> bool:
    return bool(np.isfinite(s.dropna()).all())


SCHEMA = pa.DataFrameSchema(
    {
        "UID": pa.Column(str, pa.Check.str_matches(r"^(GH|COM|IND|RES)_B\d+$"), unique=True),
        "SOURCE": pa.Column(str, pa.Check.isin(config.SOURCES)),
        "ROOFTYPE": pa.Column(str, pa.Check.isin(config.CLASSES)),
        "area_m2": pa.Column(float, positive),
        "perimeter_m": pa.Column(float, positive),
        "compactness": pa.Column(float, [pa.Check.gt(0), pa.Check.le(1)]),
        "elongation": pa.Column(float, pa.Check.ge(MIN_ELONGATION)),
        "mean_r": pa.Column(float, rgb_mean),
        "mean_g": pa.Column(float, rgb_mean),
        "mean_b": pa.Column(float, rgb_mean),
        "std_r": pa.Column(float, pa.Check.ge(0)),
        "std_g": pa.Column(float, pa.Check.ge(0)),
        "std_b": pa.Column(float, pa.Check.ge(0)),
        "brightness": pa.Column(float, [pa.Check.gt(0), pa.Check.le(255)]),
        "r_ratio": pa.Column(float, ratio),
        "g_ratio": pa.Column(float, ratio),
        "b_ratio": pa.Column(float, ratio),
        **{f: pa.Column(float, pa.Check(_finite, element_wise=False), nullable=True) for f in TEXTURE_FEATURES},
        "photo": pa.Column(nullable=True),
    },
    checks=[
        pa.Check(lambda d: np.allclose(d[["r_ratio", "g_ratio", "b_ratio"]].sum(axis=1), 1, atol=1e-6),
                 error="RGB ratios do not sum to 1"),
        pa.Check(lambda d: d["UID"].str.split("_").str[0].map(
            {"GH": "guesthouse", "COM": "commercial", "IND": "industrial", "RES": "residential"}).eq(d["SOURCE"]).all(),
                 error="UID prefix does not match SOURCE"),
        pa.Check(lambda d: d["colour_source"].eq("raster").eq(d[TEXTURE_FEATURES[0]].notna()).all(),
                 error="texture must exist exactly when colour comes from a raster"),
    ],
    strict=False,
    coerce=False,
)


def check_geometry(gdf: gpd.GeoDataFrame) -> list[str]:
    errors = []
    epsg = config.params()["data"]["crs_epsg"]
    if gdf.crs is None or gdf.crs.to_epsg() != epsg:
        errors.append(f"CRS is {gdf.crs}, expected EPSG:{epsg}")
    bad = gdf.loc[~gdf.is_valid, "UID"].tolist()
    if bad:
        errors.append(f"invalid geometry: {bad}")
    not_poly = gdf.loc[~gdf.geom_type.isin(["Polygon", "MultiPolygon"]), "UID"].tolist()
    if not_poly:
        errors.append(f"non-polygon geometry: {not_poly}")
    return errors


def check_files(df: pd.DataFrame, require_photo: bool) -> tuple[list[str], list[str]]:
    errors, warnings = [], []
    for col in ("photo", "chip"):
        if col not in df:
            continue
        paths = df[col].dropna()
        if paths.str.contains(r"^[A-Za-z]:|^/|\\", regex=True).any():
            errors.append(f"{col}: absolute or Windows-style paths found")
        missing = [p for p in paths if not (config.ROOT / p).is_file()]
        if missing:
            errors.append(f"{col}: {len(missing)} files do not exist, e.g. {missing[:3]}")
    no_photo = df.loc[df["photo"].isna(), "UID"].tolist()
    if no_photo:
        msg = f"{len(no_photo)} buildings have no photo"
        (errors if require_photo else warnings).append(msg)
    return errors, warnings


def validate(gdf: gpd.GeoDataFrame | None = None) -> dict:
    if gdf is None:
        gdf = gpd.read_file(config.processed_dir() / config.BUILDINGS_GPKG, layer="buildings")
    errors: list[str] = []
    try:
        SCHEMA.validate(pd.DataFrame(gdf.drop(columns="geometry")), lazy=True)
    except pa.errors.SchemaErrors as e:
        fc = e.failure_cases
        for (col, check), g in fc.groupby(["column", "check"], dropna=False):
            errors.append(f"{col}: {check} failed for {len(g)} rows, e.g. {g['failure_case'].head(3).tolist()}")
    errors += check_geometry(gdf)
    file_errors, warnings = check_files(gdf, config.params()["validate"]["require_photo"])
    errors += file_errors
    return {
        "passed": not errors,
        "rows": int(len(gdf)),
        "class_counts": gdf["ROOFTYPE"].value_counts().to_dict(),
        "errors": errors,
        "warnings": warnings,
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    report = validate()
    config.REPORTS_DIR.mkdir(exist_ok=True)
    (config.REPORTS_DIR / "validation.json").write_text(json.dumps(report, indent=2))
    for w in report["warnings"]:
        log.warning(w)
    for e in report["errors"]:
        log.error(e)
    log.info("validation %s", "PASSED" if report["passed"] else "FAILED")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
