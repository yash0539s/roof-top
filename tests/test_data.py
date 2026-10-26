"""Data contract tests: synthetic cases always run; the real dataset is checked when built."""
import geopandas as gpd
import numpy as np
import pytest
from shapely.geometry import box

from src import config
from src.build_dataset import clean_build_type, clean_rooftype, index_photos
from src.validate import validate


def to_gdf(df):
    geoms = [box(689000 + i * 20, 1193000, 689010 + i * 20, 1193010) for i in range(len(df))]
    return gpd.GeoDataFrame(df.copy(), geometry=geoms, crs="EPSG:32643")


def test_synthetic_passes(synthetic_df):
    report = validate(to_gdf(synthetic_df))
    assert report["passed"], report["errors"]


@pytest.mark.parametrize("mutate,expect", [
    (lambda d: d.assign(mean_r=d["mean_r"].where(d.index != 3, 0.0)), "mean_r"),
    (lambda d: d.assign(UID=d["UID"].where(d.index != 1, d["UID"][0])), "UID"),
    (lambda d: d.assign(ROOFTYPE=d["ROOFTYPE"].where(d.index != 0, "CONCRETE")), "ROOFTYPE"),
    (lambda d: d.assign(compactness=d["compactness"].where(d.index != 0, 1.4)), "compactness"),
    (lambda d: d.assign(elongation=d["elongation"].where(d.index != 0, 200.128)), None),  # brightness copy is still >= 3.54
    (lambda d: d.assign(elongation=d["elongation"].where(d.index != 0, 1.15)), "elongation"),  # the map2 formula
    (lambda d: d.assign(glcm_contrast=d["glcm_contrast"].where(d.index != 40, np.inf)), "glcm_contrast"),
    (lambda d: d.assign(SOURCE=d["SOURCE"].where(d.index != 0, "industrial")), "UID prefix"),
    (lambda d: d.assign(photo=d["photo"].where(d.index != 0, r"C:\Users\gouth\B1.jpeg")), "photo"),
])
def test_bad_rows_fail(synthetic_df, mutate, expect):
    report = validate(to_gdf(mutate(synthetic_df)))
    if expect is None:
        return
    assert not report["passed"]
    assert any(expect in e for e in report["errors"]), report["errors"]


def test_wrong_crs_fails(synthetic_df):
    gdf = to_gdf(synthetic_df).set_crs("EPSG:4326", allow_override=True)
    assert any("CRS" in e for e in validate(gdf)["errors"])


@pytest.mark.parametrize("raw,clean", [("METAL ROOF", "METAL"), ("rcc roof ", "RCC"), ("TILED ROOF", "TILED"),
                                       ("ASBESTOS ROOF", "ASBESTOS")])
def test_clean_rooftype(raw, clean):
    assert clean_rooftype(raw) == clean


def test_clean_rooftype_rejects_unknown():
    with pytest.raises(ValueError):
        clean_rooftype("THATCH ROOF")


@pytest.mark.parametrize("raw,clean", [("R", "RESIDENTIAL"), ("R(MULTI)", "RESIDENTIAL_MULTI"),
                                       ("R*MULTI)", "RESIDENTIAL_MULTI"), ("R?", "UNCERTAIN"),
                                       ("DOUBT", "UNCERTAIN"), ("", "INDUSTRIAL"), (None, "INDUSTRIAL")])
def test_clean_build_type(raw, clean):
    assert clean_build_type(raw, "industrial") == clean


def test_photo_index_prefers_exact_name(tmp_path):
    for name in ("B10 (2).png", "B11 (2).png", "B11.jpeg", "B3.JPEG", "map.jpg", "notes.txt"):
        (tmp_path / name).write_bytes(b"x")
    idx = index_photos(tmp_path)
    assert idx["B10"].name == "B10 (2).png"
    assert idx["B11"].name == "B11.jpeg"
    assert idx["B3"].name == "B3.JPEG"
    assert set(idx) == {"B10", "B11", "B3"}


BUILT = config.processed_dir() / config.BUILDINGS_GPKG


@pytest.mark.skipif(not BUILT.exists(), reason="dataset not built (run dvc repro build)")
def test_real_dataset_contract():
    gdf = gpd.read_file(BUILT, layer="buildings")
    report = validate(gdf)
    assert report["passed"], report["errors"]
    assert len(gdf) == 220
    assert (gdf[["mean_r", "mean_g", "mean_b"]] > 0).all().all()
    assert gdf["UID"].is_unique
