import math

import numpy as np
import pytest
from shapely.geometry import Point, box

from src.features import (MODEL_FEATURES, TEXTURE_FEATURES, colour_derived, colour_stats,
                          complete_features, geometry_features, polygon_features, texture_features)


def test_square_geometry():
    f = polygon_features(box(0, 0, 10, 10))
    assert f["area_m2"] == pytest.approx(100)
    assert f["perimeter_m"] == pytest.approx(40)
    assert f["elongation"] == pytest.approx(4.0)            # P/√A
    assert f["compactness"] == pytest.approx(math.pi / 4)   # 4πA/P²


def test_circle_is_the_bound():
    f = polygon_features(Point(0, 0).buffer(10, quad_segs=256))
    assert f["compactness"] == pytest.approx(1.0, abs=1e-4)
    assert f["elongation"] == pytest.approx(2 * math.sqrt(math.pi), abs=1e-3)


def test_elongated_rectangle_is_more_elongated():
    assert polygon_features(box(0, 0, 40, 5))["elongation"] > polygon_features(box(0, 0, 10, 10))["elongation"]


@pytest.mark.parametrize("area,perim", [(0, 10), (10, 0), (-1, 5)])
def test_geometry_rejects_non_positive(area, perim):
    with pytest.raises(ValueError):
        geometry_features(area, perim)


def test_colour_ratios_sum_to_one():
    d = colour_derived(85.06, 105.91, 156.42)
    assert d["r_ratio"] + d["g_ratio"] + d["b_ratio"] == pytest.approx(1)
    assert d["brightness"] == pytest.approx((85.06 + 105.91 + 156.42) / 3)


def test_colour_rejects_black():
    with pytest.raises(ValueError):
        colour_derived(0, 0, 0)


def test_colour_stats():
    px = np.array([[10, 20], [30, 30], [50, 70]], dtype=np.uint8)
    s = colour_stats(px)
    assert s["mean_r"] == 15 and s["std_g"] == 0 and s["mean_b"] == 60


def _image(pattern: np.ndarray) -> np.ndarray:
    return np.repeat(pattern[None], 3, axis=0).astype(np.uint8)


def test_texture_separates_corrugation_from_flat_and_noise():
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:64, 0:64]
    stripes = _image(128 + 100 * np.sin(xx / 1.5))                     # corrugated sheet
    flat = _image(np.full((64, 64), 180) + rng.integers(0, 3, (64, 64)))  # smooth RCC
    noise = _image(rng.integers(0, 255, (64, 64)))                     # isotropic texture
    mask = np.ones((64, 64), bool)
    s, f, n = (texture_features(img, mask) for img in (stripes, flat, noise))
    assert set(s) == set(TEXTURE_FEATURES)
    assert s["edge_coherence"] > 0.9 > n["edge_coherence"]
    assert f["glcm_homogeneity"] > n["glcm_homogeneity"]
    assert f["glcm_contrast"] < n["glcm_contrast"]


def test_texture_tiny_roof_is_nan():
    out = texture_features(np.zeros((3, 4, 4), np.uint8), np.ones((4, 4), bool))
    assert all(math.isnan(v) for v in out.values())


def test_complete_features_order_and_missing_texture():
    rec = {"area_m2": 451.66, "perimeter_m": 102.05, "mean_r": 85.06, "mean_g": 105.91, "mean_b": 156.42,
           "std_r": 12.03, "std_g": 10.59, "std_b": 9.26}
    out = complete_features(rec)
    assert list(out) == MODEL_FEATURES
    assert all(math.isnan(out[k]) for k in TEXTURE_FEATURES)
