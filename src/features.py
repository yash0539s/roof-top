"""Single source of truth for every feature formula.

Used by the dataset build, the batch predictor and the API, so a feature can
never be computed two different ways again (the ArcMap elongation bug).
"""
from __future__ import annotations

import math

import numpy as np
from shapely.geometry.base import BaseGeometry

GEOMETRY_FEATURES = ["area_m2", "perimeter_m", "compactness", "elongation"]
COLOUR_BASE = ["mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b"]
COLOUR_DERIVED = ["brightness", "r_ratio", "g_ratio", "b_ratio"]
TEXTURE_FEATURES = [
    "hue_sin", "hue_cos", "sat_mean", "val_std",
    "glcm_contrast", "glcm_homogeneity", "glcm_energy", "glcm_entropy", "glcm_correlation",
    "lbp_entropy", "lbp_flat_frac",
    "edge_density", "edge_coherence",
]
MODEL_FEATURES = GEOMETRY_FEATURES + COLOUR_BASE + COLOUR_DERIVED + TEXTURE_FEATURES

# Texture is scale dependent, so every raster is resampled to this ground
# sample distance before texture is measured (industrial imagery is ~0.21 m).
TEXTURE_GSD_M = 0.2
MIN_TEXTURE_PIXELS = 25


def geometry_features(area_m2: float, perimeter_m: float) -> dict[str, float]:
    """Compactness 4πA/P² (1 for a circle) and elongation P/√A (≥ 2√π ≈ 3.545)."""
    if area_m2 <= 0 or perimeter_m <= 0:
        raise ValueError("area_m2 and perimeter_m must be > 0")
    return {
        "area_m2": float(area_m2),
        "perimeter_m": float(perimeter_m),
        "compactness": 4.0 * math.pi * area_m2 / perimeter_m**2,
        "elongation": perimeter_m / math.sqrt(area_m2),
    }


def polygon_features(geom: BaseGeometry) -> dict[str, float]:
    """Geometry features from a polygon in a metric CRS."""
    return geometry_features(geom.area, geom.length)


def colour_derived(mean_r: float, mean_g: float, mean_b: float) -> dict[str, float]:
    total = mean_r + mean_g + mean_b
    if total <= 0:
        raise ValueError("mean RGB must be > 0")
    return {
        "brightness": total / 3.0,
        "r_ratio": mean_r / total,
        "g_ratio": mean_g / total,
        "b_ratio": mean_b / total,
    }


def colour_stats(pixels: np.ndarray) -> dict[str, float]:
    """pixels: (3, N) uint8 RGB values inside the roof."""
    px = pixels.astype(np.float64)
    mean = px.mean(axis=1)
    std = px.std(axis=1)
    out = dict(zip(COLOUR_BASE, [*mean, *std]))
    out.update(colour_derived(*mean))
    return {k: float(v) for k, v in out.items()}


def texture_features(rgb: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Texture of a roof chip at TEXTURE_GSD_M.

    rgb: (3, H, W) uint8, mask: (H, W) bool, True inside the roof.
    Corrugated sheets give strong parallel edges (high coherence), tiles give a
    fine repeating pattern (high GLCM contrast), RCC is smooth (high homogeneity).
    """
    from skimage.color import rgb2gray, rgb2hsv
    from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
    from skimage.filters import sobel_h, sobel_v

    if mask.sum() < MIN_TEXTURE_PIXELS:
        return {k: float("nan") for k in TEXTURE_FEATURES}

    img = np.moveaxis(rgb, 0, -1).astype(np.float64) / 255.0
    hsv = rgb2hsv(img)
    hue = hsv[..., 0][mask] * 2 * np.pi
    sat, val = hsv[..., 1][mask], hsv[..., 2][mask]
    out = {
        "hue_sin": float(np.mean(np.sin(hue))),
        "hue_cos": float(np.mean(np.cos(hue))),
        "sat_mean": float(sat.mean()),
        "val_std": float(val.std()),
    }

    gray = rgb2gray(img)
    # GLCM on 32 grey levels; pixels outside the roof are put in an extra level
    # that is dropped from the matrix so the background never counts.
    levels = 32
    q = np.clip((gray * levels).astype(np.int32), 0, levels - 1)
    q[~mask] = levels
    glcm = graycomatrix(q, distances=[1, 2], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=levels + 1, symmetric=True, normed=False)
    glcm = glcm[:levels, :levels].astype(np.float64)
    sums = glcm.sum(axis=(0, 1), keepdims=True)
    glcm = np.divide(glcm, sums, out=np.zeros_like(glcm), where=sums > 0)
    for prop in ("contrast", "homogeneity", "energy", "correlation"):
        vals = graycoprops(glcm, prop)
        out[f"glcm_{prop}"] = float(np.nanmean(vals))
    p = glcm.mean(axis=(2, 3))
    p = p[p > 0]
    out["glcm_entropy"] = float(-(p * np.log2(p)).sum())

    lbp = local_binary_pattern((gray * 255).astype(np.uint8), P=8, R=1, method="uniform")[mask]
    hist = np.bincount(lbp.astype(int), minlength=10) / lbp.size
    nz = hist[hist > 0]
    out["lbp_entropy"] = float(-(nz * np.log2(nz)).sum())
    out["lbp_flat_frac"] = float(hist[0] + hist[8])  # flat spots and flat areas

    gx, gy = sobel_v(gray), sobel_h(gray)
    inner = mask.copy()
    inner[0, :] = inner[-1, :] = inner[:, 0] = inner[:, -1] = False
    mag = np.hypot(gx, gy)[inner]
    out["edge_density"] = float((mag > 0.1).mean()) if mag.size else float("nan")
    # Structure-tensor coherence: 1 = all edges parallel (corrugation), 0 = isotropic.
    jxx, jyy, jxy = (gx * gx)[inner].sum(), (gy * gy)[inner].sum(), (gx * gy)[inner].sum()
    denom = jxx + jyy
    out["edge_coherence"] = float(np.sqrt((jxx - jyy) ** 2 + 4 * jxy**2) / denom) if denom > 0 else 0.0
    return out


def complete_features(record: dict) -> dict[str, float]:
    """Fill derived geometry/colour features from base fields; texture stays NaN if absent."""
    out = geometry_features(record["area_m2"], record["perimeter_m"])
    for k in COLOUR_BASE:
        out[k] = float(record[k])
    out.update(colour_derived(record["mean_r"], record["mean_g"], record["mean_b"]))
    for k in TEXTURE_FEATURES:
        v = record.get(k)
        out[k] = float("nan") if v is None else float(v)
    return out
