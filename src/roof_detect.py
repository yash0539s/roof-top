"""Find roofs in an aerial/satellite image without a trained detector.

The image is over-segmented (Felzenszwalb), adjacent segments with similar
colour are merged, and a segment is kept as a roof when it is compact and
roughly rectangular, not vegetation, not shadow and of plausible size.

    python -m src.roof_detect --evaluate     # recall against the digitised footprints
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon

log = logging.getLogger("roof_detect")

WORK_MAX_PX = 1200          # long side used for segmentation
MIN_ROOF_FRAC = 0.0006      # of the image area
MAX_ROOF_FRAC = 0.25
MIN_ROOF_PX = 60


@dataclass
class Detection:
    polygon: Polygon        # pixel coordinates (col, row) in the original image
    score: float            # 0-1 roof-likeness


def _hsv(rgb: np.ndarray) -> np.ndarray:
    from skimage.color import rgb2hsv
    return rgb2hsv(rgb.astype(np.float64) / 255.0)


def _segment(img: np.ndarray, scale: float) -> np.ndarray:
    from skimage.filters import gaussian
    from skimage.segmentation import felzenszwalb

    smooth = gaussian(img, sigma=0.6, channel_axis=-1, preserve_range=True)
    min_size = max(20, int(img.shape[0] * img.shape[1] * MIN_ROOF_FRAC * 0.5))
    return felzenszwalb(smooth / 255.0, scale=scale, sigma=0.5, min_size=min_size)


def _mask_polygon(mask: np.ndarray) -> Polygon | None:
    from skimage.measure import find_contours

    padded = np.pad(mask, 1)
    contours = find_contours(padded.astype(float), 0.5)
    if not contours:
        return None
    c = max(contours, key=len)
    poly = Polygon([(x - 1, y - 1) for y, x in c])
    if not poly.is_valid:
        poly = poly.buffer(0)
        if poly.geom_type == "MultiPolygon":
            poly = max(poly.geoms, key=lambda g: g.area)
    return poly.simplify(1.0, preserve_topology=True)


def _roof_score(poly: Polygon, mask: np.ndarray, hsv: np.ndarray, img_area: int) -> float:
    """0 when the segment is clearly not a roof, otherwise a shape/colour score."""
    area = mask.sum()
    if area < max(MIN_ROOF_PX, MIN_ROOF_FRAC * img_area) or area > MAX_ROOF_FRAC * img_area:
        return 0.0
    if poly is None or poly.is_empty or poly.area <= 0:
        return 0.0
    hull = poly.convex_hull.area
    solidity = poly.area / hull if hull else 0
    rect = poly.minimum_rotated_rectangle
    rectangularity = poly.area / rect.area if rect.area else 0
    sides = sorted(_rect_sides(rect))
    aspect = sides[1] / sides[0] if sides[0] > 0 else 99
    h, s, v = (hsv[..., i][mask] for i in range(3))
    green = ((h > 0.17) & (h < 0.45) & (s > 0.18)).mean()
    dark = (v < 0.2).mean()
    if solidity < 0.82 or rectangularity < 0.66 or aspect > 8 or green > 0.3 or dark > 0.5:
        return 0.0
    return float(np.clip(0.5 * rectangularity + 0.3 * solidity + 0.2 * (1 - green) - 0.3 * dark, 0, 1))


def _rect_sides(rect: Polygon) -> list[float]:
    xs, ys = rect.exterior.coords.xy
    return [float(np.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i])) for i in range(2)]


def _merge_similar(labels: np.ndarray, img: np.ndarray, max_colour_dist: float = 18.0) -> np.ndarray:
    """Merge adjacent segments whose mean colours are close (e.g. two halves of a gable roof)."""
    n = labels.max() + 1
    flat = labels.ravel()
    counts = np.bincount(flat, minlength=n).astype(float)
    means = np.stack([np.bincount(flat, weights=img[..., c].ravel(), minlength=n) for c in range(3)], 1)
    means /= np.maximum(counts, 1)[:, None]
    pairs = set()
    for a, b in ((labels[:, :-1], labels[:, 1:]), (labels[:-1, :], labels[1:, :])):
        diff = a != b
        pairs.update(zip(a[diff].tolist(), b[diff].tolist()))
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in pairs:
        if np.linalg.norm(means[a] - means[b]) < max_colour_dist:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
    roots = np.array([find(i) for i in range(n)])
    return roots[labels]


SCALES = (150, 400, 900)        # Felzenszwalb scales: small roofs ... large sheds
MERGE_DIST = (14.0, 20.0, 26.0)  # colour distance for merging adjacent segments, per scale
NMS_OVERLAP = 0.3


def _candidates(img: np.ndarray, hsv: np.ndarray, scale: float, merge: float) -> list[Detection]:
    labels = _merge_similar(_segment(img, scale), img, merge)
    out = []
    H, W = labels.shape
    for lab in np.unique(labels):
        mask = labels == lab
        if mask.sum() < MIN_ROOF_PX:
            continue
        rr, cc = np.nonzero(mask)
        # Segments touching the border are cut-off roofs or background.
        if rr.min() == 0 or cc.min() == 0 or rr.max() == H - 1 or cc.max() == W - 1:
            continue
        poly = _mask_polygon(mask)
        score = _roof_score(poly, mask, hsv, labels.size)
        if score > 0:
            out.append(Detection(polygon=poly, score=score))
    return out


SAM_CHECKPOINT = os.environ.get("SAM_CHECKPOINT", str(Path(__file__).resolve().parents[1] / "models" / "sam" / "mobile_sam.pt"))
SAM_MAX_PX = 1024
_sam_lock = threading.Lock()
_sam_generator = None


def sam_available() -> bool:
    if not Path(SAM_CHECKPOINT).is_file():
        return False
    try:
        import mobile_sam  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def _sam():
    """Lazily build one MobileSAM automatic mask generator (CPU)."""
    global _sam_generator
    with _sam_lock:
        if _sam_generator is None:
            import torch
            from mobile_sam import SamAutomaticMaskGenerator, sam_model_registry

            torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
            model = sam_model_registry["vit_t"](checkpoint=SAM_CHECKPOINT)
            model.to("cpu").eval()
            _sam_generator = SamAutomaticMaskGenerator(
                model, points_per_side=24, points_per_batch=64, pred_iou_thresh=0.84,
                # min_mask_region_area > 0 would need OpenCV; tiny regions are dropped by _roof_score.
                stability_score_thresh=0.88, box_nms_thresh=0.6, min_mask_region_area=0)
        return _sam_generator


def _sam_candidates(img: np.ndarray, hsv: np.ndarray) -> list[Detection]:
    import torch

    with _sam_lock, torch.inference_mode():
        masks = _sam_generator.generate(img)
    out = []
    H, W = img.shape[:2]
    for m in masks:
        mask = m["segmentation"]
        rr, cc = np.nonzero(mask)
        if rr.size < MIN_ROOF_PX or rr.min() == 0 or cc.min() == 0 or rr.max() == H - 1 or cc.max() == W - 1:
            continue
        poly = _mask_polygon(mask)
        score = _roof_score(poly, mask, hsv, mask.size)
        if score > 0:
            out.append(Detection(polygon=poly, score=0.7 * score + 0.3 * float(m["predicted_iou"])))
    return out


def detect_roofs(rgb_hwc: np.ndarray, max_roofs: int = 80, method: str = "auto") -> list[Detection]:
    """rgb_hwc: (H, W, 3) uint8. Returns roof polygons in original pixel coordinates.

    method: "sam" (MobileSAM masks), "rules" (Felzenszwalb segments) or "auto" (SAM when installed).
    """
    from shapely.affinity import scale as shp_scale
    from skimage.transform import resize

    use_sam = method == "sam" or (method == "auto" and sam_available())
    H, W = rgb_hwc.shape[:2]
    f = min(1.0, (SAM_MAX_PX if use_sam else WORK_MAX_PX) / max(H, W))
    img = rgb_hwc if f == 1.0 else resize(rgb_hwc, (round(H * f), round(W * f)), order=1,
                                           anti_aliasing=True, preserve_range=True).astype(np.uint8)
    hsv = _hsv(img)
    if use_sam:
        _sam()
        cands = _sam_candidates(img, hsv)
    else:
        cands = [d for s, m in zip(SCALES, MERGE_DIST) for d in _candidates(img, hsv, s, m)]
    # Prefer the most roof-like shape; among similar scores the larger one (whole roof over a half).
    cands.sort(key=lambda d: -(d.score + 0.05 * np.log1p(d.polygon.area)))
    kept: list[Detection] = []
    for d in cands:
        if all(d.polygon.intersection(k.polygon).area < NMS_OVERLAP * min(d.polygon.area, k.polygon.area)
               for k in kept):
            kept.append(d)
        if len(kept) >= max_roofs:
            break
    return [Detection(polygon=shp_scale(d.polygon, 1 / f, 1 / f, origin=(0, 0)), score=d.score) for d in kept]


def evaluate(gsd: float = 0.3, window_m: float = 120.0, seed: int = 0, method: str = "auto") -> dict:
    """Recall of the digitised footprints in orthophoto windows resampled to `gsd`."""
    import geopandas as gpd
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import from_bounds
    from shapely.geometry import box

    from src import config

    gdf = gpd.read_file(config.processed_dir() / config.BUILDINGS_GPKG, layer="buildings")
    rng = np.random.default_rng(seed)
    stats = {"windows": 0, "footprints": 0, "found": 0, "detections": 0, "on_footprint": 0, "by_class": {}}
    for source, cfg in config.params()["data"]["layers"].items():
        if not cfg.get("raster"):
            continue
        sub = gdf[gdf.SOURCE == source]
        with rasterio.open(config.raw_dir() / cfg["raster"]) as src:
            centres = sub.geometry.centroid.sample(min(8, len(sub)), random_state=int(rng.integers(1e6)))
            for c in centres:
                b = (c.x - window_m / 2, c.y - window_m / 2, c.x + window_m / 2, c.y + window_m / 2)
                win = from_bounds(*b, transform=src.transform).round_offsets().round_lengths()
                n = max(1, int(window_m / gsd))
                rgb = src.read([1, 2, 3], window=win, out_shape=(3, n, n), resampling=Resampling.average)
                if (rgb.sum(axis=0) == 0).mean() > 0.2:
                    continue
                t = src.window_transform(win) * rasterio.Affine.scale(win.width / n, win.height / n)
                dets = detect_roofs(np.moveaxis(rgb, 0, -1), method=method)
                inv = ~t
                det_world = [Polygon([t * p for p in d.polygon.exterior.coords]) for d in dets]
                inside = sub[sub.within(box(*b))]
                stats["windows"] += 1
                stats["detections"] += len(dets)
                stats["on_footprint"] += sum(any(d.intersection(f).area > 0.3 * d.area for f in inside.geometry)
                                             for d in det_world)
                for _, fp in inside.iterrows():
                    hit = any(fp.geometry.intersection(d).area / fp.geometry.union(d).area >= 0.3 for d in det_world)
                    stats["footprints"] += 1
                    stats["found"] += int(hit)
                    k = stats["by_class"].setdefault(fp.ROOFTYPE, [0, 0])
                    k[0] += int(hit)
                    k[1] += 1
                del inv
    stats["recall"] = round(stats["found"] / max(1, stats["footprints"]), 3)
    stats["share_of_detections_on_digitised_roofs"] = round(stats["on_footprint"] / max(1, stats["detections"]), 3)
    stats["by_class"] = {k: f"{v[0]}/{v[1]}" for k, v in stats["by_class"].items()}
    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--gsd", type=float, default=0.3)
    ap.add_argument("--method", default="auto", choices=["auto", "sam", "rules"])
    a = ap.parse_args()
    if a.evaluate:
        print(json.dumps(evaluate(a.gsd, method=a.method), indent=2))
