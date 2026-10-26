"""Read the pixels of one roof polygon from an orthophoto."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.windows import Window, from_bounds
from shapely.geometry.base import BaseGeometry


@dataclass
class RoofPixels:
    rgb: np.ndarray       # (3, H, W) uint8
    mask: np.ndarray      # (H, W) bool, True inside the roof and with valid data
    transform: Affine

    def pixels(self) -> np.ndarray:
        return self.rgb[:, self.mask]


def covers(src: rasterio.DatasetReader, geom: BaseGeometry) -> bool:
    b = src.bounds
    minx, miny, maxx, maxy = geom.bounds
    return minx >= b.left and maxx <= b.right and miny >= b.bottom and maxy <= b.top


def read_roof(src: rasterio.DatasetReader, geom: BaseGeometry, gsd: float | None = None,
              pad_px: int = 2) -> RoofPixels:
    """Read the roof window, optionally resampled (area-average) to `gsd` metres per pixel."""
    res = abs(src.transform.a)
    pad = pad_px * (gsd or res)
    minx, miny, maxx, maxy = geom.bounds
    win = from_bounds(minx - pad, miny - pad, maxx + pad, maxy + pad, transform=src.transform)
    win = win.round_offsets().round_lengths()
    win = win.intersection(Window(0, 0, src.width, src.height))
    if gsd is None or gsd <= res:
        out_h, out_w = int(win.height), int(win.width)
    else:
        scale = res / gsd
        out_h = max(1, math.ceil(win.height * scale))
        out_w = max(1, math.ceil(win.width * scale))
    rgb = src.read([1, 2, 3], window=win, out_shape=(3, out_h, out_w), resampling=Resampling.average)
    transform = src.window_transform(win) * Affine.scale(win.width / out_w, win.height / out_h)
    inside = geometry_mask([geom], out_shape=(out_h, out_w), transform=transform, invert=True)
    if not inside.any():  # roof smaller than a pixel centre grid
        inside = geometry_mask([geom], out_shape=(out_h, out_w), transform=transform,
                               invert=True, all_touched=True)
    valid = rgb.any(axis=0)  # all-zero pixels are outside the orthophoto footprint
    return RoofPixels(rgb=rgb.astype(np.uint8), mask=inside & valid, transform=transform)


def make_chip(roof: RoofPixels, size: int) -> np.ndarray:
    """Masked roof image letterboxed to size x size, HWC uint8 (background black)."""
    from skimage.transform import resize

    rows = np.where(roof.mask.any(axis=1))[0]
    cols = np.where(roof.mask.any(axis=0))[0]
    rgb = roof.rgb[:, rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    m = roof.mask[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    img = np.moveaxis(rgb, 0, -1) * m[..., None]
    h, w = m.shape
    s = size / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    small = resize(img, (nh, nw), order=1, preserve_range=True, anti_aliasing=s < 1).astype(np.uint8)
    chip = np.zeros((size, size, 3), np.uint8)
    y0, x0 = (size - nh) // 2, (size - nw) // 2
    chip[y0:y0 + nh, x0:x0 + nw] = small
    return chip
