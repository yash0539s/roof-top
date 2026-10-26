"""Classify a roof outlined on a user-uploaded image (JPG/PNG/WebP or GeoTIFF).

The user draws the roof outline in pixel coordinates. A GeoTIFF supplies its
own pixel size and location; a plain image needs the pixel size (or the roof
area) and a latitude/longitude from the user. Features are computed with the
same functions as the training build.
"""
from __future__ import annotations

import base64
import io
import math
import warnings
from dataclasses import dataclass, field

import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling
from rasterio.errors import NotGeoreferencedWarning
from rasterio.features import geometry_mask
from rasterio.io import MemoryFile
from rasterio.windows import Window
from shapely.geometry import Polygon

from src.features import TEXTURE_GSD_M, colour_stats, geometry_features, texture_features

PREVIEW_MAX_PX = 2048
# Kanjikode study area (WGS84) with a margin; outside it the model has never seen roofs.
STUDY_AREA = (76.70, 10.76, 76.79, 10.82)  # min lon, min lat, max lon, max lat
TRAINING_GSD_RANGE = (0.02, 0.25)          # metres per pixel of the training orthophotos
MAX_TEXTURE_GSD = 0.5                      # coarser than this, texture is not meaningful


class ImageError(ValueError):
    """Bad upload or outline; `field` names the form field to blame."""

    def __init__(self, field: str, message: str):
        super().__init__(message)
        self.field = field


@dataclass
class ImageInfo:
    width: int
    height: int
    bands: int
    dtype: str
    georeferenced: bool
    crs: str | None
    m_per_px: float | None
    center_lat: float | None
    center_lon: float | None
    warnings: list[str] = field(default_factory=list)


def open_image(data: bytes) -> rasterio.DatasetReader:
    if not data:
        raise ImageError("file", "the uploaded file is empty")
    mem = MemoryFile(data)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NotGeoreferencedWarning)
            src = mem.open()
    except Exception:
        raise ImageError("file", "unsupported image; upload a JPG, PNG, WebP or GeoTIFF") from None
    if src.count < 3:
        raise ImageError("file", f"the image has {src.count} band(s); an RGB image is needed")
    return src


def _is_georeferenced(src) -> bool:
    return src.crs is not None and src.transform != Affine.identity()


def _pixel_size_m(src, lat: float) -> float:
    t = src.transform
    px, py = math.hypot(t.a, t.d), math.hypot(t.b, t.e)
    if src.crs.is_geographic:
        mx = px * 111_320 * math.cos(math.radians(lat))
        my = py * 110_540
        return (mx + my) / 2
    factor = src.crs.linear_units_factor[1] if src.crs.linear_units_factor else 1.0
    return (px + py) / 2 * factor


def _pixel_to_lonlat(src, col: float, row: float) -> tuple[float, float]:
    from rasterio.warp import transform as warp

    x, y = src.transform * (col, row)
    lon, lat = warp(src.crs, "EPSG:4326", [x], [y])
    return float(lon[0]), float(lat[0])


def _stretch(src) -> tuple[np.ndarray, np.ndarray] | None:
    """Per-band (low, high) to map non-8-bit imagery to 0-255, from a decimated read."""
    if src.dtypes[0] == "uint8":
        return None
    scale = min(1.0, 1024 / max(src.width, src.height))
    sample = src.read([1, 2, 3], out_shape=(3, max(1, int(src.height * scale)), max(1, int(src.width * scale))),
                      masked=True).astype(float)
    lo = np.array([np.percentile(b.compressed(), 1) for b in sample])
    hi = np.array([np.percentile(b.compressed(), 99) for b in sample])
    return lo, np.maximum(hi, lo + 1)


def _to_uint8(arr: np.ndarray, stretch) -> np.ndarray:
    if stretch is None:
        return arr.astype(np.uint8)
    lo, hi = stretch
    out = (arr.astype(float) - lo[:, None, None]) / (hi - lo)[:, None, None] * 255
    return np.clip(out, 0, 255).astype(np.uint8)


def _valid_mask(src, window: Window, out_shape) -> np.ndarray:
    """False where the image has no data: alpha 0, nodata mask, or pure black."""
    mask = src.dataset_mask(window=window, out_shape=out_shape) > 0
    return mask


def inspect(data: bytes) -> tuple[ImageInfo, str]:
    """Metadata plus a PNG preview (data URL) at most PREVIEW_MAX_PX on the long side."""
    with open_image(data) as src:
        geo = _is_georeferenced(src)
        info = ImageInfo(width=src.width, height=src.height, bands=src.count, dtype=src.dtypes[0],
                         georeferenced=geo, crs=src.crs.to_string() if geo else None,
                         m_per_px=None, center_lat=None, center_lon=None)
        if geo:
            lon, lat = _pixel_to_lonlat(src, src.width / 2, src.height / 2)
            info.center_lon, info.center_lat = round(lon, 7), round(lat, 7)
            info.m_per_px = round(_pixel_size_m(src, lat), 5)
        if src.dtypes[0] != "uint8":
            info.warnings.append(f"{src.dtypes[0]} image was contrast-stretched to 8-bit; colours may differ from training")
        scale = min(1.0, PREVIEW_MAX_PX / max(src.width, src.height))
        shape = (3, max(1, round(src.height * scale)), max(1, round(src.width * scale)))
        rgb = _to_uint8(src.read([1, 2, 3], out_shape=shape, resampling=Resampling.average), _stretch(src))
    return info, _png_data_url(np.moveaxis(rgb, 0, -1))


def _png_data_url(rgb_hwc: np.ndarray) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgb_hwc).save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# Pixel sizes for which the main (size-aware) model is trusted.
MAIN_MODEL_GSD_RANGE = (0.01, MAX_TEXTURE_GSD)


@dataclass
class RoofResult:
    polygon: list[list[float]]   # [[col, row], ...] in original image pixels
    features: dict               # inputs for the chosen model
    model: str                   # "main" (size known) or "upload" (size-free)
    latitude: float
    longitude: float
    area_m2: float | None
    perimeter_m: float | None
    roof_image: str              # PNG data URL, background masked out
    score: float = 1.0           # detector confidence (1 for a hand-drawn outline)


@dataclass
class OutlineResult(RoofResult):
    m_per_px: float | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class AutoResult:
    roofs: list[RoofResult]
    width: int
    height: int
    georeferenced: bool
    m_per_px: float | None
    latitude: float
    longitude: float
    detector: str
    preview: np.ndarray          # (h, w, 3) uint8, for the labelled image
    preview_scale: float         # original px per preview px
    warnings: list[str]


def _check_location(latitude, longitude) -> None:
    if latitude is None or longitude is None:
        raise ImageError("latitude", "latitude and longitude are required")
    if not (-90 <= latitude <= 90):
        raise ImageError("latitude", "latitude must be between -90 and 90")
    if not (-180 <= longitude <= 180):
        raise ImageError("longitude", "longitude must be between -180 and 180")


def _common_warnings(size_known: bool, latitude: float, longitude: float, m_per_px: float | None,
                     dtype: str) -> list[str]:
    notes = []
    if not (STUDY_AREA[0] <= longitude <= STUDY_AREA[2] and STUDY_AREA[1] <= latitude <= STUDY_AREA[3]):
        notes.append("location is outside the Kanjikode study area the model was trained on; "
                     "roof types and imagery there may differ, so verify results in the field")
    if not size_known:
        notes.append("image has no georeferencing, so roof size is unknown: the size-free model is used "
                     "(macro-F1 0.78 in testing instead of 0.94)")
    elif m_per_px is not None and not (TRAINING_GSD_RANGE[0] <= m_per_px <= TRAINING_GSD_RANGE[1]):
        notes.append(f"pixel size {m_per_px:.3f} m is outside the training range "
                     f"({TRAINING_GSD_RANGE[0]}–{TRAINING_GSD_RANGE[1]} m)")
    if dtype != "uint8":
        notes.append(f"{dtype} image was contrast-stretched to 8-bit; colours may differ from training imagery")
    return notes


def _roof_on_src(src, poly: Polygon, stretch, m_per_px: float | None, geo: bool,
                 latitude: float, longitude: float, score: float = 1.0) -> RoofResult:
    """Features for one roof polygon (original pixel coords) on an open image."""
    from src.raster import RoofPixels, make_chip
    from src.upload_model import roof_features

    minc, minr, maxc, maxr = poly.bounds
    pad = 2
    c0, r0 = max(0, int(minc) - pad), max(0, int(minr) - pad)
    c1, r1 = min(src.width, math.ceil(maxc) + pad), min(src.height, math.ceil(maxr) + pad)
    win = Window(c0, r0, c1 - c0, r1 - r0)
    rgb = _to_uint8(src.read([1, 2, 3], window=win), stretch)
    inside = geometry_mask([poly], out_shape=rgb.shape[1:], transform=Affine.translation(c0, r0), invert=True)
    valid = inside & _valid_mask(src, win, rgb.shape[1:]) & rgb.any(axis=0)
    if valid.sum() < 10:
        raise ImageError("outline", "too few image pixels inside the outline; zoom in and outline the roof")
    roof_image = _png_data_url(make_chip(RoofPixels(rgb=rgb, mask=valid, transform=Affine.identity()), 256))

    if geo:
        lon_r, lat_r = _pixel_to_lonlat(src, poly.centroid.x, poly.centroid.y)
    else:
        lat_r, lon_r = latitude, longitude

    use_main = m_per_px is not None and MAIN_MODEL_GSD_RANGE[0] <= m_per_px <= MAIN_MODEL_GSD_RANGE[1]
    area_m2 = perimeter_m = None
    if m_per_px is not None:
        area_m2, perimeter_m = poly.area * m_per_px**2, poly.length * m_per_px
    if use_main:
        feats = geometry_features(area_m2, perimeter_m)
        feats.update(colour_stats(rgb[:, valid]))
        scale = min(1.0, m_per_px / TEXTURE_GSD_M)
        oh, ow = max(1, math.ceil(win.height * scale)), max(1, math.ceil(win.width * scale))
        tex_rgb = _to_uint8(src.read([1, 2, 3], window=win, out_shape=(3, oh, ow),
                                     resampling=Resampling.average), stretch)
        t = Affine.translation(c0, r0) * Affine.scale(win.width / ow, win.height / oh)
        tex_mask = geometry_mask([poly], out_shape=(oh, ow), transform=t, invert=True) & tex_rgb.any(axis=0)
        feats.update(texture_features(tex_rgb, tex_mask))
    else:
        feats = roof_features(rgb, valid, poly.area, poly.length)
    return RoofResult(polygon=[[round(x, 1), round(y, 1)] for x, y in poly.exterior.coords[:-1]],
                      features=feats, model="main" if use_main else "upload",
                      latitude=float(lat_r), longitude=float(lon_r),
                      area_m2=area_m2, perimeter_m=perimeter_m, roof_image=roof_image, score=score)


def _location_and_scale(src, latitude, longitude, notes: list[str]):
    geo = _is_georeferenced(src)
    m_per_px = None
    if geo:
        g_lon, g_lat = _pixel_to_lonlat(src, src.width / 2, src.height / 2)
        if latitude is None or longitude is None:
            latitude, longitude = g_lat, g_lon
        elif abs(latitude - g_lat) > 0.01 or abs(longitude - g_lon) > 0.01:
            notes.append(f"entered location differs from the GeoTIFF centre ({g_lat:.5f}, {g_lon:.5f}); "
                         "roof positions come from the GeoTIFF")
        m_per_px = _pixel_size_m(src, latitude)
    _check_location(latitude, longitude)
    return geo, float(latitude), float(longitude), m_per_px


def outline_features(data: bytes, outline: list[list[float]], latitude: float | None = None,
                     longitude: float | None = None, m_per_px: float | None = None,
                     roof_area_m2: float | None = None) -> OutlineResult:
    """Features for a roof the user outlined in pixel coordinates [[col, row], ...].

    Pixel size comes from a GeoTIFF, or optionally from m_per_px / roof_area_m2
    (API only); without it the size-free model is used.
    """
    if not isinstance(outline, list) or len(outline) < 3:
        raise ImageError("outline", "outline the roof with at least 3 points")
    try:
        poly = Polygon([(float(c), float(r)) for c, r in outline])
    except (TypeError, ValueError):
        raise ImageError("outline", "outline must be a list of [column, row] pixel pairs") from None
    if not poly.is_valid or poly.area < 4:
        raise ImageError("outline", "the outline crosses itself or is too small; redraw it")
    notes: list[str] = []
    with open_image(data) as src:
        minc, minr, maxc, maxr = poly.bounds
        if minc < 0 or minr < 0 or maxc > src.width or maxr > src.height:
            raise ImageError("outline", "the outline goes outside the image")
        geo, latitude, longitude, geo_mpp = _location_and_scale(src, latitude, longitude, notes)
        if m_per_px is not None and not (0.001 <= m_per_px <= 10):
            raise ImageError("m_per_px", "metres per pixel must be between 0.001 and 10")
        if m_per_px is None and roof_area_m2:
            m_per_px = math.sqrt(roof_area_m2 / poly.area)
        m_per_px = m_per_px or geo_mpp
        roof = _roof_on_src(src, poly, _stretch(src), m_per_px, geo, latitude, longitude)
        notes += _common_warnings(m_per_px is not None, latitude, longitude, m_per_px, src.dtypes[0])
    return OutlineResult(**roof.__dict__, m_per_px=m_per_px, warnings=notes)


def auto_detect(data: bytes, latitude: float | None, longitude: float | None,
                max_roofs: int = 60) -> AutoResult:
    """Find every roof in the image and compute model features for each."""
    from shapely.affinity import scale as shp_scale

    from src.roof_detect import SAM_MAX_PX, detect_roofs, sam_available

    notes: list[str] = []
    with open_image(data) as src:
        geo, latitude, longitude, m_per_px = _location_and_scale(src, latitude, longitude, notes)
        stretch = _stretch(src)
        f = min(1.0, SAM_MAX_PX / max(src.width, src.height))
        shape = (3, max(1, round(src.height * f)), max(1, round(src.width * f)))
        preview = np.moveaxis(_to_uint8(src.read([1, 2, 3], out_shape=shape, resampling=Resampling.average),
                                        stretch), 0, -1)
        k = src.width / preview.shape[1]
        if k > 4:
            notes.append(f"large image ({src.width}×{src.height} px) was searched at reduced size; "
                         "small roofs may be missed, so crop to the area of interest for best results")
        detections = detect_roofs(preview, max_roofs=max_roofs)
        roofs = []
        for d in detections:
            poly = shp_scale(d.polygon, k, k, origin=(0, 0))
            try:
                roofs.append(_roof_on_src(src, poly, stretch, m_per_px, geo, latitude, longitude, d.score))
            except ImageError:
                continue
        notes += _common_warnings(geo, latitude, longitude, m_per_px, src.dtypes[0])
        if not roofs:
            notes.append("no roofs were found; try a sharper, more zoomed-in image or draw the roof by hand")
        return AutoResult(roofs=roofs, width=src.width, height=src.height, georeferenced=geo, m_per_px=m_per_px,
                          latitude=latitude, longitude=longitude,
                          detector="MobileSAM" if sam_available() else "segmentation rules",
                          preview=preview, preview_scale=k, warnings=notes)


def labelled_image(preview: np.ndarray, scale: float, roofs: list[dict], colours: dict[str, str]) -> str:
    """PNG data URL of the preview with each roof outlined and labelled by class."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.fromarray(preview).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    size = max(11, img.width // 70)
    try:
        font = ImageFont.truetype("arialbd.ttf", size)
    except OSError:
        try:
            font = ImageFont.truetype("DejaVuSans-Bold.ttf", size)
        except OSError:
            font = ImageFont.load_default(size=size)
    for i, r in enumerate(roofs, 1):
        hexc = colours.get(r["roof_class"], "#ffffff").lstrip("#")
        rgb = tuple(int(hexc[j:j + 2], 16) for j in (0, 2, 4))
        pts = [(x / scale, y / scale) for x, y in r["polygon"]]
        draw.polygon(pts, fill=rgb + (70,), outline=rgb + (255,), width=max(2, img.width // 400))
        name = "RCC" if r["roof_class"] == "RCC" else r["roof_class"].title()
        label = f"{i} {name} {round(r['confidence'] * 100)}%"
        cx = sum(p[0] for p in pts) / len(pts)
        cy = sum(p[1] for p in pts) / len(pts)
        box = draw.textbbox((cx, cy), label, font=font, anchor="mm")
        draw.rectangle((box[0] - 3, box[1] - 2, box[2] + 3, box[3] + 2), fill=rgb + (230,))
        draw.text((cx, cy), label, fill=(255, 255, 255), font=font, anchor="mm")
    return _png_data_url(np.asarray(img))
