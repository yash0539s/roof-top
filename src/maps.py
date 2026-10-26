"""Phase 4: one map per area (PDF + PNG) from predictions.gpkg.

    python -m src.maps

Roofs are filled by predicted class over the orthophoto; buildings whose field
label differs are outlined in red, and hold-out test buildings are hatched.
For interactive checking open data/processed/predictions.gpkg in QGIS, which
picks up the embedded style.
"""
from __future__ import annotations

import logging

import geopandas as gpd
import matplotlib
import numpy as np
import rasterio
from matplotlib.patches import Patch
from rasterio.windows import from_bounds

from src import config

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

log = logging.getLogger("maps")
COLOURS = {"ASBESTOS": "#9467bd", "METAL": "#1f77b4", "RCC": "#a0a0a0", "TILED": "#d6604d"}
MAX_BACKGROUND_PX = 2000


def background(ax, raster_path, bounds) -> None:
    with rasterio.open(raster_path) as src:
        win = from_bounds(*bounds, transform=src.transform).intersection(
            rasterio.windows.Window(0, 0, src.width, src.height))
        scale = min(1.0, MAX_BACKGROUND_PX / max(win.width, win.height))
        shape = (3, max(1, int(win.height * scale)), max(1, int(win.width * scale)))
        img = src.read([1, 2, 3], window=win, out_shape=shape)
        left, bottom, right, top = rasterio.windows.bounds(win, src.transform)
    ax.imshow(np.moveaxis(img, 0, -1), extent=(left, right, bottom, top), alpha=0.85)


def draw(gdf: gpd.GeoDataFrame, source: str, raster) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(11, 8.5))
    minx, miny, maxx, maxy = gdf.total_bounds
    pad = 0.05 * max(maxx - minx, maxy - miny)
    bounds = (minx - pad, miny - pad, maxx + pad, maxy + pad)
    if raster is not None:
        background(ax, raster, bounds)
    for cls, colour in COLOURS.items():
        sub = gdf[gdf["pred_class"] == cls]
        if len(sub):
            sub.plot(ax=ax, color=colour, alpha=0.65, edgecolor="#222", linewidth=0.4)
    if "split" in gdf:
        test = gdf[gdf["split"] == "test"]
        if len(test):
            test.plot(ax=ax, facecolor="none", hatch="///", edgecolor="#222", linewidth=0)
    wrong = gdf[gdf.get("correct", 1) == 0]
    if len(wrong):
        wrong.plot(ax=ax, facecolor="none", edgecolor="#e31a1c", linewidth=1.8)
    for _, r in gdf.iterrows():
        c = r.geometry.representative_point()
        ax.annotate(r["UID"].split("_", 1)[1], (c.x, c.y), fontsize=5, ha="center", va="center")
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    ax.set_aspect("equal")
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.tick_params(labelsize=7)
    acc = gdf["correct"].mean() if "correct" in gdf else float("nan")
    ax.set_title(f"Kanjikode — {source}: predicted roof material  "
                 f"({len(gdf)} buildings, agreement with field labels {acc:.0%})", fontsize=11)
    handles = [Patch(color=c, label=k.title()) for k, c in COLOURS.items()]
    handles += [Patch(facecolor="none", edgecolor="#e31a1c", linewidth=1.8, label="Misclassified"),
                Patch(facecolor="none", hatch="///", edgecolor="#222", label="Hold-out test set")]
    ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.9)
    model = gdf["model_version"].iloc[0]
    fig.text(0.01, 0.01, f"EPSG:32643 · model {model} · predicted {gdf['predicted_at'].iloc[0]}", fontsize=7)
    fig.tight_layout()
    return fig


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    gdf = gpd.read_file(config.processed_dir() / "predictions.gpkg", layer="predictions")
    out_dir = config.REPORTS_DIR / "maps"
    out_dir.mkdir(parents=True, exist_ok=True)
    layers = config.params()["data"]["layers"]
    for source, sub in gdf.groupby("SOURCE"):
        raster = layers.get(source, {}).get("raster")
        raster = config.raw_dir() / raster if raster else None
        fig = draw(sub, source, raster)
        for ext in ("pdf", "png"):
            fig.savefig(out_dir / f"{source}.{ext}", dpi=200)
        plt.close(fig)
        log.info("map %s", source)


if __name__ == "__main__":
    main()
