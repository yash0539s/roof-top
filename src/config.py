"""Project paths and parameters loaded from params.yaml."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PARAMS_FILE = ROOT / "params.yaml"

CLASSES = ["ASBESTOS", "METAL", "RCC", "TILED"]
SOURCES = ["guesthouse", "commercial", "industrial", "residential"]


@lru_cache
def params() -> dict:
    with open(PARAMS_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f)


def raw_dir() -> Path:
    return ROOT / params()["data"]["raw_dir"]


def processed_dir() -> Path:
    return ROOT / params()["data"]["processed_dir"]


BUILDINGS_GPKG = "buildings.gpkg"
FEATURES_CSV = "features.csv"
MODELS_DIR = ROOT / "models"
PRODUCTION_DIR = MODELS_DIR / "production"
REPORTS_DIR = ROOT / "reports"
