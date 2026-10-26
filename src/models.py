"""Candidate models. All accept a DataFrame of MODEL_FEATURES that may contain NaN
(texture is missing for layers without an orthophoto) and handle class imbalance."""
from __future__ import annotations

from typing import Callable

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Ordered simplest -> most complex; a more complex model must beat the current
# choice by more than one CV standard deviation to be selected.
COMPLEXITY_ORDER = ["logreg", "random_forest", "lightgbm", "catboost", "lightgbm_tuned"]


def logreg(seed: int, **_) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000, random_state=seed)),
    ])


def random_forest(seed: int, **_) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("clf", RandomForestClassifier(n_estimators=500, min_samples_leaf=2, max_features="sqrt",
                                       class_weight="balanced_subsample", n_jobs=-1, random_state=seed)),
    ])


def lightgbm(seed: int, **params):
    from lightgbm import LGBMClassifier

    defaults = dict(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=5,
                    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0)
    defaults.update(params)
    return LGBMClassifier(class_weight="balanced", random_state=seed, n_jobs=1, verbose=-1, **defaults)


def catboost(seed: int, **params):
    from catboost import CatBoostClassifier

    defaults = dict(iterations=500, depth=4, learning_rate=0.05, l2_leaf_reg=3.0)
    defaults.update(params)
    return CatBoostClassifier(auto_class_weights="Balanced", random_seed=seed, verbose=0,
                              thread_count=1, allow_writing_files=False, **defaults)


FACTORIES: dict[str, Callable] = {
    "logreg": logreg,
    "random_forest": random_forest,
    "lightgbm": lightgbm,
    "catboost": catboost,
    "lightgbm_tuned": lightgbm,
}


def build(name: str, seed: int, **params):
    return FACTORIES[name](seed=seed, **params)


def lightgbm_search_space(trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 4, 31),
        "min_child_samples": trial.suggest_int("min_child_samples", 3, 20),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }
