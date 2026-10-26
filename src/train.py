"""Phase 3: train, compare and evaluate roof classifiers; log everything to MLflow.

    python -m src.train                 # full run
    python -m src.train --quick         # CI smoke run: fewer folds, no tuning

Protocol (same for every model, fixed seed):
  * stratified 80/20 split, the test set is used once for the selected model;
  * repeated stratified 5x5 CV on the training set for model choice;
  * leave-one-SOURCE-out CV reported separately as the spatial check;
  * imputation, scaling and tuning only ever see training folds.
Output: models/candidate/{model.joblib,metadata.json}, reports/*, an MLflow run
and a registered model version with the alias "staging".
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import warnings
from datetime import datetime, timezone

import joblib
import matplotlib
import mlflow
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score)
from sklearn.model_selection import (LeaveOneGroupOut, RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_score, train_test_split)

from src import config, models
from src.features import MODEL_FEATURES

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

log = logging.getLogger("train")
warnings.filterwarnings("ignore", message=".*X does not have valid feature names.*")
CANDIDATE_DIR = config.MODELS_DIR / "candidate"
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{(config.ROOT / 'mlflow.db').as_posix()}")


def file_md5(path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=config.ROOT,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    validation = config.REPORTS_DIR / "validation.json"
    if not validation.exists() or not json.loads(validation.read_text())["passed"]:
        raise SystemExit("data validation has not passed; run `python -m src.validate` first")
    df = pd.read_csv(config.processed_dir() / config.FEATURES_CSV)
    return df, df[MODEL_FEATURES].astype(float), df["ROOFTYPE"]


def split(df: pd.DataFrame, test_size: float, seed: int):
    # Stratify on class x source where every cell has at least 2 rows, so the
    # test set mirrors both the label mix and the area mix.
    strata = df["ROOFTYPE"] + "|" + df["SOURCE"]
    if strata.value_counts().min() < 2:
        strata = df["ROOFTYPE"]
    return train_test_split(df.index, test_size=test_size, stratify=strata, random_state=seed)


def cv_scores(est, X, y, folds: int, repeats: int, seed: int) -> np.ndarray:
    cv = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
    return cross_val_score(est, X, y, cv=cv, scoring="f1_macro", n_jobs=-1)


def grouped_cv(est, X, y, groups) -> dict:
    """Train on three areas, test on the fourth. Classes absent from training cannot be predicted."""
    out = {}
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        held = groups.iloc[te[0]]
        m = clone(est).fit(X.iloc[tr], y.iloc[tr])
        pred = m.predict(X.iloc[te])
        labels = sorted(set(y.iloc[te]))
        out[held] = {
            "n": int(len(te)),
            "accuracy": round(float(accuracy_score(y.iloc[te], pred)), 4),
            "macro_f1": round(float(f1_score(y.iloc[te], pred, labels=labels, average="macro", zero_division=0)), 4),
            "classes_unseen_in_training": sorted(set(labels) - set(y.iloc[tr])),
        }
    return out


def tune_lightgbm(X, y, trials: int, seed: int) -> dict:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    def objective(trial):
        est = models.build("lightgbm", seed, **models.lightgbm_search_space(trial))
        return cross_val_score(est, X, y, cv=cv, scoring="f1_macro", n_jobs=-1).mean()

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=trials)
    log.info("optuna best macro-F1 %.4f with %s", study.best_value, study.best_params)
    return study.best_params


def select(results: dict[str, dict]) -> str:
    """Walk from simplest to most complex; switch only on a gain larger than the current std."""
    chosen = None
    for name in models.COMPLEXITY_ORDER:
        if name not in results:
            continue
        r = results[name]
        if chosen is None or r["cv_mean"] > results[chosen]["cv_mean"] + results[chosen]["cv_std"]:
            chosen = name
    return chosen


def plot_confusion(cm: np.ndarray, labels: list[str], title: str, path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=30)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(quick: bool = False) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    p = config.params()
    tp, seed = p["train"], p["seed"]
    folds, repeats = (3, 1) if quick else (tp["cv_folds"], tp["cv_repeats"])
    trials = 0 if quick else tp["optuna_trials"]
    candidates = ["logreg", "random_forest"] if quick else list(tp["candidates"])

    df, X, y = load_data()
    groups = df["SOURCE"]
    tr_idx, te_idx = split(df, tp["test_size"], seed)
    Xtr, ytr, Xte, yte = X.loc[tr_idx], y.loc[tr_idx], X.loc[te_idx], y.loc[te_idx]
    config.REPORTS_DIR.mkdir(exist_ok=True)
    (config.processed_dir() / "split.json").write_text(json.dumps(
        {"train": df.loc[tr_idx, "UID"].tolist(), "test": df.loc[te_idx, "UID"].tolist()}, indent=1))
    data_md5 = file_md5(config.processed_dir() / config.FEATURES_CSV)
    log.info("train %d / test %d, data md5 %s", len(tr_idx), len(te_idx), data_md5)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("roof-material")
    results: dict[str, dict] = {}
    params_by_model: dict[str, dict] = {name: {} for name in candidates}
    if trials > 0:
        params_by_model["lightgbm_tuned"] = tune_lightgbm(Xtr, ytr, trials, seed)
        candidates.append("lightgbm_tuned")

    with mlflow.start_run(run_name=f"train-{datetime.now():%Y%m%d-%H%M%S}") as parent:
        mlflow.log_params({"seed": seed, "cv": f"{repeats}x{folds}", "test_size": tp["test_size"],
                           "data_md5": data_md5, "n_train": len(tr_idx), "n_test": len(te_idx),
                           "git_commit": git_commit() or "none", "quick": quick})
        for name in candidates:
            est = models.build(name, seed, **params_by_model[name])
            scores = cv_scores(est, Xtr, ytr, folds, repeats, seed)
            results[name] = {"cv_mean": float(scores.mean()), "cv_std": float(scores.std())}
            log.info("%-15s CV macro-F1 %.4f ± %.4f", name, scores.mean(), scores.std())
            with mlflow.start_run(run_name=name, nested=True):
                mlflow.log_params({"model": name, **params_by_model[name]})
                mlflow.log_metrics({"cv_macro_f1_mean": scores.mean(), "cv_macro_f1_std": scores.std()})

        best = select(results)
        best_est = models.build(best, seed, **params_by_model[best])
        log.info("selected %s", best)

        grouped = grouped_cv(best_est, X, y, groups)
        for held, r in grouped.items():
            log.info("held-out %-12s macro-F1 %.3f unseen=%s", held, r["macro_f1"], r["classes_unseen_in_training"])

        model = best_est
        if tp["calibrate"]:
            model = CalibratedClassifierCV(best_est, method="sigmoid",
                                           cv=StratifiedKFold(n_splits=min(folds, 5), shuffle=True, random_state=seed))
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        labels = list(model.classes_)
        cm = confusion_matrix(yte, pred, labels=labels)
        report = classification_report(yte, pred, labels=labels, output_dict=True, zero_division=0)
        test = {
            "accuracy": float(accuracy_score(yte, pred)),
            "macro_f1": float(f1_score(yte, pred, average="macro")),
            "per_class": {c: {k: round(report[c][k], 4) for k in ("precision", "recall", "f1-score", "support")}
                          for c in labels},
        }
        log.info("TEST accuracy %.4f macro-F1 %.4f", test["accuracy"], test["macro_f1"])

        imp = permutation_importance(model, Xte, yte, scoring="f1_macro", n_repeats=10,
                                     random_state=seed, n_jobs=-1)
        importance = (pd.Series(imp.importances_mean, index=MODEL_FEATURES)
                      .sort_values(ascending=False).round(4))

        plot_confusion(cm, labels, f"{best} — test macro-F1 {test['macro_f1']:.3f}",
                       config.REPORTS_DIR / "confusion_matrix.png")
        pd.DataFrame(results).T.sort_values("cv_mean", ascending=False).to_csv(config.REPORTS_DIR / "cv_results.csv")
        importance.to_csv(config.REPORTS_DIR / "feature_importance.csv", header=["perm_importance_macro_f1"])
        metrics = {
            "model": best,
            "test_accuracy": round(test["accuracy"], 4),
            "test_macro_f1": round(test["macro_f1"], 4),
            "cv_macro_f1_mean": round(results[best]["cv_mean"], 4),
            "cv_macro_f1_std": round(results[best]["cv_std"], 4),
            "per_class": test["per_class"],
            "confusion_matrix": {"labels": labels, "matrix": cm.tolist()},
            "grouped_cv_by_source": grouped,
            "candidates": {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in results.items()},
        }
        (config.REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

        mlflow.log_params({"selected_model": best, **{f"best_{k}": v for k, v in params_by_model[best].items()}})
        mlflow.log_metrics({"test_macro_f1": test["macro_f1"], "test_accuracy": test["accuracy"],
                            **{f"test_recall_{c}": test["per_class"][c]["recall"] for c in labels},
                            **{f"grouped_f1_{g}": r["macro_f1"] for g, r in grouped.items()}})
        for f in ("metrics.json", "confusion_matrix.png", "cv_results.csv", "feature_importance.csv"):
            mlflow.log_artifact(str(config.REPORTS_DIR / f))
        mlflow.log_artifact(str(config.REPORTS_DIR / "validation.json"))

        version = None
        if not quick:
            import skops.io

            # Our own freshly trained estimator: trust exactly the types it contains.
            trusted = skops.io.get_untrusted_types(data=skops.io.dumps(model))
            info = mlflow.sklearn.log_model(model, name="model", input_example=Xte.head(3),
                                            registered_model_name=tp["registered_model"],
                                            skops_trusted_types=trusted)
            version = info.registered_model_version
            mlflow.MlflowClient().set_registered_model_alias(tp["registered_model"], "staging", version)

        metadata = {
            "model_version": f"{tp['registered_model']}:{version}" if version else f"{tp['registered_model']}:local",
            "registered_version": version,
            "algorithm": best,
            "params": params_by_model[best],
            "calibrated": bool(tp["calibrate"]),
            "classes": labels,
            "features": MODEL_FEATURES,
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "data_md5": data_md5,
            "git_commit": git_commit(),
            "mlflow_run_id": parent.info.run_id,
            "test_macro_f1": metrics["test_macro_f1"],
            "test_accuracy": metrics["test_accuracy"],
            "cv_macro_f1_mean": metrics["cv_macro_f1_mean"],
            "cv_macro_f1_std": metrics["cv_macro_f1_std"],
            # Training feature distribution, used by the drift monitor.
            "reference_stats": {c: {"quantiles": Xtr[c].quantile(np.linspace(0, 1, 11)).round(6).tolist(),
                                    "missing_rate": round(float(Xtr[c].isna().mean()), 4)}
                                for c in MODEL_FEATURES},
            "class_distribution": ytr.value_counts(normalize=True).round(4).to_dict(),
            "trained_uids": df.loc[tr_idx, "UID"].tolist(),
            "test_uids": df.loc[te_idx, "UID"].tolist(),
        }
        if CANDIDATE_DIR.exists():
            shutil.rmtree(CANDIDATE_DIR)
        CANDIDATE_DIR.mkdir(parents=True)
        joblib.dump(model, CANDIDATE_DIR / "model.joblib")
        (CANDIDATE_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2))
        log.info("candidate saved: %s", metadata["model_version"])
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fast smoke run for CI")
    main(**vars(ap.parse_args()))
