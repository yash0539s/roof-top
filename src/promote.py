"""Quality gate: promote models/candidate to models/production, or roll back.

    python -m src.promote               # gate the current candidate
    python -m src.promote --rollback    # restore the previous production model

A candidate is promoted only if the data validation passed and its test
macro-F1 is at least the floor in params.yaml and at least the current
production model's. The previous production model is archived first.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime, timezone

from src import config

log = logging.getLogger("promote")
CANDIDATE_DIR = config.MODELS_DIR / "candidate"
ARCHIVE_DIR = config.MODELS_DIR / "archive"


def _meta(d):
    f = d / "metadata.json"
    return json.loads(f.read_text()) if f.exists() else None


def _set_alias(version, alias: str = "production") -> None:
    if version is None:
        return
    try:
        import mlflow

        from src.train import TRACKING_URI
        mlflow.set_tracking_uri(TRACKING_URI)
        mlflow.MlflowClient().set_registered_model_alias(
            config.params()["train"]["registered_model"], alias, str(version))
    except Exception as e:  # registry is optional; the file copy is the source of truth for serving
        log.warning("could not set MLflow alias: %s", e)


def gate(candidate: dict, production: dict | None, validation: dict, floor: float) -> tuple[bool, list[str]]:
    reasons = []
    if not validation.get("passed"):
        reasons.append("data validation failed")
    f1 = candidate["test_macro_f1"]
    if f1 < floor:
        reasons.append(f"test macro-F1 {f1:.4f} < floor {floor:.4f}")
    if production and f1 < production["test_macro_f1"]:
        reasons.append(f"test macro-F1 {f1:.4f} < production {production['test_macro_f1']:.4f} "
                       f"({production['model_version']})")
    return not reasons, reasons


def promote() -> dict:
    candidate, production = _meta(CANDIDATE_DIR), _meta(config.PRODUCTION_DIR)
    if candidate is None:
        raise SystemExit("no candidate model; run `python -m src.train` first")
    validation_file = config.REPORTS_DIR / "validation.json"
    validation = json.loads(validation_file.read_text()) if validation_file.exists() else {}
    ok, reasons = gate(candidate, production, validation, config.params()["promote"]["min_macro_f1"])
    decision = {
        "decided_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "candidate": candidate["model_version"],
        "candidate_test_macro_f1": candidate["test_macro_f1"],
        "production_before": production["model_version"] if production else None,
        "promoted": ok,
        "reasons": reasons,
    }
    if ok and production and production["model_version"] == candidate["model_version"]:
        decision["reasons"] = ["candidate is already in production"]
    elif ok:
        if production:
            dest = ARCHIVE_DIR / production["model_version"].replace(":", "_")
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(config.PRODUCTION_DIR, dest)
            shutil.rmtree(config.PRODUCTION_DIR)
        shutil.copytree(CANDIDATE_DIR, config.PRODUCTION_DIR)
        _set_alias(candidate.get("registered_version"))
        log.info("promoted %s", candidate["model_version"])
    else:
        log.warning("candidate %s rejected: %s", candidate["model_version"], "; ".join(reasons))
    _write(decision)
    return decision


def rollback() -> dict:
    production = _meta(config.PRODUCTION_DIR)
    archived = sorted((d for d in ARCHIVE_DIR.glob("*") if _meta(d)),
                      key=lambda d: _meta(d)["trained_at"]) if ARCHIVE_DIR.exists() else []
    if not archived:
        raise SystemExit("nothing to roll back to")
    previous = archived[-1]
    if config.PRODUCTION_DIR.exists():
        shutil.rmtree(config.PRODUCTION_DIR)
    shutil.copytree(previous, config.PRODUCTION_DIR)
    shutil.rmtree(previous)
    meta = _meta(config.PRODUCTION_DIR)
    _set_alias(meta.get("registered_version"))
    decision = {"decided_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "rollback_from": production["model_version"] if production else None,
                "rollback_to": meta["model_version"]}
    log.info("rolled back to %s; restart the API container to load it", meta["model_version"])
    _write(decision)
    return decision


def _write(decision: dict) -> None:
    config.REPORTS_DIR.mkdir(exist_ok=True)
    (config.REPORTS_DIR / "promotion.json").write_text(json.dumps(decision, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollback", action="store_true")
    args = ap.parse_args()
    result = rollback() if args.rollback else promote()
    print(json.dumps(result, indent=2))
    sys.exit(0)
