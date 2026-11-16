"""Training entry point.

    uv run python -m txlabel.train                      # defaults: data/raw CSV -> models/ + reports/
    uv run python -m txlabel.train --no-refit           # save the train-split model instead of refitting on all rows

Steps: load -> deduplicate on ref_id -> stratified train/test split -> fit on train -> evaluate on the held-out test
-> refit on 100% of the deduplicated data (the model that is served) -> save model + metadata (+ metrics and report).
The reported metrics are those of the train-only model on the held-out test split.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from sklearn.model_selection import train_test_split

from txlabel import __version__
from txlabel.artifacts import library_versions, save_artifacts
from txlabel.data import deduplicate, load_raw, sha256_file
from txlabel.evaluate import evaluate, text_report
from txlabel.model import DEFAULT_XGB_PARAMS, TransactionClassifier
from txlabel.preprocessing import NUMERIC_COLS, CATEGORICAL_COLS, TEXT_COL

log = logging.getLogger("txlabel.train")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = REPO_ROOT / "data" / "raw" / "transaction-dataset-1.csv"


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except Exception:
        return None


def _display_path(path: str | Path) -> str:
    """Repo-relative path when possible (keeps absolute, machine-specific paths out of the metadata)."""
    p = Path(path).resolve()
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return p.name


def run_training(
    data_path: str | Path = DEFAULT_DATA,
    model_dir: str | Path = REPO_ROOT / "models",
    reports_dir: str | Path = REPO_ROOT / "reports",
    seed: int = 42,
    test_size: float = 0.2,
    refit_on_all: bool = True,
    xgb_params: dict | None = None,
) -> dict:
    """Train, evaluate on the held-out test split, save artifacts. Returns the metadata dict."""
    xgb_params = xgb_params or {}
    raw = load_raw(data_path)
    df = deduplicate(raw)
    log.info("rows: %d raw, %d after removing ref_id duplicates", len(raw), len(df))

    train, test = train_test_split(df, test_size=test_size, stratify=df["minor_category"], random_state=seed)
    log.info("split: %d train / %d test (stratified on minor_category, seed=%d)", len(train), len(test), seed)

    model = TransactionClassifier(seed=seed, **xgb_params).fit(train, train["minor_category"])
    pred = model.predict(test)
    metrics = evaluate(test["minor_category"], pred["minor_category"])
    report = text_report(test["minor_category"], pred["minor_category"])
    log.info(
        "held-out test: minor accuracy=%.4f macro-F1=%.4f | major accuracy=%.4f macro-F1=%.4f",
        metrics["minor"]["accuracy"], metrics["minor"]["macro_f1"],
        metrics["major"]["accuracy"], metrics["major"]["macro_f1"],
    )

    final_model, trained_on = model, "train split only"
    if refit_on_all:
        log.info("refitting on all %d rows for the saved model (reported metrics stay those of the train-only model)", len(df))
        final_model = TransactionClassifier(seed=seed, **xgb_params).fit(df, df["minor_category"])
        trained_on = "all deduplicated rows"

    metadata = {
        "model_name": "refund rule + XGBoost",
        "txlabel_version": __version__,
        "git_commit": _git_commit(),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "saved_model_trained_on": trained_on,
        "seed": seed,
        "test_size": test_size,
        "data": {
            "path": _display_path(data_path),
            "sha256": sha256_file(data_path),
            "rows_raw": len(raw),
            "rows_deduplicated": len(df),
            "n_train": len(train),
            "n_test": len(test),
        },
        "classes": final_model.classes_,
        "features": {"text": TEXT_COL, "categorical": CATEGORICAL_COLS, "numeric": NUMERIC_COLS},
        "xgboost_params": {**DEFAULT_XGB_PARAMS, **xgb_params},
        "heldout_test_metrics_model": "trained on the train split only, evaluated on the held-out test split",
        "heldout_test_metrics": {
            t: {k: v for k, v in metrics[t].items() if k != "per_class"} for t in ("minor", "major")
        },
        "library_versions": library_versions(),
    }

    save_artifacts(final_model, metadata, model_dir)
    reports = Path(reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (reports / "classification_report.txt").write_text(report)
    log.info("saved model + metadata to %s and reports to %s", model_dir, reports_dir)
    return metadata


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Train the transaction classifier.")
    p.add_argument("--data", default=str(DEFAULT_DATA), help="path to the raw CSV")
    p.add_argument("--model-dir", default=str(REPO_ROOT / "models"), help="where to write model.joblib + metadata.json")
    p.add_argument("--reports-dir", default=str(REPO_ROOT / "reports"), help="where to write metrics and reports")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--no-refit", dest="refit_on_all", action="store_false",
                   help="save the train-split model (default: refit on all rows after evaluation)")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    run_training(args.data, args.model_dir, args.reports_dir, args.seed, args.test_size, args.refit_on_all)


if __name__ == "__main__":
    main()
