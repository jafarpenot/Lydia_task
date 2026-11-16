"""Saving / loading the trained model together with its metadata."""

from __future__ import annotations

import json
import platform
import warnings
from importlib.metadata import version
from pathlib import Path

import joblib

MODEL_FILE = "model.joblib"
METADATA_FILE = "metadata.json"
TRACKED_LIBRARIES = ("scikit-learn", "xgboost", "pandas", "numpy", "scipy", "joblib")


def library_versions() -> dict[str, str]:
    return {"python": platform.python_version(), **{lib: version(lib) for lib in TRACKED_LIBRARIES}}


def save_artifacts(model, metadata: dict, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / MODEL_FILE)
    (out / METADATA_FILE).write_text(json.dumps(metadata, indent=2, default=str))
    return out


def load_artifacts(out_dir: str | Path):
    """Return ``(model, metadata)``. Warns when the installed library versions differ from the training ones
    (a pickled scikit-learn / XGBoost model is only guaranteed to work with the versions it was trained with)."""
    out = Path(out_dir)
    metadata = json.loads((out / METADATA_FILE).read_text())
    trained = metadata.get("library_versions", {})
    current = library_versions()
    diff = {k: (trained[k], current.get(k)) for k in trained if trained[k] != current.get(k)}
    if diff:
        warnings.warn(f"library versions differ from training time (trained, current): {diff}", stacklevel=2)
    return joblib.load(out / MODEL_FILE), metadata
