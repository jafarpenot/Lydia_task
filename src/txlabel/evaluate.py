"""Metrics for both targets (minor, and major derived from the predicted minor class)."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

from txlabel.preprocessing import MINOR_TO_MAJOR


def evaluate(y_true_minor: pd.Series, y_pred_minor: pd.Series) -> dict:
    """Accuracy, macro-F1, weighted-F1 on minor and major, plus the per-class report of the minor target."""
    y_true_major = y_true_minor.map(MINOR_TO_MAJOR)
    y_pred_major = pd.Series(y_pred_minor, index=y_true_minor.index).map(MINOR_TO_MAJOR)
    return {
        "minor": {
            "accuracy": accuracy_score(y_true_minor, y_pred_minor),
            "macro_f1": f1_score(y_true_minor, y_pred_minor, average="macro"),
            "weighted_f1": f1_score(y_true_minor, y_pred_minor, average="weighted"),
            "per_class": classification_report(y_true_minor, y_pred_minor, output_dict=True, zero_division=0),
        },
        "major": {
            "accuracy": accuracy_score(y_true_major, y_pred_major),
            "macro_f1": f1_score(y_true_major, y_pred_major, average="macro"),
            "weighted_f1": f1_score(y_true_major, y_pred_major, average="weighted"),
            "per_class": classification_report(y_true_major, y_pred_major, output_dict=True, zero_division=0),
        },
    }


def text_report(y_true_minor: pd.Series, y_pred_minor: pd.Series) -> str:
    """Human-readable per-class reports for minor and major."""
    y_pred_minor = pd.Series(y_pred_minor, index=y_true_minor.index)
    minor = classification_report(y_true_minor, y_pred_minor, digits=3, zero_division=0)
    major = classification_report(
        y_true_minor.map(MINOR_TO_MAJOR), y_pred_minor.map(MINOR_TO_MAJOR), digits=3, zero_division=0
    )
    return f"=== minor_category ===\n{minor}\n=== major_category (derived from minor) ===\n{major}"
