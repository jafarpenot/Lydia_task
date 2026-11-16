"""Loading and cleaning of the raw transaction CSV."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

ID_COLUMNS = {"transaction_id": str, "member_id": str, "ref_id": str, "merchant_id": str}


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read the CSV. Empty strings are kept as '' (not NaN) and ids are read as strings."""
    return pd.read_csv(path, keep_default_na=False, dtype=ID_COLUMNS)


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicates: rows sharing a ``ref_id`` are identical except for ``transaction_id`` (see EDA, section 10.1)."""
    return df.drop_duplicates(subset="ref_id", keep="first").copy()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
