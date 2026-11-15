"""Raw transaction -> model features.

This is the ONLY place where raw columns are turned into features. Training, evaluation notebooks and the
prediction API all call :func:`build_features`, so there is no train/serve skew.
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

REFUND = "refund"
NO_MAJOR = "none"  # major category of a refund (empty in the source data)

# Strict hierarchy observed in the data (see notebooks/01_eda.ipynb, section 3.3).
MINOR_TO_MAJOR: dict[str, str] = {
    "atm": "finance",
    "betting": "entertainment",
    "hairdresser": "self_care",
    "multimedia": "entertainment",
    "others_shopping": "shopping",
    "pharmacies": "self_care",
    "refund": NO_MAJOR,
    "restaurants": "restaurants_bars_cafes",
    "supermarkets": "groceries",
    "tolls_gas_stations": "transportation",
    "transport_means": "transportation",
    "travel_means": "travel",
}

# Rare entry methods (CHIP_FALLBACK: 9 rows, MAG_STRIPE: 3, none: 1) are grouped into OTHER.
KEPT_ENTRY_METHODS = ("CHIP_CONTACTLESS", "CARD_ON_FILE", "MANUAL", "CHIP")

# Raw columns required / optional at prediction time.
REQUIRED_COLUMNS = ("amount_value", "created_at", "merchant_name")
OPTIONAL_COLUMNS = ("title", "merchant_pan_entry_method", "digitalization_type", "pfm_recurring_frequency")

TEXT_COL = "name_norm"
CATEGORICAL_COLS = ["entry_method", "digitalization"]
NUMERIC_COLS = ["abs_amount", "amount_zero", "empty_name", "recurring", "hour", "day_of_week", "day_of_month"]


def normalize_name(s: str) -> str:
    """Strip accents, lowercase, replace any non-alphanumeric run by one space, trim."""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def is_refund(raw: pd.DataFrame) -> pd.Series:
    """Deterministic refund rule: refunds are exactly the positive amounts (123/123 in the data, no exception)."""
    return pd.to_numeric(raw["amount_value"]) > 0


def _col(raw: pd.DataFrame, name: str, default: str = "") -> pd.Series:
    if name in raw.columns:
        return raw[name].fillna(default).astype(str)
    return pd.Series(default, index=raw.index)


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Build the model input from raw transaction columns (one row per transaction)."""
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    merchant = _col(raw, "merchant_name")
    title = _col(raw, "title")
    empty_name = merchant.str.strip() == ""
    text = merchant.where(~empty_name, title)  # fall back to title when merchant_name is empty

    created = pd.to_datetime(raw["created_at"], utc=True)
    amount = pd.to_numeric(raw["amount_value"])

    entry = _col(raw, "merchant_pan_entry_method")
    entry = entry.where(entry.isin(KEPT_ENTRY_METHODS), "OTHER")
    digitalization = _col(raw, "digitalization_type", "none").str.strip().str.lower()
    digitalization = digitalization.where(digitalization != "", "none")

    return pd.DataFrame(
        {
            TEXT_COL: text.map(normalize_name),
            "entry_method": entry,
            "digitalization": digitalization,
            "abs_amount": amount.abs(),
            "amount_zero": (amount == 0).astype(int),
            "empty_name": empty_name.astype(int),
            "recurring": (_col(raw, "pfm_recurring_frequency").str.strip() != "").astype(int),
            "hour": created.dt.hour,
            "day_of_week": created.dt.day_of_week,
            "day_of_month": created.dt.day,
        },
        index=raw.index,
    )
