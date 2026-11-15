import pandas as pd
import pytest

from txlabel.preprocessing import (
    MINOR_TO_MAJOR,
    NUMERIC_COLS,
    build_features,
    is_refund,
    normalize_name,
)


def _row(**kw):
    base = {
        "merchant_name": "CARREFOUR CITY",
        "title": "CARREFOUR CITY",
        "amount_value": -12.5,
        "created_at": "2026-04-28T11:45:21Z",
        "merchant_pan_entry_method": "CHIP_CONTACTLESS",
        "digitalization_type": "apple_pay",
        "pfm_recurring_frequency": "",
    }
    base.update(kw)
    return pd.DataFrame([base])


def test_normalize_name():
    assert normalize_name("UBER   *ONE MEMBERSHIP") == "uber one membership"
    assert normalize_name("DOMINO'S PIZZA") == "domino s pizza"
    assert normalize_name("Café Étoile") == "cafe etoile"
    assert normalize_name("Betclic") == normalize_name("BETCLIC")


def test_build_features_basic():
    f = build_features(_row()).iloc[0]
    assert f["name_norm"] == "carrefour city"
    assert f["entry_method"] == "CHIP_CONTACTLESS"
    assert f["digitalization"] == "apple_pay"
    assert f["abs_amount"] == 12.5
    assert f["empty_name"] == 0 and f["recurring"] == 0 and f["amount_zero"] == 0
    assert (f["hour"], f["day_of_week"], f["day_of_month"]) == (11, 1, 28)  # 2026-04-28 is a Tuesday


def test_empty_merchant_name_falls_back_to_title():
    f = build_features(_row(merchant_name="", title="RETRAIT DAB")).iloc[0]
    assert f["name_norm"] == "retrait dab"
    assert f["empty_name"] == 1


def test_rare_entry_method_grouped_and_missing_optional_columns():
    assert build_features(_row(merchant_pan_entry_method="MAG_STRIPE")).iloc[0]["entry_method"] == "OTHER"
    minimal = _row()[["merchant_name", "amount_value", "created_at"]]
    f = build_features(minimal).iloc[0]
    assert f["entry_method"] == "OTHER" and f["digitalization"] == "none"


def test_recurring_flag_and_zero_amount():
    f = build_features(_row(pfm_recurring_frequency="monthly", amount_value=-0.0)).iloc[0]
    assert f["recurring"] == 1 and f["amount_zero"] == 1


def test_missing_required_column_raises():
    with pytest.raises(ValueError):
        build_features(_row().drop(columns=["amount_value"]))


def test_refund_rule_is_amount_sign():
    raw = pd.concat([_row(amount_value=3.0), _row(amount_value=-3.0), _row(amount_value=0.0)], ignore_index=True)
    assert is_refund(raw).tolist() == [True, False, False]


def test_features_are_numeric_and_complete():
    f = build_features(_row())
    assert not f.isna().any().any()
    assert all(pd.api.types.is_numeric_dtype(f[c]) for c in NUMERIC_COLS)


def test_hierarchy_is_total():
    assert len(MINOR_TO_MAJOR) == 12
    assert MINOR_TO_MAJOR["refund"] == "none"
