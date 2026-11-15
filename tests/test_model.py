from pathlib import Path

import pandas as pd
import pytest

from txlabel.model import TransactionClassifier
from txlabel.preprocessing import MINOR_TO_MAJOR

DATA = Path(__file__).resolve().parents[1] / "data" / "raw" / "transaction-dataset-1.csv"


@pytest.fixture(scope="module")
def data():
    raw = pd.read_csv(DATA, keep_default_na=False, dtype={"ref_id": str, "member_id": str, "merchant_id": str})
    return raw.drop_duplicates("ref_id").sample(1500, random_state=0)


@pytest.fixture(scope="module")
def model(data):
    return TransactionClassifier(seed=0, n_estimators=25).fit(data, data["minor_category"])


def test_predict_output_schema_and_consistency(model, data):
    pred = model.predict(data.head(50))
    assert list(pred.columns) == ["minor_category", "major_category", "confidence"]
    assert (pred["major_category"] == pred["minor_category"].map(MINOR_TO_MAJOR)).all()
    assert pred["confidence"].between(0, 1).all()


def test_refund_rule_applies_to_positive_amounts(model, data):
    row = data.head(1).copy()
    row["amount_value"] = 4.2
    pred = model.predict(row).iloc[0]
    assert pred["minor_category"] == "refund" and pred["major_category"] == "none" and pred["confidence"] == 1.0
    row["amount_value"] = -4.2
    assert model.predict(row).iloc[0]["minor_category"] != "refund"


def test_batch_and_single_row_predictions_agree(model, data):
    """Guards against train/serve skew: predicting row by row must equal predicting the batch."""
    batch = model.predict(data.head(40))
    single = pd.concat([model.predict(data.iloc[[i]]) for i in range(40)])
    assert batch["minor_category"].tolist() == single["minor_category"].tolist()
    assert batch["confidence"].round(6).tolist() == single["confidence"].round(6).tolist()


def test_unseen_merchant_does_not_crash(model, data):
    row = data.head(1).copy()
    row["merchant_name"] = row["title"] = "TOTALLY NEW SHOP 42"
    row["amount_value"] = -9.9
    assert model.predict(row).iloc[0]["minor_category"] in MINOR_TO_MAJOR


def test_fit_rejects_labels_that_break_the_refund_rule(data):
    bad = data.copy()
    bad.iloc[0, bad.columns.get_loc("amount_value")] = 5.0  # positive amount but not labeled refund
    bad.iloc[0, bad.columns.get_loc("minor_category")] = "supermarkets"
    with pytest.raises(ValueError):
        TransactionClassifier(n_estimators=5).fit(bad, bad["minor_category"])
