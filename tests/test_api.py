from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from txlabel.api import EXAMPLE_REQUEST, app
from txlabel.artifacts import load_artifacts
from txlabel.data import load_raw
from txlabel.train import run_training

DATA = Path(__file__).resolve().parents[1] / "data" / "raw" / "transaction-dataset-1.csv"


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("api")
    sample = d / "sample.csv"
    load_raw(DATA).sample(1500, random_state=0).to_csv(sample, index=False)
    run_training(sample, d / "model", d / "reports", seed=1, refit_on_all=False, xgb_params={"n_estimators": 20})
    return d / "model"


@pytest.fixture(scope="module")
def client(model_dir):
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("TXLABEL_MODEL_DIR", str(model_dir))
        with TestClient(app) as c:
            yield c


def test_health_and_model_info(client):
    assert client.get("/health").json() == {"status": "ok"}
    info = client.get("/model-info").json()
    assert "refund" in info["classes"] and "heldout_test_metrics" in info and "library_versions" in info


def test_predict_returns_one_category_with_confidence(client):
    r = client.post("/predict", json=EXAMPLE_REQUEST)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"minor_category", "major_category", "confidence"}
    assert 0 <= body["confidence"] <= 1


def test_api_matches_direct_model_prediction(client, model_dir):
    """Parity: the API must return exactly what the model returns for the same raw row."""
    model, _ = load_artifacts(model_dir)
    rows = load_raw(DATA).sample(25, random_state=3)
    for _, row in rows.iterrows():
        payload = row.to_dict()
        payload["amount_value"] = float(payload["amount_value"])
        direct = model.predict(pd.DataFrame([row])).iloc[0]
        r = client.post("/predict", json=payload)  # full dataset row: extra fields are ignored
        assert r.status_code == 200, r.text
        assert r.json()["minor_category"] == direct["minor_category"]
        assert r.json()["major_category"] == direct["major_category"]
        assert r.json()["confidence"] == pytest.approx(float(direct["confidence"]), abs=1e-4)


def test_positive_amount_is_a_refund(client):
    r = client.post("/predict", json={**EXAMPLE_REQUEST, "amount_value": 12.5})
    assert r.json() == {"minor_category": "refund", "major_category": "none", "confidence": 1.0}


def test_minimal_payload_and_empty_merchant_name_with_title(client):
    r = client.post("/predict", json={"title": "RETRAIT DAB", "amount_value": -50, "created_at": "2026-05-02T09:00:00Z"})
    assert r.status_code == 200
    assert r.json()["minor_category"] == "atm"


def test_unseen_merchant_still_gets_a_prediction(client):
    r = client.post("/predict", json={**EXAMPLE_REQUEST, "merchant_name": "ZZZ NEVER SEEN 123", "title": ""})
    assert r.status_code == 200


def test_naive_timestamp_accepted(client):
    r = client.post("/predict", json={**EXAMPLE_REQUEST, "created_at": "2026-04-28T11:45:21"})
    assert r.status_code == 200


@pytest.mark.parametrize(
    "patch",
    [
        {"amount_value": None},
        {"amount_value": "abc"},
        {"amount_value": "nan"},
        {"created_at": "not a date"},
        {"created_at": None},
        {"merchant_name": "", "title": ""},
        {"merchant_name": "x" * 500},
    ],
)
def test_invalid_input_returns_422(client, patch):
    r = client.post("/predict", json={**EXAMPLE_REQUEST, **patch})
    assert r.status_code == 422


def test_missing_required_field_returns_422(client):
    payload = {k: v for k, v in EXAMPLE_REQUEST.items() if k != "amount_value"}
    assert client.post("/predict", json=payload).status_code == 422


def test_service_reports_not_ready_when_model_is_missing(tmp_path):
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("TXLABEL_MODEL_DIR", str(tmp_path / "does-not-exist"))
        with TestClient(app) as c:
            assert c.get("/health").status_code == 503
            assert c.post("/predict", json=EXAMPLE_REQUEST).status_code == 503
