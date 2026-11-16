"""Prediction API: one transaction in, one predicted category (with confidence) out.

    uv run uvicorn txlabel.api:app --port 8000        (or: make serve)

The model is loaded once at startup from ``$TXLABEL_MODEL_DIR`` (default: ``models/``, produced by ``make train``).
Features are built by the same code as in training (``txlabel.preprocessing``), through ``TransactionClassifier.predict``.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, model_validator

from txlabel import __version__
from txlabel.artifacts import load_artifacts

log = logging.getLogger("txlabel.api")

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "models"

EXAMPLE_REQUEST = {
    "merchant_name": "CARREFOUR CITY",
    "title": "CARREFOUR CITY",
    "amount_value": -12.5,
    "created_at": "2026-04-28T11:45:21Z",
    "merchant_pan_entry_method": "CHIP_CONTACTLESS",
    "digitalization_type": "apple_pay",
    "pfm_recurring_frequency": "",
}


class Transaction(BaseModel):
    """One card transaction. Only the fields the model uses are declared; any other field (e.g. a full dataset row) is accepted and ignored."""

    model_config = ConfigDict(extra="ignore", json_schema_extra={"examples": [EXAMPLE_REQUEST]})

    amount_value: float = Field(
        ..., allow_inf_nan=False, description="Signed amount. Negative = spending; positive = refund (deterministic rule)."
    )
    created_at: datetime = Field(..., description="ISO 8601 timestamp, e.g. 2026-04-28T11:45:21Z (naive values are read as UTC).")
    merchant_name: str = Field("", max_length=200, description="Merchant name as on the transaction. May be empty (e.g. ATM withdrawals).")
    title: str = Field("", max_length=200, description="Transaction label. Used when merchant_name is empty.")
    merchant_pan_entry_method: str | None = Field(None, max_length=50, description="e.g. CHIP_CONTACTLESS, CARD_ON_FILE, MANUAL, CHIP.")
    digitalization_type: str | None = Field(None, max_length=50, description="Wallet / token requestor, e.g. apple_pay, google_pay, none.")
    pfm_recurring_frequency: str | None = Field(None, max_length=50, description="Detected recurrence, e.g. monthly. Empty if none.")

    @model_validator(mode="after")
    def _needs_a_name(self):
        if not (self.merchant_name.strip() or self.title.strip()):
            raise ValueError("at least one of merchant_name or title must be non-empty")
        return self


class Prediction(BaseModel):
    minor_category: str = Field(..., description="Predicted fine-grained category (12 classes).")
    major_category: str = Field(..., description="Coarse category derived from the minor one ('none' for refunds).")
    confidence: float = Field(..., ge=0, le=1, description="Model probability of minor_category (1.0 for the refund rule).")


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_dir = Path(os.environ.get("TXLABEL_MODEL_DIR", DEFAULT_MODEL_DIR))
    try:
        app.state.model, app.state.metadata = load_artifacts(model_dir)
        log.info("model loaded from %s (trained %s)", model_dir, app.state.metadata.get("trained_at"))
    except Exception:  # keep the app up, report "not ready" on /health and 503 on /predict
        app.state.model, app.state.metadata = None, None
        log.exception("could not load the model from %s (did you run `make train`?)", model_dir)
    yield


app = FastAPI(
    title="Transaction labeling API",
    version=__version__,
    description="Predicts the category of a single card transaction.",
    lifespan=lifespan,
)


def _model(request: Request):
    model = request.app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail="model not available: run `make train` and restart the service")
    return model


@app.get("/health", tags=["service"])
def health(request: Request) -> dict:
    """Readiness: 200 when the model is loaded, 503 otherwise."""
    _model(request)
    return {"status": "ok"}


@app.get("/model-info", tags=["service"])
def model_info(request: Request) -> dict:
    """Metadata of the loaded model (training date, data hash, classes, held-out metrics, library versions)."""
    _model(request)
    meta = request.app.state.metadata
    keys = ("model_name", "trained_at", "git_commit", "saved_model_trained_on", "seed", "data", "classes",
            "heldout_test_metrics_model", "heldout_test_metrics", "library_versions")
    return {k: meta[k] for k in keys if k in meta}


@app.post("/predict", response_model=Prediction, tags=["prediction"])
def predict(tx: Transaction, request: Request) -> Prediction:
    """Classify one transaction."""
    model = _model(request)
    t0 = time.perf_counter()
    row = pd.DataFrame([tx.model_dump(mode="json")])
    pred = model.predict(row).iloc[0]
    out = Prediction(
        minor_category=pred["minor_category"],
        major_category=pred["major_category"],
        confidence=round(float(pred["confidence"]), 4),
    )
    log.info("predict -> %s (%.3f) in %.1f ms", out.minor_category, out.confidence, (time.perf_counter() - t0) * 1000)
    return out
