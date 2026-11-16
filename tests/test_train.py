import json
from pathlib import Path

import pandas as pd
import pytest

from txlabel.artifacts import load_artifacts
from txlabel.data import load_raw
from txlabel.train import run_training

DATA = Path(__file__).resolve().parents[1] / "data" / "raw" / "transaction-dataset-1.csv"


@pytest.fixture(scope="module")
def small_csv(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "sample.csv"
    load_raw(DATA).sample(1500, random_state=0).to_csv(path, index=False)
    return path


def test_training_saves_artifacts_and_is_reproducible(small_csv, tmp_path):
    kw = dict(data_path=small_csv, seed=1, xgb_params={"n_estimators": 20})
    m1 = run_training(model_dir=tmp_path / "m1", reports_dir=tmp_path / "r1", **kw)
    m2 = run_training(model_dir=tmp_path / "m2", reports_dir=tmp_path / "r2", **kw)

    for d in ("m1",):
        assert (tmp_path / d / "model.joblib").exists() and (tmp_path / d / "metadata.json").exists()
    assert (tmp_path / "r1" / "metrics.json").exists() and (tmp_path / "r1" / "classification_report.txt").exists()

    # same data + same seed -> same metrics
    assert m1["heldout_test_metrics"] == m2["heldout_test_metrics"]
    assert m1["data"]["sha256"] == m2["data"]["sha256"]
    assert {"python", "scikit-learn", "xgboost"} <= set(m1["library_versions"])
    assert "refund" in m1["classes"]
    assert json.loads((tmp_path / "m1" / "metadata.json").read_text())["seed"] == 1


def test_saved_model_reloads_and_predicts(small_csv, tmp_path):
    run_training(small_csv, tmp_path / "m", tmp_path / "r", seed=1, xgb_params={"n_estimators": 20})
    model, meta = load_artifacts(tmp_path / "m")
    row = load_raw(small_csv).head(5)
    pred = model.predict(row)
    assert len(pred) == 5 and set(pred.columns) == {"minor_category", "major_category", "confidence"}
    assert meta["classes"] == model.classes_


def test_refit_on_all_changes_training_set_only(small_csv, tmp_path):
    meta = run_training(small_csv, tmp_path / "m", tmp_path / "r", seed=1, refit_on_all=True, xgb_params={"n_estimators": 20})
    assert meta["saved_model_trained_on"] == "all deduplicated rows"
