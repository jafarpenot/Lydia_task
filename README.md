# Transaction labeling

Classify a card transaction into a `minor_category` (12 classes) and a `major_category` (8 classes) from the information available at
transaction time, and serve the prediction through a small local API. Task description: `docs/data-scientist-technical-test-transaction-labeling.pdf`.

**Result (held-out test, 1,601 rows):** minor accuracy **0.934**, macro-F1 **0.918**; major accuracy **0.946**, macro-F1 **0.921**.
A merchant-name lookup baseline gets 0.909 / 0.886. EDA suggests a realistic ceiling around 96% because of label noise (see below).

## Quick start

```bash
brew install libomp     # macOS only, once: XGBoost needs the OpenMP runtime
uv sync                 # creates .venv from uv.lock (Python 3.12)
make serve              # API on http://127.0.0.1:8000 (interactive docs: /docs)
```

A trained model is committed (`models/`), so the API works right after cloning. No `uv`? `pip install -r requirements.txt` and
`pip install -e .` work too (Python 3.12).

Request and response:

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d @examples/transaction.json
# {"minor_category":"supermarkets","major_category":"groceries","confidence":0.983}
```

| endpoint | purpose |
|---|---|
| `POST /predict` | one transaction in, one category (+ derived major category and confidence) out |
| `GET /health` | 200 when the model is loaded, 503 otherwise |
| `GET /model-info` | training date, data hash, classes, held-out metrics, library versions |

**Request fields**

| field | required | notes |
|---|---|---|
| `amount_value` | yes | signed amount; negative = spending, positive = refund |
| `created_at` | yes | ISO 8601 (naive timestamps are read as UTC) |
| `merchant_name`, `title` | at least one | `title` is used when `merchant_name` is empty (e.g. ATM withdrawals) |
| `merchant_pan_entry_method`, `digitalization_type`, `pfm_recurring_frequency` | no | defaults are used when missing |

Any other field (e.g. a full row of the dataset) is accepted and ignored. Invalid input returns `422` with the reason; `503` if the model is not loaded.
Examples:

```bash
# ATM withdrawal: empty merchant name, title only
curl -s -X POST localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"title":"RETRAIT DAB","amount_value":-60,"created_at":"2026-05-02T09:00:00Z"}'
# {"minor_category":"atm","major_category":"finance","confidence":0.9687}

# positive amount -> refund rule
curl -s -X POST localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"merchant_name":"APPLE.COM/BILL","amount_value":6.99,"created_at":"2026-05-02T09:00:00Z"}'
# {"minor_category":"refund","major_category":"none","confidence":1.0}
```

## Training

```bash
make train      # = uv run python -m txlabel.train   (about 10 s)
make test       # = uv run pytest -q
```

Options: `--seed`, `--test-size`, `--data`, `--model-dir`, `--reports-dir`, `--no-refit`.

What it does: load the CSV, drop the 120 exact `ref_id` duplicates, stratified 80/20 split (seed 42), fit on the train part, evaluate once on the
held-out part, **then refit on all 8,001 rows** (the model that is served), and write `models/model.joblib`, `models/metadata.json` (seed, data SHA-256,
classes, features, parameters, held-out metrics, git commit, library versions), `reports/metrics.json` and `reports/classification_report.txt`.
The reported metrics are those of the train-only model; the served model has seen all the data. The committed model must be regenerated (`make train`)
and committed again whenever the code or data change.

## Approach and key design choices

The analysis is in `notebooks/01_eda.ipynb` (exploration, with written findings) and `notebooks/02_modeling.ipynb` (baselines, XGBoost, error analysis).
The code that ships lives in `src/txlabel/`:

| module | role |
|---|---|
| `preprocessing.py` | the only place where raw columns become features; refund rule; minor to major mapping |
| `model.py` | `TransactionClassifier`: refund rule + XGBoost pipeline |
| `train.py`, `data.py`, `evaluate.py`, `artifacts.py` | training CLI, loading/dedup, metrics, model + metadata I/O |
| `api.py` | FastAPI service |

**Findings from the EDA that shaped the design**
- `refund` is exactly the positive amounts (123 of 123, no exception). It is therefore a **deterministic rule**, and the classifier is trained on the other 11 classes.
  `status == 'refunded'` matches too, but it is probably not known at transaction time, so it is not used.
- The minor to major mapping is a strict hierarchy; refunds have an empty major category in the data. **Major is derived from the predicted minor** (refund -> `none`),
  which guarantees consistent pairs. Predicting both targets with separate models was not compared yet (see "Not done").
- 120 `ref_id` values are exact duplicates (only `transaction_id` differs): removed before splitting so that a copy cannot leak across the split.
- Labels are noisy: 3.9% of non-refund rows contradict the majority label of their merchant, at a flat rate across amount, entry method, hour and merchant size.
  A pure in-sample name lookup reaches 94.7% (96.2% with the amount sign), so about **96% is a realistic ceiling**.
- `title` equals `merchant_name` except for 113 rows with an empty merchant name (mostly ATM withdrawals): one text field with a fallback, plus an "empty name" flag.
- Dropped: `transaction_id`, `ref_id`, `member_id`, `member_name`, `merchant_id` (identifiers: no signal, do not generalize, and member data could bias the model or raise privacy
  issues), `card_type`, `amount_currency`, `fees` (constant or near-constant), `status`, month (only 3 partial months), `card_name` (30 values, many FR/EN duplicates;
  to be cleaned and re-tested).

**Features and model**
- Merchant name: lowercase, accents removed, non-alphanumerics to spaces; then the presence of character 2 to 4-grams (vocabulary capped at 2,000, `min_df=3`).
  Character n-grams handle spelling variants and unseen names without any matching step.
- Entry method (rare values grouped), `digitalization_type`, recurring flag, empty-name flag, |amount|, hour, day of week, day of month.
- XGBoost (300 trees, depth 6, learning rate 0.1), fixed hyper-parameters, no tuning, no class weights: a first iteration kept simple on purpose.
- Name normalization and fuzzy matching: the EDA explores Levenshtein-based clustering of merchant names (similarity 0.85 gives 930 names -> 867 clusters, with a few bad merges
  such as names differing only by a store number). It is not used in the model; character n-grams cover the same need so far.

**Training/serving consistency.** The API calls the same `TransactionClassifier.predict` as training and the notebooks, so features are computed by one implementation.
Tests check that single-row and batch predictions agree, and that the API returns exactly what the model returns on real dataset rows.

## Results

Test split: 1,601 rows (stratified, seed 42, after deduplication). Refunds are included in the evaluation.

| model | minor accuracy | minor macro-F1 | major accuracy | major macro-F1 |
|---|---|---|---|---|
| majority class | 0.292 | 0.038 | 0.292 | 0.050 |
| refund rule + merchant-name lookup | 0.909 | 0.886 | 0.920 | 0.893 |
| **refund rule + XGBoost** | **0.934** | **0.918** | **0.946** | **0.921** |

Per-class precision/recall/F1 for both targets: `reports/classification_report.txt`. Refund is perfect (rule) and `atm` has recall 1.0. The weakest class is
`others_shopping` (recall 0.56, F1 0.70).

**Error analysis** (non-refund test rows: 1,577, of which 106 errors)

| cause | rows | errors | error rate | share of errors |
|---|---|---|---|---|
| label deviates from the merchant's majority label | 69 | 66 | 96% | 62% |
| merchant unseen in train | 91 | 26 | 29% | 25% |
| label equals the merchant's majority (genuine model error) | 1,417 | 14 | 1.0% | 13% |

- Most errors are not really model errors: when a row's label contradicts the usual label of its merchant, the model follows the merchant. If such rows are
  unpredictable, the ceiling on this split is about 95.6% (the model scores 93.3% on non-refund rows).
- The main remaining lever is **unseen merchants** (29% errors against 4 to 7% for known merchants). A random split has only 5.7% unseen merchants, so performance on
  genuinely new merchants may be lower than the headline number.
- Common confusions: restaurants vs supermarkets, betting vs multimedia, restaurants vs transport_means (possibly delivery apps or cafeterias inside supermarkets),
  `others_shopping` vs supermarkets / multimedia. The most confident errors are of this kind, so confidence does not flag them.
- About 99% of the model's importance is on the merchant-name n-grams. A quick ablation (single split, run outside the notebooks, not reproducible from the repo yet) gave
  0.9353 accuracy with the name alone and 0.9328 with all features: the structured features add nothing measurable.
  The hourly patterns visible in the EDA reflect which merchants are visited when, not time itself.

## Limitations and honest caveats

- **No validation set.** Only a train/test split was used. The test set was never used to fit or tune anything (fixed hyper-parameters, no early stopping), but the error analysis
  and the ablation were done on test rows, so any future model or feature decision based on them would leak into the test score. Planned fix: keep the test set sealed,
  do selection and analysis with 5-fold stratified cross-validation on the training part (mean and std), and evaluate on the test set once at the end.
- **A single random split** of 1,601 rows: small classes have 24 to 39 test rows, so per-class numbers and differences under about 1 point are not reliable. No confidence intervals.
- Grouped-by-merchant and temporal splits were not evaluated. The data covers about 3 months (2026-03-13 to 2026-06-10), with partial first and last months.
- The served model is refit on all data, so it was not scored itself.
- Hyper-parameters are untuned and there are no class weights.
- Amounts contain implausible outliers (65 above 10,000 EUR); tree models are robust to them, but the values were not investigated further.
- The refund rule assumes the sign convention holds in production (positive = money in).
- The committed model is a pickle: it is only guaranteed to load with the library versions in `models/metadata.json` (`uv.lock` pins them; the API warns on a mismatch).

## Not done (ideas)

1. **Unseen merchants:** nearest-known-merchant features (rapidfuzz), embeddings, handling of truncated names; test whether the Levenshtein clusters add anything over the n-grams.
2. **Evaluation:** cross-validation with confidence intervals, merchant-grouped and temporal splits; make the ablation a notebook cell.
3. **Targets:** compare predicting minor only (current), two independent models, a hierarchical model, and a multi-output model.
4. **Model:** stacking (logistic regression on the n-grams, out-of-fold probabilities into XGBoost), class weights or per-class thresholds for small classes, calibration and an
   "unsure" fallback for low-confidence predictions.
5. **Data:** clean and re-add `card_name` (merge FR / EN duplicates); investigate the deviating labels with domain knowledge (cafeterias in supermarkets, delivery apps) to
   separate noise from real second behaviours; reconsider keeping the structured features, which add no measurable value.

## Production notes

- **Model serving:** the model is loaded once at startup; `/health` reports readiness (503 until a model is available) and `/model-info` exposes what is running.
  Inference is one small DataFrame per request (about 50 ms per request with curl overhead on a laptop); batch inference would be a straightforward extension.
- **Reproducibility:** `uv.lock` pins dependencies, fixed seed, data hash and library versions are stored in the model metadata, tests cover training reproducibility and parity.
- **Packaging:** a `Dockerfile` is provided (see below). For other environments a portable format (e.g. ONNX) would remove the dependency on matching library versions.
- **Monitoring and retraining (not implemented):** log predictions and confidence (without personal data), track the share of unseen merchant names and the confidence distribution as drift
  signals, and retrain on a schedule or on drift. New merchants are the main failure mode, so a feedback loop (user corrections or manual review of low-confidence predictions) would
  feed the next training set.
- **Privacy:** member identifiers and names are not used as features and are ignored by the API.

## Docker

```bash
docker build -t txlabel .
docker run --rm -p 8000:8000 txlabel
```

The image installs the locked dependencies and serves the committed model.
Note: the Dockerfile was written but **not built or run** by the author (the Docker daemon was not running on the development machine); the non-Docker path above was tested from a fresh clone.

## Repository layout

```
data/raw/            the provided CSV
notebooks/           01_eda.ipynb, 02_modeling.ipynb (executed, with findings)
src/txlabel/         package: preprocessing, model, training, evaluation, artifacts, API
tests/               33 tests (features, model, training, API)
models/, reports/    committed trained model + metadata, held-out metrics
examples/            example request
docs/                task PDF and the working plan / decision log (PLAN.md)
```
