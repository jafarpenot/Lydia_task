# Project plan and decision log

Task: Lydia data scientist technical test, transaction labeling (see `docs/data-scientist-technical-test-transaction-labeling.pdf`).
Predict `minor_category` (12 classes) and `major_category` (8 classes) from one transaction, expose it through a local prediction API,
and write a README (approach, results, how to run training and the API). Deliver as a Git repo link by email within 7 days
(email must also state time spent and a self-rated quality). Suggested effort: about half a day; unfinished work is fine if the rest is explained.

## Status

| Step | Status |
|---|---|
| 1. Repo and environment setup | done (commit `4fd4208`) |
| 2. EDA notebook (`notebooks/01_eda.ipynb`) | done (commit `0f3fe0b`) |
| 3. Evaluation design (split, metrics) | first version done: stratified random 80/20 after dedup, macro-F1 primary; grouped/temporal splits still open |
| 4. Baselines + first XGBoost (`notebooks/02_modeling.ipynb`) | done: accuracy 0.934, macro-F1 0.918 (lookup baseline 0.909 / 0.886) |
| 5. Error analysis | brief version done (in notebook 02, section 5); ideas listed in its section 6 |
| 6. Training pipeline in `src/txlabel/` + CLI | done: `make train` (train/eval on held-out test, then refit on all rows for the saved model; metadata + reports) |
| 7. Prediction API (FastAPI) + tests | done: `make serve`; `POST /predict` (one category + confidence + derived major), `GET /health`, `GET /model-info`; 33 tests pass |
| 8. README, Dockerfile, final fresh-clone check | done, except: Dockerfile not built/run (Docker daemon was off); fresh clone tested with uv (33 tests pass, API answers, retrain reproduces 0.9338 / 0.9177) and with pip + requirements.txt (model loads and predicts). Remaining: GitHub push, email |

## Decisions made

- Local git repo only for now. The user will create a GitHub repo later and we push then (no remote exists yet).
- The raw CSV is committed (`data/raw/`), so reviewers can reproduce everything.
- Tooling: **uv** (`pyproject.toml` + `uv.lock`), plus an exported `requirements.txt` for pip users. Python 3.12.
- Docker: include a Dockerfile, but at the end.
- Start simple (baseline first) and iterate.
- Notebooks explore; code that ships lives in `src/txlabel/` and is imported by notebooks, training and the API.

## Open decisions (deliberately deferred)

1. **Split strategy.** Candidates: random stratified (primary), merchant-grouped (stress test for unseen merchants), temporal (only about 3 months of data, partial first and last month).
   Whatever we pick must keep `ref_id` duplicates together (or dedupe first). Test set stays untouched until the final evaluation.
2. **Target strategy.** The user wants to compare predicting both targets vs minor only (major derived from the strict mapping). Options: minor only + mapping,
   two independent models, hierarchical, multi-output. Choose on measured results.
3. **Major for `refund`.** All 123 refund rows have an empty `major_category`. Options: extra class (`refund`/`none`), or exclude them from the major task.
4. Whether Levenshtein/fuzzy name clustering adds anything over character n-gram TF-IDF (measure it).

## README checklist (document everything clearly at the end)

- Setup: uv, Python 3.12, `brew install libomp` on macOS (XGBoost), what happens without it.
- Approach: refund as a deterministic rule (amount > 0, 123/123); XGBoost on the 11 other classes; major derived from minor via the strict hierarchy (refund -> `none`).
- Features and why: char 2-4-gram presence of the normalized merchant name (2,000 cap), amount, entry method (rare values grouped), digitalization, recurring, empty-name, hour/weekday/day-of-month. Dropped columns and why (see EDA notes).
- Dedup on `ref_id` (120 exact duplicates) before splitting.
- Label noise and the ~96% ceiling; the error analysis; the structured-features ablation (name alone is as good as everything).
- **Evaluation caveats (decided to leave as is for now, document honestly):**
  - Only a train/test split was used, no validation set. The test set was never used to fit or tune (fixed hyper-parameters, no early stopping), but the error analysis and the ablation were done on test rows.
    Any future model/feature decision based on them would leak into the test score.
  - Planned fix: keep the test set sealed; do model selection, ablations and error analysis with 5-fold stratified CV on the training part (mean +- std), evaluate on test once at the end.
    Alternative: a 60/20/20 split (noisier for small classes: 24 to 39 rows per class on a 20% slice).
  - Single split of 1,601 test rows: per-class numbers and differences under about 1 point are not reliable; no confidence intervals yet.
  - A random split has only 5.7% unseen merchants; performance on new merchants is worse (29% error vs 4-7%). Grouped-by-merchant and temporal splits still to evaluate.
- Ideas not done: see `notebooks/02_modeling.ipynb`, section 6.
- Production notes: pickle tied to library versions (recorded in metadata), parity tests, monitoring and drift, retraining, new merchants, feedback loop, portable formats.
- The served model is refit on all deduplicated rows; reported metrics are those of the train-only model on the held-out test split (default of `make train`, `--no-refit` to disable).
- API contract: required `amount_value`, `created_at`, and at least one of `merchant_name` / `title`; optional entry method, digitalization, recurring; extra fields ignored; 422 on invalid input, 503 if the model is not loaded.
- Decide whether to commit `models/model.joblib` (about 5 MB) so the API runs right after cloning, or require `make train` first.
- Email answers: time spent and quality rating.

## Key EDA findings (details in `notebooks/01_eda.ipynb`, section 12)

- 8,121 rows, 2026-03-13 to 2026-06-10. Strict minor to major hierarchy; `refund` has empty major.
- `refund` is exactly the positive amounts (and `status == 'refunded'`, which we should not use as a feature: probably unknown at transaction time).
- 120 `ref_id` pairs are exact duplicates (only `transaction_id` differs): deduplicate before splitting.
- Label noise: 3.9% of non-refund rows contradict their merchant's majority label. A name lookup gives 94.7%, with amount sign 96.2%: **realistic accuracy ceiling is about 96%**.
- Drop: `card_type`, `amount_currency`, `fees`, `month`, ids, `member_name`, `card_name`, `status`. Keep: merchant text (merchant_name, fallback title), amount sign, log |amount|,
  entry method, digitalization type, recurring flag, hour, empty-name flag.
- 65 amounts above 10,000 EUR are implausible; use log scale or tree models.
- Normalization: lowercase, strip accents, non-alphanumerics to spaces (960 raw names to 930). Levenshtein clustering at 0.85 gives 867 clusters, with a few bad merges.
- Primary metric: macro-F1 (imbalance up to 20x); accuracy interpreted against the ~96% ceiling.

## Planned work

### 3. Evaluation design
Pick the split(s) and metrics (macro-F1, per-class precision/recall, confusion matrices). Fixed seeds. Deduplicate first.

### 4. Baselines then iteration
1. Majority class (floor).
2. Rules: positive amount means refund, plus exact merchant lookup.
3. TF-IDF character n-grams on the normalized name + logistic regression (main baseline).
4. Add structured features; try linear SVM, LightGBM/XGBoost; class weights.
5. Compare target strategies; decide refund handling for major.
6. Optional: fuzzy-match feature, embeddings (low priority).

### 5. Error analysis
Confusion pairs, errors by merchant frequency (seen vs rare), label noise vs missing features, confidence and an "unsure" fallback.

### 6. Training pipeline
- One sklearn `Pipeline` (preprocessing + model) taking the **raw** schema, saved as a single artifact (joblib) + metadata JSON (versions, metrics, classes, data hash).
- CLI: `python -m txlabel.train` (and `make train`). Config in a file or dataclass, not scattered constants.
- Training/serving parity: one implementation imported by both sides; preprocessing inside the pipeline; parity tests
  (batch vs single prediction, API vs direct pipeline, golden examples); library versions pinned and recorded in metadata.
- Moving notebook code to the package: prototype with pure functions, move to `src/`, delete from the notebook, import it back, and check that notebook metrics equal CLI metrics.

### 7. Prediction API
- FastAPI + uvicorn, model loaded once at startup.
- `POST /predict` (one transaction in; minor and major, confidences, top-k alternatives, model version out), `GET /health`, optionally `GET /model-info`.
- Pydantic input schema with only the fields the model uses, clear 422 errors, handling of unseen merchants and missing values.
- Logging without personal data; tests with TestClient.

### 8. README and submission
README with approach, assumptions, results, error analysis, how to run training and the API (example request/response), and a "next steps in production" section
(monitoring and drift, retraining, new merchants, feedback loop, portable model formats). Then a fresh-clone run of install, train, serve, request.
Dockerfile at the end. Final email: repo link, time spent, quality rating.

## How to resume
Run `uv sync`, open `notebooks/01_eda.ipynb`, and continue with step 3. Reading this file plus section 12 of the notebook gives the full context.
