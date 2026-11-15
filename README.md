# Transaction labeling

Work in progress. Classifies card transactions into `minor_category` / `major_category`
(see `docs/` for the task description). Full write-up to come.

## Setup

```bash
uv sync            # creates .venv from uv.lock (Python 3.12)
```

On macOS, XGBoost needs the OpenMP runtime: `brew install libomp` (once per machine).
Without it, `import xgboost` fails with a `Symbol not found: ___kmpc_dispatch_deinit` error.
