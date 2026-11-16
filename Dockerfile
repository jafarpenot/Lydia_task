FROM python:3.12-slim

# XGBoost needs the OpenMP runtime on Linux
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1

# dependencies first (cached layer), then the code and the committed model
COPY pyproject.toml uv.lock .python-version README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev

COPY models ./models

ENV PATH="/app/.venv/bin:$PATH" TXLABEL_MODEL_DIR=/app/models
EXPOSE 8000
CMD ["uvicorn", "txlabel.api:app", "--host", "0.0.0.0", "--port", "8000"]
