.PHONY: install train serve test

install:
	uv sync

train:
	uv run python -m txlabel.train

serve:
	uv run uvicorn txlabel.api:app --host 127.0.0.1 --port 8000

test:
	uv run pytest -q
