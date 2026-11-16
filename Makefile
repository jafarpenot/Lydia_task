.PHONY: install train test

install:
	uv sync

train:
	uv run python -m txlabel.train

test:
	uv run pytest -q
