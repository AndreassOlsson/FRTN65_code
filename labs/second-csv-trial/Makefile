# Run from this folder. `uv sync` first.
.PHONY: data test explore sweep

data: data/credit-g.csv

data/credit-g.csv:
	uv run python -m creditg.data

test: data
	uv run pytest -q
