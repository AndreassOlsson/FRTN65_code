# Run from this folder. `uv sync` first.
.PHONY: data test explore sweep

data: data/credit-g.csv

data/credit-g.csv:
	uv run python -m creditg.data

test: data
	uv run pytest -q

explore: data
	uv run python -m creditg.explore

# every registry method on the same 25 splits; about an hour on 4 cores
sweep: data
	uv run python -m creditg.sweep
