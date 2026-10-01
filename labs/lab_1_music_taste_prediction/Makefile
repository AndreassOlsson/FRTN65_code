# Run from this folder. `uv sync` first.
URL := http://handsonml.control.lth.se/data
FILES := data/training_data.csv data/songs_to_classify.csv

.PHONY: data test explore baselines sweep clean-data

data: $(FILES)

data/%.csv:
	mkdir -p data
	curl -fsSL -o $@ $(URL)/$*.csv

test: data
	uv run pytest -q

explore: data
	MPLBACKEND=Agg uv run python -m songtaste.explore

baselines: data
	uv run python -m songtaste.evaluate

# every other registry entry; reuses 02-baselines.csv, about 40 minutes on 4 cores
sweep: data
	uv run python -m songtaste.report

clean-data:
	rm -rf data
