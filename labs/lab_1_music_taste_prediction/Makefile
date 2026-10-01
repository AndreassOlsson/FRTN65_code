# Run from this folder. `uv sync` first.
URL := http://handsonml.control.lth.se/data
FILES := data/training_data.csv data/songs_to_classify.csv

.PHONY: data test explore baselines clean-data

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

clean-data:
	rm -rf data
