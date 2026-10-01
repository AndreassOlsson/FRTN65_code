# Run from this folder. `uv sync` first.
URL := http://handsonml.control.lth.se/data
FILES := data/training_data.csv data/songs_to_classify.csv

.PHONY: data test explore clean-data

data: $(FILES)

data/%.csv:
	mkdir -p data
	curl -fsSL -o $@ $(URL)/$*.csv

test: data
	uv run pytest -q

explore: data
	MPLBACKEND=Agg uv run python -m songtaste.explore

clean-data:
	rm -rf data
