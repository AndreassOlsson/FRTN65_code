# Run from this folder. `uv sync` first.
URL := http://handsonml.control.lth.se/data
FILES := data/training_data.csv data/songs_to_classify.csv
FAMILY ?= linear

.PHONY: data test explore baselines sweep variants predict handin clean-data

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

# the variants stage under the screening protocol, one family at a time
# (linear, knn, svm, trees); appends to results/04-variants.csv, skips what is there
variants: data
	uv run python -m songtaste.variants --family $(FAMILY)

# the one read of songs_to_classify.csv: refit the method 03-sweep.md chose,
# write results/submission-<date>.txt and print it; about a minute
predict: data
	uv run python -m songtaste.predict

# handin/Andreas-Olsson.zip from a clean checkout of this folder
handin:
	bash handin/build.sh

clean-data:
	rm -rf data
