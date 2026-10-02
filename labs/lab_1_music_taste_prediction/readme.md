# Lab 1 Music Taste Prediction

## How to run

From this folder, with [uv](https://docs.astral.sh/uv/) installed:

```sh
uv sync        # the environment, from pyproject.toml and uv.lock
make data      # fetches training_data.csv and songs_to_classify.csv into data/ (gitignored)
make test      # the contract, the protocol, the decision rule and the submission string
make explore   # writes figures/ and prints the tables behind results/01-exploration.md
make baselines # dummy, logistic regression and kNN through the protocol: results/02-baselines.csv
make sweep     # every other method on the same splits: results/03-sweep.csv; about 36 minutes on 4 cores
make predict   # refits the chosen method and writes results/submission-<date>.txt; about a minute
make variants FAMILY=linear   # the variants stage under the screening protocol, one family (linear, knn, svm, trees)
make synthesis # the stage closed: the promoted variant under the full protocol, the rule on the union, the cross-family figures; about an hour the first time
```

So `uv sync && make data && make test && make sweep && make predict`
reproduces everything: the sweep's CSV byte for byte, the figure, the
decision and the submission string.

`DESIGN.md` is the design the code follows. The package is
`src/songtaste/`, one module per stage: `data.py` the contract,
`explore.py` the exploration, `features.py` the preprocessor,
`models.py` the registry of methods, `evaluate.py` the protocol,
`report.py` the comparison and the decision, `predict.py` the final
refit and the submission, `variants.py` the same families with their
knobs turned (scaler, log, encoding, feature subset, interactions,
regularisation, kernel) under a screening protocol, `diagnostics.py`
the curves and importances that say why (validation and learning
curves, permutation importance, out-of-fold ROC and confusion, the
decision rule under its neighbours), `synthesis.py` the stage's closing
steps (promotion to the full protocol, the rule on the union with the
sweep, the feature screen, importance and learning curves across six
methods, into `results/04-variants.md` and `figures/04-synthesis/`). `notebooks/` are the places to look at
things (`04_variants.ipynb` drives the variants stage a family at a
time, each run appending to `results/04-variants.csv`), `results/` holds the written findings (`protocol.md` first,
then one file per stage), `instructions/` the files given to us (the
spec, the sample code, a sample of the data's shape). Data files are
never committed.

## The hand-in

`handin/` holds what goes to Canvas. `slides.md` is the source of the
presentation and `slides.pptx` is built from it
(`uv run --with python-pptx python handin/make_slides.py`, which
pulls python-pptx in for that one run). `bash handin/build.sh` (or
`make handin`) zips the slides with everything above, minus `data/`,
`.venv/` and `instructions/`, into `handin/Andreas-Olsson.zip`. The
zip is a build product and is not committed.

The submission string is the newest `results/submission-<date>.txt`:
200 characters, one per row of `songs_to_classify.csv` in the file's
order, 1 like and 0 dislike. `songs_to_classify.csv` is read for
prediction only there, by the method `results/03-sweep.md` chose
(the random forest; the variants stage later moved the rule's choice to
`knn_top4`, `results/protocol.md` 2026-10-02, and the string was not
remade)
(`data.py` and `explore.py` check its contract and describe it, and
nothing else opens it).

## About the training data

This is the readme file to the files songs_to_classify.csv and
training_data.csv.

The files contain the (unlabeled) test data and the (labeled)
training data, respectively.

The columns represent features, as specified by the header and
documented in the instructions. The column "label"
(training_data.csv. only) is encoded as 1 = like, 0 = dislike.

The files can be loaded into Python using, e.g., panda as

import pandas as pd
training=pd.read_csv('training_data.csv', sep=',')

The files were created by Andreas Svensson in August 2018 using
Spotipy (https://spotipy.readthedocs.io/)
