# Lab 1 Music Taste Prediction

## How to run

From this folder, with [uv](https://docs.astral.sh/uv/) installed:

```sh
uv sync        # the environment, from pyproject.toml and uv.lock
make data      # fetches training_data.csv and songs_to_classify.csv into data/ (gitignored)
make test      # the data contract against both files
make explore   # writes figures/ and prints the tables behind results/01-exploration.md
```

`DESIGN.md` is the design the code follows. The package is
`src/songtaste/` (`data.py` the contract, `explore.py` the exploration),
`notebooks/` are the places to look at things, `results/` holds the
written findings, `instructions/` the files given to us (the spec, the
sample code, a sample of the data's shape). Data files are never
committed. The final hand-in will be bundled as a zip once done.

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
