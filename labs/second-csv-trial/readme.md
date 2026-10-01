# Second CSV trial: credit-g through songtaste

A second binary tabular problem put through lab 1's package as it
stands, to see which of its seams were real (LIF-181). OpenML's
credit-g: 1000 loan applicants, 7 numeric and 13 categorical
attributes, good or bad credit risk, and a cost matrix that makes a
missed bad applicant five times as costly as a wrongly refused good one.

## How to run

From this folder, with [uv](https://docs.astral.sh/uv/) installed:

```sh
uv sync        # installs this folder and lab 1's songtaste from ../lab_1_music_taste_prediction
make data      # OpenML dataset 31 into data/credit-g.csv (gitignored)
make test      # the contract, the cost scorer, the protocol through songtaste
make explore   # figures/01-*.png and the tables behind results/01-exploration.md
make sweep     # every registry method on 25 splits: results/sweep.csv, figures/sweep-by-split.png; about an hour
```

Only `src/creditg/` is this folder's own: `data.py` the spec and the
schema, `explore.py` and `sweep.py` the drivers that call songtaste.
`results/protocol.md` was committed before any score; `results/sweep.md`
holds the reading and the rule's decision. What the trial found about
the package is in lab 1's `DESIGN.md` ("2026-10-01: the second CSV").
