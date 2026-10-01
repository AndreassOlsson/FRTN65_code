# Second CSV trial

OpenML credit-g (1000 loan applicants, good or bad credit risk) put
through lab 1's `songtaste` package where it stands, to test which of
its seams were real (LIF-181). Nothing of the package is copied: it is
imported through a uv path source.

    uv sync && make data && make test && make explore && make sweep

- `src/creditg/data.py`: the contract, a `FeatureSpec` and a pandera
  schema; `make data` fetches OpenML dataset 31 into gitignored `data/`.
- `results/01-exploration.md`: what the data looks like.
- `results/protocol.md`: the protocol and the decision rule, committed
  before any model was scored.
- `results/sweep.md`, `results/sweep.csv`: every registry method, the
  paired comparison and the decision.

What had to change in the package, and what that says about where it
belongs, is the diary in the vault's record
(`initiatives/Modelling and Learning from Data/efforts/lab1-music-classifier/record/05-second-csv.md`).
