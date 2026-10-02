# Lab 1 as one worked instance of the ML lifecycle

Planned 2026-10-01 in life-os as the effort `lab1-music-classifier`
(sketch in the vault at
`initiatives/Modelling and Learning from Data/efforts/lab1-music-classifier/sketch.md`).
This file is the design the code in this folder follows; the vault
holds the records and thinking.

## What this is for

The lab asks for exploration, preprocessing, feature selection, several
classifiers tuned and evaluated by cross-validation, one chosen for
production with a motivation, and a 5 to 15 slide hand-in with
reproducing code. Done here as a small package plus thin notebooks, so
that the mechanism (contract, features, protocol, comparison, report)
is reusable on the next tabular problem and the notebooks stay the
place for looking at things.

## The shape

Package `songtaste` in `src/`, managed with `uv`, tests in `tests/`,
data fetched from the course server into gitignored `data/`.

| stage | stands on | what we write |
|---|---|---|
| contract | pandera schema, a FeatureSpec dataclass | the spec for these 13 columns (numeric; categorical: key, mode, time_signature) |
| explore | seaborn, pandas describe | a short module calling them with the spec |
| features | sklearn ColumnTransformer, OneHotEncoder, StandardScaler | the builder from the spec |
| models | sklearn estimators and Pipeline | a registry: name, factory, search space |
| protocol | sklearn cross_validate, RepeatedStratifiedKFold, GridSearchCV inside the folds | the protocol as one function, results as a tidy DataFrame |
| compare | sklearn's model-comparison example (corrected resampled t-test), pandas | the paired comparison and the decision rule |
| report | pandas, seaborn, python-pptx | the comparison table, the figures, the prediction string |
| tracking | a results CSV per stage in git, seeds fixed | nothing more for one lab |

Rules the tasks follow:

- No layer of our own where a maintained package does the job; the
  work is using it correctly.
- The protocol (metric, folds, seeds, tuning inside the folds) and the
  decision rule are written in `results/protocol.md` before any model
  result exists, and never edited after.
- Model selection uses only `training_data.csv`. `songs_to_classify.csv`
  is touched once, by the chosen model, to produce the submission string.
- Config is a dataclass. One dataset does not earn a config framework.
- The notebooks are Andreas's working surface; a pipeline task reads
  them and does not rewrite them.

## Stages

1. data-in: skeleton, contract, exploration and a written finding.
2. protocol-and-baselines: features, protocol, registry; dummy,
   logistic regression, kNN.
3. the-sweep: LDA, QDA, tree, random forest, bagging, boosting, SVM;
   paired comparison; the decision.
4. ship: submission string, slides, the zip. Due 2026-10-04.
5. second-csv: after the deadline, a second binary CSV problem through
   the same package, recording what had to change.

## 2026-10-01: the second CSV

Task 5 put OpenML's credit-g through this package as it stood, from
`labs/second-csv-trial/`, importing `songtaste` by a uv path source
(LIF-181; the diary is the vault record `05-second-csv.md`).

What held untouched: `features.py`, the registry's mechanism in
`models.py`, `run_protocol` and `run_many`, the corrected errors, and
`report`'s comparison, paired tests and decision rule. What had to
change, each a parameter where lab 1 had a constant, none altering a
lab 1 number: `FeatureSpec.classes` (the label's values and names, the
positive class last) replacing explore's use of `LABELS`; explore's
tables saying rows, its categorical figure wrapping; `Protocol.scorers`
so a score need not be an sklearn name; `Protocol.rope` and the
figure's axis taken from the protocol.

**The package stays here.** The split into a repo-level package is
not yet earned: the second consumer is a trial that exists to test it,
the hand-in zip reproduces from this folder alone until the
resubmission window closes (2026-10-23), and the trial left open which
half is generic. A third dataset earns the move if it is real work
(not a test) and shows three things: that the registry's search spaces
can be shared (on credit-g, tuned on cost, kNN, the tree and RBF SVM
chose the edges of lab 1's grids); whether a decision-threshold stage
belongs in the protocol (no method beat rejecting every applicant at
the default threshold); and how missing values enter the preprocessor
(neither dataset had any). Until then the generic functions keep
taking `spec` and `protocol` explicitly, and the lab-1 defaults in
their signatures are a known hazard, failing loudly on foreign columns.

## 2026-10-02: the variants layer

His exploration (`exploration.ipynb`) asked two things the stages above
do not answer: which features are robustly important across methods,
and what each family is sensitive to (regularisation, scaling, feature
engineering, kernels). Added as a layer over the package, not a second
one: `variants.py` builds a `Method` from a registry entry plus what
differs (scaler, log1p columns, encoder, feature subset, interactions,
estimator parameters, grid), so `evaluate.run_protocol` scores a variant
as it scores a method; `diagnostics.py` wraps sklearn's
`validation_curve`, `learning_curve`, `permutation_importance` and
`cross_val_predict` with the spec and protocol passed in;
`explore.feature_screen` is his single-and-pair tree screen with equal
capacity for singles and pairs and a shuffled-feature null. Two
protocols, declared in `protocol.md` (2026-10-02): a screen at two
repeats for the wide grid, promotion to the full protocol for the best
variant of a family that beats its base by more than the error of the
gap. `features.build_preprocessor` gained three optional arguments for
this and is unchanged at its defaults; `run_protocol` takes a `Method`
beside a name. The notebook `04_variants.ipynb` runs a family at a time
and every run lands in `results/04-variants.csv`.
