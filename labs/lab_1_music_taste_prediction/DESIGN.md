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
