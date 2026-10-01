# Evaluation protocol

Written 2026-10-01 (LIF-178), before any model has been fitted or
scored on this data. Every method in this lab goes through it, through
one function, `songtaste.evaluate.run_protocol`. It is never edited
after this commit: a later change is a new dated section at the bottom
saying what changed and why, and the numbers produced before it stay
as they were.

## What is scored

- **Accuracy is the primary metric**, because accuracy on the 200 songs
  is what the leaderboard scores. Every decision is made on it.
- **Balanced accuracy and ROC AUC are reported beside it**, never
  decided on. Task 1 found a 60/40 like/dislike split (`01-exploration.md`),
  enough that a model can look decent by leaning on "like"; balanced
  accuracy shows that, and AUC shows how well a model ranks songs
  regardless of where its threshold sits.
- The floor is the majority-class dummy, about 0.60 accuracy. A method
  is read as distance above it, not as a raw number.

## The data a method sees

- Only `training_data.csv`, loaded through `songtaste.data.load_training`.
- **Exact duplicate rows are dropped before any split** (features and
  label identical, first kept): 750 rows become 736. Task 1 found 12
  songs repeated 14 times; left in, a repeated song can sit in a
  training fold and its own validation fold, and cross-validated
  accuracy is inflated, most for kNN and deep trees. Dropping them
  settles that with one line and keeps plain stratified folds; grouping
  the folds by song would need a repeated grouped splitter sklearn does
  not ship, which means a loop of our own.
- **`songs_to_classify.csv` is never read before task 4.** `make data`
  downloads it beside the training file, which is not reading it; no
  code in `src/` outside `data.py` and `explore.py` (task 1's, which
  check its contract) and no notebook from this task on opens it. Task 4
  reads it once, with the chosen model, to produce the submission.

## The resampling

- **Outer loop: `RepeatedStratifiedKFold(n_splits=5, n_repeats=5,
  random_state=65)`.** 25 train/validation splits, each stratified so
  every fold keeps the 60/40. Each method gets one row per outer split,
  so every method is scored on exactly the same 25 splits and
  comparisons between methods are paired.
- **Anything tuned is tuned inside each outer split, never on the whole
  set.** The method's pipeline is wrapped in `GridSearchCV` over its
  declared search space, scoring accuracy, with inner folds
  `StratifiedKFold(n_splits=5, shuffle=True, random_state=65)`; the
  search sees only that outer split's training part, refits the best
  setting on it, and the outer validation part scores that refit. The
  chosen hyperparameters are recorded per outer split, so their spread
  across splits is visible.
- **Preprocessing lives inside the pipeline**, so scaling and encoding
  are learned from the training part of each split only.
- **`cross_validate` and `GridSearchCV` do all of it.** There is no
  cross-validation loop of our own.
- **One seed, 65, in one place**: the `Protocol` dataclass in
  `songtaste/evaluate.py`, which hands out both fold objects and the
  seed any estimator with randomness takes. The same seed regenerates
  the same results file, byte for byte (`make baselines`). Fit times
  are measured but kept out of the results file, because wall-clock
  time is not reproducible from a seed.

## Uncertainty

Per method, the mean over the 25 outer splits, and its **corrected
standard error**, `std * sqrt(1/25 + n_val/n_train)`, with `n_val/n_train`
= 1/4 for 5 folds (Nadeau and Bengio's correction, the one sklearn's
model-comparison example uses). The 25 scores are not independent,
since the splits share most of their training data, and the plain
`std/sqrt(25)` would understate the error by about half. Between two
methods, the same correction is applied to their 25 paired per-split
differences.

## The decision rule for task 3

**Among the non-dummy methods whose mean accuracy is within one
corrected standard error of the best method's, where the error is that
of their 25 paired per-split differences from the best, task 3 chooses
the one with the fewest searched hyperparameters, ties broken by the
order below.**

Simplicity order for ties, simplest first, fixed now for every method
the sketch names: logistic regression, LDA, QDA, kNN, decision tree,
linear SVM, RBF SVM, bagging, random forest, boosting. The best method
itself always qualifies, so the rule always returns one method. The
chosen method is then refit, by the same `GridSearchCV` over the
deduplicated training set, for task 4.
