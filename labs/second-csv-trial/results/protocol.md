# Evaluation protocol: credit-g

Written 2026-10-01 (LIF-181), before any model has been fitted or
scored on this data. It is lab 1's protocol
(`../lab_1_music_taste_prediction/results/protocol.md`) with what this
dataset changes, and it runs through the same function,
`songtaste.evaluate.run_protocol`, with a `Protocol` built in
`creditg/sweep.py`. Never edited after this commit; a change is a new
dated section at the bottom.

## What is scored

- **The primary score is the expected misclassification cost per
  applicant**, from the cost matrix OpenML publishes with the dataset:
  calling a bad applicant good costs 5, calling a good one bad costs 1,
  a right call costs 0. It is what the dataset's owner says a mistake
  costs, so it is what a choice should be made on. It enters the
  protocol as the scorer `neg_cost` (sklearn's convention: negated, so
  higher is better), and the tidy results hold it negated; the written
  results report the cost itself.
- **Accuracy, balanced accuracy and ROC AUC are reported beside it**,
  never decided on.
- **Two floors, read against.** The registry's dummy always says
  "good", the majority: accuracy 0.70, cost 0.3 x 5 = 1.50. Saying
  "bad" to everyone costs 0.7 x 1 = 0.70 and is the real floor on cost;
  it is not in the registry (no new method is added) and is stated
  here instead.
- **Every method predicts at its default threshold.** The cost matrix
  makes the cost-minimising cut at P(bad) = 1/6, not 1/2, and moving it
  is a decision stage the package does not have
  (`TunedThresholdClassifierCV` would be it). Adding one is a new
  method, out of this task's bounds, so methods are compared as they
  stand and the threshold is named in the record as a missing seam.
  Expect few methods, maybe none, to beat 0.70 at the default cut.

## The data a method sees

- `data/credit-g.csv` through `creditg.data.load`, all 1000 rows.
  There are no exact duplicates (`01-exploration.md`), so lab 1's
  de-duplication step changes nothing and is not applied.
- There is no held-out file; this trial does no final fit and no
  prediction.

## The resampling

As lab 1, unchanged: outer `RepeatedStratifiedKFold(n_splits=5,
n_repeats=5, random_state=65)`, every method on the same 25 splits;
anything with a search space tuned by `GridSearchCV` inside each outer
training part, inner `StratifiedKFold(n_splits=5, shuffle=True,
random_state=65)`, scoring the primary score (cost); preprocessing
inside the pipeline; `cross_validate` and `GridSearchCV` do all of it;
seed 65 in one place, so the same seed regenerates the same CSV.

The registry's search spaces are used as lab 1 wrote them, sized for
13 columns where credit-g encodes to 61; whether they still bracket
the chosen settings is read from the per-split choices, not adjusted
in advance.

## Uncertainty

As lab 1: mean over the 25 splits and Nadeau and Bengio's corrected
standard error, on scores and on paired per-split differences.
**The region of practical equivalence is 0.05 of cost per applicant**,
one extra bad loan waved through per hundred applicants; it only
shapes the Bayesian probabilities reported, not the decision.

## The decision rule

Lab 1's, on cost: **among the non-dummy methods whose mean cost is
within one corrected standard error (of the paired per-split
difference) of the cheapest method's, choose the one with the fewest
searched hyperparameters, ties broken by songtaste.report's simplicity
order.** The trial chooses a method to show the decision stage works
on a second dataset; nothing is refit or shipped.
