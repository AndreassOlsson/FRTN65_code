# The sweep on credit-g, and the decision

Written 2026-10-01 (LIF-181) from `make sweep`: every registry method
through `songtaste.evaluate.run_many` with the protocol in
`protocol.md` (committed before this run), tidy rows in `sweep.csv`,
per-split figure in `../figures/sweep-by-split.png`. Cost is the mean
cost per applicant under the dataset's matrix (bad called good 5, good
called bad 1), lower is better; `sweep.csv` holds it negated as
`neg_cost`.

| method | cost | accuracy | ROC AUC | searched |
|---|---|---|---|---|
| qda | 0.820 ± 0.069 | 0.738 | 0.768 | 1 |
| boosting | 0.836 ± 0.064 | 0.753 | 0.776 | 3 |
| svm_rbf | 0.854 ± 0.057 | 0.742 | 0.771 | 2 |
| lda | 0.865 ± 0.060 | 0.751 | 0.781 | 0 |
| adaboost | 0.872 ± 0.057 | 0.753 | 0.780 | 2 |
| logreg | 0.876 ± 0.061 | 0.750 | 0.779 | 1 |
| svm_linear | 0.876 ± 0.056 | 0.749 | 0.778 | 1 |
| bagging | 0.896 ± 0.053 | 0.752 | 0.777 | 1 |
| rf | 0.909 ± 0.058 | 0.758 | 0.782 | 2 |
| tree | 1.002 ± 0.073 | 0.683 | 0.665 | 2 |
| knn | 1.016 ± 0.051 | 0.704 | 0.648 | 2 |
| dummy | 1.500 | 0.700 | 0.500 | 0 |

(± is the corrected standard error over the 25 splits.)

## The decision, by the rule

Among non-dummy methods within one corrected error of the paired
difference to the cheapest (qda), six qualify: lda, qda, svm_linear,
svm_rbf, adaboost, boosting. The fewest searched hyperparameters is
lda's zero, so **the rule chooses `lda`**, at 0.865 per applicant,
0.046 above qda with a paired error of 0.055 (p = 0.41).

## What it says

- **Nothing beats saying "bad" to everyone.** That costs 0.70 per
  applicant (`protocol.md`, "Two floors"); the cheapest method costs
  0.82. The protocol predicted it: every method calls at P(bad) = 1/2
  where the matrix puts the cut at 1/6, so each waves through bad
  applicants at five times the price of turning away good ones. The
  ranking by cost is a ranking of which methods happen to lean towards
  "bad" at the default cut, which is why QDA, whose separate class
  covariances push it that way (as in lab 1), comes first by cost and
  ninth by accuracy. The missing piece is a decision stage between
  probability and call, not a better model.
- **By accuracy the order is lab 1's**: rf on top, the ensembles and
  linear models within a point of each other, tree and knn at the
  bottom, knn picking one neighbour on 14 of 25 splits (61 encoded
  columns, most of them one-hot, make distance mean little).
- **The grids, sized for lab 1's 28 columns, do not bracket credit-g's
  choices.** Modal settings on the grid's edge: boosting's learning
  rate 0.3 and 300 iterations, adaboost's learning rate 1.0 and 300
  stumps, svm_linear's C = 10, rf's `max_features` 0.5, knn's one
  neighbour, bagging's leaf of 1. Almost all sit at the least
  regularised end, so the true optimum may lie past the grid. The
  protocol said to read this off the choices rather than adjust in
  advance; it is a leak of lab 1 into the registry, recorded in the
  vault's diary, and not fixed here because widening a lab 1 grid
  would change lab 1's sweep.
- The choice is not refit or shipped; the trial only shows the
  decision stage running on a second dataset.
