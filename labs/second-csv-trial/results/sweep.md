# The sweep on credit-g, and the decision

Run 2026-10-01 by `make sweep` (`creditg.sweep`), every songtaste
registry method on the 25 splits `protocol.md` fixed, about 65 minutes
on 4 shared cores. Per-split rows in `sweep.csv` (the cost negated, as
`neg_cost`), the figure `../figures/sweep-by-split.png`. Below, cost is
reported as cost: mean per applicant, lower is better, ± the corrected
standard error.

| method | cost | accuracy | balanced acc. | AUC | searched | gap to qda | corrected se of gap |
|---|---|---|---|---|---|---|---|
| qda | 0.820 ± 0.069 | 0.738 | 0.680 | 0.768 | 1 | | |
| boosting | 0.836 ± 0.064 | 0.753 | 0.683 | 0.776 | 3 | 0.016 | 0.053 |
| svm_rbf | 0.854 ± 0.057 | 0.742 | 0.674 | 0.771 | 2 | 0.034 | 0.054 |
| lda | 0.865 ± 0.060 | 0.751 | 0.675 | 0.781 | 0 | 0.046 | 0.055 |
| adaboost | 0.872 ± 0.057 | 0.753 | 0.675 | 0.780 | 2 | 0.052 | 0.065 |
| logreg | 0.876 ± 0.061 | 0.750 | 0.672 | 0.779 | 1 | 0.056 | 0.053 |
| svm_linear | 0.876 ± 0.056 | 0.749 | 0.672 | 0.778 | 1 | 0.057 | 0.065 |
| bagging | 0.896 ± 0.053 | 0.752 | 0.669 | 0.777 | 1 | 0.077 | 0.068 |
| rf | 0.909 ± 0.058 | 0.758 | 0.668 | 0.782 | 2 | 0.089 | 0.056 |
| tree | 1.002 ± 0.073 | 0.683 | 0.611 | 0.665 | 2 | 0.182 | 0.085 |
| knn | 1.016 ± 0.051 | 0.704 | 0.617 | 0.648 | 2 | 0.196 | 0.076 |
| dummy (always good) | 1.500 | 0.700 | 0.500 | 0.500 | 0 | 0.680 | 0.069 |

## The decision

**The chosen method is linear discriminant analysis (`lda`).** QDA is
the cheapest; six methods sit within one corrected error of it (qda,
boosting, svm_rbf, lda, adaboost, svm_linear), and among them LDA
searches nothing, so the rule takes it. Nothing is refit or shipped;
the decision is here to show the stage works on a second dataset.

## The reading

- **Nothing beats saying no to everyone.** Every method costs more
  than 0.70, the cost of rejecting every applicant, as the protocol
  expected: at the default threshold a method calls an applicant bad
  only when it is more likely bad than good, while the cost matrix
  wants it called bad from a one-in-six chance. The ranking above is
  real (the methods are compared on the same splits); the absolute
  level is what a missing threshold stage costs, not what the methods
  can do.
- **The ranking on cost is not the ranking on accuracy.** The forest
  has the best accuracy and AUC and is ninth on cost; QDA is mid-table
  on accuracy and first on cost. Lower accuracy at lower cost can only
  mean it trades costly misses (bad called good, 5) for cheap false
  alarms (good called bad, 1), which is what the cost matrix pays for.
  Deciding on accuracy here would have chosen the wrong thing; on lab
  1, where the metric was accuracy, the question never came up.
- **Grids tuned for lab 1 hit their edges.** Tuning on cost, kNN chose
  1 neighbour on 14 of 25 splits (the grid's lower edge), the tree no
  depth limit, RBF SVM C = 100 (upper edge) on 10. The cost metric
  rewards high-variance fits that say bad more often, and the
  registry's spaces were sized for accuracy on 13 columns. Not
  adjusted (the protocol said so in advance); it is a finding about
  where the registry stops being generic.
- **The band is wide.** Corrected errors of 0.05 to 0.07 on gaps of
  0.02 to 0.06: with 300 bad applicants, five-to-one costs and 25
  correlated splits, most of the field is indistinguishable, and the
  ROPE probabilities (p_rope 0.4 to 0.6 against qda) say the same.
