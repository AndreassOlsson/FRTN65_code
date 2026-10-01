# 02 Baselines

Three methods through the protocol (`protocol.md`, committed before
any of these numbers existed): the majority-class dummy, L2 logistic
regression with `C` searched, and kNN with `n_neighbors` and `weights`
searched. 736 songs after dropping exact duplicates, 25 outer splits
(5 folds times 5 repeats, seed 65), every search inside the outer
training part. `make baselines` regenerates `02-baselines.csv`
byte for byte and prints every table below; `notebooks/02_baselines.ipynb`
calls the same functions.

## Scores

Mean over the 25 splits, with the corrected standard error
(`protocol.md`, "Uncertainty").

| method | accuracy | balanced accuracy | ROC AUC |
|---|---|---|---|
| dummy | 0.602 ± 0.002 | 0.500 ± 0.000 | 0.500 ± 0.000 |
| logreg | **0.814 ± 0.019** | 0.801 ± 0.018 | 0.886 ± 0.013 |
| knn | 0.795 ± 0.016 | 0.768 ± 0.018 | 0.880 ± 0.011 |

Paired against the best (logreg) on the same splits:

| method | accuracy gap | corrected error of the gap | splits where logreg wins / ties |
|---|---|---|---|
| knn | -0.0190 | 0.0185 | 16 / 3 of 25 |
| dummy | -0.2122 | 0.0193 | 25 / 0 |

## What the searches chose

| method | setting | splits |
|---|---|---|
| logreg | C = 0.01 | 8 |
| logreg | C = 0.032 | 12 |
| logreg | C = 0.1 | 5 |
| knn | n_neighbors from 5 to 51, median 21 | 25 |
| knn | weights = distance / uniform | 16 / 9 |

The full per-split choice is the `params` column of the CSV. Fit times
(seconds per outer split, inner search included, on the machine that ran
it, not in the CSV): dummy 0.01, logreg 1.5, knn 2.2.

## Reading

Both real methods sit about 20 points above the floor: logistic
regression gets 81% of songs right against the dummy's 60%, and the gap
is more than ten corrected errors wide, so this is not fold luck. The
balanced accuracy says it is not bought by leaning on "like" either:
0.80 against the dummy's 0.50 means both classes are recognised, the
disliked ones nearly as well as the liked. Logistic regression's search
was remarkably settled, picking C between 0.01 and 0.1 on every split,
which is strong shrinkage; with 13 features, half of them weak and
three of the strong ones carrying one axis, a model that holds most
coefficients near zero and lets the loud-versus-acoustic and the
speechiness directions through is close to what the exploration
predicted would work. The ceiling question for task 3 is how much of
the remaining 19% is reachable at all with these features.

kNN's weakness with mixed types showed, but mildly. Its accuracy is two
points below logistic regression's, its balanced accuracy three, so the
loss falls more on the minority class, and its search never settled:
the chosen neighbourhood ranged from 5 to 51 songs across splits, which
is what a flat validation curve looks like. Distance here mixes ten
standardised numeric axes with 0/1 one-hot columns (twelve for key
alone), so a song's neighbours are partly decided by sharing a key,
which the exploration found to be mostly noise, and the weak numeric
features count as much as the strong ones. Yet its AUC is nearly
logistic regression's (0.880 against 0.886), so it ranks songs about as
well and loses at the threshold. Under the protocol's decision rule kNN
falls just outside: its gap of 0.0190 is a hair over one corrected error
(0.0185), so it would not qualify against logistic regression, and
since logistic regression is also the simpler of the two (one searched
hyperparameter against two), nothing in task 3's choice turns on that
hair.
