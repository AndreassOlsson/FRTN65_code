# 04 Variants: the kNN family

kNN has one idea, that a song is liked when the songs closest to it
are, so everything it knows sits in what "closest" means. The baseline
measures it as Euclidean distance over ten standardised numeric
features and eighteen one-hot columns for key, mode and meter. The
seven variants change that space and nothing else: how the numbers are
scaled, which axes are in it, and which distance is taken.

All numbers here are from **the screen** (`protocol.md`, 2026-10-02):
the protocol's 5 stratified folds at seed 65, repeated twice, so ten
outer splits, with `n_neighbors` and `weights` searched inside each
outer training part on the baseline's grid. They are not the sweep's
numbers and are never to be read beside 03-sweep's 0.795 as if they
were; the baseline itself scores 0.801 here only because these are ten
of the sweep's twenty-five splits, a different sample. `make variants
FAMILY=knn` writes `04-variants-knn.csv` (70 rows) and prints the table
in under three minutes on 4 cores.

## The family under the screen

Mean over the ten splits with the corrected standard error; "gap" is
the mean paired difference to `knn` on the same splits, with the
corrected error of that difference, and the last column counts the
splits the variant wins, ties and loses against `knn`.

| variant | what changes | accuracy | balanced accuracy | ROC AUC | gap to knn | W / T / L |
|---|---|---|---|---|---|---|
| knn_top4 | only speechiness, loudness, acousticness, energy | **0.821 ± 0.015** | 0.804 ± 0.019 | 0.877 ± 0.014 | +0.020 ± 0.018 | 8 / 0 / 2 |
| knn_quantile | rank to normal instead of standardising | 0.820 ± 0.012 | 0.800 ± 0.011 | 0.895 ± 0.011 | +0.019 ± 0.018 | 6 / 3 / 1 |
| knn_numeric | one-hot columns dropped | 0.814 ± 0.008 | 0.786 ± 0.008 | 0.883 ± 0.015 | +0.013 ± 0.013 | 6 / 2 / 2 |
| knn_manhattan | L1 distance (`p=1`) | 0.810 ± 0.014 | 0.784 ± 0.017 | 0.895 ± 0.013 | +0.010 ± 0.012 | 6 / 1 / 3 |
| knn | the baseline | 0.801 ± 0.011 | 0.774 ± 0.012 | 0.874 ± 0.013 | | |
| knn_log | log1p on duration, speechiness, instrumentalness, liveness | 0.796 ± 0.011 | 0.771 ± 0.013 | 0.876 ± 0.010 | -0.005 ± 0.008 | 3 / 3 / 4 |
| knn_robust | median and IQR instead of mean and sd | 0.758 ± 0.013 | 0.722 ± 0.014 | 0.825 ± 0.021 | -0.043 ± 0.017 | 0 / 0 / 10 |

Every change that shrinks or reshapes the space towards the strong
features helped by one to two points, none by much more than its own
error, and the two that were meant to tame the tails did nothing or
did harm. The spread between the best and worst variant (six points)
is three times the spread among the ones that helped, so the family's
story is more about what breaks a distance than about what perfects
one.

## What the one-hot columns cost a Euclidean distance

Under the baseline's encoding, two songs in different keys differ by
one in two of the twelve key columns, which adds 2 to their squared
distance, as much as being 1.4 standard deviations apart on one
numeric feature. Over random pairs of songs, the
one-hot columns make up **13% of the squared distance** (key alone 8%,
mode 3%, meter 2%), while the exploration found key nearly worthless
and mode weak. Permutation importance agrees from the model's side
(`figures/04-knn/permutation-importance.png`): shuffling key, mode or
time_signature costs the baseline under one point each (0.007, 0.006,
0.001). Dropping them (`knn_numeric`) gains 0.013 ± 0.013 and wins six
splits of ten, a gain the size of its own error. It is the right
direction and a small effect, because the same arithmetic says the
larger dilution was never the one-hot columns: the six weak numeric
features (tempo, liveness, valence, danceability, duration,
instrumentalness) carry about **half** of every squared distance, and
the four strong ones only 35%. A standardised distance gives every
axis the same vote, and most of the axes have nothing to say.

That is the k-NN topic's warning about units turned one step further
(`Book Topics/01 Regression and generalisation/k-NN.md`, "Input
normalisation": "whatever units your inputs happen to be measured in
*become* your notion of similarity"). Standardising fixes the units;
it cannot fix that an irrelevant axis, once scaled to unit variance,
moves neighbours as much as a relevant one.

## Whether a rank or robust scaling fixes the tails

The exploration found duration, speechiness, instrumentalness and
liveness to be a spike with a long tail, and predicted the tails would
dominate a distance unless scaled robustly or log-transformed. The
screen splits that prediction three ways.

- **Robust scaling broke it** (-0.043 ± 0.017, worse on all ten
  splits). The reason is in the spike: instrumentalness has a median
  and an interquartile range of essentially zero, so dividing by its
  IQR sends its few non-zero songs out to a value of about 400. After
  `RobustScaler` its standard deviation is about **110** against about
  1 for every other feature, so the distance is instrumentalness and
  little else. Robust scaling protects against outliers in a
  bell-shaped feature; on a feature that is all spike, it divides by
  nothing.
- **The log did nothing** (-0.005 ± 0.008, three wins and four
  losses). `log1p` on a feature that lives in 0 to 1 is almost the
  identity: on speechiness, instrumentalness and liveness the logged
  values correlate 0.996 to 0.999 with the raw ones and their skew
  barely moves (instrumentalness 2.49 to 2.41). Only duration, in
  milliseconds, is actually reshaped. So the variant tested "log1p on
  duration", and the answer is that it does not matter to a distance.
- **The rank transform helped** (+0.019 ± 0.018, six wins, one loss)
  and is the only variant to lift ROC AUC clearly (0.895 against
  0.874). Mapping each feature to normal scores through its ranks
  gives every axis the same bounded shape, so a spiky feature cannot
  outvote the others and a dense region of the data is spread out
  where neighbours are decided. The AUC gain, larger than the accuracy
  gain, says the rank-scaled neighbours order songs better even where
  the vote at 0.5 is unchanged.

The exploration's advice not to clip the tails still stands: the rank
transform keeps every song's order, it only stops the tail from
setting the scale.

## Whether four strong features beat thirteen

They do, by the most of any variant: `knn_top4` at 0.821, +0.020 ±
0.018 over the baseline, winning eight splits of ten. Read with the
distance arithmetic above, this is the same finding as `knn_numeric`
taken all the way: removing the 65% of the distance that came from
features with little to say. Where the gain lands is in the confusion
matrices (`figures/04-knn/oof-roc-confusion.png`, out-of-fold, every
song predicted once): top4 gets 210 of the 293 disliked songs right
against the baseline's 189, at the cost of five liked songs, so
balanced accuracy rises more than accuracy (0.804 against 0.774). The
baseline's neighbourhoods were being pulled towards the majority
"like" by axes that do not separate the classes; in the four-feature
space the disliked songs, which are loud, energetic and talky, form a
region of their own.

What top4 gives up shows in the permutation matrix: danceability is
the baseline's second most important feature by permutation (0.041,
above acousticness, loudness and energy), and top4 cannot use it. Its ROC AUC
(0.877) is no better than the baseline's, below the quantile and
Manhattan variants', so top4 wins at the threshold more than it ranks
songs better.

## Whether Manhattan changes anything

A little, and in the same direction as the rank transform: +0.010 ±
0.012 in accuracy, the gap inside its error, with ROC AUC up to 0.895.
L1 sums absolute differences instead of squaring them, so one feature
far apart counts linearly rather than quadratically and a single
tail-heavy axis has less pull. It is a milder version of what the
quantile transform does to the tails, and it shows the same signature:
better ranking, a small and uncertain gain in accuracy.

## Whether the neighbourhood search settles

In the sweep the baseline's search never settled, and it does not here
either. The table is the `params` column of `04-variants-knn.csv`, the
`n_neighbors` each split's inner search chose (splits 0 to 4 are the
first repeat's folds, 5 to 9 the second's), with how often `distance`
weighting won over `uniform`.

| variant | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | median | distinct | distance / uniform |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| knn | 7 | 21 | 7 | 51 | 31 | 31 | 7 | 5 | 7 | 21 | 14 | 5 | 8 / 2 |
| knn_robust | 7 | 5 | 41 | 5 | 7 | 41 | 51 | 51 | 31 | 31 | 31 | 5 | 5 / 5 |
| knn_quantile | 21 | 11 | 31 | 7 | 15 | 21 | 9 | 41 | 15 | 11 | 15 | 7 | 5 / 5 |
| knn_log | 7 | 9 | 7 | 7 | 31 | 41 | 7 | 5 | 7 | 21 | 7 | 6 | 7 / 3 |
| knn_numeric | 9 | 11 | 31 | 21 | 5 | 41 | 41 | 31 | 31 | 31 | 31 | 6 | 10 / 0 |
| knn_top4 | 7 | 7 | 7 | 15 | 11 | 11 | 7 | 15 | 5 | 11 | 9 | 4 | 8 / 2 |
| knn_manhattan | 15 | 11 | 21 | 31 | 9 | 15 | 21 | 15 | 41 | 9 | 15 | 6 | 9 / 1 |

Only `knn_top4` settles: every split picks between 5 and 15, against
5 to 51 for the baseline and for dropping the one-hot axes. The
validation curves show why (`figures/04-knn/validation-n_neighbors.png`,
`weights` at its default `uniform`, one `plot_validation_curve` call
per variant on shared axes). For the baseline and for `knn_numeric` the
validation line is flat from k = 9 to k = 51, within 0.013 of its
peak, so any k in that range is as good as any other and the inner
search's choice is decided by which songs are in the inner folds. For
`knn_top4` the line has a real peak, 0.827 at k = 9, falling to 0.808
at k = 3 and 0.806 at k = 51. In the bias-variance terms of
`Book Topics/01 Regression and generalisation/Bias, variance and the generalisation gap.md`,
the U of E_new is there for top4 and flattened out for the baseline:
in thirteen dimensions with most of them noise, a small neighbourhood
is not much more specific than a large one, because "near" in that
space is mostly decided by the noise axes. Once the space holds only
features that separate the classes, a small k is genuinely local and a
large k genuinely averages structure away, and the trade-off the book
describes reappears with a clear optimum.

## Learning curves: what more songs would buy

`figures/04-knn/learning-curves.png`, with the inner search run at
every training size. The training line is high for both (0.91 to
0.97) only because `distance` weighting, which wins most searches,
scores every training song by itself; it is not a measure of fit here,
and the reading is in the validation line. The baseline's is still
rising at the full training part, from 0.758 at 117 songs to 0.795 at
588: more songs would still help it, which is what a method spending
half its distance on noise axes needs to average the noise away. The
four-feature space reaches 0.82 by 352 songs and stays there (0.820,
0.824, 0.820): it has learnt what four features allow, and further
gains would have to come from a better space, not more data.

## The threshold

At the protocol's 0.5 the baseline is at 0.800 out-of-fold, and the
best threshold on the same scores (0.525) gives 0.807; for top4 0.822
against 0.826 at 0.575. Both are inside the noise of one out-of-fold
pass, so the decision threshold is not where kNN's accuracy is lost.

## Promotion

`variants.promoted` promotes **`knn_top4`**: the family's best mean
(0.821), with a paired gap to `knn` of 0.020 against a corrected error
of 0.018, so it clears the rule, by 0.002. That margin is thin and
the full protocol is where it is tested; `knn_quantile` is behind it by
0.001 with a gap (0.019 ± 0.018) that would also have cleared, and is
not promoted only because the rule takes one variant per family. Even
if top4's screening number held under the full protocol it would sit
about a point below the sweep's random forest, so the question task 5
puts to it is whether it joins the band of 03-sweep's decision rule,
not whether it wins.

## Figures

- `figures/04-knn/validation-n_neighbors.png`: validation curves for
  `n_neighbors` on knn, knn_numeric, knn_top4.
- `figures/04-knn/learning-curves.png`: learning curves, tuned, for
  knn and knn_top4.
- `figures/04-knn/permutation-importance.png`: permutation importance,
  tuned, knn, knn_numeric and knn_top4 as one matrix.
- `figures/04-knn/oof-roc-confusion.png`: out-of-fold ROC for knn and
  knn_top4, and each one's confusion at 0.5.

Each is one `songtaste.diagnostics` call per variant on the screen's
folds, the calls `notebooks/04_variants.ipynb` makes with these variant
names: `validation_curve_table` over k in 1, 3, 5, 9, 15, 21, 31, 41,
51, 71, 101; `learning_curve_table(..., tuned=True)`;
`permutation_table(..., n_repeats=10, tuned=True)` into
`importance_matrix`; `oof_predictions` into `plot_roc`, `plot_confusion`
and `threshold_table`. About two minutes for all four on 4 cores.
