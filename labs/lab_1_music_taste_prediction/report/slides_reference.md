# Lab 1 slides, reference content

Reference for building the deck by hand. One section per slide: title,
the figure or table, the body in slide words, and notes for the oral.
Every number is from `modelling_results.csv` (5 repeats unless marked
screen) or from the notebooks. Twelve slides, 5 to 15 is the limit.

The red line, which the title of every slide should be able to say:
*look at the data first (exploration), then decide how to judge before
judging (protocol), then let every method and every variant of it go
through the same judge (modelling), then pick by a rule you wrote
before you saw the numbers, and say when you step outside it.*

---

## 1. Title

**Predicting which songs Andreas likes**

- 736 songs after dropping 14 duplicates, 13 Spotify audio features, 60 % liked
- 12 methods and 51 variants of them through one evaluation protocol
- Handed in: a random forest on the 10 numeric features, 0.833 ± 0.018 cross-validated accuracy
- FRTN65 Lab 1, Andreas Olsson, October 2026

Notes: the one sentence version. Four ensembles are indistinguishable on this data; the protocol's rule picks bagging as the simplest of them; I hand in the forest without the three discrete columns, and slide 10 says why.

---

## 2. Why two notebooks: look first, then judge

No figure, or the two notebook titles side by side.

- `initial_exploration.ipynb`: unstructured on purpose. Plots, a feature screen, a few guesses written down as next steps
- `modelling.ipynb`: structured on purpose. Definitions up top (data, protocol, preprocessing, methods, variants, evaluation, comparison, diagnostics), runs below
- The exploration decides *what* to try (which columns are spiky, which four features are strong, that pairs matter). The modelling decides *how to compare* and never changes that mid-way
- Everything that varies between methods is a dict entry; everything that must not vary (folds, seed, metric, tuning inside the folds) is one function

Notes: the point of the split is that exploration is allowed to be messy and the comparison is not. Mention that the structure made 51 variants cheap: adding one is one line, and it gets the same folds, tables and plots as everything else.

---

## 3. The data: one axis, quiet-acoustic-wordless versus loud-energetic-talky

Figure: `feature_overviews.png` (crop to acousticness, energy, loudness, speechiness, plus mode or key). Small inset: `class_balance.png`.

- 60/40 like/dislike, so "always like" scores 0.60: that is the floor every number is measured from
- Liked songs: quiet (loudness), acoustic, almost no speech. Disliked: loud, energetic, speechy, more danceable
- key, mode, time_signature barely separate the classes
- Several features are a spike plus a long tail (duration, speechiness, instrumentalness, liveness): a plain z-score is set by the tail

Notes: densities are normalised within class so the 60/40 does not read as separation. The spiky four are why robust/quantile/log scaling became variants later.

---

## 4. Which features repeat each other, which one stands alone

Figure: `feature_correlations.png` (Spearman, clustered).

- energy, loudness, acousticness form one block (energy~loudness 0.85, energy~acousticness −0.74): three columns, one direction
- speechiness is the strongest single relation to the label (−0.52) and sits outside that block
- tempo, liveness, duration, instrumentalness near zero against everything
- Spearman (ranks) rather than Pearson because of the tails

Notes: correlation is pairwise and monotone. It cannot see interactions, which is why the next slide exists. Also why a heavily regularised linear model does well: the main axis is linear and L2 shares weight across the three correlated columns.

---

## 5. Pairs carry more than their parts

Figure: `decision_tree_roc_auc_pair_gain.png` left, one or two panels of `decision_tree_roc_auc_best_gain_pairs.png` right (energy + speechiness, acousticness + speechiness). `decision_tree_roc_auc_single.png` as a build or backup.

- Screen: a depth-limited decision tree, 5-fold CV, ROC AUC, on each feature alone and on each pair
- Gain = pair AUC minus the better single. energy + speechiness +0.095, acousticness + speechiness +0.081, loudness + speechiness +0.078
- Read: speechiness splits the songs the loud/acoustic axis cannot, and the other way round
- Caveat stated: one model, one depth; the modelling notebook asks the same question of every method (permutation importance, slide 11)

Notes: this is where "feature engineering" variants came from (interactions, top4). Be ready for "is the gain just extra tree depth": singles and pairs were run at the same depth in the final version, and the small gains at the bottom of the bar chart are noise-level.

---

## 6. The protocol, written before any number

No figure. Keep it a list; this is the slide the examiner will ask about.

- **Metric:** accuracy, because the leaderboard scores accuracy. Balanced accuracy and ROC AUC reported beside it, never decided on
- **Splits:** 5-fold stratified, repeated 5 times, one seed: 25 outer splits, the same 25 for every method, so every comparison is paired
- **Tuning:** grid search with 5 inner folds, inside each outer training part only. The validation part never sees the search
- **Preprocessing inside the pipeline:** scaling and encoding fit on the training part of each split
- **Uncertainty:** the 25 splits share most of their songs, so the plain std/√25 is too small. Corrected standard error: std · √(1/25 + n_val/n_train) (Nadeau and Bengio); the same correction on paired per-split differences
- **Decision rule:** among the methods within one corrected error of the best (error of the paired gap), take the one with the fewest tuned hyperparameters; ties by a simplicity order fixed in advance
- **Test file:** read once, by the model handed in

Notes: why corrected: with 5 folds, n_val/n_train = 1/4 and never shrinks with more repeats, so 5 repeats already give 93 % of the precision infinite repeats would. Why accuracy: the leaderboard. Why paired: the boxes on slide 9 overlap almost entirely and most of that spread is which songs landed in a fold, which every method shares.

---

## 7. Methods, variants, and why the screen exists

Table: method, what it does, what is tuned (from `METHODS` and the `SIMPLICITY` comments). Twelve rows; mark xgboost as outside the course list, included as a check on the boosting implementation.

- A **variant** = a method plus one change: scaler (standard, robust, quantile, none), log1p on the spiky four, encoding of the discrete columns (one-hot, grouped rare levels, dropped), feature subset (the four strong ones), pairwise interactions, another penalty or kernel, another grid
- 51 variants in four families. Running all of them at 5 repeats would be hours, so: **screen** at 2 repeats (same seed, same first two repeats), **promote** to 5 repeats only the best variant of each method, and only if it beats its own base by more than the error of that gap
- Screen numbers are never shown next to full numbers

Notes: "simplicity" in the rule means fewest tuned hyperparameters, not fewest parameters. Reason: among methods the data cannot tell apart, prefer the one whose score had the fewest chances to look good by search. Parameter count measures capacity, and for ensembles it is a poor guide (more trees never overfit more). The simplicity order (logreg, lda, qda, knn, tree, svm linear, svm rbf, bagging, rf, boosting, xgboost) is a stated preference for ties, and it agrees with the tuned count here.

---

## 8. What the searches chose

Figure: `effect_of_sweeping_different_hyperparams_for_different_methods.png`.

Small table (modal setting over 25 splits):

| method | chosen most often | splits |
|---|---|---|
| logreg | C = 0.01 | 13 / 25 |
| svm_linear | C = 0.01 | 19 / 25 |
| svm_rbf | C = 1, gamma = 0.1 | 8 / 25 |
| knn | k = 11, distance weights | 5 / 25 |
| tree | depth 2, leaf 1 | 5 / 25 |
| bagging | min leaf 3 | 9 / 25 |
| rf | max_features 0.25, leaf 1 | most |

- The linear models want heavy shrinkage; the trees want shallow single trees and deep averaged ones
- kNN's validation line is flat from k = 5 to 50 while the training line collapses: the neighbourhood size barely matters
- RBF gamma: flat, then a cliff at gamma = 1 (every song is its own island)

Notes: the figure is validation curves at the screen's folds, one knob at a time with the rest at defaults.

---

## 9. Every method on the same 25 splits

Figure: `outer_validation_fold_mean_accuracy_all_variants.png`.

Table (5 repeats, mean ± corrected error):

| variant | accuracy | balanced acc. | ROC AUC | tuned |
|---|---|---|---|---|
| xgboost | 0.834 ± 0.016 | 0.824 | 0.905 | 3 |
| rf_numeric | 0.833 ± 0.018 | 0.825 | 0.912 | 2 |
| rf | 0.833 ± 0.018 | 0.825 | 0.911 | 2 |
| bagging | 0.828 ± 0.017 | 0.819 | 0.905 | 1 |
| boosting | 0.826 ± 0.015 | 0.814 | 0.902 | 3 |
| svm_rbf_top4 | 0.824 ± 0.016 | 0.808 | 0.878 | 2 |
| knn_top4 | 0.824 ± 0.015 | 0.805 | 0.880 | 2 |
| svm_rbf_numeric | 0.821 ± 0.016 | 0.804 | 0.881 | 2 |
| logreg_elastic | 0.817 ± 0.018 | 0.805 | 0.888 | 1 |
| svm_rbf | 0.814 ± 0.016 | 0.801 | 0.887 | 2 |
| svm_linear | 0.813 ± 0.017 | 0.800 | 0.888 | 1 |
| adaboost | 0.809 ± 0.017 | 0.796 | 0.890 | 2 |
| logreg | 0.807 ± 0.019 | 0.794 | 0.887 | 1 |
| knn | 0.802 ± 0.012 | 0.776 | 0.883 | 2 |
| tree | 0.800 ± 0.017 | 0.786 | 0.845 | 2 |
| lda | 0.795 ± 0.016 | 0.782 | 0.879 | 0 |
| qda | 0.783 ± 0.017 | 0.788 | 0.879 | 1 |
| dummy | 0.602 ± 0.002 | 0.500 | 0.500 | 0 |

Notes: every real method is 18 to 23 points above the floor. Read the boxes: from xgboost down to logreg they overlap, which is the whole reason for the paired table on the next slide.

---

## 10. Paired against the best, the rule, and the override

Table (paired per-split differences against xgboost, the best mean):

| variant | gap to best | corrected error | p | within 1 error |
|---|---|---|---|---|
| rf | −0.001 | 0.008 | 0.91 | yes |
| rf_numeric | −0.001 | 0.007 | 0.90 | yes |
| bagging | −0.006 | 0.008 | 0.48 | yes |
| boosting | −0.008 | 0.009 | 0.37 | yes |
| svm_rbf_top4 | −0.010 | 0.013 | 0.48 | yes |
| knn_top4 | −0.010 | 0.016 | 0.52 | yes |
| svm_rbf_numeric | −0.013 | 0.013 | 0.31 | no |
| logreg_elastic | −0.017 | 0.011 | 0.11 | no |
| logreg | −0.027 | 0.012 | 0.035 | no |
| lda | −0.039 | 0.011 | 0.002 | no |

- **The rule picks bagging:** seven variants inside the band, bagging has the fewest tuned hyperparameters (one)
- **Sensitivity:** within 2 errors the band widens to logreg_elastic and the rule would pick it; best mean alone picks xgboost. The strictness was fixed before the numbers
- **Handed in: rf_numeric.** The leaderboard counts expected accuracy only. Among the indistinguishable ensembles the forest has the best balanced accuracy and AUC of the course methods, and dropping key, mode and time_signature changes nothing the data can detect (gap to rf +0.000 ± 0.007 on the screen, identical means at 5 repeats) while every diagnostic rates those three columns at zero
- bagging and rf_numeric disagree on 10 of the 200 test songs

Notes: say the override as an override. "Under the rule I wrote first, bagging. For the competition I take the forest without the three columns the model never uses. Both strings are in the hand-in." If asked why not xgboost: not a course method, and not better than the forest by anything measurable.

---

## 11. Why: bias on the left, variance on the right, and what every model leans on

Figure: `learning_curves_for_different_methods.png` left, `permutation_importances_across_different_methods.png` right.

- Training accuracy of tree, rf and boosting sits at 1.0 because at their defaults they fit until pure; read the validation line
- logreg: the two lines have met, bias limited, more data would not help
- tree: the biggest gap, variance limited, 0.76 on validation
- rf: same zero training error, validation at 0.84 and still climbing at 590 songs: averaging bought eight points of generalisation from the same fitted-to-purity trees
- Permutation importance: speechiness is the one feature every method leans on (0.09 to 0.11 drop when shuffled), then acousticness and loudness; key, mode, time_signature are zero for every method

Notes: this is the bias-variance motivation for the ensemble choice in one picture, and the bottom three rows of the heatmap are the evidence for dropping the discrete columns.

---

## 12. What the variants taught, and the production model

Figure: `oof_roc_and_confusion_logreg_vs_best.png`.

Screen findings (2 repeats, gap to the method's own base):

- kNN on the four strong features: +0.022 ± 0.013. The 18 one-hot axes drown the distance
- RBF SVM on the four strong features: +0.020 ± 0.019; on numeric only +0.019 ± 0.011
- L1 / elastic-net logistic regression: +0.018 / +0.019 over L2, and L1 keeps the same few features
- Robust scaling costs kNN and the RBF SVM four points (−0.039, −0.041): the IQR of the spiky columns is tiny, so the tails explode
- Scaling does nothing for trees: tree_noscale equals tree split for split

Production: rf_numeric refit on all 736 songs with the same grid search (300 trees, max_features 0.25, leaf 1); 128 of 200 test songs predicted liked; out-of-fold confusion shows the ensemble's gain over logistic regression is mostly on disliked songs. Expect 0.80 to 0.86 on the 200 songs.

Notes: the four screen bullets are the "regularisation, scaling, feature engineering" motivations the rubric asks for, each with a number. End on the honest range.

---

## Disclosure line (last slide or README)

AI assistance (Claude) was used to understand how professional ML projects structure the explore / protocol / compare lifecycle, for help with some of the plotting code, and for the paired-comparison logic (corrected standard error, decision rule). The exploration, the modelling notebook, the variant choices, the runs and the conclusions are my own.

---

# The hand-in package

One zip, named `Andreas-Olsson.zip` as the spec asks, containing:

- `slides.pdf`: the deck exported to PDF (keep the native file for yourself)
- `initial_exploration.ipynb` and `modelling.ipynb`, executed, outputs kept, so a reviewer sees the tables without running 40 minutes
- `modelling_results.csv`: every number in the slides regenerates from it, and `load_results()` fills the notebook from it
- `figures/`: the twelve PNGs the slides use
- `submission_rf_numeric.txt` (what went to the leaderboard) and `submission_best_choice.txt` (what the rule picked), both 200 characters
- `README.md`, short: Python 3.12, scikit-learn 1.9, numpy, pandas, scipy, seaborn, matplotlib, xgboost, adjustText; run the notebooks from the repository root (paths are `labs/lab_1_music_taste_prediction/...`); put `training_data.csv` and `songs_to_classify.csv` in `labs/lab_1_music_taste_prediction/data/`; `load_results()` reads the cached results, delete the csv to recompute; the disclosure line above

Leave out: `data/` (the course's files), `professional_reference/`, `src/`, `tests/`, `.venv/`, `instructions/`, `comments.txt`.

Before zipping, three checks: open the PDF and count the slides (5 to 15); search the notebooks for the leaderboard password (it should not be in a cell); confirm both submission files are exactly 200 characters of 0 and 1.

After the hand-in: book the oral slot on the Canvas calendar (two per slot), peer reviews due 2026-10-11, resubmission window to 10-23.
