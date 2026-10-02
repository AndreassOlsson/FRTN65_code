<!--
The source of handin/slides.pptx. Rebuild with
`uv run --with python-pptx python handin/make_slides.py`.
Slides are separated by a line holding only `---`. In each: `# title`,
then any of `## subtitle`, bullets, one image, one table, stat lines
(`> 0.830 | label`), and `Note:` lines, which become speaker notes.
Figures are the ones the code drew, read from figures/, never redrawn.
-->

# Predicting which songs Andreas likes
## FRTN65 Lab 1, Andreas Olsson, October 2026
> 736 | labelled songs after dropping duplicates
> 11 | methods through one protocol, then 44 variants of them
> 0.830 | random forest, the best mean
> 0.826 | kNN on four features, level with it and chosen by the rule

Note: Ten minutes. The short version: eleven methods and 44 variants, one evaluation protocol written before any result, and at the top two models that are level on accuracy: a random forest and a kNN on the four strong features. The rule written in advance picks the simpler one.

---

# The data: one axis, loud versus acoustic
![](../figures/01-numeric-by-label.png)
- 750 songs, 13 Spotify features, 60% liked, so always saying "like" scores 0.60
- Liked songs are quiet, acoustic and nearly free of speech
- Energy, loudness and acousticness are close to one feature (rank correlation up to 0.85)
- Speechiness is the strong signal that is not that axis
- Clean but not tidy: 12 songs repeated, heavy tails in duration, speechiness, instrumentalness

Note: Every song quieter than -20 dB is liked. 36 of the 43 songs above 0.33 speechiness, meaning rap or spoken parts, are disliked.

---

# (1) The methods: lines, distances, densities
| method | what it does | tuned |
|---|---|---|
| Logistic regression | linear log-odds, L2 shrinkage | C |
| LDA | Gaussian classes, one shared covariance, so a linear boundary | nothing |
| QDA | a covariance per class, so a quadratic boundary | shrinkage |
| kNN | majority vote among the k closest songs | k, weighting |
| Linear SVM | widest-margin hyperplane, soft margin | C |
| RBF SVM | the same margin in a kernel space, curved boundary | C, gamma |

Note: The first four are the classic families from the course; the two SVMs differ only in the kernel. All share the same preprocessing.

---

# (1) The methods: trees and their ensembles
| method | what it does | tuned |
|---|---|---|
| Decision tree | recursive threshold splits, one tree | depth, leaf size |
| Bagging | 300 deep trees on bootstrap samples, averaged | leaf size |
| Random forest | bagging, but each split sees a random subset of features | features per split, leaf size |
| Gradient boosting | shallow trees added one by one, each fixing the last one's errors | rate, depth, rounds |
| AdaBoost | stumps, reweighting the songs the previous ones got wrong | rate, rounds |
| Dummy | always "like", the floor | nothing |

Note: Bagging versus random forest is a controlled pair: the only difference is the random feature subsets, so their gap is what decorrelating the trees buys.

---

# (2) How the inputs went in
![](../figures/01-categorical-by-label.png)
- All 13 features in every method of the comparison; the variants then tried subsets, and only the distance and covariance methods gained from dropping to the four strong ones
- 10 numeric features, standardised
- key, mode and time_signature treated as qualitative and one-hot encoded: key 11 is not "more" than key 0, and meters are distinct, not points on a scale
- 14 exact duplicate rows dropped before any split, so a song cannot sit in its own validation fold
- Scaling and encoding live inside each model's pipeline, learned on the training part only

Note: Trees do not need the scaler, but it changes no split, so every method keeps one pipeline shape.

---

# (2) What the knobs did, family by family
| family | moved it beyond the noise (screen, gap ± error) | did nothing |
|---|---|---|
| Logistic regression | nothing; only C, which the search already finds | L1, elastic net, log, quantile, grouped or dropped categoricals |
| LDA, QDA | fewer parameters: shrinkage +0.012 ± 0.008, QDA on 4 features +0.015 ± 0.009 | QDA without the categoricals |
| kNN | 4 strong features only +0.020 ± 0.018 (promoted); robust scaling -0.043 ± 0.017 | log, Manhattan |
| SVM | robust scaling -0.048 ± 0.015 | polynomial kernel, log, 4 features against its own base |
| Trees | averaging: bagging +0.027 ± 0.015 over one tree, the forest +0.0095 ± 0.0066 over bagging | fewer features, grouped levels, depth limits, slower boosting |

Note: Screen numbers, ten splits, each variant against its own base on the same splits. Two lessons cross families. kNN and the RBF SVM measure distances, so they are the ones the input space matters to: robust scaling broke both, because instrumentalness is so spiked at zero that its interquartile range is nothing and dividing by it makes that one feature the whole distance. And the covariance methods and kNN gained from fewer features, while the line and the forest did not care.

---

# (2) Which features matter, across six methods
![](../figures/04-synthesis/permutation-importance.png)
- Drop in held-out accuracy when one feature is shuffled, each method tuned inside each of 10 splits
- Speechiness is first for all six, worth 0.09 to 0.11 to every one of them
- Acousticness is second-tier for all six; loudness and energy too, but they are twins, so methods split the credit
- Tempo, key, mode, meter and valence are worth nothing to any method
- Weak features matter only to kNN, which is being misled by them, not informed

Note: Robust across models means the same answer from a line, a covariance, a distance, a kernel and two kinds of tree ensemble. Permutation importance undercounts a feature with a correlated twin: LDA barely uses loudness because energy and acousticness carry the same axis for it.

---

# (2) How parameters were tuned, and the rule written first
> 25 | outer splits: 5-fold stratified CV, repeated 5 times, seed 65
> 5 | inner folds: grid search inside each outer training part only
> 1 | decision rule, committed before any model was fitted
- Accuracy decides, since the leaderboard scores accuracy; balanced accuracy and ROC AUC are reported beside it
- Every method sees the same 25 splits, so comparisons are paired
- Errors use the Nadeau and Bengio correction, because the 25 splits share most of their data
- The rule: of the methods within one corrected error of the best, take the one with the fewest tuned parameters

Note: Nested CV means the score of a tuned method never saw its own validation fold during tuning. Writing the rule first is what makes the choice a judgment rather than picking the biggest number after the fact.

---

# (3) Evaluation: the boxes overlap, the pairs do not
![](../figures/03-accuracy-by-split.png)
- Each box is one method's 25 per-split accuracies, the triangle its mean
- From rf down to LDA the boxes overlap almost entirely
- Most of the spread is which songs land in a fold, and every method shares that
- So methods are compared on their per-split differences, not their boxes

Note: This is why the comparison is paired. Read naively, everything above LDA looks the same.

---

# (3) Paired against the best method
| method | accuracy | gap to rf | corrected error | p | rf wins of 25 |
|---|---|---|---|---|---|
| Random forest | 0.830 | | | | |
| kNN, 4 features | 0.826 | -0.005 | 0.014 | 0.75 | 14 |
| Bagging | 0.820 | -0.011 | 0.007 | 0.15 | 18 |
| Gradient boosting | 0.819 | -0.011 | 0.009 | 0.24 | 19 |
| Logistic regression | 0.814 | -0.016 | 0.015 | 0.29 | 18 |
| RBF SVM | 0.813 | -0.017 | 0.015 | 0.28 | 17 |
| LDA | 0.799 | -0.031 | 0.014 | 0.04 | 19 |
| kNN | 0.795 | -0.035 | 0.014 | 0.02 | 22 |
| Decision tree | 0.790 | -0.041 | 0.017 | 0.03 | 22 |
| QDA | 0.783 | -0.047 | 0.020 | 0.02 | 22 |

Note: p is the corrected resampled t-test. The kNN on four features is the one variant the screen promoted, run here under the full protocol on the same 25 splits; it is the only method inside the one-error band. AdaBoost (0.808) and linear SVM (0.807) sit between RBF SVM and LDA and are left off for space; the full tables are in results/03-sweep.md and results/04-variants.md.

---

# (3) The decision rule under its neighbours
| rule | qualifying | chosen |
|---|---|---|
| within 1 corrected error of the best (written first) | kNN on 4 features, random forest | kNN on 4 features |
| within the rope, gap under one point | kNN on 4 features, random forest | kNN on 4 features |
| within 2 corrected errors | 8 methods | logistic regression |
| t-test not significant at 5% | the same 8 | logistic regression |
| best mean, no band | random forest | random forest |

Note: Same 25 splits, the eleven methods plus the promoted variant. Ties go to the fewest tuned parameters, then the simpler method; both top models tune two, and kNN is the simpler. Three answers from five reasonable rules is the honest picture: the top eight are within about two errors of each other. Before the variants the strict rule picked the forest alone; the rule did not change, the table got one row longer.

---

# (4) Conclusion: two models level at the top, read as bias and variance
> 0.830 | random forest
> 0.826 | kNN on speechiness, loudness, acousticness, energy
- A line captures the main axis, loud versus acoustic, and nothing we turned moved logistic regression past 0.815
- Trees find the thresholds a line misses, and pay in variance; averaging 300 of them, with random feature subsets, pays it down
- kNN was last of the real methods until its distance was measured in the four features that matter; then its search settled and it gained three points
- On the 15 splits the variant screen never saw, the four-feature kNN is 0.001 behind the forest
- The forest ranks songs better (ROC AUC 0.907 against 0.879); the rule decides on accuracy, as the leaderboard does

Note: The rule picks the four-feature kNN: level with the forest on accuracy, the same number of tuned parameters, and a model you can explain in a sentence, a song is liked when the seven to eleven songs nearest to it on those four features mostly were.

---

# (4) How sure: a small edge, honestly sized
> ~1 | song in 200: the forest's expected edge over the four-feature kNN
> 0.80 to 0.86 | what to expect on the 200 hidden songs
- The two are within a point of each other with probability about one half, and the forest wins 14 of the 25 splits
- A looser rule, "not significantly worse at 5%", would hold eight methods and pick logistic regression
- The strictness was fixed before the numbers, which is what makes this a choice and not a sweep
- The variant was picked on 10 of the 25 splits; the other 15 agree with them, so the promotion did not manufacture it

Note: If I had to defend the forest or logistic regression instead, I could: all three are within noise. The protocol said in advance which end of that trade-off I would take.

---

# Production: one refit, one read of the test file
> 200 | songs predicted by the random forest, 123 liked
- The submitted string was made by the forest on 2026-10-01, before the variants; the four-feature kNN joined the comparison after it
- Refit on all 736 songs with the same grid search: 300 trees, 25% of features per split, at least 3 songs per leaf
- songs_to_classify.csv read once, by this model, after the decision
- The string is in results/submission-2026-10-01.txt, one character per test song in file order
- 4 test songs also appear in the training file; the forest predicts their known labels

Note: make predict reproduces the string byte for byte.

---

# The code: a small package, thin notebooks
- `src/songtaste/` holds the mechanism as one module per stage (data contract, features, protocol, comparison, prediction), each standing on scikit-learn, pandera and seaborn rather than code of my own, and `tests/` checks the contract, the protocol, the rule and the string.
- The notebooks are where I looked at things and `results/` is where every number in these slides is written down, so `uv sync && make data && make test && make sweep && make synthesis && make predict` reproduces all of it.
- AI use, per the lab's policy: the package, tests and first drafts of the written results and these slides were produced with Claude (Anthropic) coding agents, working from a plan and a design I set and approved.

Note: The sweep takes about 36 minutes on four cores, the four variant families and their synthesis about an hour and a quarter more; everything else runs in a minute or two.
