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
> 11 | methods through one protocol
> 0.830 | cross-validated accuracy of the chosen random forest

Note: Ten minutes. The short version: eleven methods, one evaluation protocol written before any result, and a random forest that wins by a small but consistent margin.

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
- All 13 features used, none dropped: the weak ones cost little and the forest ignores noise well
- 10 numeric features, standardised
- key, mode and time_signature treated as qualitative and one-hot encoded: key 11 is not "more" than key 0, and meters are distinct, not points on a scale
- 14 exact duplicate rows dropped before any split, so a song cannot sit in its own validation fold
- Scaling and encoding live inside each model's pipeline, learned on the training part only

Note: Trees do not need the scaler, but it changes no split, so every method keeps one pipeline shape.

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
| Bagging | 0.820 | -0.011 | 0.007 | 0.15 | 18 |
| Gradient boosting | 0.819 | -0.011 | 0.009 | 0.24 | 19 |
| Logistic regression | 0.814 | -0.016 | 0.015 | 0.29 | 18 |
| RBF SVM | 0.813 | -0.017 | 0.015 | 0.28 | 17 |
| LDA | 0.799 | -0.031 | 0.014 | 0.04 | 19 |
| kNN | 0.795 | -0.035 | 0.014 | 0.02 | 22 |
| Decision tree | 0.790 | -0.041 | 0.017 | 0.03 | 22 |
| QDA | 0.783 | -0.047 | 0.020 | 0.02 | 22 |

Note: p is the corrected resampled t-test. AdaBoost (0.808) and linear SVM (0.807) sit between RBF SVM and LDA and are left off for space; the full table is in results/03-sweep.md. Four methods miss the one-error band by a hair.

---

# (4) Conclusion: the random forest, read as bias and variance
> 0.830 | random forest, ± 0.014 over 25 splits
- A line captures the main axis; logistic regression reaches 0.814 with heavy shrinkage
- What a line misses are thresholds and interactions (every song under -20 dB is liked), which trees find
- One tree finds them but pays in variance (0.790); averaging 300 trees keeps the low bias and pays the variance down (bagging 0.820)
- Random feature subsets decorrelate the trees and add one more point (0.830)
- kNN and QDA were last: distances mixed with one-hot columns, and too many covariance parameters for 240 dislikes

Note: Boosting reaches the same place from the other side: low-variance learners, bias removed step by step. It ties with bagging.

---

# (4) How sure: a small edge, honestly sized
> ~3 | songs in 200: the forest's expected edge over logistic regression
> 0.80 to 0.86 | what to expect on the 200 hidden songs
- Only the forest falls inside the one-error band, so the rule picks it outright
- A looser rule, "not significantly worse at 5%", would hold seven methods and pick logistic regression
- The strictness was fixed before the numbers, which is what makes this a choice and not a sweep
- First on accuracy, balanced accuracy and AUC alike, so the win is not bought by leaning on "like"

Note: If I had to defend logistic regression instead, I could: it is within noise. The protocol said in advance which end of that trade-off I would take.

---

# Production: one refit, one read of the test file
> 200 | songs predicted, 123 liked
- Refit on all 736 songs with the same grid search: 300 trees, 25% of features per split, at least 3 songs per leaf
- The inner search scored that setting at 0.837; the setting barely matters, the grid's surface was flat across the sweep
- songs_to_classify.csv read once, by this model, after the decision
- The string is in results/submission-2026-10-01.txt, one character per test song in file order
- 4 test songs also appear in the training file; the forest predicts their known labels

Note: make predict reproduces the string byte for byte.

---

# The code: a small package, thin notebooks
- `src/songtaste/` holds the mechanism as one module per stage (data contract, features, protocol, comparison, prediction), each standing on scikit-learn, pandera and seaborn rather than code of my own, and `tests/` checks the contract, the protocol, the rule and the string.
- The notebooks are where I looked at things and `results/` is where every number in these slides is written down, so `uv sync && make data && make test && make sweep && make predict` reproduces all of it.
- AI use, per the lab's policy: the package, tests and first drafts of the written results and these slides were produced with Claude (Anthropic) coding agents, working from a plan and a design I set and approved.

Note: The sweep takes about 36 minutes on four cores; everything else runs in a minute or two.
