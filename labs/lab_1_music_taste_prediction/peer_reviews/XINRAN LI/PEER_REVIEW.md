# Peer review by Andreas

Mandatory peer review ("kamratgranskning")

Do not submit your peer review as a "new attempt"---use the built-in system in Canvas to submit your peer review.

You need to peer review two other Lab 1 hand-ins which you will be assigned via Canvas.

A good peer review should comment on the following:

- Does the hand-in contain a presentation or report, and the code needed to reproduce the findings?
- Has data been properly investigated and handled?
- Has a reasonable selection of features been done, and motivated?
- Has a reasonable choice of modeling method(s) been done?
- Are choices of parameters described and motivated?
- Has the performance of each method been properly evaluated?
- Is the code readable and well-documented?

Your peer review must contain detailed motivations. I.e., only leaving yes/no answers (or similar) to the questions above is not acceptable and will lead to a failing grade on your peer review.

## My initial thoughts

Good coding practice, it makes it easy to try out multiple combinations.

Very clear presentation. Dataset was inspected, features were preprocessed, validation and holdout split, explicit selection rule.

Selects best model by holdout, retrains by all.

Has good intuition of the models and their hyperparams.

Performed well structured hyperparameter search.

Computed cross validation comparisons with fold standard deviation, i.e. taking a statistical approach to comparing models. Good.

Good declaration of limitations: small dataset, artist IDs unavailable, target accuracy unknown.

## The plain text review comment to submit in canvas

Overall: a clear, disciplined hand-in. The pipeline has the right shape (inspect, split, shared folds, select by cross-validation, check once on a holdout, refit on everything, predict) and the twelve slides say what was done and why without padding. What I list below would make the evidence stronger; I did not find anything wrong.

Presentation and code: yes to both. The deck covers the data, the preprocessing, the validation scheme, each method, the search, the comparison, the holdout result and the final submission, and main.py reproduces all of it with one command, seeds fixed. One gap on the reproduce side: there is no requirements file, and OneHotEncoder(sparse_output=...) needs scikit-learn 1.2 or newer, so on an older install the script fails on the first model. One line with the versions you ran would make "python main.py" literal.

Data: properly handled for a problem this size. Before touching anything the script checks that labels are exactly 0 and 1, that every feature is finite, and that no two identical feature rows disagree on the label; then the 14 duplicates are dropped before the split, so the same record cannot sit on both sides of it. Class balance (443/293) and the overlap in energy and instrumentalness are shown. The exploration itself is one slide, which is on the thin side: a like rate per key, mode and time_signature, or a look at which features the forest actually uses, would have cost one cell and would tell the reader where the signal is.

Features: motivated, but not really selected. Keeping all 13, standardizing the 10 numeric ones for the distance and margin based models and leaving them raw for the trees, and one-hot encoding the three categoricals so the model is not told that key 11 is far from key 0, is all correct and each gets its reason on the slide. What is missing is any test of whether a subset does as well: no feature set comparison and no importances, so the report cannot say which of the 13 carry the prediction. "Keep all to limit the search" is an honest choice on 736 rows; I would just present it as a choice not to select rather than as a selection.

Methods: a reasonable spread. Logistic regression (linear), kNN (local), RBF SVM (kernel), random forest and gradient boosting (two ensembles) cover the main families from the course, and each method slide says in a few lines what the model does and what its parameters control, which shows you understand them rather than only ran them.

Parameters: described and mostly motivated. The search table lists every grid, the C values are log spaced, the role of each parameter (k and the weighting, C and gamma, depth and leaf size, tree count against learning rate) is stated, and 45 settings x 5 folds tells the reader the size of the search. I would add one sentence on why those ranges and not others (why gamma 0.01 and 0.1 next to scale, why depth 1 and 2 for boosting), and say that the forest's 150 trees is fixed rather than tuned, which is fine for a forest but should be stated.

Evaluation: the strongest part. The same five folds for every setting, preprocessing fitted inside each fold, a majority baseline (60%), the fold standard deviation next to every mean, an explicit note that the CV score of a search winner is optimistic, and a holdout untouched until the choice was frozen. The holdout reports accuracy, balanced accuracy, F1, ROC-AUC and the confusion matrix, so the 81% is not hiding a bad minority class. Two things would make the comparison itself stronger. A: only the winner is scored on the holdout. The other four are already fitted, so scoring them costs nothing and would answer the question slide 9 raises, whether the 0.33 percentage point lead over gradient boosting survives at all. B: the fold SD is the right instinct; the next step is the standard error of the mean over the five folds (SD/sqrt(5)), or, since the folds are shared, the paired per-fold difference between the two ensembles. With a fold SD around 3.6 points a 0.33 point gap is noise. The slides say so in words; the number would say it in numbers.

Code: readable and well documented. One file, a docstring saying how to run it and what comes out, a comment on every step, and the models dict (pipeline plus grid per model) makes adding or removing a method a two line change. Preprocessing lives inside the sklearn Pipeline, so there is no leakage by construction, and bad input raises with a message.

Limitations are declared honestly (small data, one split, gaps smaller than the noise, no artist IDs so no grouping by artist, target accuracy unknown) and the AI use is stated. Good work.
