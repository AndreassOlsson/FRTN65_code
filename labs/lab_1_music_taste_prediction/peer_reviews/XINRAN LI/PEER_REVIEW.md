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

Great job overall! But could use a little more motivation with certain choices and it would be interesting to see some more feature selection.

Presentation and code: yes to both. The deck covers all it needs, and we can reproduce the code, because the source code and datasets were provided, and the seeds are fixed. However, there is no python requirements file, which is inconvinent for people like me who use Python venvs instead of anaconda.

Data: properly handled for a problem this size. Before touching anything the script checks that labels are exactly 0 and 1, that every feature is finite, and that no two identical feature rows disagree on the label. Then the 14 duplicates are dropped before the split, so the same record cannot sit on both sides of it. Class balance (443/293) and the overlap in energy and instrumentalness are shown. The exploration itself is one slide, which is on the thin side: a like rate per key, mode and time_signature, or a look at which features the forest actually uses, would be nice to have.

Features: motivated, but not really selected. Keeping all 13, standardizing the 10 numeric ones for the distance and margin based models and leaving them raw for the trees, and one-hot encoding the categoricals is all correct and each gets its reason on the slide. What is missing is any test of whether a subset of these features does as well or worse. I.e. there is no real feature set comparison, so the report does not really say which features are the most important.

Methods and Parameters: a reasonable spread. Logistic regression (linear), kNN (local), RBF SVM (kernel), random forest and gradient boosting (two ensembles) cover the main families from the course. The parameters chosen for the search are good, for example, log spaced C, but they are not necessarily motivated, which could be looked into.

Evaluation: personally, I think this is one of the strongest parts. The same five folds for every setting, preprocessing fitted inside each fold, a majority baseline (60%), the fold standard deviation next to every mean, an explicit note that the CV score of a search winner is optimistic, and a holdout untouched until the choice was frozen. The holdout reports accuracy, balanced accuracy, F1, ROC-AUC and the confusion matrix, so the 81% is not hiding a bad minority class. One thing I think maybe could make it even stronger, would be if you also fit the runner-ups on the full dataset for comparison. Right now it seems like you only fit the best choice on the total dataset. It would be interesting to see if those 0.33 pp still holds. My guess is not necessarily, because you had SD of 3.63 so its quite hard to tell which model is genuiunely better.

Code: readable and well documented. And the models dict (pipeline plus grid per model) makes adding or removing a method very easy.

The limitations mentioned were a nice bonus!

Good job Xinran!
