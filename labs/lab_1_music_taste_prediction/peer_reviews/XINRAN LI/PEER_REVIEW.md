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

Great job overall! You did a very pragmatic job on the method side, and while you did not make any mistakes on the data side, it would have been nice to see a bit more data exploration and feature selection, so that we could better understand the nature of the problem, like what kind of songs Andreas seemed to like etc.

Presentation and code: yes to both. The deck covers what it needs to, and the code reproduces since the source, the datasets and the seeds are all there. However, there is no requirements file, which is a bit inconvenient for people like me who use venvs instead of anaconda.

Data: properly handled. The script sanity checks the labels, the features and the duplicates before doing anything, and the duplicates are dropped before the split so nothing ends up on both sides of it. The exploration itself felt a bit thin. Some more plots on the feature distributions and feature importance after training would be nice to have.

Features: motivated, but not really selected. The preprocessing (scaling for the distance based models, raw values for the trees, one-hot for the categoricals) is correct and explained on the slide. But all 13 features are kept without testing whether a subset does as well or worse, so the report never really says which features matter. That would have been interesting to see.

Methods and parameters: a reasonable spread, from logistic regression and kNN to RBF SVM and the two tree ensembles, and the method slides show that you understand what each parameter does. The search grids look sensible (log spaced C for example), but why those ranges in particular is not really motivated, which could be looked into.

Evaluation: personally I think this is the strongest part. Same folds for every model, preprocessing fitted inside each fold, a majority baseline, the fold SD next to every mean, and a holdout that was left alone until the choice was made. One thing that maybe could make it even stronger: right now only the winner is scored on the holdout. It would be interesting to see the runner-ups there too, to check whether those 0.33 pp still hold. My guess is not necessarily, because with an SD of 3.63 it is quite hard to tell which model is genuinely better.

Code: readable and well documented. And the models dict (pipeline plus grid per model) makes adding or removing a method very easy.

The limitations slide was a nice bonus!

Good job Xinran!
