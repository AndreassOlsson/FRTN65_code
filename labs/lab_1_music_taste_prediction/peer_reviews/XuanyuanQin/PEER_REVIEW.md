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

Good modular code structure (but since you seem to use those models in the same way, perhaps it would be better code quality to apply the SOLID swe principles, abstracting away your own "model" interface etc)

Good that you selected based on a hold out test set and then refit on the entire labelled dataset in time for training.

Good with a programmatic outlier check, instead of only looking at the plots.

Overall a clear presentation.

## The plain text review comment to submit in canvas

Overall: a careful hand-in. The order of operations is right (dedupe, split, explore only the development set, fix the feature sets and grids, select by cross-validation, score the holdout once, refit on everything), the report is honest about how little separates the models, and the notebook writes every intermediate result to disk so a reader can check the numbers instead of trusting them. What I have below is mostly about the shape of the code and about showing the noise next to the means.

Presentation and code: yes to both. The report walks from the data overview to the final predictions and the AI-use statement, and the notebook plus the lecture_methods folder reproduce it, with a requirements.txt and a timestamped run folder holding the split indices and the decision file, which is more than most hand-ins do. Three things get in the way of "run it and get what was submitted". A: RUN_EXPORT is False, so the notebook as handed in does not produce the 200 character string; the reader has to find the flag. B: the last cell imports google.colab and fails anywhere but Colab. C: the README's "Running and outputs" section contains only the pip line, so what lecture_runs will contain is written nowhere. Small things, but that question is literally about them.

Data: properly investigated and handled. The 14 exact duplicates are dropped under a stated assumption, with a guard that raises if identical features ever disagree on the label; the split is stratified and seeded; and the exploration (describe table, per-class histograms for all ten numeric features, like rate per key, mode and time_signature, a scatter matrix) is done on the development set only, so the holdout is unseen even by your eyes. I liked the outlier handling in particular: the 1.5 IQR rule is run programmatically, it flags 41 songs by duration, and you then give a reason not to remove them (a statistical flag is not evidence of an error). That is the right call; most people either skip the check or act on it blindly.

Features: selected and motivated. Three candidate sets (the five numeric features that showed a class difference in the histograms, those plus the categoricals, all 13) are fixed before CV, each model picks its own set on the same folds, and the outcome is reported: trees did best with the five, the linear models with all 13, and the categoricals helped nobody. That is a test rather than an assertion. One caveat you partly state yourself: set A was chosen by eye from the development histograms and then validated by CV on the same development data, so the five are the five that looked good here, not a best subset. The holdout is clean, so this is fine, just worth knowing.

Methods: reasonable, and each has a stated purpose (a simple linear baseline, a second way to fit a linear boundary, flexible combinations of features, a second kind of ensemble). The one I would question is linear SVM next to logistic regression: on standardized inputs they draw nearly the same boundary, and the CV numbers agree to 0.2 percentage points, so the fourth model told you little the first had not. A kernel SVM or kNN would have been a more informative fourth.

Parameters: described and motivated. The candidate table gives every grid with its reason (C by factors of ten, 100 against 300 trees for stability, sqrt against 0.8 features per split, leaf size, depth 1 against 2 for boosting), the same five folds score every candidate, and preprocessing is fitted inside the pipeline so nothing leaks from the validation fold. The grids are small, which is fine for this amount of data, and the chosen values are in the results table.

Evaluation: properly done and honestly reported. Selection is by CV only, the choice is written to decision_before_holdout.json before the holdout is scored (so the claim is checkable), all four models are scored on the holdout rather than only the winner, the majority baseline (60%) is shown, and the conclusion says plainly that random forest leads boosting by 0.17 points in CV, that three models tie on the holdout, and that no clear winner was shown. The confusion matrix and per-class precision and recall are there for the chosen model, and the importances come with the right caveats. What I would add: the fold standard deviation is computed in every trials table but the report shows only the means, so a reader of the report alone cannot see that 0.17 points sits inside a fold noise of about 3. Put the SD (or SD/sqrt(5) as a standard error) next to each CV mean and the "no clear advantage" sentence becomes a number.

Code: readable and documented, with one structural issue. The notebook is sectioned, every step has a markdown note on what it does and why, the modules have docstrings, and everything is written to disk. The issue is that logistic.py, svm_linear.py, random_forest.py and gradient_boosting.py are four copies of the same run function: the loop over feature sets and ParameterGrid, the table, the stable sort and the refit are identical, and only build_model and the grid differ. The consequence is that any change to the loop (a metric, a bug) has to be made in four places. A: one shared run(build_model, grid, name, ...) with each module reduced to its build_model and grid would cut the package to a third. B: GridSearchCV already does this loop, including the final refit, so most of run re-implements something sklearn ships. The modular intent is good; the duplication is what to fix.

Good work.
