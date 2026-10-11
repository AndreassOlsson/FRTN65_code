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

Great job overall! Very clear report, and my only real points of feedback are that I wanted the fold SD included in the final table, and that you had some violation of DRY in the lecture_methods.

Presentation and code: yes to both. The report walks through everything from the data to the final predictions, and the notebook plus the lecture_methods folder reproduce it, with a requirements file included. Two small things though: the export cell is switched off by default, so the notebook as handed in does not actually produce the submission string, and the last cell only works in Colab.

Data: properly investigated and handled. The duplicates are removed with the assumption stated, the split is stratified, and all the exploration is done on the development set only. I especially liked the outlier check, since it is done programmatically rather than by eyeballing plots, and you then give a reason for not removing anything, which I think is the right call.

Features: selected and motivated. Comparing three candidate feature sets on the same folds, and reporting that the trees did best with the five numeric ones while the linear models wanted all 13, is an actual test. Nice.

Methods and parameters: reasonable choices, and each one has a stated purpose. The parameter grids are reasonable and motivated. However, Logistic Regression adn Linear SVM are quite similar, so it would have been interesting to see something a little more different, like kernel SVM or kNN in the comparison.

Evaluation: properly done and honestly reported. Selection by CV only, the decision saved before the holdout is scored, all four models scored on the holdout, and a conclusion which states that no model clearly wins. One thing I would have liked: having the fold SD included in the final comparison table, placed alongside the mean CV accuracy. With the SD next to each mean it would be easier to see how much of that 0.17 pp lead is noise. My guess is most of it, since the SD in the tables is around 3 pp.

Code: readable and documented, but one thing I would change. The four model files are basically copies of each other, with the same run loop and only the model and the grid differing, so any change to the loop has to be made in four places. I.e. it could be one shared run function, or just sklearn GridSearchCV, which already does that loop. But if you did it for learning purposes then there is nothign to be said here.

Good job Xuanyuan!
