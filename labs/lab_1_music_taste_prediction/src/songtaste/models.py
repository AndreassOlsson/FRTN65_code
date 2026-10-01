"""The model registry: a name, a factory returning the full pipeline
(preprocessor, then estimator), and the grid to search.

sklearn's estimator interface already is the strategy pattern, so a
method here is only what differs between methods: which estimator, and
which of its hyperparameters the protocol tunes. Search-space keys are
pipeline parameter names (`clf__C`), so the grid goes straight into
GridSearchCV. An entry with an empty space is scored as is.

Every entry starts with the same preprocessor, the tree-based ones
included. Trees split on thresholds and do not care about scale, so
for them the scaler is a monotone relabelling that changes no split;
keeping it costs a few milliseconds and keeps one pipeline shape for
every method, where a flag would be a second code path to get wrong.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from songtaste.data import SPEC, FeatureSpec
from songtaste.features import build_preprocessor


@dataclass(frozen=True)
class Method:
    """`factory(spec, seed)` returns an unfitted pipeline; `space` is the
    grid over it."""

    factory: Callable[[FeatureSpec, int], Pipeline]
    space: dict[str, list] = field(default_factory=dict)

    def pipeline(self, spec: FeatureSpec = SPEC, seed: int = 0) -> Pipeline:
        return self.factory(spec, seed)


def _pipe(spec: FeatureSpec, estimator) -> Pipeline:
    return Pipeline([("pre", build_preprocessor(spec)), ("clf", estimator)])


REGISTRY: dict[str, Method] = {
    # the floor: always "like", the majority class
    "dummy": Method(lambda spec, seed: _pipe(spec, DummyClassifier(strategy="most_frequent"))),
    # L2-regularised; C is the inverse strength, searched on a log grid
    # from heavy shrinkage to almost none
    "logreg": Method(
        lambda spec, seed: _pipe(spec, LogisticRegression(max_iter=5000, random_state=seed)),
        {"clf__C": [float(c) for c in np.logspace(-3, 3, 13)]},
    ),
    "knn": Method(
        lambda spec, seed: _pipe(spec, KNeighborsClassifier()),
        {
            "clf__n_neighbors": [1, 3, 5, 7, 9, 11, 15, 21, 31, 41, 51],
            "clf__weights": ["uniform", "distance"],
        },
    ),
    # nothing searched: one shared covariance, which is what makes it
    # linear; the svd solver copes with the one-hot columns summing to one
    "lda": Method(lambda spec, seed: _pipe(spec, LinearDiscriminantAnalysis())),
    # a covariance per class from ~240 dislikes over 28 columns, some of
    # them one-hot and collinear, is singular without shrinkage, so the
    # shrinkage towards the identity is what is searched
    "qda": Method(
        lambda spec, seed: _pipe(spec, QuadraticDiscriminantAnalysis()),
        {"clf__reg_param": [0.01, 0.05, 0.1, 0.25, 0.5, 0.9]},
    ),
    "tree": Method(
        lambda spec, seed: _pipe(spec, DecisionTreeClassifier(random_state=seed)),
        {
            "clf__max_depth": [2, 3, 4, 6, 8, None],
            "clf__min_samples_leaf": [1, 5, 10, 20],
        },
    ),
    # enough trees that adding more only changes the noise, so the count
    # is fixed and the two knobs that set each tree's variance are searched
    "rf": Method(
        lambda spec, seed: _pipe(spec, RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed)),
        {
            "clf__max_features": ["sqrt", 0.25, 0.5],
            "clf__min_samples_leaf": [1, 3, 10],
        },
    ),
    # a random forest that looks at every column at every split, so its
    # gap to rf is what decorrelating the trees buys
    "bagging": Method(
        lambda spec, seed: _pipe(
            spec,
            BaggingClassifier(DecisionTreeClassifier(), n_estimators=300, n_jobs=-1, random_state=seed),
        ),
        {"clf__estimator__min_samples_leaf": [1, 3, 10]},
    ),
    # sklearn's histogram gradient boosting: the same additive model of
    # shallow trees as classic gradient boosting, faster, and the one
    # sklearn recommends; adaboost below is the textbook variant
    "boosting": Method(
        lambda spec, seed: _pipe(spec, HistGradientBoostingClassifier(random_state=seed)),
        {
            "clf__learning_rate": [0.03, 0.1, 0.3],
            "clf__max_depth": [2, 3, None],
            "clf__max_iter": [100, 300],
        },
    ),
    # stumps, reweighting the songs the last ones got wrong
    "adaboost": Method(
        lambda spec, seed: _pipe(
            spec, AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), random_state=seed)
        ),
        {"clf__learning_rate": [0.1, 0.3, 1.0], "clf__n_estimators": [100, 300]},
    ),
    "svm_linear": Method(
        lambda spec, seed: _pipe(spec, SVC(kernel="linear")),
        {"clf__C": [0.001, 0.01, 0.1, 1.0, 10.0]},
    ),
    # gamma around sklearn's "scale" default (about 1/28 here) and C
    # from soft to hard margin
    "svm_rbf": Method(
        lambda spec, seed: _pipe(spec, SVC(kernel="rbf")),
        {"clf__C": [0.1, 1.0, 10.0, 100.0], "clf__gamma": [0.01, 0.03, 0.1, 0.3]},
    ),
}


def get(name: str) -> Method:
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"no method {name!r}; the registry has {sorted(REGISTRY)}") from None
