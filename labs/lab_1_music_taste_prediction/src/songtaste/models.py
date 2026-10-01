"""The model registry: a name, a factory returning the full pipeline
(preprocessor, then estimator), and the grid to search.

sklearn's estimator interface already is the strategy pattern, so a
method here is only what differs between methods: which estimator, and
which of its hyperparameters the protocol tunes. Search-space keys are
pipeline parameter names (`clf__C`), so the grid goes straight into
GridSearchCV. An entry with an empty space is scored as is.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

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
}


def get(name: str) -> Method:
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"no method {name!r}; the registry has {sorted(REGISTRY)}") from None
