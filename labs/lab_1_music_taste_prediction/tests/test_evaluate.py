"""The protocol: the fold count, tuning that never sees the validation
part, and seeds that reproduce. Needs `make data` first."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline

from songtaste import evaluate, models
from songtaste.evaluate import Protocol, run_protocol, training_xy

QUICK = Protocol(n_repeats=2)


@pytest.fixture(scope="module")
def xy():
    return training_xy()


def test_duplicates_dropped(xy):
    X, y = xy
    assert len(X) == 736
    assert not pd.concat([X, y], axis=1).duplicated().any()


def test_one_row_per_outer_split(xy):
    rows = run_protocol("dummy", *xy)
    assert len(rows) == 25
    assert set(zip(rows["repeat"], rows["fold"])) == {(r, f) for r in range(5) for f in range(5)}
    # the floor: always "like", so accuracy is the like share of each fold
    assert rows["accuracy"].between(0.59, 0.62).all()


def test_seed_reproduces(xy):
    first = run_protocol("knn", *xy, protocol=QUICK).drop(columns="fit_time")
    again = run_protocol("knn", *xy, protocol=QUICK).drop(columns="fit_time")
    pd.testing.assert_frame_equal(first, again)


def test_other_seed_other_splits(xy):
    a = run_protocol("logreg", *xy, protocol=QUICK)
    b = run_protocol("logreg", *xy, protocol=Protocol(n_repeats=2, seed=1))
    assert not np.allclose(a["accuracy"], b["accuracy"])


class Spy(ClassifierMixin, BaseEstimator):
    """Records which songs each fit saw, by their row id in column 0."""

    seen: list[frozenset] = []

    def __init__(self, alpha=0):
        self.alpha = alpha

    def fit(self, X, y):
        Spy.seen.append(frozenset(np.asarray(X)[:, 0].astype(int)))
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.full(len(X), self.classes_[-1])

    def predict_proba(self, X):
        return np.tile([0.0, 1.0], (len(X), 1))


def test_tuning_never_sees_the_validation_part(xy, monkeypatch):
    X, y = xy
    ids = pd.DataFrame({"row": np.arange(len(X))})
    space = {"clf__alpha": [0, 1, 2]}
    monkeypatch.setitem(
        models.REGISTRY, "spy", models.Method(lambda spec, seed: Pipeline([("clf", Spy())]), space)
    )
    Spy.seen = []
    run_protocol("spy", ids, y, protocol=QUICK)

    # per outer split: one fit per inner fold and candidate, then the refit
    per_split = QUICK.inner_splits * len(space["clf__alpha"]) + 1
    splits = list(QUICK.outer_cv().split(ids, y))
    assert len(Spy.seen) == per_split * len(splits)
    for i, (train, test) in enumerate(splits):
        fits = Spy.seen[i * per_split : (i + 1) * per_split]
        assert all(fit <= set(train) for fit in fits)
        assert all(not fit & set(test) for fit in fits)
        assert fits[-1] == set(train)  # the refit is on the whole outer training part


def test_registry_has_the_three_baselines():
    assert set(evaluate.BASELINES) <= set(models.REGISTRY)
    assert not models.get("dummy").space
    assert set(models.get("logreg").space) == {"clf__C"}
    assert set(models.get("knn").space) == {"clf__n_neighbors", "clf__weights"}
