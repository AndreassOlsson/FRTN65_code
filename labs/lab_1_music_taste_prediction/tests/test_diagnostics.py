"""Diagnostics: shapes and invariants on a one-repeat protocol."""

import numpy as np
import pandas as pd
import pytest

from songtaste import diagnostics as d
from songtaste import evaluate, explore
from songtaste.evaluate import RESULTS, Protocol

QUICK = Protocol(n_repeats=1)


@pytest.fixture(scope="module")
def xy():
    return evaluate.training_xy(QUICK)


def test_validation_curve_one_row_per_value_and_split(xy):
    t = d.validation_curve_table("logreg", "clf__C", [0.01, 100.0], *xy, protocol=QUICK)
    assert len(t) == 2 * QUICK.n_outer
    # heavy shrinkage underfits the training part less than it overfits: train >= validation on average
    assert t.groupby("value")["train"].mean().loc[100.0] >= t.groupby("value")["validation"].mean().loc[100.0]
    d.plot_validation_curve(t)


def test_learning_curve_grows_the_training_set(xy):
    t = d.learning_curve_table("tree", *xy, protocol=QUICK, sizes=(0.5, 1.0))
    assert sorted(t["n_train"].unique())[0] < sorted(t["n_train"].unique())[1]
    d.plot_learning_curves([t])


def test_permutation_matrix_orders_by_mean_rank(xy):
    t = d.permutation_table(["logreg", "lda"], *xy, protocol=QUICK, n_repeats=2)
    m = d.importance_matrix(t)
    assert m.shape == (13, 2)
    assert m.index[0] in ("speechiness", "loudness", "acousticness", "energy")
    d.plot_importance_matrix(m)


def test_oof_predictions_cover_every_song_once(xy):
    X, y = xy
    o = d.oof_predictions("logreg", X, y, QUICK)
    assert len(o) == len(X) and o["is_proba"].all()
    assert o["score"].between(0, 1).all()
    s = d.oof_predictions("svm_linear", X, y, QUICK, tuned=False)
    assert not s["is_proba"].any()
    tt = d.threshold_table(o)
    assert tt["accuracy"].max() >= 0.75
    d.plot_roc([o, s])
    d.plot_confusion(o)


def test_decision_sensitivity_reproduces_the_protocol_rule():
    sweep = pd.read_csv(RESULTS / "03-sweep.csv")
    t = d.decision_sensitivity(sweep)
    assert t.loc["within 1 corrected se (protocol.md)", "chosen"] == "rf"
    assert len(t) == 5


def test_feature_screen_equal_capacity_and_null(xy):
    X, y = xy
    s = explore.feature_screen(X, y, protocol=QUICK, null_draws=2)
    assert list(s["singles"].index[:1])[0] in ("speechiness", "loudness", "acousticness", "energy")
    assert len(s["pairs"]) == 13 * 12 // 2
    assert np.allclose(s["pairs"]["gain"], s["pairs"]["pair"] - s["pairs"]["best_single"], atol=2e-3)
    assert len(s["null"]) == 2 and s["null"].abs().max() < 0.05
    explore.plot_feature_screen(s)
