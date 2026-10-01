"""The trial's protocol through songtaste: the cost scorer, the floor it
implies, and seeds that reproduce. Needs `make data` first."""

import dataclasses

import matplotlib
import numpy as np
import pandas as pd
import pytest
from songtaste import evaluate, explore

from creditg.data import SPEC, load
from creditg.sweep import PROTOCOL, cost, xy

matplotlib.use("Agg")
QUICK = dataclasses.replace(PROTOCOL, n_repeats=1)


def test_cost_matrix():
    assert cost(["bad"], ["good"]) == 5
    assert cost(["good"], ["bad"]) == 1
    assert cost(["good", "bad"], ["good", "bad"]) == 0


def test_majority_floor_costs_five_times_the_bad_share():
    rows = evaluate.run_protocol("dummy", *xy(), QUICK, SPEC)
    np.testing.assert_allclose(-rows["neg_cost"], 5 * (1 - rows["accuracy"]))
    assert (-rows["neg_cost"]).mean() == pytest.approx(1.5)


def test_seed_reproduces():
    first = evaluate.run_protocol("logreg", *xy(), QUICK, SPEC).drop(columns="fit_time")
    again = evaluate.run_protocol("logreg", *xy(), QUICK, SPEC).drop(columns="fit_time")
    pd.testing.assert_frame_equal(first, again)


def test_exploration_runs_on_worded_levels():
    df = load()
    assert explore.class_balance_table(df, SPEC)["rows"].to_dict() == {"good": 700, "bad": 300}
    assert "bad share" in explore.categorical_table(df, SPEC).columns
    for make in explore.FIGURE_MAKERS.values():
        make(df, SPEC)
