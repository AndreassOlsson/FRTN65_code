"""The comparison and the decision: the rule applied as protocol.md
wrote it, the paired test as sklearn's example computes it, and the
sweep's CSV holding every registry entry on the same 25 splits."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from songtaste import evaluate, models, report
from songtaste.evaluate import PROTOCOL


def rows_from(acc: dict[str, list[float]]) -> pd.DataFrame:
    """Per-split accuracies by method as the protocol's tidy rows."""
    out = []
    for name, scores in acc.items():
        for i, a in enumerate(scores):
            out.append({"method": name, "repeat": i // 5, "fold": i % 5, "accuracy": a, "params": "{}"})
    return pd.DataFrame(out)


rng = np.random.default_rng(0)
NOISE = rng.normal(0, 0.02, 25)


def test_best_wins_when_alone_in_the_band():
    rows = rows_from({"dummy": [0.6] * 25, "rf": list(0.85 + NOISE), "logreg": list(0.75 + NOISE)})
    choice, band = report.decide(rows)
    assert choice == "rf"
    assert list(band.index) == ["rf"]


def test_fewest_searched_wins_inside_the_band():
    # lda searches nothing, so inside the band it beats rf (two) and logreg (one)
    jitter = rng.normal(0, 0.01, 25)
    rows = rows_from(
        {"rf": list(0.83 + NOISE), "logreg": list(0.828 + NOISE + jitter), "lda": list(0.826 + NOISE - jitter)}
    )
    choice, band = report.decide(rows)
    assert set(band.index) == {"rf", "logreg", "lda"}
    assert choice == "lda"


def test_ties_on_count_go_by_the_simplicity_order():
    # logreg and svm_linear each search one hyperparameter; logreg comes first
    jitter = rng.normal(0, 0.01, 25)
    rows = rows_from({"svm_linear": list(0.83 + NOISE), "logreg": list(0.829 + NOISE + jitter)})
    assert report.decide(rows)[0] == "logreg"


def test_dummy_never_chosen():
    rows = rows_from({"dummy": [0.6] * 25, "tree": list(0.6 + NOISE)})
    assert report.decide(rows)[0] == "tree"


def test_corrected_t_test_matches_the_formula():
    a, b = 0.82 + NOISE, 0.80 + NOISE + rng.normal(0, 0.01, 25)
    table = report.paired(rows_from({"a": list(a), "b": list(b)}))
    diff = b - a
    t = diff.mean() / np.sqrt((1 / 25 + 1 / 4) * diff.var(ddof=1))
    assert table.loc["b", "p"] == pytest.approx(2 * stats.t.sf(abs(t), df=24))
    probs = table.loc["b", ["p_best_better", "p_rope", "p_method_better"]]
    assert probs.sum() == pytest.approx(1)


def test_every_method_family_is_registered_and_ordered():
    named = {"lda", "qda", "tree", "rf", "bagging", "boosting", "svm_linear", "svm_rbf"}
    assert named <= set(models.REGISTRY)
    assert set(models.REGISTRY) - {"dummy"} == set(report.SIMPLICITY)
    assert set(models.get("qda").space) == {"clf__reg_param"}


def test_sweep_csv_has_every_entry_on_the_same_splits():
    rows = pd.read_csv(evaluate.RESULTS / "03-sweep.csv")
    assert set(rows["method"]) == set(models.REGISTRY)
    splits = {(r, f) for r in range(PROTOCOL.n_repeats) for f in range(PROTOCOL.n_splits)}
    for _, g in rows.groupby("method"):
        assert len(g) == PROTOCOL.n_outer
        assert set(zip(g["repeat"], g["fold"])) == splits


def test_sweep_reuses_the_baselines_rows():
    sweep = pd.read_csv(evaluate.RESULTS / "03-sweep.csv")
    base = pd.read_csv(evaluate.RESULTS / "02-baselines.csv")
    reused = sweep[sweep["method"].isin(evaluate.BASELINES)].reset_index(drop=True)
    pd.testing.assert_frame_equal(reused, base)
