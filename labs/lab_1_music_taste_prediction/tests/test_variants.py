"""Variants: every one builds and fits, a plain variant is its registry
entry, feature subsets reach the estimator, and the results file
accumulates without recomputing. Needs `make data` first."""

import pandas as pd
import pytest

from songtaste import evaluate, variants
from songtaste.evaluate import Protocol

QUICK = Protocol(n_repeats=1)


@pytest.fixture(scope="module")
def xy():
    return evaluate.training_xy()


@pytest.mark.parametrize("name", sorted(variants.VARIANTS))
def test_every_variant_builds_and_fits(name, xy):
    X, y = xy
    v = variants.get(name)
    pipe = v.pipeline(seed=65).fit(X.iloc[:300], y.iloc[:300])
    assert set(pipe.predict(X.iloc[:5])) <= {0, 1}
    space = v.method().space
    if space:
        pipe.set_params(**{k: vals[0] for k, vals in space.items()}).fit(X.iloc[:300], y.iloc[:300])


def test_every_registry_method_has_a_plain_variant():
    bases = {v.base for v in variants.VARIANTS.values()}
    assert bases == {"logreg", "lda", "qda", "knn", "svm_linear", "svm_rbf", "tree", "rf", "bagging", "boosting", "adaboost"}
    for base in bases:
        assert base in variants.VARIANTS and variants.get(base).base == base


def test_plain_variant_is_the_registry_entry(xy):
    X, y = xy
    a = evaluate.run_protocol("knn", X, y, QUICK).drop(columns="fit_time")
    b = variants.run_variants(["knn"], X, y, QUICK, path=None, verbose=False)
    b = b.drop(columns=["fit_time", "variant", "family", "base", "n_repeats"])
    pd.testing.assert_frame_equal(a, b)


def test_feature_subset_and_encoder_reach_the_estimator(xy):
    X, y = xy
    top4 = variants.get("logreg_top4").pipeline().fit(X, y)
    assert set(top4.named_steps["pre"].get_feature_names_out()) == set(variants.TOP4)  # spec order, not TOP4 order
    numeric = variants.get("logreg_numeric").pipeline().fit(X, y)
    assert len(numeric.named_steps["pre"].get_feature_names_out()) == 10
    grouped = variants.get("logreg_grouped").pipeline().fit(X, y)
    names = list(grouped.named_steps["pre"].get_feature_names_out())
    assert any("infrequent" in n for n in names)
    inter = variants.get("logreg_interactions").pipeline().fit(X, y)
    assert len(inter.named_steps["pre"].get_feature_names_out()) == 10 + 45 + 18


def test_no_scaler_changes_no_split(xy):
    X, y = xy
    a = variants.run_variants(["tree", "tree_none"], X, y, QUICK, path=None, verbose=False)
    wide = a.pivot(index=["repeat", "fold"], columns="variant", values="accuracy")
    assert (wide["tree"] == wide["tree_none"]).all()


def test_results_file_accumulates_and_skips(xy, tmp_path):
    X, y = xy
    path = tmp_path / "v.csv"
    first = variants.run_variants(["lda"], X, y, QUICK, path=path, verbose=False)
    assert len(first) == 5 and path.exists()
    again = variants.run_variants(["lda", "lda_top4"], X, y, QUICK, path=path, verbose=False)
    assert set(again["variant"]) == {"lda", "lda_top4"}
    stored = pd.read_csv(path)
    assert len(stored) == 10 and "fit_time" not in stored
    # a run under another protocol is kept apart by n_repeats
    variants.run_variants(["lda"], X, y, Protocol(n_repeats=2), path=path, verbose=False)
    assert sorted(pd.read_csv(path)["n_repeats"].unique()) == [1, 2]


def test_summary_and_promotion(xy):
    X, y = xy
    rows = variants.run_variants(["lda", "lda_top4", "knn", "knn_top4"], X, y, QUICK, path=None, verbose=False)
    table = variants.summarize(rows, QUICK)
    assert list(table.columns[:2]) == ["family", "base"]
    assert table.loc["lda", "gap to base"] == 0 and table.loc["knn", "gap to base"] == 0
    assert set(table.index) == {"lda", "lda_top4", "knn", "knn_top4"}
    prom = variants.promoted(rows, QUICK)
    assert set(prom.index) <= set(table.index)
    assert (prom["gap to base"] > prom["se of gap"]).all()


def test_catalogue_names_every_variant():
    cat = variants.catalogue()
    assert set(cat.index) == set(variants.VARIANTS)
    assert set(cat["family"]) == set(variants.FAMILIES)
