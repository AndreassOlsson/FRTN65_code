"""Diagnostics: the plots that say why a method or a variant scored as
it did. Each one is an sklearn routine (`validation_curve`,
`learning_curve`, `permutation_importance`, `cross_val_predict` with
the display classes) called with the spec and protocol passed in, its
output tidied into a frame, and one seaborn or sklearn call to draw it.

All of them take a registry name or a `Variant` name, so a baseline
and a variant are diagnosed the same way, and all of them use the
protocol's outer folds, so a curve is averaged over the same splits
the scores came from.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, learning_curve, validation_curve

from songtaste import evaluate, models, report, variants
from songtaste.data import SPEC, FeatureSpec
from songtaste.evaluate import Protocol

SCREEN = variants.SCREEN


def resolve(name: str) -> models.Method:
    """A variant's Method when the name is a variant, else the registry's."""
    if name in variants.VARIANTS:
        return variants.get(name).method()
    return models.get(name)


def _pipeline(name: str, protocol: Protocol, spec: FeatureSpec):
    return resolve(name).pipeline(spec, protocol.seed)


# ---- validation curve: one hyperparameter swept, the rest at their defaults


def validation_curve_table(
    name: str, param: str, grid, X, y, protocol: Protocol = SCREEN, spec: FeatureSpec = SPEC
) -> pd.DataFrame:
    """Train and validation score for each value of one pipeline
    parameter (`clf__C`, say), one row per value and outer split."""
    train, val = validation_curve(
        _pipeline(name, protocol, spec), X, y, param_name=param, param_range=list(grid),
        cv=protocol.outer_cv(), scoring=protocol.scorer(protocol.primary), n_jobs=-1,
    )
    rows = []
    for i, value in enumerate(grid):
        for s in range(train.shape[1]):
            rows.append({"method": name, "param": param, "value": value, "split": s, "train": train[i, s], "validation": val[i, s]})
    return pd.DataFrame(rows)


def plot_validation_curve(table: pd.DataFrame, log_x: bool | None = None, ax=None):
    long = table.melt(id_vars=["method", "param", "value", "split"], var_name="set", value_name="score")
    ax = sns.lineplot(long, x="value", y="score", hue="set", errorbar=("se", 1), marker="o", ax=ax)
    values = table["value"]
    if log_x is None:
        log_x = pd.api.types.is_numeric_dtype(values) and values.min() > 0 and values.max() / values.min() > 50
    if log_x:
        ax.set_xscale("log")
    ax.set(title=f"{table['method'].iat[0]}: {table['param'].iat[0]}", ylabel=f"{long['score'].name}")
    return ax


# ---- learning curve: does more data help, which is the bias-variance question in one plot


def learning_curve_table(
    name: str, X, y, protocol: Protocol = SCREEN, spec: FeatureSpec = SPEC, sizes=(0.2, 0.4, 0.6, 0.8, 1.0), tuned: bool = False
) -> pd.DataFrame:
    """Train and validation score against the number of training songs.
    `tuned` wraps the pipeline in the protocol's inner search (slow);
    otherwise the estimator runs at its defaults."""
    est = evaluate.estimator(resolve(name), protocol, spec) if tuned else _pipeline(name, protocol, spec)
    n, train, val = learning_curve(
        est, X, y, train_sizes=list(sizes), cv=protocol.outer_cv(), scoring=protocol.scorer(protocol.primary),
        shuffle=True, random_state=protocol.seed, n_jobs=-1,
    )
    rows = []
    for i, size in enumerate(n):
        for s in range(train.shape[1]):
            rows.append({"method": name, "n_train": int(size), "split": s, "train": train[i, s], "validation": val[i, s]})
    return pd.DataFrame(rows)


def plot_learning_curve(table: pd.DataFrame, ax=None):
    long = table.melt(id_vars=["method", "n_train", "split"], var_name="set", value_name="score")
    ax = sns.lineplot(long, x="n_train", y="score", hue="set", errorbar=("se", 1), marker="o", ax=ax)
    ax.set(title=f"{table['method'].iat[0]}: learning curve", xlabel="training songs")
    return ax


def plot_learning_curves(tables: list[pd.DataFrame], ncols: int = 3):
    n = len(tables)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.4 * nrows), sharey=True, constrained_layout=True)
    for ax, t in zip(np.atleast_1d(axes).ravel(), tables):
        plot_learning_curve(t, ax=ax)
        ax.get_legend().remove()
    for ax in np.atleast_1d(axes).ravel()[n:]:
        ax.set_visible(False)
    return fig


# ---- permutation importance: which features a fitted method actually leans on, scored on held-out songs


def permutation_table(
    names, X, y, protocol: Protocol = SCREEN, spec: FeatureSpec = SPEC, n_repeats: int = 10, tuned: bool = False
) -> pd.DataFrame:
    """For each method and each outer split: fit on the training part,
    then the drop in validation score when one original column is
    shuffled. One row per method, split and feature. `tuned` fits the
    inner search per split (slow); otherwise the defaults."""
    rows = []
    for name in names:
        for s, (tr, va) in enumerate(protocol.outer_cv().split(X, y)):
            est = evaluate.estimator(resolve(name), protocol, spec) if tuned else _pipeline(name, protocol, spec)
            est.fit(X.iloc[tr], y.iloc[tr])
            pi = permutation_importance(
                est, X.iloc[va], y.iloc[va], scoring=protocol.scorer(protocol.primary),
                n_repeats=n_repeats, random_state=protocol.seed, n_jobs=-1,
            )
            for f, m in zip(X.columns, pi.importances_mean):
                rows.append({"method": name, "split": s, "feature": f, "importance": float(m)})
    return pd.DataFrame(rows)


def importance_matrix(table: pd.DataFrame) -> pd.DataFrame:
    """Mean importance per feature (rows) and method (columns), features
    ordered by their mean rank across methods, so the robustly important
    ones are on top."""
    wide = table.groupby(["feature", "method"])["importance"].mean().unstack("method")
    rank = wide.rank(ascending=False).mean(axis=1)
    return wide.loc[rank.sort_values().index]


def plot_importance_matrix(matrix: pd.DataFrame, ax=None):
    ax = sns.heatmap(matrix, annot=True, fmt=".3f", cmap="Blues", cbar=False, ax=ax)
    ax.set(title="permutation importance (drop in accuracy on held-out songs)", xlabel="", ylabel="")
    return ax


# ---- out-of-fold predictions: ROC, confusion, and whether the threshold matters


def oof_predictions(name: str, X, y, protocol: Protocol = SCREEN, spec: FeatureSpec = SPEC, tuned: bool = True) -> pd.DataFrame:
    """Out-of-fold scores for every song, from one stratified k-fold at
    the protocol's seed (one repeat; every song predicted exactly once)."""
    est = evaluate.estimator(resolve(name), protocol, spec) if tuned else _pipeline(name, protocol, spec)
    cv = StratifiedKFold(n_splits=protocol.n_splits, shuffle=True, random_state=protocol.seed)
    pipe = est.estimator if hasattr(est, "estimator") else est
    method = "predict_proba" if hasattr(pipe.named_steps["clf"], "predict_proba") else "decision_function"
    score = cross_val_predict(est, X, y, cv=cv, method=method, n_jobs=-1)
    if score.ndim == 2:
        score = score[:, 1]
    return pd.DataFrame({"method": name, "y": y.to_numpy(), "score": score, "is_proba": method == "predict_proba"})


def plot_roc(oofs: list[pd.DataFrame], ax=None):
    ax = ax or plt.gca()
    for o in oofs:
        RocCurveDisplay.from_predictions(o["y"], o["score"], name=o["method"].iat[0], ax=ax)
    ax.set(title="out-of-fold ROC")
    return ax


def plot_confusion(oof: pd.DataFrame, threshold: float | None = None, ax=None):
    thr = (0.5 if oof["is_proba"].iat[0] else 0.0) if threshold is None else threshold
    pred = (oof["score"] >= thr).astype(int)
    d = ConfusionMatrixDisplay.from_predictions(oof["y"], pred, display_labels=list(SPEC.classes.values()), ax=ax, colorbar=False)
    d.ax_.set(title=f"{oof['method'].iat[0]}: out-of-fold confusion at {thr:g}")
    return d.ax_


def threshold_table(oof: pd.DataFrame, grid=None) -> pd.DataFrame:
    """Accuracy at each threshold on the out-of-fold scores: whether the
    0.5 the protocol decides at is near the best."""
    if grid is None:
        grid = np.linspace(0.2, 0.8, 25) if oof["is_proba"].iat[0] else np.quantile(oof["score"], np.linspace(0.1, 0.9, 25))
    return pd.DataFrame(
        {"threshold": grid, "accuracy": [accuracy_score(oof["y"], (oof["score"] >= t).astype(int)) for t in grid],
         "share_liked": [float((oof["score"] >= t).mean()) for t in grid]}
    )


# ---- the decision rule, under its neighbours


def decision_sensitivity(rows: pd.DataFrame, protocol: Protocol = evaluate.PROTOCOL) -> pd.DataFrame:
    """What the sweep's choice would be under neighbouring rules, so
    the report can say how much turns on the strictness fixed in
    protocol.md. `rows` are 03-sweep.csv's, or those joined with the
    promoted variants' full-protocol rows (`synthesis.union_rows`)."""
    paired = report.paired(rows, protocol).drop(index="dummy", errors="ignore")
    best = paired["gap"].idxmax()
    out = []
    rules = {
        "within 1 corrected se (protocol.md)": lambda t: t[-t["gap"] <= t["se"]],
        "within 2 corrected se": lambda t: t[-t["gap"] <= 2 * t["se"]],
        "corrected t-test p > 0.05": lambda t: t[t["p"] > 0.05],
        "within the rope (gap > -0.01)": lambda t: t[t["gap"] > -protocol.rope],
        "best mean, no band": lambda t: t.loc[[best]],
    }
    for label, rule in rules.items():
        band = rule(paired).copy()
        band["searched"] = [report.n_searched(n) for n in band.index]
        band["order"] = [report.simplicity_rank(n) for n in band.index]
        band = band.sort_values(["searched", "order"])
        out.append({"rule": label, "qualifying": ", ".join(band.index), "chosen": band.index[0]})
    return pd.DataFrame(out).set_index("rule")
