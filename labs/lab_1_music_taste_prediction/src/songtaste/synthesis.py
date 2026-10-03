"""The variants stage closed: what the four families promoted, run
under the full protocol and joined to the sweep's comparison, and the
cross-family readings `results/04-variants.md` is written from.

Each step is a call the other modules already make, with the six
methods the synthesis reads across (`SIX`) and the folds of the
protocol that applies:

- promotion: `variants.promoted` over every screening file, then
  `variants.run_variants` under the full protocol into
  `results/04-variants-full.csv` (skipped when the rows are there);
- the union: 03-sweep's rows plus the promoted variants' full rows,
  through `report.comparison`, `report.paired` and `report.decide`, and
  `diagnostics.decision_sensitivity` on the sweep and on the union;
- the feature screen: `explore.feature_screen` under the screen;
- permutation importance across the six, searched inside each split;
- the learning curves of the six, searched inside each split.

Every table behind a figure is written beside it in
`figures/04-synthesis/` as a CSV, and a step whose CSV is there is read
back, not refitted, so `python -m songtaste.synthesis` (`make
synthesis`) reprints everything in seconds once it has run. The first
run takes about 20 minutes on 4 cores, most of it the searched
learning curves and permutation importances.
"""

import argparse
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from songtaste import diagnostics, evaluate, explore, report, variants
from songtaste.evaluate import PROTOCOL, RESULTS
from songtaste.variants import SCREEN

FIGURES = report.FIGURES / "04-synthesis"
FULL_FILE = RESULTS / "04-variants-full.csv"
SIX = ("logreg", "lda", "knn", "svm_rbf", "rf", "boosting")


def promote(rerun: bool = False) -> pd.DataFrame:
    """The promoted variants' rows under the full protocol."""
    names = list(variants.promoted(variants.load_all()).index)
    if not names:
        return pd.DataFrame()
    X, y = evaluate.training_xy(PROTOCOL)
    return variants.run_variants(names, X, y, PROTOCOL, path=FULL_FILE, rerun=rerun)


def union_rows() -> pd.DataFrame:
    """03-sweep's rows, then each promoted variant's full-protocol rows
    as one more method, on the same 25 splits."""
    sweep = pd.read_csv(RESULTS / "03-sweep.csv")
    if not FULL_FILE.exists():
        return sweep
    full = pd.read_csv(FULL_FILE)
    full = full[full["n_repeats"] == PROTOCOL.n_repeats].assign(method=lambda f: f["variant"])
    return pd.concat([sweep, full[sweep.columns]], ignore_index=True)


def _cached(name: str, make) -> pd.DataFrame:
    path = FIGURES / f"{name}.csv"
    if path.exists():
        return pd.read_csv(path)
    table = make()
    FIGURES.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, float_format="%.6f", lineterminator="\n")
    return table


def feature_screen() -> dict:
    X, y = evaluate.training_xy(SCREEN)
    parts = {}

    def run():
        out = explore.feature_screen(X, y, protocol=SCREEN)
        parts["singles"] = out["singles"].reset_index()
        parts["null"] = out["null"].to_frame("gain")
        return out["pairs"]

    pairs = _cached("feature-screen-pairs", run)
    singles = _cached("feature-screen-singles", lambda: parts["singles"])
    null = _cached("feature-screen-null", lambda: parts["null"])
    screen = {"singles": singles.set_index("feature"), "pairs": pairs, "null": null["gain"]}
    fig = explore.plot_feature_screen(screen)
    fig.savefig(FIGURES / "feature-screen.png", dpi=150)
    plt.close(fig)
    return screen


def permutation() -> pd.DataFrame:
    X, y = evaluate.training_xy(SCREEN)
    table = _cached("permutation", lambda: diagnostics.permutation_table(SIX, X, y, SCREEN, n_repeats=10, tuned=True))
    matrix = diagnostics.importance_matrix(table)[list(SIX)]
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    diagnostics.plot_importance_matrix(matrix, ax=ax)
    fig.savefig(FIGURES / "permutation-importance.png", dpi=150)
    plt.close(fig)
    return matrix


def learning_curves() -> pd.DataFrame:
    X, y = evaluate.training_xy(SCREEN)

    def run():
        return pd.concat([diagnostics.learning_curve_table(n, X, y, SCREEN, tuned=True) for n in SIX], ignore_index=True)

    table = _cached("learning-curves", run)
    fig = diagnostics.plot_learning_curves([table[table["method"] == n] for n in SIX], ncols=3)
    handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper right")
    fig.savefig(FIGURES / "learning-curves.png", dpi=150)
    plt.close(fig)
    return table


def sensitivity(union: pd.DataFrame) -> pd.DataFrame:
    def run():
        sweep = pd.read_csv(RESULTS / "03-sweep.csv")
        a = diagnostics.decision_sensitivity(sweep).reset_index().assign(table="03-sweep")
        b = diagnostics.decision_sensitivity(union).reset_index().assign(table="03-sweep and promoted")
        return pd.concat([a, b], ignore_index=True)

    return _cached("decision-sensitivity", run)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", choices=["promote", "union", "screen", "permutation", "learning", "sensitivity"], action="append")
    args = ap.parse_args(argv)
    steps = set(args.only or ["promote", "union", "screen", "permutation", "learning", "sensitivity"])
    FIGURES.mkdir(parents=True, exist_ok=True)
    wide = {"display.width": 220, "display.max_columns": 30, "display.max_colwidth": 90}
    with pd.option_context(*[x for kv in wide.items() for x in kv]):
        if "promote" in steps:
            print("promoted under the screen:")
            print(variants.promoted(variants.load_all()).round(3).to_string(), end="\n\n")
            promote()
        union = union_rows()
        if "union" in steps:
            fig, ax = plt.subplots(figsize=(6.4, 4.8))
            plt.sca(ax)
            report.plot_splits(union, FIGURES / "accuracy-by-split.png")
            plt.close(fig)
            print(report.comparison(union).round(3).to_string(), end="\n\n")
            print(report.paired(union).round(4).to_string(), end="\n\n")
            choice, band = report.decide(union)
            print("qualifying under the rule:\n" + band.round(4).to_string())
            print(f"\nchosen on the union: {choice}\n")
        if "sensitivity" in steps:
            print(sensitivity(union).to_string(index=False), end="\n\n")
        if "screen" in steps:
            s = feature_screen()
            print(s["singles"].to_string(), end="\n\n")
            print(s["pairs"].head(10).to_string(), end="\n\n")
            print(f"null max {s['null'].max():.3f}, mean {s['null'].mean():.3f}\n")
        if "permutation" in steps:
            print(permutation().round(3).to_string(), end="\n\n")
        if "learning" in steps:
            t = learning_curves()
            print(t.groupby(["method", "n_train"])[["train", "validation"]].mean().round(3).unstack("method").to_string())


if __name__ == "__main__":
    main(sys.argv[1:])
