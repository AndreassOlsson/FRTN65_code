"""The sweep and what is read from it: the comparison table, the
per-split figure, the paired tests against the best method, and the
decision by the rule `results/protocol.md` wrote down before any number.

The paired tests follow sklearn's example "Statistical comparison of
models using grid search": on the 25 per-split accuracy differences
between a method and the best, the corrected resampled t-test (Nadeau
and Bengio) and its Bayesian reading, a Student t posterior over the
mean difference with the same corrected scale. The arithmetic is
scipy's t distribution and `evaluate.corrected_se`, nothing of our own.

`python -m songtaste.report` (`make sweep`) runs every registry entry
that is not a baseline through `evaluate.run_many`, reuses the
baselines' rows from `02-baselines.csv`, writes `03-sweep.csv` and the
figure, and prints every table `03-sweep.md` quotes.
"""

import sys

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import seaborn as sns
from scipy import stats

from songtaste import evaluate, models
from songtaste.evaluate import PROTOCOL, RESULTS, Protocol

FIGURES = RESULTS.parent / "figures"

# protocol.md's tie order, simplest first, as fixed there. adaboost is
# not in it, since the sketch did not name it: it is boosting's textbook
# variant and ranks with boosting, after it.
SIMPLICITY = (
    "logreg",
    "lda",
    "qda",
    "knn",
    "tree",
    "svm_linear",
    "svm_rbf",
    "bagging",
    "rf",
    "boosting",
    "adaboost",
)


def n_searched(name: str) -> int:
    """How many hyperparameters the protocol tunes for a method, or for a
    promoted variant, from its own grid (protocol.md, 2026-10-02)."""
    if name in models.REGISTRY:
        return len(models.get(name).space)
    from songtaste import variants

    return len(variants.get(name).method().space)


def simplicity_rank(name: str) -> float:
    """A method's place in SIMPLICITY. A promoted variant takes its base
    method's place, just after it: it is that method with a knob turned
    (protocol.md, 2026-10-02 and 2026-10-03)."""
    if name in SIMPLICITY:
        return float(SIMPLICITY.index(name))
    from songtaste import variants

    return SIMPLICITY.index(variants.get(name).base) + 0.5


def sweep_rows(protocol: Protocol = PROTOCOL) -> pd.DataFrame:
    """The baselines' saved rows, then every other registry entry through
    the protocol on the same splits."""
    base = pd.read_csv(RESULTS / "02-baselines.csv")
    X, y = evaluate.training_xy(protocol)
    names = [n for n in models.REGISTRY if n not in evaluate.BASELINES]
    new = evaluate.run_many(names, X, y, protocol)
    return pd.concat([base, new], ignore_index=True)


def paired(rows: pd.DataFrame, protocol: Protocol = PROTOCOL) -> pd.DataFrame:
    """Every method against the best non-dummy one on the same splits:
    the mean accuracy gap, its corrected error, the corrected t-test's
    two-sided p-value, and the posterior probabilities that the best is
    better, that the two are practically equal (within the protocol's rope), and that
    the method is better."""
    wide = rows.pivot_table(index=["repeat", "fold"], columns="method", values=protocol.primary, sort=False)
    best = wide.drop(columns="dummy", errors="ignore").mean().idxmax()
    out = []
    for name in wide.columns:
        diff = wide[name] - wide[best]
        k = len(diff)
        gap, se = float(diff.mean()), evaluate.corrected_se(diff, protocol)
        if name == best or se == 0:
            out.append({"method": name, "gap": gap, "se": se, "p": 1.0, "p_best_better": 0.0, "p_rope": 1.0})
            continue
        posterior = stats.t(df=k - 1, loc=gap, scale=se)
        out.append(
            {
                "method": name,
                "gap": gap,
                "se": se,
                "p": float(2 * stats.t.sf(abs(gap / se), df=k - 1)),
                "p_best_better": float(posterior.cdf(-protocol.rope)),
                "p_rope": float(posterior.cdf(protocol.rope) - posterior.cdf(-protocol.rope)),
            }
        )
    frame = pd.DataFrame(out).set_index("method")
    frame["p_method_better"] = 1 - frame["p_best_better"] - frame["p_rope"]
    return frame.sort_values("gap", ascending=False).rename_axis(f"vs {best}")


def decide(rows: pd.DataFrame, protocol: Protocol = PROTOCOL) -> tuple[str, pd.DataFrame]:
    """protocol.md's rule: among non-dummy methods whose mean accuracy is
    within one corrected error (of the paired difference) of the best's,
    the one with the fewest searched hyperparameters, ties by SIMPLICITY.
    Returns the choice and the qualifying set, in the order the rule
    ranked it."""
    table = paired(rows, protocol).drop(index="dummy", errors="ignore")
    band = table[-table["gap"] <= table["se"]].copy()
    band["searched"] = [n_searched(n) for n in band.index]
    band["order"] = [simplicity_rank(n) for n in band.index]
    band = band.sort_values(["searched", "order"])
    return band.index[0], band.drop(columns="order")


def _modal(params: pd.Series) -> str:
    """The setting picked on most splits, and on how many."""
    counts = params.value_counts()
    return f"{counts.index[0]} ({counts.iloc[0]}/{len(params)})"


def comparison(rows: pd.DataFrame, protocol: Protocol = PROTOCOL) -> pd.DataFrame:
    """Mean and corrected error per score, the number of searched
    hyperparameters and the most chosen setting, best accuracy first."""
    table = evaluate.summarize(rows, protocol)
    table.columns = [f"{score} {stat}" for score, stat in table.columns]
    table["searched"] = [n_searched(n) for n in table.index]
    table["most chosen"] = rows.groupby("method", sort=False)["params"].agg(_modal)
    return table.sort_values(f"{protocol.primary} mean", ascending=False)


def plot_splits(rows: pd.DataFrame, path=None, protocol: Protocol = PROTOCOL):
    """Per-split primary score by method, best mean on top; one seaborn
    call."""
    order = rows.groupby("method")[protocol.primary].mean().sort_values(ascending=False).index
    ax = sns.boxplot(rows, x=protocol.primary, y="method", order=order, color="0.85", showmeans=True)
    ax.set(xlabel=f"{protocol.primary} on each of the {protocol.n_outer} outer validation folds", ylabel="")
    ax.figure.tight_layout()
    if path is not None:
        ax.figure.savefig(path, dpi=150)
    return ax


def main(argv=None) -> None:
    rows = sweep_rows()
    evaluate.write_results(rows, RESULTS / "03-sweep.csv")
    FIGURES.mkdir(exist_ok=True)
    plot_splits(rows, FIGURES / "03-accuracy-by-split.png")
    choice, band = decide(rows)
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.max_colwidth", 80):
        print(comparison(rows).round(3).to_string(), end="\n\n")
        print(paired(rows).round(4).to_string(), end="\n\n")
        print(evaluate.chosen_params(rows).to_string(index=False), end="\n\n")
        print(rows.groupby("method", sort=False)["fit_time"].mean().round(2).to_string(), end="\n\n")
        print("qualifying under the rule:\n" + band.round(4).to_string())
    print(f"\nchosen: {choice}")


if __name__ == "__main__":
    main(sys.argv[1:])
