"""Every registry method through the protocol `results/protocol.md`
states, then the comparison, the paired tests and the decision: all of
it songtaste's, called with this dataset's spec and protocol.

`python -m creditg.sweep` (`make sweep`) writes `results/sweep.csv` and
`figures/sweep-by-split.png` and prints the tables `results/sweep.md`
quotes.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer
from songtaste import evaluate, models, report
from songtaste.evaluate import Protocol

from creditg.data import SPEC, load

ROOT = Path(__file__).resolve().parents[2]
RESULTS, FIGURES = ROOT / "results", ROOT / "figures"


def cost(y, pred) -> float:
    """Mean cost per applicant under OpenML's matrix: a bad applicant
    called good costs 5, a good one called bad costs 1."""
    y, pred = np.asarray(y), np.asarray(pred)
    return float((5 * ((y == "bad") & (pred == "good")) + ((y == "good") & (pred == "bad"))).mean())


PROTOCOL = Protocol(
    primary="neg_cost",
    scores=("neg_cost", "accuracy", "balanced_accuracy", "roc_auc"),
    scorers={"neg_cost": make_scorer(cost, greater_is_better=False)},
    drop_duplicates=False,
    rope=0.05,
)


def xy() -> tuple[pd.DataFrame, pd.Series]:
    df = load()
    return df[list(SPEC.features)], df[SPEC.label]


def main(argv=None) -> None:
    X, y = xy()
    rows = evaluate.run_many(list(models.REGISTRY), X, y, PROTOCOL, SPEC)
    evaluate.write_results(rows, RESULTS / "sweep.csv")
    FIGURES.mkdir(exist_ok=True)
    report.plot_splits(rows, FIGURES / "sweep-by-split.png", PROTOCOL)
    choice, band = report.decide(rows, PROTOCOL)
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.max_colwidth", 80):
        print(report.comparison(rows, PROTOCOL).round(3).to_string(), end="\n\n")
        print(report.paired(rows, PROTOCOL).round(4).to_string(), end="\n\n")
        print(evaluate.chosen_params(rows).to_string(index=False), end="\n\n")
        print(rows.groupby("method", sort=False)["fit_time"].mean().round(2).to_string(), end="\n\n")
        print("qualifying under the rule:\n" + band.round(4).to_string())
    print(f"\nchosen: {choice}")


if __name__ == "__main__":
    main(sys.argv[1:])
