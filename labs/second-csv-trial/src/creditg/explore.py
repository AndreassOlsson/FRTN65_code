"""Exploration: songtaste's explore functions with credit-g's spec.
`python -m creditg.explore` (`make explore`) writes the figures to
`figures/` and prints the tables `results/01-exploration.md` is written
from. There is no test file, so the duplicate check runs on one."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pandas as pd
from songtaste import explore

from creditg.data import SPEC, load

FIGURES = Path(__file__).resolve().parents[2] / "figures"


def main() -> None:
    df = load()
    for path in explore.save_all(df, SPEC, FIGURES):
        print("wrote", path.relative_to(FIGURES.parent))
    with pd.option_context("display.width", 140):
        print("\n## class balance\n" + explore.class_balance_table(df, SPEC).to_string())
        print("\n## separation\n" + explore.separation(df, SPEC).to_string())
        print("\n## outliers\n" + explore.outliers(df, SPEC).to_string())
        print("\n## categoricals\n" + explore.categorical_table(df, SPEC).to_string(index=False))
        print("\n## duplicates\n" + pd.Series(explore.duplicates(df, df.iloc[:0], SPEC)).to_string())


if __name__ == "__main__":
    main()
