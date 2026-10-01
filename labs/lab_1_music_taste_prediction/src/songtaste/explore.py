"""Exploration: each function takes a frame and the spec and hands the
drawing to seaborn and the arithmetic to pandas.

The figures answer three questions before any model exists: how
unbalanced the classes are, which features separate like from dislike
on their own, and what the oddities are that scaling and encoding must
survive. `python -m songtaste.explore` (`make explore`) writes every
figure to `figures/` and prints the tables the finding is written from;
the notebook calls the same functions for looking.
"""

from pathlib import Path

import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score

from songtaste.data import SPEC, FeatureSpec, load_test, load_training

FIGURES = Path(__file__).resolve().parents[2] / "figures"


def _named(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """The label as words, so legends say like and dislike."""
    return df.assign(**{spec.label: df[spec.label].map(spec.classes)})


def _long(df: pd.DataFrame, spec: FeatureSpec, columns) -> pd.DataFrame:
    return _named(df, spec).melt(id_vars=spec.label, value_vars=list(columns), var_name="feature")


# tables


def class_balance_table(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    counts = df[spec.label].map(spec.classes).value_counts()
    return pd.DataFrame({"rows": counts, "share": (counts / counts.sum()).round(3)})


def separation(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """How far each numeric feature alone pulls the classes apart.

    The univariate AUC is the chance that a random row of the positive
    class (a liked song) scores higher on the feature than a random row
    of the other: 0.5 is no separation, and the distance from 0.5 is
    what matters (below 0.5 means the positive class scores lower).
    """
    medians = df.groupby(spec.label)[list(spec.numeric)].median().T.rename(columns=spec.classes)
    positive = df[spec.label] == spec.positive
    auc = pd.Series({c: roc_auc_score(positive, df[c]) for c in spec.numeric}, name="auc")
    out = medians.join(auc)
    out["strength"] = (out["auc"] - 0.5).abs()
    return out.sort_values("strength", ascending=False).round(3)


def outliers(df: pd.DataFrame, spec: FeatureSpec, k: float = 3.0) -> pd.DataFrame:
    """Values beyond k interquartile ranges from the quartiles, per
    numeric feature: far enough out that a scaler or a distance-based
    model will feel them."""
    x = df[list(spec.numeric)]
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    iqr = q3 - q1
    far = (x < q1 - k * iqr) | (x > q3 + k * iqr)
    out = pd.DataFrame({"min": x.min(), "median": x.median(), "max": x.max(), "far_out": far.sum()})
    return out.round(3)


def categorical_table(df: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """For every value of every categorical: how many rows, and what
    share of them are of the positive class (for songs, liked)."""
    share = f"{spec.classes[spec.positive]} share"
    positive = (df[spec.label] == spec.positive).rename(share)
    rows = []
    for c in spec.categorical:
        g = positive.groupby(df[c]).agg(["size", "mean"]).set_axis(["rows", share], axis=1)
        rows.append(g.rename_axis("value").reset_index().assign(feature=c))
    return pd.concat(rows)[["feature", "value", "rows", share]].round(3).reset_index(drop=True)


def duplicates(train: pd.DataFrame, test: pd.DataFrame, spec: FeatureSpec) -> dict:
    features = list(spec.features)
    return {
        "training rows that repeat another": int(train.duplicated(features).sum()),
        "distinct rows repeated in training": int(train[train.duplicated(features, keep=False)].drop_duplicates(features).shape[0]),
        "test rows that repeat another": int(test.duplicated(features).sum()),
        "distinct rows in both files": int(train[features].drop_duplicates().merge(test[features].drop_duplicates()).shape[0]),
    }


# figures


def plot_class_balance(df: pd.DataFrame, spec: FeatureSpec):
    return sns.catplot(_named(df, spec), x=spec.label, kind="count", order=list(spec.classes.values()), height=3.5, aspect=1.1)


def plot_numeric_by_label(df: pd.DataFrame, spec: FeatureSpec):
    """One density per class per numeric feature, each normalised within
    its class so the 60/40 imbalance does not read as separation."""
    return sns.displot(
        _long(df, spec, spec.numeric),
        x="value",
        hue=spec.label,
        hue_order=list(spec.classes.values()),
        col="feature",
        col_wrap=5,
        kind="kde",
        common_norm=False,
        fill=True,
        cut=0,
        height=2.4,
        facet_kws={"sharex": False, "sharey": False},
    )


def plot_categorical_by_label(df: pd.DataFrame, spec: FeatureSpec):
    """The categoricals as proportions within each class: pandas counts
    the share, seaborn draws one panel per feature with its own axis."""
    long = _long(df, spec, spec.categorical)
    shares = (
        long.groupby([spec.label, "feature"])["value"]
        .value_counts(normalize=True)
        .rename("proportion")
        .reset_index()
    )
    return sns.catplot(
        shares,
        x="value",
        y="proportion",
        hue=spec.label,
        hue_order=list(spec.classes.values()),
        col="feature",
        kind="bar",
        height=3,
        sharex=False,
        sharey=False,
    )


def plot_correlation(df: pd.DataFrame, spec: FeatureSpec, method: str = "spearman"):
    """Rank correlation between the numeric features and the label.
    Spearman, because duration, speechiness and instrumentalness are far
    from normal and a few extreme songs would drive Pearson."""
    positive = (df[spec.label] == spec.positive).astype(int)
    corr = df[list(spec.numeric)].assign(**{spec.label: positive}).corr(method=method)
    grid = sns.clustermap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, vmin=-1, vmax=1, figsize=(9, 9), annot_kws={"size": 7})
    return grid


FIGURE_MAKERS = {
    "class-balance": plot_class_balance,
    "numeric-by-label": plot_numeric_by_label,
    "categorical-by-label": plot_categorical_by_label,
    "correlation": plot_correlation,
}


def save_all(df: pd.DataFrame, spec: FeatureSpec, out: Path = FIGURES) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, make in FIGURE_MAKERS.items():
        path = out / f"01-{name}.png"
        make(df, spec).savefig(path, dpi=120, bbox_inches="tight")
        paths.append(path)
    return paths


def main() -> None:
    train, test = load_training(), load_test()
    for path in save_all(train, SPEC):
        print("wrote", path.relative_to(FIGURES.parent))
    print("\n## class balance\n" + class_balance_table(train, SPEC).to_string())
    print("\n## separation\n" + separation(train, SPEC).to_string())
    print("\n## outliers (training)\n" + outliers(train, SPEC).to_string())
    print("\n## outliers (test)\n" + outliers(test, SPEC).to_string())
    print("\n## categoricals\n" + categorical_table(train, SPEC).to_string(index=False))
    print("\n## duplicates\n" + pd.Series(duplicates(train, test, SPEC)).to_string())


if __name__ == "__main__":
    main()
