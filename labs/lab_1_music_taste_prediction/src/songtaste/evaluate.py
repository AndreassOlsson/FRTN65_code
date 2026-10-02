"""The evaluation protocol as one function, `results/protocol.md` in code.

`run_protocol` puts one registry method through the repeated stratified
folds with sklearn's `cross_validate`; a method with a search space is
wrapped in `GridSearchCV`, so tuning happens inside each outer training
part and never sees its validation part. The result is a tidy frame,
one row per outer split. The seed and both fold objects come from
`Protocol` and nowhere else.

`python -m songtaste.evaluate` (`make baselines`) runs the three
baselines and writes `results/02-baselines.csv`.
"""

import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_validate,
)

from songtaste import models
from songtaste.data import SPEC, FeatureSpec, load_training

RESULTS = Path(__file__).resolve().parents[2] / "results"
BASELINES = ("dummy", "logreg", "knn")


@dataclass(frozen=True)
class Protocol:
    """protocol.md's numbers. Change one only by a dated section there.

    A score is named by an sklearn scorer string, or by a key of
    `scorers` when no built-in scorer measures it (a cost matrix, say);
    either way, higher is better."""

    seed: int = 65
    n_splits: int = 5
    n_repeats: int = 5
    inner_splits: int = 5
    primary: str = "accuracy"
    scores: tuple[str, ...] = ("accuracy", "balanced_accuracy", "roc_auc")
    drop_duplicates: bool = True
    # a gap in the primary score inside this is no difference (report.paired)
    rope: float = 0.01
    scorers: Mapping[str, object] = field(default_factory=dict)

    def scorer(self, name: str):
        return self.scorers.get(name, name)

    def outer_cv(self) -> RepeatedStratifiedKFold:
        return RepeatedStratifiedKFold(n_splits=self.n_splits, n_repeats=self.n_repeats, random_state=self.seed)

    def inner_cv(self) -> StratifiedKFold:
        return StratifiedKFold(n_splits=self.inner_splits, shuffle=True, random_state=self.seed)

    @property
    def n_outer(self) -> int:
        return self.n_splits * self.n_repeats

    @property
    def test_train_ratio(self) -> float:
        """n_val / n_train for one outer split, for the corrected error."""
        return 1 / (self.n_splits - 1)


PROTOCOL = Protocol()


def training_xy(protocol: Protocol = PROTOCOL, spec: FeatureSpec = SPEC) -> tuple[pd.DataFrame, pd.Series]:
    """The labelled songs a method may see: the training file, exact
    duplicates dropped (protocol.md, 'The data a method sees')."""
    df = load_training()
    if protocol.drop_duplicates:
        df = df.drop_duplicates().reset_index(drop=True)
    return df[list(spec.features)], df[spec.label]


def estimator(name: str | models.Method, protocol: Protocol = PROTOCOL, spec: FeatureSpec = SPEC):
    """The method's pipeline, inside a GridSearchCV when it has a space.
    `name` is a registry name, or a `Method` built elsewhere (a variant)."""
    method = name if isinstance(name, models.Method) else models.get(name)
    pipe = method.pipeline(spec, protocol.seed)
    if not method.space:
        return pipe
    return GridSearchCV(pipe, method.space, scoring=protocol.scorer(protocol.primary), cv=protocol.inner_cv(), refit=True)


def _chosen(fitted) -> str:
    """The hyperparameters a fold's search picked, as stable JSON."""
    if not isinstance(fitted, GridSearchCV):
        return "{}"
    params = {k.removeprefix("clf__"): v for k, v in fitted.best_params_.items()}
    return json.dumps(params, sort_keys=True, default=float)


def run_protocol(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    protocol: Protocol = PROTOCOL,
    spec: FeatureSpec = SPEC,
    method: models.Method | None = None,
) -> pd.DataFrame:
    """One method through the outer folds: one row per split, with
    method, repeat, fold, each score, the chosen hyperparameters and
    the fit time (seconds, including the inner search). `method`, when
    given, is scored in place of the registry entry and `name` labels
    the rows (how `songtaste.variants` runs a variant through the same
    protocol)."""
    cv = cross_validate(
        estimator(method if method is not None else name, protocol, spec),
        X,
        y,
        cv=protocol.outer_cv(),
        scoring={score: protocol.scorer(score) for score in protocol.scores},
        return_estimator=True,
        error_score="raise",
    )
    split = np.arange(protocol.n_outer)
    rows = pd.DataFrame(
        {
            "method": name,
            "repeat": split // protocol.n_splits,
            "fold": split % protocol.n_splits,
            **{score: cv[f"test_{score}"] for score in protocol.scores},
            "params": [_chosen(est) for est in cv["estimator"]],
            "fit_time": cv["fit_time"],
        }
    )
    return rows


def run_many(names, X, y, protocol: Protocol = PROTOCOL, spec: FeatureSpec = SPEC) -> pd.DataFrame:
    return pd.concat([run_protocol(n, X, y, protocol, spec) for n in names], ignore_index=True)


def corrected_se(scores: pd.Series, protocol: Protocol = PROTOCOL) -> float:
    """Nadeau and Bengio's corrected standard error of a mean over the
    outer splits (protocol.md, 'Uncertainty'). Works on per-split scores
    and on paired per-split differences alike."""
    return float(np.sqrt((1 / len(scores) + protocol.test_train_ratio) * scores.var(ddof=1)))


def summarize(rows: pd.DataFrame, protocol: Protocol = PROTOCOL) -> pd.DataFrame:
    """Mean and corrected standard error per method and score."""
    out = {}
    for score in protocol.scores:
        g = rows.groupby("method", sort=False)[score]
        out[(score, "mean")] = g.mean()
        out[(score, "se")] = g.apply(corrected_se, protocol=protocol)
    return pd.DataFrame(out)


def paired_to_best(rows: pd.DataFrame, protocol: Protocol = PROTOCOL) -> pd.DataFrame:
    """Each method's mean accuracy gap to the best method, on the same
    splits, with the corrected error of that paired difference."""
    wide = rows.pivot_table(index=["repeat", "fold"], columns="method", values=protocol.primary)
    best = wide.mean().idxmax()
    diff = wide.sub(wide[best], axis=0)
    return pd.DataFrame(
        {"gap_to_best": diff.mean(), "se_of_gap": diff.apply(corrected_se, protocol=protocol)}
    ).rename_axis(f"best: {best}")


def chosen_params(rows: pd.DataFrame) -> pd.DataFrame:
    """How often each hyperparameter setting was picked across splits."""
    tuned = rows[rows["params"] != "{}"]
    return tuned.groupby(["method", "params"]).size().rename("splits").reset_index()


def write_results(rows: pd.DataFrame, path: Path) -> None:
    """Everything but the fit time, which no seed reproduces."""
    rows.drop(columns="fit_time").to_csv(path, index=False, float_format="%.6f", lineterminator="\n")


def main(argv=None) -> None:
    X, y = training_xy()
    rows = run_many(BASELINES, X, y)
    write_results(rows, RESULTS / "02-baselines.csv")
    with pd.option_context("display.width", 140, "display.max_columns", 20, "display.float_format", "{:.3f}".format):
        print(f"{len(X)} songs after dropping duplicates\n")
        print(summarize(rows), end="\n\n")
        print(paired_to_best(rows), end="\n\n")
        print(chosen_params(rows).to_string(index=False), end="\n\n")
        print(rows.groupby("method", sort=False)["fit_time"].agg(["mean", "sum"]))


if __name__ == "__main__":
    main(sys.argv[1:])
