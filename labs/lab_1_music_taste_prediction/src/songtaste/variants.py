"""Variants: the same method families with their knobs turned, through
the same protocol.

A `Variant` is a registry method plus what differs from the baseline
pipeline: the numeric scaler, log1p on the spiky columns, how the
categoricals are encoded, a feature subset, interaction features, and
any change to the estimator or its grid. `method()` builds it as a
`models.Method`, so `evaluate.run_protocol` scores it exactly as it
scores a registry entry, and the rows carry the variant's name.

Variants are grouped in families (`FAMILIES`) so the notebook can run
one family at a time; `run_variants` appends to one results file and
skips what is already there, so a family can be run in slices across
sittings and every slice lands in the same table.

Two protocols apply (results/protocol.md, section 2026-10-02): the
screening protocol `SCREEN` (5 folds, 2 repeats, same seed) for the
wide grid here, and the full protocol for whatever a family promotes.
Nothing in this module changes a number in 02-baselines or 03-sweep.

`python -m songtaste.variants --family linear` (or `--all`) runs a
family under the screen and writes `results/04-variants.csv`.
"""

import argparse
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    PolynomialFeatures,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

from songtaste import evaluate, models
from songtaste.data import SPEC, FeatureSpec
from songtaste.evaluate import PROTOCOL, RESULTS, Protocol
from songtaste.features import build_preprocessor

VARIANTS_FILE = RESULTS / "04-variants.csv"

# the screening protocol: the same folds and seed, two repeats instead
# of five, so ten outer splits per variant (protocol.md, 2026-10-02)
SCREEN = Protocol(n_repeats=2)

# the four strongest features of the exploration (01-exploration.md),
# and the spiky ones a log tames
TOP4 = ("speechiness", "loudness", "acousticness", "energy")
SPIKY = ("duration", "speechiness", "instrumentalness", "liveness")


def _scaler(kind: str, seed: int):
    if kind == "standard":
        return StandardScaler()
    if kind == "robust":
        return RobustScaler()
    if kind == "quantile":
        # a rank transform to a normal shape; n_quantiles below the
        # smallest training part (about 470 rows under the screen)
        return QuantileTransformer(output_distribution="normal", n_quantiles=200, random_state=seed)
    if kind == "none":
        return "passthrough"
    raise ValueError(f"unknown scaler {kind!r}")


def _encoder(kind: str):
    if kind == "onehot":
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    if kind == "onehot-grouped":
        # levels rarer than 3% of the training part (key's thin ones,
        # time_signature 1 and 5) share one "infrequent" column
        return OneHotEncoder(handle_unknown="infrequent_if_exist", sparse_output=False, min_frequency=0.03)
    if kind == "drop":
        return "drop"
    raise ValueError(f"unknown encoder {kind!r}")


@dataclass(frozen=True)
class Variant:
    """One way of running a method. `base` names the registry entry
    whose estimator and grid this starts from."""

    name: str
    base: str
    family: str
    scaler: str = "standard"
    log_columns: tuple[str, ...] = ()
    encoder: str = "onehot"
    features: tuple[str, ...] | None = None  # None means every feature
    interactions: bool = False  # pairwise products of the numeric columns
    estimator_params: dict = field(default_factory=dict)  # set on the base estimator
    space: dict | None = None  # replaces the base grid when given
    note: str = ""

    def spec_for(self, spec: FeatureSpec = SPEC) -> FeatureSpec:
        if self.features is None:
            return spec
        keep = set(self.features)
        unknown = keep - set(spec.features)
        if unknown:
            raise KeyError(f"{self.name}: not in the spec: {sorted(unknown)}")
        return replace(
            spec,
            numeric=tuple(c for c in spec.numeric if c in keep),
            categorical=tuple(c for c in spec.categorical if c in keep),
        )

    def pipeline(self, spec: FeatureSpec = SPEC, seed: int = 0) -> Pipeline:
        spec = self.spec_for(spec)
        numeric = _scaler(self.scaler, seed)
        if self.interactions:
            numeric = make_pipeline(
                clone(numeric) if numeric != "passthrough" else StandardScaler(),
                PolynomialFeatures(degree=2, interaction_only=True, include_bias=False),
                StandardScaler(),
            )
        pre = build_preprocessor(spec, numeric=numeric, categorical=_encoder(self.encoder), log_columns=self.log_columns)
        base = models.get(self.base).pipeline(spec, seed)
        clf = clone(base.named_steps["clf"])
        if self.estimator_params:
            clf.set_params(**self.estimator_params)
        return Pipeline([("pre", pre), ("clf", clf)])

    def method(self) -> models.Method:
        space = models.get(self.base).space if self.space is None else self.space
        return models.Method(lambda spec, seed: self.pipeline(spec, seed), dict(space))


def _logreg_grid():
    return {"clf__C": [float(c) for c in np.logspace(-3, 3, 13)]}


VARIANTS: dict[str, Variant] = {}


def _add(*variants: Variant) -> None:
    for v in variants:
        if v.name in VARIANTS:
            raise ValueError(f"duplicate variant {v.name}")
        VARIANTS[v.name] = v


# ---- linear and discriminant: what regularisation, scaling and features do to a line

_add(
    Variant("logreg", "logreg", "linear", note="the baseline: L2, standard scaling, all features"),
    Variant(
        "logreg_l1", "logreg", "linear",
        estimator_params={"solver": "saga", "l1_ratio": 1.0, "max_iter": 20000},
        note="L1: coefficients go to exactly zero, so the fit selects features",
    ),
    Variant(
        "logreg_elastic", "logreg", "linear",
        estimator_params={"solver": "saga", "l1_ratio": 0.5, "max_iter": 20000},
        note="elastic net, half L1 half L2",
    ),
    Variant("logreg_robust", "logreg", "linear", scaler="robust", note="median and IQR scaling: the tails pull less"),
    Variant("logreg_quantile", "logreg", "linear", scaler="quantile", note="rank to normal: no tails at all"),
    Variant("logreg_log", "logreg", "linear", log_columns=SPIKY, note="log1p on the spiky four, then standard"),
    Variant("logreg_top4", "logreg", "linear", features=TOP4, note="only the four strong features"),
    Variant(
        "logreg_numeric", "logreg", "linear", encoder="drop",
        note="categoricals dropped: what key, mode and meter are worth to a line",
    ),
    Variant("logreg_grouped", "logreg", "linear", encoder="onehot-grouped", note="rare key and meter levels grouped"),
    Variant(
        "logreg_interactions", "logreg", "linear", interactions=True,
        note="pairwise products of the numeric features, so the line can bend along pairs",
    ),
    Variant("lda", "lda", "linear", note="the baseline"),
    Variant(
        "lda_shrink", "lda", "linear",
        estimator_params={"solver": "lsqr", "shrinkage": "auto"}, space={},
        note="Ledoit-Wolf shrinkage of the shared covariance",
    ),
    Variant("lda_top4", "lda", "linear", features=TOP4),
    # reg_param 0.1 is only the default an untuned diagnostic fits at
    # (0 is singular on the one-hot columns); the grid overrides it
    Variant("qda", "qda", "linear", estimator_params={"reg_param": 0.1}, note="the baseline"),
    Variant("qda_numeric", "qda", "linear", encoder="drop", estimator_params={"reg_param": 0.1},
            note="no one-hot columns in the class covariances"),
    Variant("qda_top4", "qda", "linear", features=TOP4, estimator_params={"reg_param": 0.1}),
)

# ---- kNN: a distance is only as good as the space it is measured in

_add(
    Variant("knn", "knn", "knn", note="the baseline: standard scaling, one-hot, Euclidean"),
    Variant("knn_robust", "knn", "knn", scaler="robust"),
    Variant("knn_quantile", "knn", "knn", scaler="quantile"),
    Variant("knn_log", "knn", "knn", log_columns=SPIKY),
    Variant("knn_numeric", "knn", "knn", encoder="drop", note="no one-hot axes in the distance"),
    Variant("knn_top4", "knn", "knn", features=TOP4),
    Variant("knn_manhattan", "knn", "knn", estimator_params={"p": 1}, note="L1 distance"),
)

# ---- SVM: the kernel, the margin, and the scale the kernel sees

_add(
    Variant("svm_linear", "svm_linear", "svm", note="the baseline"),
    Variant("svm_rbf", "svm_rbf", "svm", note="the baseline"),
    Variant(
        "svm_poly", "svm_rbf", "svm",
        estimator_params={"kernel": "poly", "coef0": 1.0},
        space={"clf__C": [0.1, 1.0, 10.0], "clf__degree": [2, 3], "clf__gamma": ["scale"]},
        note="polynomial kernel, degree 2 and 3",
    ),
    Variant("svm_rbf_robust", "svm_rbf", "svm", scaler="robust"),
    Variant("svm_rbf_quantile", "svm_rbf", "svm", scaler="quantile"),
    Variant("svm_rbf_log", "svm_rbf", "svm", log_columns=SPIKY),
    Variant("svm_rbf_top4", "svm_rbf", "svm", features=TOP4),
    Variant("svm_rbf_numeric", "svm_rbf", "svm", encoder="drop"),
)

# ---- trees and ensembles: scaling should not matter, features and depth should

_add(
    Variant("tree", "tree", "trees", note="the baseline"),
    Variant("tree_none", "tree", "trees", scaler="none", note="no scaler: must match the baseline split for split"),
    Variant("tree_top4", "tree", "trees", features=TOP4),
    Variant("rf", "rf", "trees", note="the baseline"),
    Variant("rf_top4", "rf", "trees", features=TOP4),
    Variant("rf_numeric", "rf", "trees", encoder="drop"),
    Variant("rf_grouped", "rf", "trees", encoder="onehot-grouped"),
    Variant(
        "rf_shallow", "rf", "trees",
        space={"clf__max_features": ["sqrt", 0.25, 0.5], "clf__max_depth": [3, 5, 8]},
        note="depth-limited trees instead of leaf-limited",
    ),
    Variant("bagging", "bagging", "trees", note="the baseline"),
    Variant("boosting", "boosting", "trees", note="the baseline"),
    Variant(
        "boosting_slow", "boosting", "trees",
        space={"clf__learning_rate": [0.01, 0.03], "clf__max_depth": [2, 3], "clf__max_iter": [300, 600]},
        note="smaller steps, more of them",
    ),
    Variant("boosting_top4", "boosting", "trees", features=TOP4),
    Variant("adaboost", "adaboost", "trees", note="the baseline"),
)

FAMILIES: dict[str, tuple[str, ...]] = {}
for _v in VARIANTS.values():
    FAMILIES.setdefault(_v.family, ())
    FAMILIES[_v.family] += (_v.name,)


def get(name: str) -> Variant:
    try:
        return VARIANTS[name]
    except KeyError:
        raise KeyError(f"no variant {name!r}; see songtaste.variants.FAMILIES") from None


def catalogue() -> pd.DataFrame:
    """Every variant as a row: what it changes against its base."""
    rows = []
    for v in VARIANTS.values():
        rows.append(
            {
                "variant": v.name,
                "family": v.family,
                "base": v.base,
                "scaler": v.scaler,
                "log": ",".join(v.log_columns),
                "encoder": v.encoder,
                "features": "all" if v.features is None else ",".join(v.features),
                "interactions": v.interactions,
                "estimator": ", ".join(f"{k}={val}" for k, val in v.estimator_params.items()),
                "searched": len(v.method().space),
                "note": v.note,
            }
        )
    return pd.DataFrame(rows).set_index("variant")


def load_results(path: Path = VARIANTS_FILE) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def run_variants(
    names,
    X: pd.DataFrame,
    y: pd.Series,
    protocol: Protocol = SCREEN,
    spec: FeatureSpec = SPEC,
    path: Path | None = VARIANTS_FILE,
    rerun: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Each named variant through the protocol, one row per outer split,
    tagged with the variant, its family and the protocol's repeat count.
    With `path`, rows are appended to that file and a variant already
    in it under the same protocol is skipped unless `rerun`. Returns the
    rows of the named variants (fresh or loaded)."""
    done = load_results(path) if path is not None else pd.DataFrame()
    have = set()
    if len(done):
        mask = done["n_repeats"] == protocol.n_repeats
        have = set(done.loc[mask, "variant"])
    out = []
    for name in names:
        v = get(name)
        if name in have and not rerun:
            out.append(done[(done["variant"] == name) & (done["n_repeats"] == protocol.n_repeats)])
            continue
        if verbose:
            print(f"{name:22s} ...", end="", flush=True)
        rows = evaluate.run_protocol(name, X, y, protocol, spec, method=v.method())
        rows.insert(0, "variant", name)
        rows.insert(1, "family", v.family)
        rows.insert(2, "base", v.base)
        rows["n_repeats"] = protocol.n_repeats
        if verbose:
            print(f" {rows[protocol.primary].mean():.3f} ± {evaluate.corrected_se(rows[protocol.primary], protocol):.3f}"
                  f"  ({rows['fit_time'].sum():.0f}s)")
        if path is not None:
            kept = rows.drop(columns="fit_time")
            if len(done):
                done = done[~((done["variant"] == name) & (done["n_repeats"] == protocol.n_repeats))]
            done = pd.concat([done, kept], ignore_index=True)
            Path(path).parent.mkdir(exist_ok=True)
            done.to_csv(path, index=False, float_format="%.6f", lineterminator="\n")
        out.append(rows)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def summarize(rows: pd.DataFrame, protocol: Protocol = SCREEN) -> pd.DataFrame:
    """Per variant: mean and corrected error of each score, and the gap
    to its own base variant on the same splits (positive means the
    variant helped)."""
    table = evaluate.summarize(rows.drop(columns="method").rename(columns={"variant": "method"}), protocol).rename_axis("variant")
    table.columns = [f"{score} {stat}" for score, stat in table.columns]
    meta = rows.groupby("variant")[["family", "base"]].first()
    table = meta.join(table)
    wide = rows.pivot_table(index=["repeat", "fold"], columns="variant", values=protocol.primary)
    gaps, ses = {}, {}
    for name in wide.columns:
        base = meta.loc[name, "base"]
        if base in wide.columns and base != name:
            d = wide[name] - wide[base]
            gaps[name], ses[name] = float(d.mean()), evaluate.corrected_se(d, protocol)
        else:
            gaps[name], ses[name] = 0.0, 0.0
    table["gap to base"] = pd.Series(gaps)
    table["se of gap"] = pd.Series(ses)
    return table.sort_values(["family", f"{protocol.primary} mean"], ascending=[True, False])


def promoted(rows: pd.DataFrame, protocol: Protocol = SCREEN) -> pd.DataFrame:
    """protocol.md's promotion rule: per family, the best variant, kept
    only if it beats its base by more than the corrected error of the
    paired difference. These are the ones worth the full protocol."""
    table = summarize(rows, protocol)
    best = table.loc[table.groupby("family")[f"{protocol.primary} mean"].idxmax()]
    return best[best["gap to base"] > best["se of gap"]]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--family", action="append", choices=sorted(FAMILIES), help="run this family (repeatable)")
    ap.add_argument("--variant", action="append", help="run this variant (repeatable)")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--full", action="store_true", help="the full protocol instead of the screen")
    ap.add_argument("--rerun", action="store_true")
    args = ap.parse_args(argv)
    names = []
    for f in args.family or ([] if not args.all else sorted(FAMILIES)):
        names += list(FAMILIES[f])
    names += args.variant or []
    if not names:
        ap.error("name a --family, a --variant, or --all")
    protocol = PROTOCOL if args.full else SCREEN
    X, y = evaluate.training_xy(protocol)
    rows = run_variants(names, X, y, protocol, rerun=args.rerun)
    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.max_colwidth", 60):
        print()
        print(summarize(rows, protocol).round(3).to_string())
        print("\npromoted (beat their base by more than the error of the gap):")
        print(promoted(rows, protocol).round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1:])
