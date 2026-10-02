"""Features: the preprocessor every model's pipeline starts with, built
from the spec so no column is named twice.

Numeric columns are standardised, categorical ones one-hot encoded.
That is all on purpose. The exploration finding (results/01-exploration.md)
notes that duration and the spiky features would suit a robust or
quantile scaling for distance-based models; that is a variant a
registry entry can ask for through `numeric`, measured as its own
method, not a change to every baseline.
"""

import numpy as np
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from songtaste.data import FeatureSpec


def build_preprocessor(
    spec: FeatureSpec,
    numeric: TransformerMixin | str | None = None,
    categorical: TransformerMixin | str | None = None,
    log_columns: tuple[str, ...] = (),
) -> ColumnTransformer:
    """StandardScaler on the numeric columns, OneHotEncoder on the
    categorical ones; a category unseen in training encodes as all
    zeros instead of failing. Dense output, since some estimators
    (kNN with some metrics, QDA) do not take sparse input.

    The three optional arguments are how a variant (`songtaste.variants`)
    asks for something else without a second builder: another numeric
    transformer (or a Pipeline of them, or "passthrough"), another
    encoder for the categoricals, and numeric columns to take log1p of
    before that transformer, for the spiky ones the exploration found.
    With the defaults this is exactly the baselines' preprocessor.
    """
    numeric = StandardScaler() if numeric is None else numeric
    categorical = (
        OneHotEncoder(handle_unknown="ignore", sparse_output=False) if categorical is None else categorical
    )
    plain = [c for c in spec.numeric if c not in log_columns]
    logged = [c for c in spec.numeric if c in log_columns]
    branches = [("numeric", numeric, plain)]
    if logged:
        branches.append(("log", make_pipeline(FunctionTransformer(np.log1p, feature_names_out="one-to-one"), _clone(numeric)), logged))
    branches.append(("categorical", categorical, list(spec.categorical)))
    return ColumnTransformer(branches, remainder="drop", verbose_feature_names_out=False)


def _clone(transformer):
    if isinstance(transformer, str):
        return transformer
    from sklearn.base import clone

    return clone(transformer)
