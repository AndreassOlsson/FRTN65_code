"""Features: the preprocessor every model's pipeline starts with, built
from the spec so no column is named twice.

Numeric columns are standardised, categorical ones one-hot encoded.
That is all on purpose. The exploration finding (results/01-exploration.md)
notes that duration and the spiky features would suit a robust or
quantile scaling for distance-based models; that is a variant a
registry entry can ask for through `numeric`, measured as its own
method, not a change to every baseline.
"""

from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from labs.lab_1_music_taste_prediction.professional_reference.src.songtaste.data import (
    FeatureSpec,
)


def build_preprocessor(
    spec: FeatureSpec, numeric: TransformerMixin | None = None
) -> ColumnTransformer:
    """StandardScaler on the numeric columns, OneHotEncoder on the
    categorical ones; a category unseen in training encodes as all
    zeros instead of failing. Dense output, since some estimators
    (kNN with some metrics, QDA) do not take sparse input."""
    return ColumnTransformer(
        [
            (
                "numeric",
                numeric if numeric is not None else StandardScaler(),
                list(spec.numeric),
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(spec.categorical),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
