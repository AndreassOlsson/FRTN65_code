"""Prepare numerical and categorical song features for classification."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL = ["key", "mode", "time_signature"]


def make_preprocessor(features, scale=True, drop=None):
    """Build column-specific transformations for the selected raw features.

    Fitting happens inside each model's Pipeline, so cross-validation learns
    imputation, scaling, and category mappings from training folds only.
    """
    numeric = [name for name in features if name not in CATEGORICAL]
    categorical = [name for name in features if name in CATEGORICAL]
    return ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler() if scale else "passthrough"),
        ]), numeric),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop=drop)),
        ]), categorical),
    ])
