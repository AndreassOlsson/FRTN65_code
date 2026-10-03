"""The contract against the real files. Needs `make data` first."""

import pandas as pd
import pandera.errors
import pytest

from songtaste.data import (
    DATA_DIR,
    SPEC,
    TEST_FILE,
    TRAINING_FILE,
    TRAINING_SCHEMA,
    load_test,
    load_training,
)


@pytest.fixture(scope="module")
def raw_training():
    path = DATA_DIR / TRAINING_FILE
    if not path.exists():
        pytest.fail(f"{path} missing: run `make data` first")
    return pd.read_csv(path)


def test_spec_covers_every_non_label_column(raw_training):
    features = set(raw_training.columns) - {SPEC.label}
    assert set(SPEC.features) == features
    assert SPEC.label in raw_training.columns


def test_spec_types_each_column_once():
    assert len(SPEC.features) == len(set(SPEC.features)) == 13
    assert not set(SPEC.numeric) & set(SPEC.categorical)
    assert SPEC.label not in SPEC.features


def test_test_file_has_the_features_and_no_label():
    raw = pd.read_csv(DATA_DIR / TEST_FILE)
    assert set(raw.columns) == set(SPEC.features)


def test_training_validates():
    df = load_training()
    assert df.shape == (750, 14)


def test_test_validates():
    df = load_test()
    assert df.shape == (200, 13)


@pytest.mark.parametrize(
    "column, value",
    [("acousticness", 1.5), ("mode", 2), ("key", 12), ("loudness", 3.0), ("label", 2)],
)
def test_schema_rejects_out_of_range(raw_training, column, value):
    bad = raw_training.copy()
    bad.loc[0, column] = value
    with pytest.raises(pandera.errors.SchemaError):
        TRAINING_SCHEMA.validate(bad)


def test_schema_rejects_contradicting_duplicates(raw_training):
    row = raw_training.iloc[[0]].assign(label=1 - raw_training.label.iloc[0])
    with pytest.raises(pandera.errors.SchemaError):
        TRAINING_SCHEMA.validate(pd.concat([raw_training, row]))


def test_schema_rejects_unknown_column(raw_training):
    # pandera collects several column problems into SchemaErrors
    with pytest.raises((pandera.errors.SchemaError, pandera.errors.SchemaErrors)):
        TRAINING_SCHEMA.validate(raw_training.assign(extra=0))
