"""The contract against the real file. Needs `make data` first."""

import pandas as pd
import pandera.errors
import pytest

from creditg.data import DATA_DIR, FILE, LEVELS, SCHEMA, SPEC, load


@pytest.fixture(scope="module")
def raw():
    path = DATA_DIR / FILE
    if not path.exists():
        pytest.fail(f"{path} missing: run `make data` first")
    return pd.read_csv(path)


def test_spec_covers_every_column(raw):
    assert set(SPEC.features) | {SPEC.label} == set(raw.columns)
    assert not set(SPEC.numeric) & set(SPEC.categorical)


def test_loads_and_validates():
    df = load()
    assert df.shape == (1000, 21)
    assert df[SPEC.label].value_counts().to_dict() == {"good": 700, "bad": 300}


def test_every_level_occurs(raw):
    for column, levels in LEVELS.items():
        assert set(raw[column]) == set(levels), column


def test_unknown_level_fails_loudly(raw):
    broken = raw.copy()
    broken.loc[0, "housing"] = "boat"
    with pytest.raises(pandera.errors.SchemaError):
        SCHEMA.validate(broken)
