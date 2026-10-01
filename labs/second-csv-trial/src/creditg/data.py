"""The data contract for OpenML's credit-g: which columns exist, what
kind each one is, and what values they may take.

1000 loan applicants from a German bank, 20 attributes, each labelled
`good` or `bad` credit risk (70/30). `make data` writes OpenML dataset
31 (pinned by id, so by version) to `data/credit-g.csv`, which is
gitignored. The categorical columns hold OpenML's string levels
(`checking_status` is `<0`, `0<=X<200`, ...), not integer codes, and
the label is a string too; the contract keeps both as they come.

The spec reuses songtaste's `FeatureSpec`; the schema is pandera's, as
in lab 1, because a schema is the part of a contract that is about
this file and nothing else.
"""

from pathlib import Path

import pandas as pd
import pandera.pandas as pa

from songtaste.data import FeatureSpec

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
FILE = "credit-g.csv"
OPENML_ID = 31

LEVELS = {
    "checking_status": ["<0", "0<=X<200", ">=200", "no checking"],
    "credit_history": [
        "no credits/all paid",
        "all paid",
        "existing paid",
        "delayed previously",
        "critical/other existing credit",
    ],
    "purpose": [
        "new car",
        "used car",
        "furniture/equipment",
        "radio/tv",
        "domestic appliance",
        "repairs",
        "education",
        "retraining",
        "business",
        "other",
    ],
    "savings_status": ["<100", "100<=X<500", "500<=X<1000", ">=1000", "no known savings"],
    "employment": ["unemployed", "<1", "1<=X<4", "4<=X<7", ">=7"],
    "personal_status": ["male div/sep", "female div/dep/mar", "male single", "male mar/wid"],
    "other_parties": ["none", "co applicant", "guarantor"],
    "property_magnitude": ["real estate", "life insurance", "car", "no known property"],
    "other_payment_plans": ["bank", "stores", "none"],
    "housing": ["rent", "own", "for free"],
    "job": ["unemp/unskilled non res", "unskilled resident", "skilled", "high qualif/self emp/mgmt"],
    "own_telephone": ["none", "yes"],
    "foreign_worker": ["yes", "no"],
}

# the bank's question is which applicants default, so bad is the
# positive class, last, the one shares and AUCs are measured towards
LABELS = {"good": "good", "bad": "bad"}

# small integer counts and 1-4 bands (installment as a share of income,
# years at the residence) are kept numeric: their order means something
# and a scaled number carries it, where one-hot would throw it away
SPEC = FeatureSpec(
    numeric=(
        "duration",
        "credit_amount",
        "installment_commitment",
        "residence_since",
        "age",
        "existing_credits",
        "num_dependents",
    ),
    categorical=tuple(LEVELS),
    label="class",
    classes=LABELS,
)


def _count(lo: int, hi: int) -> pa.Column:
    return pa.Column(int, pa.Check.in_range(lo, hi), nullable=False)


SCHEMA = pa.DataFrameSchema(
    {
        # months
        "duration": _count(1, 120),
        # Deutsche Mark
        "credit_amount": pa.Column(int, pa.Check.gt(0), nullable=False),
        "installment_commitment": _count(1, 4),
        "residence_since": _count(1, 4),
        "age": _count(18, 100),
        "existing_credits": _count(1, 4),
        "num_dependents": _count(1, 2),
        **{c: pa.Column(str, pa.Check.isin(levels), nullable=False) for c, levels in LEVELS.items()},
        "class": pa.Column(str, pa.Check.isin(list(LABELS)), nullable=False),
    },
    strict=True,
    coerce=False,
)


def load(path: Path | str | None = None) -> pd.DataFrame:
    """The 1000 applicants, validated."""
    return SCHEMA.validate(pd.read_csv(path or DATA_DIR / FILE))


def fetch(path: Path | str | None = None) -> Path:
    """OpenML dataset 31 as a plain CSV, the columns in OpenML's order."""
    from sklearn.datasets import fetch_openml

    path = Path(path or DATA_DIR / FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    fetch_openml(data_id=OPENML_ID, as_frame=True).frame.to_csv(path, index=False, lineterminator="\n")
    return path


if __name__ == "__main__":
    print("wrote", fetch())
