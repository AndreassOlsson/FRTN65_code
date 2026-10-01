"""The data contract: which columns exist, what kind each one is, and
what values they may take.

Every later stage reads columns through `SPEC` rather than naming them,
and every frame enters the package through `load_training` or
`load_test`, which validate against the pandera schema. A file that
breaks the contract fails loudly on load instead of quietly skewing a
model three stages later.

Ranges come from Table 1 of the lab spec (`instructions/lab1.pdf`,
taken from the Spotify audio-features documentation). One deviation
from that table: the files call the duration column `duration`, not
`duration_ms`. The unit is still milliseconds.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pandera.pandas as pa

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
TRAINING_FILE = "training_data.csv"
TEST_FILE = "songs_to_classify.csv"


@dataclass(frozen=True)
class FeatureSpec:
    """Names every feature column once, as numeric or categorical.

    Categorical here means the integer codes name a class, and the
    distance between two codes carries no meaning: key 11 (B) is not
    "more" than key 0 (C), and pitch classes wrap around anyway. Mode is
    binary (1 major, 0 minor). time_signature is the arguable one: it is
    a count of beats per bar, so it could be read as a number, but the
    handful of values it takes (1, 3, 4, 5) are distinct meters rather
    than points on a scale, and 1 is what Spotify reports when it could
    not settle a meter. So it is typed categorical; the exploration
    finding argues it out.

    `classes` maps each label value to the word figures and tables use
    for it, negative class first and positive last; the positive class
    is the one shares and AUCs are measured towards.
    """

    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    label: str
    classes: Mapping[object, str]

    @property
    def features(self) -> tuple[str, ...]:
        return self.numeric + self.categorical

    @property
    def positive(self):
        return list(self.classes)[-1]

    def kind(self, column: str) -> str:
        if column in self.numeric:
            return "numeric"
        if column in self.categorical:
            return "categorical"
        if column == self.label:
            return "label"
        raise KeyError(f"{column!r} is not in the spec")


LABELS = {0: "dislike", 1: "like"}

SPEC = FeatureSpec(
    numeric=(
        "acousticness",
        "danceability",
        "duration",
        "energy",
        "instrumentalness",
        "liveness",
        "loudness",
        "speechiness",
        "tempo",
        "valence",
    ),
    categorical=("key", "mode", "time_signature"),
    label="label",
    classes=LABELS,
)


def _unit() -> pa.Column:
    """A Spotify confidence or perceptual measure, from 0.0 to 1.0."""
    return pa.Column(float, pa.Check.in_range(0.0, 1.0), nullable=False)


_COLUMNS = {
    "acousticness": _unit(),
    "danceability": _unit(),
    # milliseconds; positive is all the spec promises
    "duration": pa.Column(int, pa.Check.gt(0), nullable=False),
    "energy": _unit(),
    "instrumentalness": _unit(),
    # pitch class notation, 0 = C up to 11 = B
    "key": pa.Column(int, pa.Check.isin(range(12)), nullable=False),
    "liveness": _unit(),
    # dB; the spec gives -60 to 0 as the typical range
    "loudness": pa.Column(float, pa.Check.in_range(-60.0, 0.0), nullable=False),
    # 1 major, 0 minor (the spec says "string"; the files hold 0/1)
    "mode": pa.Column(int, pa.Check.isin([0, 1]), nullable=False),
    "speechiness": _unit(),
    # beats per minute; positive, no upper bound in the spec
    "tempo": pa.Column(float, pa.Check.gt(0.0), nullable=False),
    # beats per bar; Spotify documents 3 to 7, and uses 1 when unsure
    "time_signature": pa.Column(int, pa.Check.isin(range(1, 8)), nullable=False),
    "valence": _unit(),
}


def _same_label_for_same_song(df: pd.DataFrame) -> bool:
    """Rows with identical features must carry one label, or the file
    contradicts itself."""
    return bool((df.groupby(list(SPEC.features))[SPEC.label].nunique() <= 1).all())


def schema(labelled: bool) -> pa.DataFrameSchema:
    columns = dict(_COLUMNS)
    checks = []
    if labelled:
        columns[SPEC.label] = pa.Column(int, pa.Check.isin(list(LABELS)), nullable=False)
        checks.append(pa.Check(_same_label_for_same_song, error="duplicate songs disagree on label"))
    return pa.DataFrameSchema(columns, checks=checks, strict=True, coerce=False)


TRAINING_SCHEMA = schema(labelled=True)
TEST_SCHEMA = schema(labelled=False)


def load_training(path: Path | str | None = None) -> pd.DataFrame:
    """The 750 labelled songs, validated. The only file model selection
    may use."""
    return TRAINING_SCHEMA.validate(pd.read_csv(path or DATA_DIR / TRAINING_FILE))


def load_test(path: Path | str | None = None) -> pd.DataFrame:
    """The 200 songs to classify, validated. Touched once, by the chosen
    model."""
    return TEST_SCHEMA.validate(pd.read_csv(path or DATA_DIR / TEST_FILE))
