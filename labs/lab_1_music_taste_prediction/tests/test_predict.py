"""The submission: the method the sweep decided, a string of exactly 200
zeros and ones, one per test song in the file's row order. Needs
`make data` first, and `make predict` for the committed-file check."""

from datetime import date

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from songtaste import predict
from songtaste.data import SPEC, load_test
from songtaste.evaluate import RESULTS


class LoudIsDisliked(ClassifierMixin, BaseEstimator):
    """A stand-in model whose answer for a row depends on that row alone,
    so the string's order can be checked against the frame's."""

    def predict(self, X):
        return (X["loudness"] < -8).astype(int).to_numpy()


@pytest.fixture(scope="module")
def test_songs():
    return load_test()


def test_reads_the_decision():
    assert predict.chosen_method() == "rf"


def test_decision_must_be_unambiguous(tmp_path):
    two = tmp_path / "03-sweep.md"
    two.write_text("The chosen method is the random forest (`rf`).\nThe chosen method is LDA (`lda`).\n")
    with pytest.raises(ValueError):
        predict.chosen_method(two)


def test_one_character_per_song_in_row_order(test_songs):
    string = predict.prediction_string(LoudIsDisliked(), test_songs)
    assert len(string) == predict.N_TEST == len(test_songs)
    expected = "".join("1" if v < -8 else "0" for v in test_songs["loudness"])
    assert string == expected
    # reversing the rows reverses the string: position i is row i
    assert predict.prediction_string(LoudIsDisliked(), test_songs.iloc[::-1]) == string[::-1]


def test_only_the_features_reach_the_model(test_songs):
    seen = {}

    class Spy(ClassifierMixin, BaseEstimator):
        def predict(self, X):
            seen["columns"] = list(X.columns)
            return np.zeros(len(X), dtype=int)

    predict.prediction_string(Spy(), test_songs)
    assert seen["columns"] == list(SPEC.features)


def test_write_refuses_a_malformed_string(tmp_path):
    with pytest.raises(ValueError):
        predict.write_submission("01" * 99, results=tmp_path)
    with pytest.raises(ValueError):
        predict.write_submission("2" * 200, results=tmp_path)
    path = predict.write_submission("01" * 100, date(2026, 10, 4), results=tmp_path)
    assert path.name == "submission-2026-10-04.txt"
    assert path.read_text() == "01" * 100 + "\n"


def test_committed_submissions_are_well_formed():
    files = sorted(RESULTS.glob("submission-*.txt"))
    assert files, "no submission yet; run `make predict`"
    for path in files:
        text = path.read_text()
        assert len(text) == 201 and text.endswith("\n"), path.name
        assert set(text[:-1]) <= {"0", "1"}, path.name
