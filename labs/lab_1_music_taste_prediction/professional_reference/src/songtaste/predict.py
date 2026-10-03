"""The one read of the test songs: refit the chosen method and write the
submission string.

The method's name is read from the decision in `results/03-sweep.md`,
so the choice is made once, there, and not again here. Its
hyperparameters are chosen the way `protocol.md` allows for the final
fit: the same `GridSearchCV` over the same grid, on the whole
deduplicated training set (736 songs; the protocol drops the 14 exact
duplicates before anything sees the data, and the final fit is no
exception). The refit then predicts the 200 songs of
`songs_to_classify.csv`, in the file's row order, as one line of 200
characters, each 0 (dislike) or 1 (like).

`python -m songtaste.predict` (`make predict`) writes
`results/submission-<YYYY-MM-DD>.txt` and prints the string.
"""

import re
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GridSearchCV

from labs.lab_1_music_taste_prediction.professional_reference.src.songtaste import (
    evaluate,
)
from labs.lab_1_music_taste_prediction.professional_reference.src.songtaste.data import (
    SPEC,
    load_test,
)
from labs.lab_1_music_taste_prediction.professional_reference.src.songtaste.evaluate import (
    PROTOCOL,
    RESULTS,
    Protocol,
)

DECISION = RESULTS / "03-sweep.md"
N_TEST = 200
_CHOSEN = re.compile(r"The chosen method is [^(]*\(`(\w+)`\)")


def chosen_method(path: Path = DECISION) -> str:
    """The registry name the sweep's decision names, e.g. `rf`."""
    found = _CHOSEN.findall(Path(path).read_text())
    if len(set(found)) != 1:
        raise ValueError(
            f"{path} should name exactly one chosen method, found {sorted(set(found))}"
        )
    return found[0]


def fit_final(name: str, protocol: Protocol = PROTOCOL):
    """The method refit on every training song a method may see, its
    hyperparameters picked by the protocol's inner search."""
    X, y = evaluate.training_xy(protocol)
    return evaluate.estimator(name, protocol).fit(X, y)


def prediction_string(model, X: pd.DataFrame) -> str:
    """One character per row of X, in X's order: 1 like, 0 dislike."""
    return "".join(str(int(p)) for p in model.predict(X[list(SPEC.features)]))


def write_submission(
    string: str, day: date | None = None, results: Path = RESULTS
) -> Path:
    """`results/submission-<day>.txt`, the string and a newline."""
    if len(string) != N_TEST or set(string) - {"0", "1"}:
        raise ValueError(
            f"a submission is {N_TEST} characters of 0 and 1, got {len(string)}"
        )
    path = results / f"submission-{(day or date.today()).isoformat()}.txt"
    path.write_text(string + "\n")
    return path


def main(argv=None) -> None:
    name = chosen_method()
    model = fit_final(name)
    string = prediction_string(model, load_test())
    path = write_submission(string)
    if isinstance(model, GridSearchCV):
        print(
            f"{name}, refit with {evaluate._chosen(model)} (inner accuracy {model.best_score_:.3f})"
        )
    print(
        f"{string.count('1')} liked of {len(string)}, written to {path.relative_to(RESULTS.parent)}\n"
    )
    print(string)


if __name__ == "__main__":
    main(sys.argv[1:])
