"""Compare five classifiers and predict song preferences for Lab 1.

Run: python main.py
Outputs: model scores, evaluation metrics and predictions in results/.
"""
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

FOLDER = Path(__file__).resolve().parent
SEED = 42
CATEGORICAL = ["key", "mode", "time_signature"]


def make_pipeline(classifier, numeric, scale, sparse=False):
    """Combine numeric preprocessing and one-hot encoding with a classifier."""
    preparation = ColumnTransformer([
        ("numeric", StandardScaler() if scale else "passthrough", numeric),
        ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=sparse), CATEGORICAL),
    ], sparse_threshold=1.0 if sparse else 0.0)
    return Pipeline([("prepare", preparation), ("model", classifier)])


def main():
    # Read the labels and 13 input features, then check the data.
    output = FOLDER / "results"
    output.mkdir(exist_ok=True)
    raw = pd.read_csv(FOLDER / "training_data.csv")
    features = raw.columns.drop("label").tolist()
    if set(raw.label.unique()) != {0, 1}:
        raise ValueError("Labels must contain both 0 and 1, with no missing values.")
    if not np.isfinite(raw[features].to_numpy()).all():
        raise ValueError("Input features contain missing or nonfinite values.")
    if raw.groupby(features).label.nunique().gt(1).any():
        raise ValueError("Identical feature records have conflicting labels.")

    # Deduplicate before splitting to keep identical records out of both sets.
    data = raw.drop_duplicates(subset=features)
    X, y = data[features], data.label.astype(int)
    numeric = [name for name in features if name not in CATEGORICAL]
    print(f"Original rows: {len(raw)}; unique rows: {len(data)}; features: {len(features)}.", flush=True)
    print("Class counts:", y.value_counts().sort_index().to_dict(), flush=True)

    # Reserve 20% for evaluation. Use the remaining 80% for model selection.
    X_dev, X_hold, y_dev, y_hold = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    # Reuse the same five validation folds for every model.
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X_dev, y_dev))

    # The model__ prefix addresses parameters of the pipeline's model step.
    models = {
        "Logistic regression": (
            make_pipeline(LogisticRegression(max_iter=2000), numeric, scale=True, sparse=True),
            {"model__C": [0.01, 0.1, 1, 10]},
        ),
        "k-NN": (
            make_pipeline(KNeighborsClassifier(), numeric, scale=True),
            {"model__n_neighbors": [3, 5, 9, 15, 25, 41], "model__weights": ["uniform", "distance"]},
        ),
        "RBF SVM": (
            make_pipeline(SVC(kernel="rbf"), numeric, scale=True),
            {"model__C": [0.1, 1, 10, 100], "model__gamma": ["scale", 0.01, 0.1]},
        ),
        "Random forest": (
            make_pipeline(RandomForestClassifier(n_estimators=150, random_state=SEED, n_jobs=1),
                          numeric, scale=False),
            {"model__max_depth": [4, 8, None], "model__min_samples_leaf": [1, 3, 8]},
        ),
        "Gradient boosting": (
            make_pipeline(GradientBoostingClassifier(random_state=SEED), numeric, scale=False),
            {"model__n_estimators": [100, 200], "model__learning_rate": [0.03, 0.1], "model__max_depth": [1, 2]},
        ),
    }

    # Each fold fits preprocessing on its own training portion.
    comparison, searches = [], {}
    for name, (pipeline, grid) in models.items():
        print(f"Comparing {name} ...", flush=True)
        search = GridSearchCV(pipeline, grid, scoring="accuracy", cv=folds,
                              n_jobs=1, error_score="raise")
        search.fit(X_dev, y_dev)
        index = search.best_index_
        comparison.append({
            "model": name,
            "cv_accuracy": float(search.best_score_),
            "cv_std": float(search.cv_results_["std_test_score"][index]),
            "best_parameters": search.best_params_,
        })
        searches[name] = search
        print(f"  Mean five-fold accuracy: {search.best_score_:.2%}; parameters: {search.best_params_}", flush=True)

    # Choose by mean CV accuracy before evaluating the holdout.
    selected = max(comparison, key=lambda row: row["cv_accuracy"])
    selected_model = searches[selected["model"]].best_estimator_
    hold_predictions = selected_model.predict(X_hold)
    scores = (selected_model.predict_proba(X_hold)[:, 1] if hasattr(selected_model, "predict_proba")
              else selected_model.decision_function(X_hold))
    baseline = DummyClassifier(strategy="most_frequent").fit(X_dev, y_dev)
    baseline_cv = cross_val_score(baseline, X_dev, y_dev, cv=folds, scoring="accuracy")
    holdout = {
        "correct": int((hold_predictions == y_hold.to_numpy()).sum()),
        "size": len(y_hold),
        "accuracy": float(accuracy_score(y_hold, hold_predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_hold, hold_predictions)),
        "f1_like": float(f1_score(y_hold, hold_predictions)),
        "roc_auc": float(roc_auc_score(y_hold, scores)),
        "confusion_matrix": confusion_matrix(y_hold, hold_predictions, labels=[0, 1]).tolist(),
        "baseline_accuracy": float(baseline.score(X_hold, y_hold)),
    }
    print(f"Selected: {selected['model']}; holdout accuracy: {holdout['accuracy']:.2%}", flush=True)

    # Refit with the selected parameters on all unique labeled records.
    final_model = clone(selected_model).fit(X, y)
    test = pd.read_csv(FOLDER / "songs_to_classify.csv")
    if set(test.columns) != set(features) or not np.isfinite(test[features].to_numpy()).all():
        raise ValueError("The test file has invalid columns or missing or nonfinite values.")
    predictions = final_model.predict(test[features]).astype(int)
    submission = "".join(str(value) for value in predictions)
    assert len(submission) == len(test) and set(submission) <= {"0", "1"}
    (output / "submission.txt").write_text(submission + "\n", encoding="utf-8")
    pd.DataFrame({"song_row": np.arange(1, len(test) + 1), "label": predictions}).to_csv(
        output / "predictions.csv", index=False)

    # Save the model comparison and final evaluation for the report.
    summary = {
        "models": comparison, "baseline_cv_accuracy": float(baseline_cv.mean()),
        "selected_model": selected["model"], "selected_parameters": selected["best_parameters"],
        "holdout": holdout,
        "prediction_counts": {str(k): int(v) for k, v in pd.Series(predictions).value_counts().sort_index().items()},
    }
    (output / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved predictions for {len(test)} songs: {output / 'submission.txt'}", flush=True)


if __name__ == "__main__":
    main()
