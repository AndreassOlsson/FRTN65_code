"""Compare gradient boosting settings for song preference classification."""
import pandas as pd
from sklearn.model_selection import ParameterGrid, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from .preprocessing import make_preprocessor


def build_model(features, params):
    """Create a pipeline with encoded categories and boosted trees."""
    preprocessing = make_preprocessor(features, scale=False, drop=None)
    model = GradientBoostingClassifier(random_state=42, **params)
    return Pipeline([("preprocess", preprocessing), ("model", model)])


def run(X_train, y_train, feature_sets, cv, jobs=-1):
    """Choose a feature set and boosting parameters by five-fold accuracy."""
    # Tree count and learning rate jointly control the size of each update.
    parameter_grid = {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.05, 0.1],
        "max_depth": [1, 2],
    }
    rows = []

    # Use the same folds for every feature-set and parameter combination.
    for set_name, features in feature_sets.items():
        for params in ParameterGrid(parameter_grid):
            model = build_model(features, params)
            scores = cross_val_score(
                model, X_train[features], y_train,
                cv=cv, scoring="accuracy", n_jobs=jobs, error_score="raise",
            )
            row = {"feature_set": set_name, "params": params,
                   "cv_accuracy": scores.mean(), "cv_std": scores.std()}
            row.update({f"fold_{i+1}": score for i, score in enumerate(scores)})
            rows.append(row)
        print(f"GradientBoosting: finished {set_name}", flush=True)

    # Resolve equal scores by the order in which candidates were evaluated.
    table = pd.DataFrame(rows).sort_values(
        "cv_accuracy", ascending=False, kind="stable"
    ).reset_index(drop=True)
    best = table.iloc[0]
    features = list(feature_sets[best["feature_set"]])
    best_model = build_model(features, best["params"])

    # Cross-validation does not leave a fitted model; fit the selected setup.
    best_model.fit(X_train[features], y_train)
    return {"name": "GradientBoosting", "model": best_model, "features": features,
            "feature_set": best["feature_set"], "params": best["params"],
            "cv_accuracy": float(best["cv_accuracy"]), "cv_std": float(best["cv_std"]),
            "trials": table}

