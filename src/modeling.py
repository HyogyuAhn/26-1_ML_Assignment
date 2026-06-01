import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from evaluate import regression_metrics, score_row
from utils import RANDOM_STATE

def make_candidate_models():
    return {
        "Mean Baseline": DummyRegressor(strategy="mean"),
        "Linear Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "KNN Regressor": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", KNeighborsRegressor(n_neighbors=7)),
            ]
        ),
        "Decision Tree": DecisionTreeRegressor(max_depth=6, min_samples_leaf=20, random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "AdaBoost": AdaBoostRegressor(
            estimator=DecisionTreeRegressor(max_depth=2, min_samples_leaf=10, random_state=RANDOM_STATE),
            n_estimators=100,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
        ),
    }


def fit_base_models(X_train, y_train, X_val, y_val, feature_set, target):
    rows = []
    fitted = {}
    for name, model in make_candidate_models().items():
        model.fit(X_train, y_train)
        rows.append(score_row(model, X_val, y_val, name, "validation", feature_set, target))
        fitted[name] = model
    return pd.DataFrame(rows), fitted


def _group_cv(groups_train):
    subject_count = len(pd.Series(groups_train).unique())
    n_splits = min(3, subject_count)
    if n_splits < 3:
        n_splits = 2
    return GroupKFold(n_splits=n_splits)


def tune_models(X_train, y_train, groups_train, X_val, y_val, feature_set, target):
    cv = _group_cv(groups_train)
    specs = {
        "Random Forest": (
            RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1),
            {
                "n_estimators": [120, 200],
                "max_depth": [8, 10, None],
                "min_samples_leaf": [1, 3, 5],
                "max_features": ["sqrt", 1.0],
            },
        ),
        "AdaBoost": (
            AdaBoostRegressor(
                estimator=DecisionTreeRegressor(random_state=RANDOM_STATE),
                random_state=RANDOM_STATE,
            ),
            {
                "estimator__max_depth": [1, 2],
                "estimator__min_samples_leaf": [10],
                "n_estimators": [60, 100],
                "learning_rate": [0.1, 0.5],
            },
        ),
    }

    tuning_rows = []
    tuned = {}
    for name, (model, param_grid) in specs.items():
        grid = GridSearchCV(
            model,
            param_grid=param_grid,
            scoring="neg_mean_absolute_error",
            cv=cv,
            n_jobs=1,
            refit=True,
            return_train_score=True,
        )
        grid.fit(X_train, y_train, groups=groups_train)
        display_name = f"Tuned {name}"
        tuned[display_name] = grid.best_estimator_
        val_pred = grid.best_estimator_.predict(X_val)
        val_metrics = regression_metrics(y_val, val_pred)
        tuning_rows.append(
            {
                "model": name,
                "target": target,
                "feature_set": feature_set,
                "best_parameters": grid.best_params_,
                "cv_mean_mae": -grid.best_score_,
                "validation_mae": val_metrics["mae"],
                "validation_rmse": val_metrics["rmse"],
                "validation_r2": val_metrics["r2"],
            }
        )

    return pd.DataFrame(tuning_rows), tuned


def select_best_model(validation_results, fitted_models):
    best_row = validation_results.sort_values(["mae", "rmse", "r2"], ascending=[True, True, False]).iloc[0]
    return best_row["model"], fitted_models[best_row["model"]], best_row


def refit_selected_model(model, X_train, X_val, y_train, y_val):
    X_train_val = pd.concat([X_train, X_val])
    y_train_val = pd.concat([y_train, y_val])
    new_model = clone(model)
    new_model.fit(X_train_val, y_train_val)
    return new_model


def make_prediction_table(model, X_test, y_test, meta_test, target):
    y_pred = model.predict(X_test)
    table = meta_test.copy()
    table["target"] = target
    table["y_true"] = y_test.to_numpy()
    table["y_pred"] = y_pred
    table["residual"] = table["y_true"] - table["y_pred"]
    table["abs_error"] = np.abs(table["residual"])
    return table.sort_values("abs_error", ascending=False)


def summarize_error_by_subject(predictions):
    rows = []
    for subject, part in predictions.groupby("subject#"):
        metrics = regression_metrics(part["y_true"], part["y_pred"])
        rows.append(
            {
                "subject#": int(subject),
                "rows": len(part),
                "true_mean": part["y_true"].mean(),
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
            }
        )
    return pd.DataFrame(rows).sort_values("mae", ascending=False)


def summarize_error_by_severity(predictions):
    table = predictions.copy()
    table["severity_bin"] = pd.qcut(
        table["y_true"],
        q=3,
        labels=["low", "medium", "high"],
        duplicates="drop",
    )
    rows = []
    for name, part in table.groupby("severity_bin", observed=False):
        metrics = regression_metrics(part["y_true"], part["y_pred"])
        rows.append(
            {
                "severity_bin": name,
                "rows": len(part),
                "true_min": part["y_true"].min(),
                "true_max": part["y_true"].max(),
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
            }
        )
    return pd.DataFrame(rows)


def summarize_error_by_time(predictions):
    table = predictions.copy()
    table["test_time_bin"] = pd.qcut(
        table["test_time"],
        q=3,
        labels=["early", "middle", "late"],
        duplicates="drop",
    )
    rows = []
    for name, part in table.groupby("test_time_bin", observed=False):
        metrics = regression_metrics(part["y_true"], part["y_pred"])
        rows.append(
            {
                "test_time_bin": name,
                "rows": len(part),
                "test_time_min": part["test_time"].min(),
                "test_time_max": part["test_time"].max(),
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
            }
        )
    return pd.DataFrame(rows)
