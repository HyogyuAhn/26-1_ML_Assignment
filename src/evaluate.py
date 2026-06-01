import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def _safe_pearson(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _safe_spearman(y_true, y_pred):
    true_rank = pd.Series(y_true).rank()
    pred_rank = pd.Series(y_pred).rank()
    if true_rank.std() == 0 or pred_rank.std() == 0:
        return np.nan
    return float(true_rank.corr(pred_rank))


def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mse)),
        "r2": r2_score(y_true, y_pred),
        "pearson": _safe_pearson(y_true, y_pred),
        "spearman": _safe_spearman(y_true, y_pred),
    }


def score_row(model, X, y, model_name, split_name, feature_set, target):
    y_pred = model.predict(X)
    row = {
        "model": model_name,
        "target": target,
        "feature_set": feature_set,
        "split": split_name,
    }
    row.update(regression_metrics(y, y_pred))
    return row
