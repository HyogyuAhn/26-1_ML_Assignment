import atexit
import logging
import os
import shutil
import tempfile

logging.getLogger("matplotlib").setLevel(logging.ERROR)

if "MPLCONFIGDIR" not in os.environ:
    _mpl_dir = tempfile.mkdtemp(prefix="parkinsons_mpl_", dir=tempfile.gettempdir())
    os.environ["MPLCONFIGDIR"] = _mpl_dir
    atexit.register(lambda: shutil.rmtree(_mpl_dir, ignore_errors=True))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance

from utils import PRIMARY_TARGET, RANDOM_STATE, SECONDARY_TARGET, VOICE_FEATURES, figure_path, table_path


def save_eda_figures(df):
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    sns.histplot(df[PRIMARY_TARGET], bins=30, kde=True, color="#4C78A8", ax=ax)
    ax.set_title("total_UPDRS distribution")
    ax.set_xlabel(PRIMARY_TARGET)
    ax.set_ylabel("Recordings")
    fig.tight_layout()
    fig.savefig(figure_path("eda", "target_distribution_total_updrs.png"), dpi=170)
    plt.close(fig)

    subject_counts = df["subject#"].value_counts().sort_index()
    subject_table = subject_counts.rename_axis("subject#").reset_index(name="recordings")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(subject_table["subject#"].astype(str), subject_table["recordings"], color="#59A14F")
    ax.set_title("Recordings per subject")
    ax.set_xlabel("subject#")
    ax.set_ylabel("Recordings")
    ax.tick_params(axis="x", labelrotation=90)
    fig.tight_layout()
    fig.savefig(figure_path("eda", "recordings_per_subject.png"), dpi=170)
    plt.close(fig)

    corr_cols = VOICE_FEATURES + ["age", "sex", "test_time", SECONDARY_TARGET, PRIMARY_TARGET]
    corr = df[corr_cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, cmap="vlag", center=0, linewidths=0.2, ax=ax)
    ax.set_title("Voice feature and target correlation")
    fig.tight_layout()
    fig.savefig(figure_path("eda", "feature_correlation_heatmap.png"), dpi=170)
    plt.close(fig)

    selected = ["Jitter(%)", "Shimmer", "NHR", "HNR", "RPDE", "DFA", "PPE"]
    fig, axes = plt.subplots(3, 3, figsize=(10, 8))
    axes = axes.flatten()
    for idx, feature in enumerate(selected):
        sns.histplot(df[feature], bins=28, color="#F28E2B", ax=axes[idx])
        axes[idx].set_title(feature)
        axes[idx].set_xlabel("")
        axes[idx].set_ylabel("")
    for idx in range(len(selected), len(axes)):
        axes[idx].axis("off")
    fig.suptitle("Selected voice feature distributions", y=1.01)
    fig.tight_layout()
    fig.savefig(figure_path("eda", "selected_voice_features_distribution.png"), dpi=170)
    plt.close(fig)


def save_model_comparison_plot(validation_results, target):
    ordered = validation_results.sort_values("mae", ascending=False)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.barh(ordered["model"], ordered["mae"], color="#4C78A8")
    ax.set_xlabel("Validation MAE")
    ax.set_title(f"Model comparison for {target}")
    fig.tight_layout()
    fig.savefig(figure_path("modeling", f"model_comparison_{target.lower()}.png"), dpi=170)
    plt.close(fig)


def save_prediction_plots(predictions, target):
    fig, ax = plt.subplots(figsize=(5.8, 5.4))
    ax.scatter(predictions["y_true"], predictions["y_pred"], alpha=0.7, color="#4C78A8")
    lo = min(predictions["y_true"].min(), predictions["y_pred"].min())
    hi = max(predictions["y_true"].max(), predictions["y_pred"].max())
    ax.plot([lo, hi], [lo, hi], color="#E15759", linewidth=1.4)
    ax.set_title(f"Predicted vs actual {target}")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    fig.tight_layout()
    fig.savefig(figure_path("modeling", f"predicted_vs_actual_{target.lower()}.png"), dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.scatter(predictions["y_pred"], predictions["residual"], alpha=0.7, color="#59A14F")
    ax.axhline(0, color="#E15759", linewidth=1.2)
    ax.set_title(f"Residual plot for {target}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual, true - predicted")
    fig.tight_layout()
    fig.savefig(figure_path("modeling", f"residual_plot_{target.lower()}.png"), dpi=170)
    plt.close(fig)


def save_error_analysis_plots(error_by_subject, target):
    ordered = error_by_subject.sort_values("mae", ascending=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.barh(ordered["subject#"].astype(str), ordered["mae"], color="#B07AA1")
    ax.set_title(f"Error by test subject for {target}")
    ax.set_xlabel("MAE")
    ax.set_ylabel("subject#")
    fig.tight_layout()
    fig.savefig(figure_path("analysis", f"error_by_subject_{target.lower()}.png"), dpi=170)
    plt.close(fig)


def _unwrap_estimator(model):
    if hasattr(model, "named_steps"):
        return model.named_steps.get("model") or model
    return model


def save_feature_importance(model, X_test, y_test, feature_names, target):
    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="neg_mean_absolute_error",
        n_repeats=5,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    table = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
            "method": "permutation_importance_neg_mae",
        }
    ).sort_values("importance_mean", ascending=False)

    estimator = _unwrap_estimator(model)
    if hasattr(estimator, "feature_importances_") and len(estimator.feature_importances_) == len(feature_names):
        table["tree_importance"] = estimator.feature_importances_

    table.to_csv(table_path("analysis", f"permutation_importance_{target.lower()}.csv"), index=False)

    top = table.head(12).sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.barh(top["feature"], top["importance_mean"], color="#59A14F")
    ax.set_xlabel("Permutation importance, MAE increase")
    ax.set_title(f"Feature importance for {target}")
    fig.tight_layout()
    fig.savefig(figure_path("analysis", f"feature_importance_{target.lower()}.png"), dpi=170)
    plt.close(fig)
    return table
