import os
import sys

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
sys.dont_write_bytecode = True

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from data import load_raw_data, load_split, make_group_split, make_xy_groups
from evaluate import regression_metrics, score_row
from modeling import (
    fit_base_models,
    make_prediction_table,
    refit_selected_model,
    select_best_model,
    summarize_error_by_severity,
    summarize_error_by_subject,
    summarize_error_by_time,
    tune_models,
)
from plots import (
    save_error_analysis_plots,
    save_feature_importance,
    save_model_comparison_plot,
    save_prediction_plots,
)
from utils import (
    PRIMARY_TARGET,
    RANDOM_STATE,
    setup,
    table_path,
)

print("모델 실험 시작")

output_dir = setup()
X_train, X_val, X_test, y_train, y_val, y_test, groups_train, groups_val, groups_test, meta_train, meta_val, meta_test = load_split(output_dir)
print(f"subject 수: train {groups_train.nunique()}, validation {groups_val.nunique()}, test {groups_test.nunique()}")

feature_set = "voice_only"
target = PRIMARY_TARGET

base_results, base_models = fit_base_models(X_train, y_train, X_val, y_val, feature_set, target)

tuning_results, tuned_models = tune_models(X_train, y_train, groups_train, X_val, y_val, feature_set, target)

tuned_validation_rows = []
for name, model in tuned_models.items():
    tuned_validation_rows.append(score_row(model, X_val, y_val, name, "validation", feature_set, target))

validation_results = pd.concat([base_results, pd.DataFrame(tuned_validation_rows)], ignore_index=True)
validation_results = validation_results.sort_values(["mae", "rmse", "r2"], ascending=[True, True, False])
validation_results.to_csv(table_path("modeling", "model_comparison_total_updrs.csv"), index=False)

tuning_results = tuning_results.sort_values("validation_mae")
tuning_results.to_csv(table_path("modeling", "best_params_total_updrs.csv"), index=False)
save_model_comparison_plot(validation_results, target)

fitted_models = {**base_models, **tuned_models}
candidate_results = validation_results[validation_results["model"] != "Mean Baseline"].copy()
best_name, best_model, best_row = select_best_model(candidate_results, fitted_models)
baseline_row = validation_results[validation_results["model"] == "Mean Baseline"].iloc[0]
print(f"- 평균 baseline validation MAE: {baseline_row['mae']:.4f}")
print(f"- baseline이 아닌 모델 중 validation MAE 기준 제일 괜찮은 모델: {best_name}")

best_model = refit_selected_model(best_model, X_train, X_val, y_train, y_val)

test_row = score_row(best_model, X_test, y_test, best_name, "test", feature_set, target)

baseline_test_model = clone(base_models["Mean Baseline"])
baseline_test_model.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
baseline_test_row = score_row(baseline_test_model, X_test, y_test, "Mean Baseline", "test", feature_set, target)
pd.DataFrame([baseline_test_row, test_row]).to_csv(
    table_path("modeling", "test_baseline_comparison_total_updrs.csv"),
    index=False,
)

predictions = make_prediction_table(best_model, X_test, y_test, meta_test, target)

error_by_subject = summarize_error_by_subject(predictions)
error_by_severity = summarize_error_by_severity(predictions)
error_by_time = summarize_error_by_time(predictions)

error_by_subject.to_csv(table_path("analysis", "error_by_subject_total_updrs.csv"), index=False)
error_by_severity.to_csv(table_path("analysis", "error_by_severity_bin_total_updrs.csv"), index=False)
error_by_time.to_csv(table_path("analysis", "error_by_test_time_bin_total_updrs.csv"), index=False)

save_prediction_plots(predictions, target)
save_error_analysis_plots(error_by_subject, target)
save_feature_importance(best_model, X_test, y_test, X_test.columns, target)

df = load_raw_data()
ablation_rows = []
for ablation_feature_set in ["voice_only", "voice_plus_context"]:
    (
        Xa_train,
        Xa_val,
        Xa_test,
        ya_train,
        ya_val,
        ya_test,
        ga_train,
        ga_val,
        ga_test,
        ma_train,
        ma_val,
        ma_test,
    ) = make_group_split(df, target=PRIMARY_TARGET, feature_set=ablation_feature_set)
    ablation_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=3,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    ablation_model.fit(Xa_train, ya_train)
    val_row = score_row(ablation_model, Xa_val, ya_val, "Random Forest fixed", "validation", ablation_feature_set, PRIMARY_TARGET)
    val_row["subjects"] = ga_val.nunique()
    ablation_rows.append(val_row)

    ablation_test_model = clone(ablation_model)
    ablation_test_model.fit(pd.concat([Xa_train, Xa_val]), pd.concat([ya_train, ya_val]))
    test_ablation_row = score_row(
        ablation_test_model,
        Xa_test,
        ya_test,
        "Random Forest fixed",
        "test",
        ablation_feature_set,
        PRIMARY_TARGET,
    )
    test_ablation_row["subjects"] = ga_test.nunique()
    ablation_rows.append(test_ablation_row)
pd.DataFrame(ablation_rows).to_csv(table_path("analysis", "feature_set_ablation.csv"), index=False)

X_all, y_all, groups_all, meta_all = make_xy_groups(df, target=PRIMARY_TARGET, feature_set="voice_only")
X_train_val_row, X_test_row, y_train_val_row, y_test_row = train_test_split(
    X_all,
    y_all,
    test_size=0.15,
    random_state=RANDOM_STATE,
)
X_train_row, X_val_row, y_train_row, y_val_row = train_test_split(
    X_train_val_row,
    y_train_val_row,
    test_size=0.1765,
    random_state=RANDOM_STATE,
)
row_model = clone(best_model)
row_model.fit(pd.concat([X_train_row, X_val_row]), pd.concat([y_train_row, y_val_row]))
row_pred = row_model.predict(X_test_row)
group_pred = best_model.predict(X_test)
leakage_rows = [
    {
        "split_type": "group_by_subject",
        "note": "main_result",
        **regression_metrics(y_test, group_pred),
    },
    {
        "split_type": "random_row_split",
        "note": "diagnostic_only_subject_leakage_possible",
        **regression_metrics(y_test_row, row_pred),
    },
]
pd.DataFrame(leakage_rows).to_csv(table_path("analysis", "group_split_vs_random_split.csv"), index=False)

print("실험 종료")
print(f"- 최종 모델: {best_name}")
print(f"- Validation MAE: {best_row['mae']:.4f}")
print(f"- Test MAE: {test_row['mae']:.4f}")
print(f"- Test RMSE: {test_row['rmse']:.4f}")
print(f"- Test R2: {test_row['r2']:.4f}")
