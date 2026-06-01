import sys

sys.dont_write_bytecode = True

import pandas as pd

from data import load_raw_data, make_group_split, save_split
from plots import save_eda_figures
from utils import GROUP_COLUMN, PRIMARY_TARGET, VOICE_FEATURES, setup, table_path

print("전처리 시작")

output_dir = setup()
df = load_raw_data()
print(f"- Parkinsons telemonitoring 데이터 {len(df)}개 recording, {df[GROUP_COLUMN].nunique()}명")

(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    groups_train,
    groups_val,
    groups_test,
    meta_train,
    meta_val,
    meta_test,
) = make_group_split(df, target=PRIMARY_TARGET, feature_set="voice_only")

save_split(
    output_dir,
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    groups_train,
    groups_val,
    groups_test,
    meta_train,
    meta_val,
    meta_test,
    target=PRIMARY_TARGET,
)

summary = pd.DataFrame(
    [
        {
            "rows": len(df),
            "columns": df.shape[1],
            "subjects": df[GROUP_COLUMN].nunique(),
            "target": PRIMARY_TARGET,
            "feature_set": "voice_only",
            "features": len(VOICE_FEATURES),
            "missing_values": int(df.isna().sum().sum()),
            "duplicate_rows": int(df.duplicated().sum()),
            "total_updrs_mean": df[PRIMARY_TARGET].mean(),
            "total_updrs_std": df[PRIMARY_TARGET].std(),
            "total_updrs_min": df[PRIMARY_TARGET].min(),
            "total_updrs_max": df[PRIMARY_TARGET].max(),
            "train_subjects": groups_train.nunique(),
            "validation_subjects": groups_val.nunique(),
            "test_subjects": groups_test.nunique(),
            "train_rows": len(X_train),
            "validation_rows": len(X_val),
            "test_rows": len(X_test),
        }
    ]
)
summary.to_csv(table_path("eda", "dataset_summary.csv"), index=False)

split_rows = []
for split_name, X_part, y_part, groups_part in [
    ("train", X_train, y_train, groups_train),
    ("validation", X_val, y_val, groups_val),
    ("test", X_test, y_test, groups_test),
]:
    split_rows.append(
        {
            "split": split_name,
            "rows": len(X_part),
            "subjects": groups_part.nunique(),
            "target_mean": y_part.mean(),
            "target_std": y_part.std(),
            "target_min": y_part.min(),
            "target_max": y_part.max(),
        }
    )
pd.DataFrame(split_rows).to_csv(table_path("eda", "split_summary.csv"), index=False)

save_eda_figures(df)

print("전처리 종료")
print(f"- Feature 수: {len(VOICE_FEATURES)}")
print(f"- Train/Validation/Test row 수: {len(X_train)}/{len(X_val)}/{len(X_test)}")
print(f"- Train/Validation/Test subject 수: {groups_train.nunique()}/{groups_val.nunique()}/{groups_test.nunique()}")
