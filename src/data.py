from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from utils import (
    DATA_PATH,
    GROUP_COLUMN,
    KEY_ERROR_COLUMNS,
    PRIMARY_TARGET,
    RANDOM_STATE,
    TARGET_COLUMNS,
    get_feature_columns,
)

def load_raw_data(path=DATA_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"데이터 파일을 찾지 못함: {path}")

    df = pd.read_csv(path)
    df.columns = [str(col).strip() for col in df.columns]

    needed = [GROUP_COLUMN, "age", "sex", "test_time"] + TARGET_COLUMNS
    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼이 없음: {missing}")

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna().copy()
    df[GROUP_COLUMN] = df[GROUP_COLUMN].astype(int)
    df.index.name = "source_index"
    return df


def make_xy_groups(df, target=PRIMARY_TARGET, feature_set="voice_only"):
    if target not in TARGET_COLUMNS:
        raise ValueError(f"target은 {TARGET_COLUMNS} 중 하나여야 함")

    feature_columns = get_feature_columns(feature_set)
    missing_features = [col for col in feature_columns if col not in df.columns]
    if missing_features:
        raise ValueError(f"데이터에 없는 feature가 있음: {missing_features}")

    X = df[feature_columns].copy()
    y = df[target].copy()
    groups = df[GROUP_COLUMN].copy()
    meta_cols = [col for col in KEY_ERROR_COLUMNS + TARGET_COLUMNS if col in df.columns]
    meta = df[meta_cols].copy()
    return X, y, groups, meta


def assert_group_disjoint(groups_train, groups_val, groups_test):
    train_subjects = set(groups_train)
    val_subjects = set(groups_val)
    test_subjects = set(groups_test)

    if train_subjects.intersection(val_subjects):
        raise ValueError("train과 validation에 같은 subject가 섞임")
    if train_subjects.intersection(test_subjects):
        raise ValueError("train과 test에 같은 subject가 섞임")
    if val_subjects.intersection(test_subjects):
        raise ValueError("validation과 test에 같은 subject가 섞임")


def make_group_split(df, target=PRIMARY_TARGET, feature_set="voice_only"):
    X, y, groups, meta = make_xy_groups(df, target=target, feature_set=feature_set)

    first = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=RANDOM_STATE)
    train_val_idx, test_idx = next(first.split(X, y, groups=groups))

    X_train_val = X.iloc[train_val_idx]
    y_train_val = y.iloc[train_val_idx]
    groups_train_val = groups.iloc[train_val_idx]
    meta_train_val = meta.iloc[train_val_idx]

    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]
    groups_test = groups.iloc[test_idx]
    meta_test = meta.iloc[test_idx]

    second = GroupShuffleSplit(n_splits=1, test_size=0.1765, random_state=RANDOM_STATE)
    train_rel, val_rel = next(second.split(X_train_val, y_train_val, groups=groups_train_val))

    X_train = X_train_val.iloc[train_rel]
    y_train = y_train_val.iloc[train_rel]
    groups_train = groups_train_val.iloc[train_rel]
    meta_train = meta_train_val.iloc[train_rel]

    X_val = X_train_val.iloc[val_rel]
    y_val = y_train_val.iloc[val_rel]
    groups_val = groups_train_val.iloc[val_rel]
    meta_val = meta_train_val.iloc[val_rel]

    assert_group_disjoint(groups_train, groups_val, groups_test)
    return X_train, X_val, X_test, y_train, y_val, y_test, groups_train, groups_val, groups_test, meta_train, meta_val, meta_test


def save_split(output_dir, X_train, X_val, X_test, y_train, y_val, y_test, groups_train, groups_val, groups_test, meta_train, meta_val, meta_test, target=PRIMARY_TARGET):
    processed = Path(output_dir) / "processed"

    X_train.to_csv(processed / "X_train.csv", index=True, index_label="source_index")
    X_val.to_csv(processed / "X_val.csv", index=True, index_label="source_index")
    X_test.to_csv(processed / "X_test.csv", index=True, index_label="source_index")

    y_train.rename(target).to_csv(processed / "y_train.csv", index=True, index_label="source_index")
    y_val.rename(target).to_csv(processed / "y_val.csv", index=True, index_label="source_index")
    y_test.rename(target).to_csv(processed / "y_test.csv", index=True, index_label="source_index")

    groups_train.rename(GROUP_COLUMN).to_csv(processed / "groups_train.csv", index=True, index_label="source_index")
    groups_val.rename(GROUP_COLUMN).to_csv(processed / "groups_val.csv", index=True, index_label="source_index")
    groups_test.rename(GROUP_COLUMN).to_csv(processed / "groups_test.csv", index=True, index_label="source_index")

    meta_train.to_csv(processed / "meta_train.csv", index=True, index_label="source_index")
    meta_val.to_csv(processed / "meta_val.csv", index=True, index_label="source_index")
    meta_test.to_csv(processed / "meta_test.csv", index=True, index_label="source_index")


def load_split(output_dir, target=PRIMARY_TARGET):
    processed = Path(output_dir) / "processed"
    needed = [
        "X_train.csv",
        "X_val.csv",
        "X_test.csv",
        "y_train.csv",
        "y_val.csv",
        "y_test.csv",
        "groups_train.csv",
        "groups_val.csv",
        "groups_test.csv",
        "meta_train.csv",
        "meta_val.csv",
        "meta_test.csv",
    ]
    missing = [name for name in needed if not (processed / name).exists()]
    if missing:
        raise FileNotFoundError(f"전처리 결과가 부족합니다: {missing}. 먼저 python3 src/preprocess.py 를 실행하세요.")

    X_train = pd.read_csv(processed / "X_train.csv", index_col="source_index")
    X_val = pd.read_csv(processed / "X_val.csv", index_col="source_index")
    X_test = pd.read_csv(processed / "X_test.csv", index_col="source_index")

    y_train = pd.read_csv(processed / "y_train.csv", index_col="source_index")[target]
    y_val = pd.read_csv(processed / "y_val.csv", index_col="source_index")[target]
    y_test = pd.read_csv(processed / "y_test.csv", index_col="source_index")[target]

    groups_train = pd.read_csv(processed / "groups_train.csv", index_col="source_index")[GROUP_COLUMN].astype(int)
    groups_val = pd.read_csv(processed / "groups_val.csv", index_col="source_index")[GROUP_COLUMN].astype(int)
    groups_test = pd.read_csv(processed / "groups_test.csv", index_col="source_index")[GROUP_COLUMN].astype(int)

    meta_train = pd.read_csv(processed / "meta_train.csv", index_col="source_index")
    meta_val = pd.read_csv(processed / "meta_val.csv", index_col="source_index")
    meta_test = pd.read_csv(processed / "meta_test.csv", index_col="source_index")

    assert_group_disjoint(groups_train, groups_val, groups_test)
    return X_train, X_val, X_test, y_train, y_val, y_test, groups_train, groups_val, groups_test, meta_train, meta_val, meta_test
