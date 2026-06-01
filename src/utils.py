import os
import warnings
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "datasets" / "telemonitoring_parkinsons_updrs.data.csv"
OUTPUT_DIR = ROOT_DIR / "results"

RANDOM_STATE = 42
GROUP_COLUMN = "subject#"
PRIMARY_TARGET = "total_UPDRS"
SECONDARY_TARGET = "motor_UPDRS"
TARGET_COLUMNS = [PRIMARY_TARGET, SECONDARY_TARGET]

CONTEXT_COLUMNS = ["age", "sex", "test_time"]

VOICE_FEATURES = [
    "Jitter(%)",
    "Jitter(Abs)",
    "Jitter:RAP",
    "Jitter:PPQ5",
    "Jitter:DDP",
    "Shimmer",
    "Shimmer(dB)",
    "Shimmer:APQ3",
    "Shimmer:APQ5",
    "Shimmer:APQ11",
    "Shimmer:DDA",
    "NHR",
    "HNR",
    "RPDE",
    "DFA",
    "PPE",
]

FEATURE_SETS = {
    "voice_only": VOICE_FEATURES,
    "voice_plus_context": VOICE_FEATURES + CONTEXT_COLUMNS,
}

KEY_ERROR_COLUMNS = [
    GROUP_COLUMN,
    "age",
    "sex",
    "test_time",
    "Jitter(%)",
    "Shimmer",
    "NHR",
    "HNR",
    "RPDE",
    "DFA",
    "PPE",
]

FORBIDDEN_FEATURES = [GROUP_COLUMN, PRIMARY_TARGET, SECONDARY_TARGET]


def setup():
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    for folder in [
        "processed",
        "tables",
        "figures",
        "tables/eda",
        "tables/modeling",
        "tables/analysis",
        "figures/eda",
        "figures/modeling",
        "figures/analysis",
    ]:
        (OUTPUT_DIR / folder).mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def table_path(section, filename):
    return OUTPUT_DIR / "tables" / section / filename


def figure_path(section, filename):
    return OUTPUT_DIR / "figures" / section / filename


def target_tag(target):
    return target.lower()


def check_feature_leakage(columns):
    leaked = sorted(set(columns).intersection(FORBIDDEN_FEATURES))
    if leaked:
        raise ValueError(f"Feature에 들어가면 안 되는 column이 있음: {leaked}")


def get_feature_columns(feature_set):
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"알 수 없는 feature set: {feature_set}")
    columns = FEATURE_SETS[feature_set]
    check_feature_leakage(columns)
    return columns
