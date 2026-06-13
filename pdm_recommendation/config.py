from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_RAW   = BASE_DIR / "data" / "raw"
DATA_PROC  = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

RAW_FILES = {
    "telemetry"  : DATA_RAW / "PdM_telemetry.csv",
    "errors"     : DATA_RAW / "PdM_errors.csv",
    "failures"   : DATA_RAW / "PdM_failures.csv",
    "maint"      : DATA_RAW / "PdM_maint.csv",
    "machines"   : DATA_RAW / "PdM_machines.csv",
}

PROC_FILES = {
    "merged"      : DATA_PROC / "merged_features.parquet",
    "interaction" : DATA_PROC / "interaction_matrix.parquet",
    "sensor_norm" : DATA_PROC / "sensor_normalized.parquet",
}

MODEL_FILES = {
    "sim_matrix"  : MODELS_DIR / "similarity_matrix.npy",
    "weights"     : MODELS_DIR / "weights_config.json",
}

# ── Sensor columns ─────────────────────────────────────────────────────────────
SENSOR_COLS    = ["volt", "rotate", "pressure", "vibration"]
COMPONENT_COLS = ["comp1", "comp2", "comp3", "comp4"]

# ── Rolling window untuk feature engineering ──────────────────────────────────
ROLLING_WINDOW_HOURS = 24

# ── Knowledge-Based thresholds ─────────────────────────────────────────────────
# Dikalibrasi dari distribusi aktual EDA (p1 dan p99 dari 876.100 baris)
KB_THRESHOLDS = {
    "volt"      : {"min": 135.4, "max": 208.1, "comp": ["comp1", "comp3"]},
    "rotate"    : {"min": 316.8, "max": 565.5, "comp": ["comp2", "comp3"]},
    "pressure"  : {"min": 76.8,  "max": 131.7, "comp": ["comp1", "comp4"]},
    "vibration" : {"max": 54.2,                "comp": ["comp2", "comp4"]},
}

# ── Usia komponen (hari) sebelum dianggap overdue ─────────────────────────────
# Dikalibrasi dari data maint: ~800 records per komponen / 100 mesin / 365 hari
# = rata-rata penggantian tiap ~45 hari per mesin
COMP_AGE_THRESHOLDS = {
    "comp1": 45,
    "comp2": 45,
    "comp3": 45,
    "comp4": 45,
}

# ── Hybrid weights (default, di-override oleh weights_config.json) ────────────
DEFAULT_WEIGHTS = {
    "cbf": 0.35,
    "cf" : 0.40,
    "kb" : 0.25,
}

# ── Evaluation ────────────────────────────────────────────────────────────────
EVAL_K_VALUES = [1, 3, 5]

# ── Streamlit ─────────────────────────────────────────────────────────────────
APP_TITLE          = "Sistem Rekomendasi Suku Cadang Mesin Industri"
DEFAULT_MACHINE_ID = 1