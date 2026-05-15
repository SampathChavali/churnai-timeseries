"""
config.py
---------
Single source of truth for all paths, hyperparameters, and feature definitions
for the ChurnAI Time-Series sub-system.

This is a focused, self-contained version of the larger ChurnAI project that
trains and serves the LSTM time-series churn model only. It does NOT require
the Kaggle Telco CSV — a synthetic customer base is generated.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Resolve BASE_DIR relative to this file so the project is portable
# (works on macOS dev, Linux/Streamlit Cloud, Docker, etc. — no hardcoded paths).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR        = os.path.join(BASE_DIR, "data")
OUTPUTS_DIR     = os.path.join(BASE_DIR, "outputs")
PLOTS_DIR       = os.path.join(OUTPUTS_DIR, "plots")
SAVED_DIR       = os.path.join(OUTPUTS_DIR, "saved")

CUSTOMERS_CSV   = os.path.join(DATA_DIR, "customers.csv")
TIMESERIES_CSV  = os.path.join(DATA_DIR, "time_series.csv")
TIMESERIES_SAMPLE_CSV = os.path.join(DATA_DIR, "time_series_sample.csv")
PREDICTIONS_CSV = os.path.join(DATA_DIR, "predictions.csv")

LSTM_MODEL_PATH = os.path.join(SAVED_DIR, "lstm_encoder.pt")
TS_SCALER_PATH  = os.path.join(SAVED_DIR, "ts_scaler.joblib")
METRICS_JSON    = os.path.join(OUTPUTS_DIR, "metrics.json")

for _d in (DATA_DIR, OUTPUTS_DIR, PLOTS_DIR, SAVED_DIR):
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------------------
# Synthetic customer base
# ---------------------------------------------------------------------------
N_CUSTOMERS      = 2000
CHURN_RATE       = 0.26  # matches the Kaggle Telco baseline
RANDOM_SEED      = 42

# ---------------------------------------------------------------------------
# Time-series schema
# ---------------------------------------------------------------------------
SEQ_LEN = 12  # 12 monthly snapshots per customer

TS_FEATURES = [
    "data_gb",
    "call_minutes",
    "monthly_charge",
    "support_tickets",
    "login_count",
    "service_outages",
]
TS_INPUT_DIM = len(TS_FEATURES)  # 6

# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------
LSTM_HIDDEN     = 128
LSTM_LAYERS     = 2
LSTM_EMBED_DIM  = 128
LSTM_DROPOUT    = 0.3
LSTM_LR         = 1e-3
LSTM_EPOCHS     = 15
LSTM_BATCH      = 64

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
TEST_SIZE        = 0.15
VAL_SIZE         = 0.15
CHURN_POS_WEIGHT = 2.5

# ---------------------------------------------------------------------------
# Risk thresholds for the UI
# ---------------------------------------------------------------------------
RISK_HIGH   = 0.65
RISK_MEDIUM = 0.35
