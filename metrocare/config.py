"""Shared constants for MetroCare."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MODEL_DIR = ROOT / "models"

MINUTE_FILE = DATA_DIR / "metropt3_1min.parquet"
SCORES_FILE = DATA_DIR / "risk_scores.parquet"
QUALITY_FILE = DATA_DIR / "quality_report.json"
METRICS_FILE = MODEL_DIR / "metrics.json"
SAMPLE_UPLOAD = DATA_DIR / "sample_upload.csv"

ANALOG = ["TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Oil_temperature", "Motor_current"]
DIGITAL = ["COMP", "DV_eletric", "Towers", "MPG", "LPS", "Pressure_switch", "Oil_level", "Caudal_impulses"]
SENSORS = ANALOG + DIGITAL

SENSOR_INFO = {
    "TP2": ("bar", "Pressure on the compressor"),
    "TP3": ("bar", "Pressure at the pneumatic panel"),
    "H1": ("bar", "Pressure drop when the cyclonic separator filter discharges"),
    "DV_pressure": ("bar", "Pressure drop when the towers discharge air dryers"),
    "Reservoirs": ("bar", "Downstream pressure of the reservoirs (should be close to TP3)"),
    "Oil_temperature": ("°C", "Oil temperature on the compressor"),
    "Motor_current": ("A", "Motor current (≈0 A off, ≈4 A loaded, ≈7 A starting)"),
    "COMP": ("on/off", "Air intake valve signal — active when there is no air intake"),
    "DV_eletric": ("on/off", "Compressor outlet valve signal"),
    "Towers": ("on/off", "Which air-drying tower is active"),
    "MPG": ("on/off", "Starts the compressor when pressure drops below 8.2 bar"),
    "LPS": ("on/off", "Low pressure signal — active below 7 bar"),
    "Pressure_switch": ("on/off", "Pressure discharge detected in the pilot-controlled valve"),
    "Oil_level": ("on/off", "Oil level below expected values"),
    "Caudal_impulses": ("on/off", "Air flow pulses from the flowmeter"),
    "duty_cycle": ("share", "Share of the minute the compressor motor was running (Motor_current > 1 A)"),
}

# Failure reports written by Metro do Porto maintenance staff (Davari et al., 2021)
FAILURES = pd.DataFrame(
    {
        "id": [1, 2, 3, 4],
        "start": pd.to_datetime(["2020-04-18 00:00", "2020-05-29 23:30", "2020-06-05 10:00", "2020-07-15 14:30"]),
        "end": pd.to_datetime(["2020-04-18 23:59", "2020-05-30 06:00", "2020-06-07 14:30", "2020-07-15 19:00"]),
        "type": ["Air leak"] * 4,
        "severity": ["High stress"] * 4,
    }
)

PRE_FAILURE_HOURS = 2      # extra "warning" label before each failure
LEAD_LOOKBACK_HOURS = 24   # alerts this long before a failure count as early warnings
SPLIT_DATE = pd.Timestamp("2020-06-01")  # train on Feb–May, test on Jun–Sep

MODEL_NAMES = {
    "logreg": "Logistic Regression",
    "rf": "Random Forest",
    "gb": "Gradient Boosting",
    "iforest": "Isolation Forest",
    "hybrid": "Hybrid (LogReg + IsoForest)",
}
POST_HOC_MODELS = {"hybrid"}  # added after inspecting test results — flagged in the app
