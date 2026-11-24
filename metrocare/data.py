"""Cached loaders used by every page of the app."""
from __future__ import annotations

import json

import joblib
import pandas as pd
import streamlit as st

from .config import METRICS_FILE, MINUTE_FILE, MODEL_DIR, QUALITY_FILE, SCORES_FILE, SPLIT_DATE
from .features import build_features
from .scoring import smooth


def artifacts_ready() -> bool:
    return MINUTE_FILE.exists() and SCORES_FILE.exists() and METRICS_FILE.exists()


@st.cache_data(show_spinner="Loading sensor data…")
def load_minutes() -> pd.DataFrame:
    return pd.read_parquet(MINUTE_FILE)


@st.cache_data(show_spinner=False)
def load_scores() -> pd.DataFrame:
    raw = pd.read_parquet(SCORES_FILE)
    out = pd.DataFrame(index=raw.index)
    for c in raw.columns:
        out[c] = smooth(raw[c]).astype("float32")
    return out


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    return json.loads(METRICS_FILE.read_text())


@st.cache_data(show_spinner=False)
def load_quality() -> dict:
    return json.loads(QUALITY_FILE.read_text()) if QUALITY_FILE.exists() else {}


@st.cache_resource(show_spinner="Loading model…")
def load_model(key: str, fold: str = "fwd"):
    return joblib.load(MODEL_DIR / f"{key}_{fold}.joblib")


def fold_for(t: pd.Timestamp) -> str:
    """Which trained model produced the score at time t (each minute is scored out-of-sample)."""
    return "fwd" if t >= SPLIT_DATE else "bwd"


@st.cache_data(show_spinner=False)
def features_at(t: pd.Timestamp) -> pd.DataFrame:
    m = load_minutes()
    window = m.loc[t - pd.Timedelta(hours=2): t]
    X = build_features(window)
    return X.loc[[t]] if t in X.index else X.tail(1)


@st.cache_data(show_spinner=False)
def daily_summary() -> pd.DataFrame:
    m = load_minutes()
    s = load_scores()
    d = m[["duty_cycle", "Oil_temperature", "Motor_current", "TP2"]].resample("1D").mean()
    d = d.join(s.resample("1D").max().add_suffix("_max"))
    return d.dropna(how="all")


def window(df: pd.DataFrame, start, end) -> pd.DataFrame:
    return df.loc[pd.Timestamp(start): pd.Timestamp(end)]


def auto_resample(df: pd.DataFrame, max_points: int = 6000) -> tuple[pd.DataFrame, str]:
    """Pick a resolution that keeps charts fast."""
    if df.empty:
        return df, "1 min"
    span_min = (df.index.max() - df.index.min()).total_seconds() / 60
    for rule, label, mins in (("1min", "1 min", 1), ("5min", "5 min", 5), ("15min", "15 min", 15), ("1h", "1 hour", 60), ("6h", "6 hours", 360)):
        if span_min / mins <= max_points:
            break
    # Keep empty bins as NaN so lines break across periods when the train was offline
    return df.resample(rule).mean(), label


def with_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Regular 1-minute grid (NaN where no data) so charts don't draw lines across gaps."""
    return df.asfreq("1min") if len(df) else df
