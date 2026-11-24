"""Turning raw sensor readings into model features (shared by training and the app)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ANALOG, DIGITAL, FAILURES, SENSORS

WINDOWS = {"10m": 10, "1h": 60}
MOTOR_ON_AMPS = 1.0
MAX_FILL_MINUTES = 5


class SchemaError(ValueError):
    pass


def standardise_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Validate an uploaded/raw MetroPT-style table and index it by timestamp."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.drop(columns=[c for c in df.columns if c.lower().startswith("unnamed") or c == ""], errors="ignore")
    ts_col = next((c for c in df.columns if c.lower() in ("timestamp", "time", "datetime", "date")), None)
    if ts_col is None:
        raise SchemaError("No 'timestamp' column found.")
    missing = [s for s in SENSORS if s not in df.columns]
    if missing:
        raise SchemaError("Missing sensor columns: " + ", ".join(missing))
    df["timestamp"] = pd.to_datetime(df[ts_col], errors="coerce")
    bad = df["timestamp"].isna().sum()
    df = df.dropna(subset=["timestamp"])
    if df.empty:
        raise SchemaError("No readable timestamps.")
    for s in SENSORS:
        df[s] = pd.to_numeric(df[s], errors="coerce")
    df = df[["timestamp"] + SENSORS].drop_duplicates().sort_values("timestamp")
    df.attrs["bad_timestamps"] = int(bad)
    return df.set_index("timestamp")


def to_minutes(raw: pd.DataFrame) -> pd.DataFrame:
    """Average readings per minute, add duty cycle, fill short gaps, drop long ones."""
    raw = raw.copy()
    raw["duty_cycle"] = (raw["Motor_current"] > MOTOR_ON_AMPS).astype("float32")
    raw["n_readings"] = 1
    agg = {c: "mean" for c in SENSORS + ["duty_cycle"]}
    agg["n_readings"] = "sum"
    m = raw.resample("1min").agg(agg)
    m.loc[m["n_readings"] == 0, SENSORS + ["duty_cycle"]] = np.nan
    m[SENSORS + ["duty_cycle"]] = m[SENSORS + ["duty_cycle"]].ffill(limit=MAX_FILL_MINUTES)
    m["offline"] = m["TP2"].isna()
    return m.astype({c: "float32" for c in SENSORS + ["duty_cycle"]})


def build_features(m: pd.DataFrame) -> pd.DataFrame:
    """Rolling 10-minute and 1-hour summaries on a regular 1-minute index."""
    m = m.asfreq("1min")
    cols = {}
    for name, w in WINDOWS.items():
        mp = max(2, w // 2)
        for s in ANALOG:
            r = m[s].rolling(w, min_periods=mp)
            cols[f"{s}_mean_{name}"] = r.mean()
            cols[f"{s}_std_{name}"] = r.std()
            cols[f"{s}_min_{name}"] = r.min()
            cols[f"{s}_max_{name}"] = r.max()
            cols[f"{s}_slope_{name}"] = (m[s] - m[s].shift(w)) / w
        for s in DIGITAL + ["duty_cycle"]:
            cols[f"{s}_mean_{name}"] = m[s].rolling(w, min_periods=mp).mean()
    X = pd.DataFrame(cols, index=m.index).astype("float32")
    valid = m["TP2"].notna() & X.notna().all(axis=1)
    return X[valid]


def feature_signal(feature: str) -> str:
    """Map a feature name back to the sensor it was computed from."""
    for s in sorted(ANALOG + DIGITAL + ["duty_cycle"], key=len, reverse=True):
        if feature.startswith(s + "_"):
            return s
    return feature


def pretty_feature(feature: str) -> str:
    sig = feature_signal(feature)
    rest = feature[len(sig) + 1:]
    stat, _, win = rest.rpartition("_")
    stat_name = {"mean": "average", "std": "variability", "min": "minimum", "max": "maximum", "slope": "trend"}.get(stat, stat)
    if sig == "duty_cycle":
        return f"Duty cycle ({win})"
    return f"{sig} {stat_name} ({win})"


def make_labels(index: pd.DatetimeIndex, pre_hours: float = 0) -> pd.Series:
    y = pd.Series(0, index=index, dtype="int8")
    for f in FAILURES.itertuples():
        y[(index >= f.start - pd.Timedelta(hours=pre_hours)) & (index <= f.end)] = 1
    return y


def in_failure(index: pd.DatetimeIndex) -> np.ndarray:
    return make_labels(index, 0).to_numpy().astype(bool)
