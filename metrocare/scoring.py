"""Risk-score post-processing and the evaluation metrics from section 5.2 of the proposal."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FAILURES, LEAD_LOOKBACK_HOURS

SMOOTH = "15min"
EPISODE_GAP = pd.Timedelta("30min")


def smooth(risk: pd.Series) -> pd.Series:
    """15-minute rolling mean so one noisy minute does not trigger an alert."""
    return risk.rolling(SMOOTH, min_periods=1).mean()


def model_scores(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.risk(X)


def episodes(mask: pd.Series) -> pd.DataFrame:
    """Group alert minutes into episodes (alerts less than 30 min apart are one episode)."""
    t = mask.index[mask.to_numpy()]
    if len(t) == 0:
        return pd.DataFrame(columns=["start", "end", "minutes"])
    new = np.r_[True, np.diff(t.values) > EPISODE_GAP.to_timedelta64()]
    gid = np.cumsum(new)
    df = pd.DataFrame({"t": t, "g": gid}).groupby("g")["t"].agg(["min", "max", "count"])
    return df.rename(columns={"min": "start", "max": "end", "count": "minutes"}).reset_index(drop=True)


def near_failure(index: pd.DatetimeIndex) -> np.ndarray:
    m = np.zeros(len(index), dtype=bool)
    for f in FAILURES.itertuples():
        m |= (index >= f.start - pd.Timedelta(hours=LEAD_LOOKBACK_HOURS)) & (index <= f.end)
    return m


def lead_times(risk: pd.Series, threshold: float) -> list[dict]:
    """For each failure inside the scored period: first alert in the 24 h before it, until it ends."""
    out = []
    for f in FAILURES.itertuples():
        if not (risk.index.min() <= f.start <= risk.index.max()):
            continue
        win = risk[(risk.index >= f.start - pd.Timedelta(hours=LEAD_LOOKBACK_HOURS)) & (risk.index <= f.end)]
        hit = win[win >= threshold]
        first = hit.index.min() if len(hit) else None
        out.append({
            "failure": int(f.id),
            "detected": first is not None,
            "first_alert": str(first) if first is not None else None,
            "lead_h": round((f.start - first).total_seconds() / 3600, 2) if first is not None else None,
            "peak_risk": round(float(win.max()), 3) if len(win) else None,
        })
    return out


def evaluate(risk: pd.Series, y: pd.Series, threshold: float) -> dict:
    from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

    pred = (risk >= threshold).astype(int)
    alert = risk >= threshold
    fa = episodes(alert & ~near_failure(risk.index))
    days = (risk.index.max() - risk.index.min()).total_seconds() / 86400
    leads = lead_times(risk, threshold)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    return {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y, risk)) if y.sum() else float("nan"),
        "false_alarms_per_day": float(len(fa) / max(days, 1e-9)),
        "false_alarm_episodes": int(len(fa)),
        "failures_detected": int(sum(l["detected"] for l in leads)),
        "failures_total": len(leads),
        "mean_lead_h": float(np.mean([l["lead_h"] for l in leads if l["detected"]])) if any(l["detected"] for l in leads) else None,
        "leads": leads,
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "days": round(days, 1),
    }


def status(risk: float, watch: float, alert: float) -> str:
    if risk >= alert:
        return "ALERT"
    if risk >= watch:
        return "WATCH"
    return "NORMAL"
