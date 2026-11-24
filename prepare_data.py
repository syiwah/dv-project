"""Step 1–2 of the MetroCare pipeline: load, check and clean MetroPT-3, then save 1-minute averages.

Usage:  python prepare_data.py [path/to/MetroPT3(AirCompressor).csv]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from metrocare.config import ANALOG, DATA_DIR, FAILURES, MINUTE_FILE, QUALITY_FILE, RAW_DIR, SAMPLE_UPLOAD, SENSORS
from metrocare.features import standardise_raw, to_minutes


def find_csv() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1])
    hits = sorted(RAW_DIR.rglob("*.csv"))
    if not hits:
        sys.exit(f"No CSV found in {RAW_DIR}. Download MetroPT-3 from https://doi.org/10.24432/C5VW3R")
    return hits[0]


def main():
    path = find_csv()
    print(f"Reading {path} ...")
    raw_in = pd.read_csv(path)
    n_raw = len(raw_in)
    dup_rows = int(raw_in.drop(columns=raw_in.columns[0]).duplicated().sum()) if raw_in.columns[0].lower().startswith("unnamed") or raw_in.columns[0] == "" else int(raw_in.duplicated().sum())
    missing = int(raw_in[SENSORS].isna().sum().sum())

    raw = standardise_raw(raw_in)
    gaps = raw.index.to_series().diff().dt.total_seconds()
    interval = float(gaps.median())
    # Pressures cannot really be negative; tiny negatives near zero are sensor offset, larger ones are impossible
    negatives = {}
    for s in ANALOG:
        neg = raw[s] < 0
        if neg.any():
            negatives[s] = {"count": int(neg.sum()), "min": round(float(raw[s].min()), 3),
                            "impossible": int((raw[s] < -0.1).sum())}
    constant = [s for s in SENSORS if raw[s].nunique() <= 1]

    # Flag (not delete) spikes: a single reading that jumps away from BOTH neighbours.
    # A 3-point median ignores normal on/off steps of the compressor and only reacts to isolated jumps.
    spikes = {}
    for s in ANALOG:
        x = raw[s]
        dev = (x - x.rolling(3, center=True).median()).abs()
        span = float(x.quantile(0.99) - x.quantile(0.01)) or 1.0
        spikes[s] = int((dev > 0.25 * span).sum())

    print("Resampling to 1-minute averages ...")
    m = to_minutes(raw)
    offline = int(m["offline"].sum())
    m = m[~m["offline"]].drop(columns="offline")
    m.to_parquet(MINUTE_FILE)

    report = {
        "file": path.name,
        "raw_rows": n_raw,
        "rows_after_cleaning": int(len(raw)),
        "duplicate_rows": dup_rows,
        "missing_values": missing,
        "start": str(raw.index.min()),
        "end": str(raw.index.max()),
        "median_interval_s": interval,
        "gaps_over_5min": int((gaps > 300).sum()),
        "longest_gap_h": round(float(gaps.max()) / 3600, 1),
        "negative_values": negatives,
        "constant_columns": constant,
        "spikes_flagged": spikes,
        "minutes_kept": int(len(m)),
        "minutes_offline": offline,
        "sensor_stats": raw[SENSORS].describe().T.round(3).reset_index().rename(columns={"index": "sensor"}).to_dict("records"),
    }
    QUALITY_FILE.write_text(json.dumps(report, indent=2))

    # A small raw file in the original format, for trying the upload page (6 h around failure 4)
    f4 = FAILURES.iloc[3]
    sample = raw.loc[f4.start - pd.Timedelta(hours=4): f4.start + pd.Timedelta(hours=2)].reset_index()
    sample.to_csv(SAMPLE_UPLOAD, index=False)

    print(json.dumps({k: v for k, v in report.items() if k != "sensor_stats"}, indent=2))
    print(f"Saved {MINUTE_FILE} ({MINUTE_FILE.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    main()
