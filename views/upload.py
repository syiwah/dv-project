import pandas as pd
import streamlit as st

from metrocare.charts import risk_chart, signals_chart
from metrocare.config import MODEL_NAMES, SAMPLE_UPLOAD, SENSORS
from metrocare.data import load_model, with_gaps
from metrocare.features import SchemaError, build_features, standardise_raw, to_minutes
from metrocare.scoring import episodes, model_scores, smooth, status
from metrocare.ui import card, footer, hero, note, section

model, watch, alert = st.session_state.model, st.session_state.watch, st.session_state.alert

hero("Tools", "Upload & Predict",
     "Upload a new sensor log in the MetroPT-3 format and MetroCare will clean it, build features, score every minute "
     "and give you the results as a CSV.")

c1, c2 = st.columns([1.6, 1])
with c1:
    up = st.file_uploader("Sensor CSV", type=["csv"], help="Columns: timestamp + the 15 MetroPT-3 sensors. Any sampling rate; at least ~1 hour of data.")
with c2:
    st.markdown("**Required columns**")
    st.code("timestamp, " + ", ".join(SENSORS), language=None, wrap_lines=True)
    if SAMPLE_UPLOAD.exists():
        st.download_button("⬇️ Sample file (6 h before/at failure 4)", SAMPLE_UPLOAD.read_bytes(), "metrocare_sample.csv", "text/csv",
                           width="stretch")
use_sample = st.button("▶ Try it with the sample file", disabled=not SAMPLE_UPLOAD.exists())
if use_sample:
    st.session_state.upload_sample = True
source = up if up is not None else (SAMPLE_UPLOAD if st.session_state.get("upload_sample") else None)

if source is None:
    note("No file yet. Download the sample above, or press <b>Try it with the sample file</b>.")
    footer()
    st.stop()


@st.cache_data(show_spinner="Cleaning, building features and scoring…")
def run(data: bytes | str, key: str):
    raw = standardise_raw(pd.read_csv(data if isinstance(data, str) else pd.io.common.BytesIO(data)))
    m = to_minutes(raw)
    m = m[~m["offline"]].drop(columns="offline")
    X = build_features(m)
    if X.empty:
        raise SchemaError("Not enough continuous data to build features — upload at least about an hour of readings.")
    risk = smooth(pd.Series(model_scores(load_model(key, "fwd"), X), index=X.index))
    return raw.attrs.get("bad_timestamps", 0), len(raw), m, risk


try:
    payload = up.getvalue() if up is not None else str(SAMPLE_UPLOAD)
    bad, n_raw, m, risk = run(payload, model)
except SchemaError as e:
    st.error(f"Couldn't read this file: {e}")
    st.stop()
except Exception as e:
    st.error(f"Something went wrong while processing the file: {e}")
    st.stop()

ep = episodes(risk >= alert)
latest = float(risk.iloc[-1])
k = st.columns(4)
with k[0]: card("Rows read", f"{n_raw:,}", f"{bad} unreadable timestamps dropped" if bad else "all timestamps valid")
with k[1]: card("Minutes scored", f"{len(risk):,}", f"{risk.index.min():%d %b %H:%M} → {risk.index.max():%d %b %H:%M}")
with k[2]: card("Peak risk", f"{risk.max():.2f}", f"at {risk.idxmax():%d %b %H:%M}")
with k[3]: card("Status at end of file", status(latest, watch, alert), f"risk {latest:.2f} · {len(ep)} alert episode(s)")

scored = with_gaps(pd.DataFrame({"risk_" + model: risk}))
st.plotly_chart(risk_chart(scored, [model], watch, alert, 320, title=f"Risk score · {MODEL_NAMES[model]}"), width="stretch")
st.plotly_chart(signals_chart(with_gaps(m.loc[risk.index.min():]), ["duty_cycle", "Motor_current", "TP2", "Oil_temperature"], 120),
                width="stretch")
st.caption("The model was trained on Metro do Porto data from Feb–May 2020. The first hour of any file is used to warm up the rolling features.")

section("Results")
out = m.loc[risk.index].copy()
out.insert(0, "risk", risk.round(4))
out.insert(1, "status", [status(v, watch, alert) for v in risk])
st.dataframe(out.reset_index(), hide_index=True, width="stretch", height=320,
             column_config={"risk": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")})
st.download_button("⬇️ Download predictions (CSV)", out.reset_index().to_csv(index=False).encode(), "metrocare_predictions.csv", "text/csv", type="primary")
footer()
