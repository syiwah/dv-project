import datetime as dt

import pandas as pd
import streamlit as st

from metrocare.charts import risk_chart
from metrocare.config import FAILURES, LEAD_LOOKBACK_HOURS, MODEL_NAMES
from metrocare.data import auto_resample, load_scores
from metrocare.scoring import episodes, near_failure
from metrocare.ui import card, footer, hero, note, section

scores = load_scores()
model, watch, alert = st.session_state.model, st.session_state.watch, st.session_state.alert
col = f"risk_{model}"

hero("Objective 2 · Predict", "Failure Risk",
     "The model's risk score for every minute, with the alert threshold you set in the sidebar. "
     "Each alert episode is checked against the maintenance reports, so you can see real warnings versus false alarms.")

c1, c2 = st.columns([1.4, 1])
dmin, dmax = scores.index.min().date(), scores.index.max().date()
with c1:
    rng = st.date_input("Date range", (dmin, dmax), min_value=dmin, max_value=dmax)
with c2:
    compare = st.toggle("Overlay all four models", value=False)
start, end = (rng if isinstance(rng, (tuple, list)) and len(rng) == 2 else (dmin, dmax))
view = scores.loc[pd.Timestamp(start): pd.Timestamp(end) + pd.Timedelta(days=1)]

ep = episodes(view[col] >= alert)
if len(ep):
    ep["peak risk"] = [float(view.loc[r.start:r.end, col].max()) for r in ep.itertuples()]
    ep["duration"] = (ep["end"] - ep["start"] + pd.Timedelta(minutes=1))
    ep["verdict"] = ["✅ Early warning / failure" if near_failure(pd.DatetimeIndex([r.start, r.end])).any() else "❌ False alarm" for r in ep.itertuples()]

days = max((view.index.max() - view.index.min()).total_seconds() / 86400, 1e-9) if len(view) else 1
n_fa = int((ep["verdict"] == "❌ False alarm").sum()) if len(ep) else 0
fail_in = FAILURES[(FAILURES.start >= view.index.min()) & (FAILURES.start <= view.index.max())] if len(view) else FAILURES.iloc[:0]
caught = 0
for f in fail_in.itertuples():
    w = view.loc[f.start - pd.Timedelta(hours=LEAD_LOOKBACK_HOURS): f.end, col]
    caught += int((w >= alert).any())

k = st.columns(4)
with k[0]: card("Alert episodes", f"{len(ep)}", f"in {days:.0f} days")
with k[1]: card("False alarms / day", f"{n_fa / days:.2f}", f"{n_fa} episodes not near a failure")
with k[2]: card("Failures caught", f"{caught} / {len(fail_in)}", f"alert within {LEAD_LOOKBACK_HOURS} h before or during")
with k[3]: card("Time in alert", f"{(view[col] >= alert).mean():.1%}" if len(view) else "–", "share of minutes above threshold")

plot, res = auto_resample(view, 8000)
if res != "1 min":
    plot = view.resample({"5 min": "5min", "15 min": "15min", "1 hour": "1h", "6 hours": "6h"}[res]).max()
st.plotly_chart(risk_chart(plot, list(MODEL_NAMES) if compare else [model], watch, alert, height=420,
                           title=f"{MODEL_NAMES[model] if not compare else 'All models'} · risk score ({res} peaks)"), width="stretch")
note("Scores for Feb–May come from a model trained on Jun–Sep, and scores for Jun–Sep from a model trained on Feb–May, "
     "so every minute you see is an <b>out-of-sample</b> prediction. Official metrics use only the forward direction "
     "(train on earlier months, test on later ones).")

section("Alert episodes")
if len(ep):
    show = ep[["start", "end", "duration", "minutes", "peak risk", "verdict"]].sort_values("start", ascending=False)
    st.dataframe(show, hide_index=True, width="stretch",
                 column_config={"peak risk": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
                                "minutes": st.column_config.NumberColumn("alert minutes")})
    st.download_button("⬇️ Download alert log (CSV)", show.to_csv(index=False).encode(), "metrocare_alerts.csv", "text/csv")
else:
    st.success("No alerts above the threshold in this range.")

footer()
