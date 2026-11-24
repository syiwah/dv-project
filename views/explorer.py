import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from metrocare.charts import base_layout, heatmap, signals_chart
from metrocare.config import ANALOG, DIGITAL, FAILURES, SENSOR_INFO
from metrocare.data import auto_resample, load_minutes
from metrocare.features import in_failure
from metrocare.ui import COLORS, footer, hero, note, section

m = load_minutes()
ALL = ANALOG + ["duty_cycle"] + DIGITAL

hero("Objective 1 · Explore", "Sensor Explorer",
     "Plot any of the 15 APU sensors over time with failure periods shaded, compare normal and failure behaviour, "
     "and see which signals move together.")

c1, c2 = st.columns([2, 1.2])
with c1:
    sigs = st.multiselect("Signals", ALL, default=["TP2", "Motor_current", "Oil_temperature", "duty_cycle"],
                          format_func=lambda s: f"{s} ({SENSOR_INFO[s][0]})")
with c2:
    preset = st.selectbox("Period", ["Whole dataset"] + [f"Around failure {i}" for i in FAILURES.id] + ["Custom"], index=4)
dmin, dmax = m.index.min().date(), m.index.max().date()
if preset == "Whole dataset":
    start, end = dmin, dmax
elif preset == "Custom":
    rng = st.date_input("Date range", (dt.date(2020, 7, 10), dt.date(2020, 7, 17)), min_value=dmin, max_value=dmax)
    start, end = (rng if isinstance(rng, (tuple, list)) and len(rng) == 2 else (dmin, dmax))
else:
    f = FAILURES.iloc[int(preset.split()[-1]) - 1]
    start, end = (f.start - pd.Timedelta(days=3)).date(), (f.end + pd.Timedelta(days=1)).date()

view = m.loc[pd.Timestamp(start): pd.Timestamp(end) + pd.Timedelta(days=1)]
tabs = st.tabs(["📈 Time series", "⚖️ Normal vs failure", "🔗 Relationships", "🗓️ Daily patterns", "📋 Data table"])

with tabs[0]:
    if not sigs:
        st.info("Pick at least one signal.")
    else:
        plot_df, res = auto_resample(view[sigs])
        st.caption(f"{len(view):,} minutes in range · drawn at **{res}** resolution for speed · drag to zoom, double-click to reset.")
        st.plotly_chart(signals_chart(plot_df, sigs), width="stretch")

with tabs[1]:
    section("How each signal differs during reported failures")
    fail = in_failure(m.index)
    pick = st.multiselect("Signals to compare", ALL, default=["Motor_current", "duty_cycle", "TP2", "Oil_temperature", "DV_pressure", "Reservoirs"], key="cmp")
    if pick:
        samp = m[pick].copy()
        samp["Condition"] = np.where(fail, "Failure", "Normal")
        normal = samp[samp.Condition == "Normal"].sample(min(40000, (~fail).sum()), random_state=0)
        both = pd.concat([normal, samp[samp.Condition == "Failure"]])
        cols = st.columns(3)
        for i, s in enumerate(pick):
            fig = go.Figure()
            for cond, colr in (("Normal", COLORS["normal"]), ("Failure", COLORS["failure"])):
                fig.add_trace(go.Violin(y=both.loc[both.Condition == cond, s], name=cond, line_color=colr, box_visible=True,
                                        meanline_visible=True, points=False, spanmode="hard"))
            cols[i % 3].plotly_chart(base_layout(fig, 280, f"{s} ({SENSOR_INFO[s][0]})", legend=False), width="stretch")
        summary = pd.DataFrame({
            "Normal mean": samp[samp.Condition == "Normal"][pick].mean(),
            "Failure mean": samp[samp.Condition == "Failure"][pick].mean(),
        })
        summary["Change"] = (summary["Failure mean"] - summary["Normal mean"]) / summary["Normal mean"].abs().replace(0, np.nan)
        st.dataframe(summary.style.format({"Normal mean": "{:.3f}", "Failure mean": "{:.3f}", "Change": "{:+.0%}"})
                     .background_gradient(subset=["Change"], cmap="RdYlGn_r", vmin=-1, vmax=1), width="stretch")
        note("Expected pattern from the proposal: a leak makes the compressor run longer to keep pressure up, "
             "so <b>duty cycle</b> and <b>motor current</b> rise while pressures such as <b>TP2</b> struggle to hold.")

with tabs[2]:
    section("Correlation between signals")
    fail = in_failure(m.index)
    corr_sigs = ANALOG + ["duty_cycle", "COMP", "MPG", "LPS"]
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(heatmap(m.loc[~fail, corr_sigs].sample(60000, random_state=0).corr().round(2), "Normal operation"), width="stretch")
    with c2:
        st.plotly_chart(heatmap(m.loc[fail, corr_sigs].corr().round(2), "During failures"), width="stretch")
    section("Scatter: any two signals")
    a, b = st.columns(2)
    x = a.selectbox("X axis", ALL, index=ALL.index("Motor_current"))
    y = b.selectbox("Y axis", ALL, index=ALL.index("Oil_temperature"))
    sc = view[[x, y]].copy()
    sc["fail"] = in_failure(sc.index)
    sc = pd.concat([sc[~sc.fail].sample(min(15000, (~sc.fail).sum()), random_state=1), sc[sc.fail]])
    fig = go.Figure()
    for flag, name, colr, op in ((False, "Normal", COLORS["accent"], 0.25), (True, "Failure", COLORS["failure"], 0.7)):
        d = sc[sc.fail == flag]
        fig.add_trace(go.Scattergl(x=d[x], y=d[y], mode="markers", name=name, marker=dict(size=4, color=colr, opacity=op)))
    fig.update_xaxes(title=x); fig.update_yaxes(title=y)
    st.plotly_chart(base_layout(fig, 420).update_layout(hovermode="closest"), width="stretch")

with tabs[3]:
    section("Hour-of-day × date heatmap")
    sig = st.selectbox("Signal", ALL, index=ALL.index("duty_cycle"), key="heat")
    h = m[sig].resample("1h").mean().to_frame("v")
    h["date"], h["hour"] = h.index.date, h.index.hour
    piv = h.pivot_table(index="hour", columns="date", values="v")
    fig = go.Figure(go.Heatmap(z=piv.values, x=pd.to_datetime(piv.columns), y=piv.index, colorscale="Viridis",
                               hovertemplate="%{x|%d %b} %{y}:00 → %{z:.3f}<extra></extra>", colorbar=dict(thickness=10)))
    for f in FAILURES.itertuples():
        fig.add_vrect(x0=f.start.normalize(), x1=f.end.normalize() + pd.Timedelta(days=1), line=dict(color=COLORS["failure"], width=2), fillcolor="rgba(0,0,0,0)")
    fig.update_yaxes(title="Hour of day", autorange="reversed")
    st.plotly_chart(base_layout(fig, 420, legend=False).update_layout(hovermode="closest"), width="stretch")
    st.caption("Red boxes mark failure days. Night-time gaps show when the train is parked; operating patterns repeat daily.")

with tabs[4]:
    show = view[sigs] if sigs else view
    st.dataframe(show.tail(5000), width="stretch", height=420)
    st.download_button("⬇️ Download this range (1-minute averages, CSV)", show.to_csv().encode(),
                       f"metropt3_{start}_{end}.csv", "text/csv")

footer()
