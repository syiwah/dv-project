import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from metrocare.charts import MODEL_COLORS, base_layout, gauge, shade_failures, timeline_strip
from metrocare.config import FAILURES, MODEL_NAMES, SENSOR_INFO, SENSORS
from metrocare.data import daily_summary, load_minutes, load_scores, with_gaps
from metrocare.scoring import status
from metrocare.ui import COLORS, card, footer, hero, section, status_box

m, scores = load_minutes(), load_scores()
model, watch, alert = st.session_state.model, st.session_state.watch, st.session_state.alert
col = f"risk_{model}"

hero("Overview", "Air Production Unit — live health",
     "Replay any moment from February to September 2020 and see the compressor exactly as a maintenance controller would have: "
     "status, failure risk, how hard the compressor is working, and every sensor reading.")

tmin, tmax = scores.index.min().to_pydatetime(), scores.index.max().to_pydatetime()
if "replay" not in st.session_state:
    st.session_state.replay = tmax


def jump(ts):
    st.session_state.replay = pd.Timestamp(ts).to_pydatetime()


st.markdown("<span style='color:#8B9BB8;font-size:.85rem'>Jump to a moment</span>", unsafe_allow_html=True)
c = st.columns(6)
c[0].button("⏭ Latest", on_click=jump, args=(tmax,), width="stretch")
for i, f in enumerate(FAILURES.itertuples(), 1):
    c[i].button(f"F{f.id} −3 h", on_click=jump, args=(f.start - pd.Timedelta(hours=3),), width="stretch",
                help=f"3 hours before failure {f.id} was reported ({f.start:%d %b %Y %H:%M})")
c[5].button("🔴 In F4", on_click=jump, args=(FAILURES.iloc[3].start + pd.Timedelta(hours=1),), width="stretch",
            help="One hour into failure 4")

t = st.slider("Replay moment", min_value=tmin, max_value=tmax, step=pd.Timedelta(minutes=10).to_pytimedelta(),
              format="DD MMM YYYY, HH:mm", key="replay")
t = pd.Timestamp(t)
t = scores.index[scores.index.get_indexer([t], method="ffill")[0]] if t >= scores.index[0] else scores.index[0]
risk = float(scores.at[t, col])
state = status(risk, watch, alert)

now = m.loc[:t].iloc[-1]
last_h = m.loc[t - pd.Timedelta(hours=1): t]
prev_day = m.loc[t - pd.Timedelta(hours=25): t - pd.Timedelta(hours=1)]

st.caption(f"Showing **{t:%A, %d %B %Y · %H:%M}** · model: **{MODEL_NAMES[model]}**")
left, mid = st.columns([1.15, 1])
with left:
    status_box(state, risk)
    in_fail = FAILURES[(FAILURES.start <= t) & (FAILURES.end >= t)]
    nxt = FAILURES[FAILURES.start > t]
    if len(in_fail):
        st.markdown(f"<div class='mc-note warn'>⚠️ This moment is <b>inside reported failure {int(in_fail.iloc[0].id)}</b> (air leak).</div>", unsafe_allow_html=True)
    elif len(nxt):
        hrs = (nxt.iloc[0].start - t).total_seconds() / 3600
        if hrs <= 48:
            st.markdown(f"<div class='mc-note warn'>Hindsight: failure {int(nxt.iloc[0].id)} was reported <b>{hrs:.1f} hours</b> after this moment.</div>", unsafe_allow_html=True)
with mid:
    st.plotly_chart(gauge(risk, watch, alert), width="stretch", config={"displayModeBar": False})

k = st.columns(5)
duty_h, duty_d = last_h["duty_cycle"].mean(), prev_day["duty_cycle"].mean()
with k[0]:
    card("Duty cycle · 1 h", f"{duty_h:.0%}", "vs previous 24 h", (duty_h - duty_d) * 100, fmt="{:+.0f} pts")
for box, s in zip(k[1:], ["Motor_current", "Oil_temperature", "TP2", "Reservoirs"]):
    unit = SENSOR_INFO[s][0]
    with box:
        card(s.replace("_", " "), f"{now[s]:.2f} {unit}", "vs 24 h average", float(now[s] - prev_day[s].mean()) if len(prev_day) else None,
             bad_when_up=s in ("Motor_current", "Oil_temperature"))

section("Last 24 hours")
day = with_gaps(m.loc[t - pd.Timedelta(hours=24): t])
sday = with_gaps(scores.loc[t - pd.Timedelta(hours=24): t])
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.55, 0.45], vertical_spacing=0.06,
                    subplot_titles=("Failure risk", "Compressor duty cycle & motor current"))
fig.add_trace(go.Scatter(x=sday.index, y=sday[col], name="Risk", line=dict(color=MODEL_COLORS[model], width=2), fill="tozeroy",
                         fillcolor="rgba(34,211,238,.10)"), row=1, col=1)
fig.add_hline(y=alert, line=dict(color=COLORS["alert"], dash="dash", width=1), row=1, col=1)
fig.add_hline(y=watch, line=dict(color=COLORS["watch"], dash="dot", width=1), row=1, col=1)
duty15 = day["duty_cycle"].rolling("15min").mean()
fig.add_trace(go.Scatter(x=duty15.index, y=duty15, name="Duty cycle (15-min avg)", line=dict(color=COLORS["accent2"], width=1.8)), row=2, col=1)
fig.add_trace(go.Scatter(x=day.index, y=day["Motor_current"] / 10, name="Motor current ÷ 10 (A)", line=dict(color="#34D399", width=0.8), opacity=0.6), row=2, col=1)
fig.update_yaxes(range=[0, 1.02], row=1, col=1)
fig.update_yaxes(range=[0, 1.02], tickformat=".0%", row=2, col=1)
for f in FAILURES.itertuples():
    if f.end >= day.index.min() and f.start <= t:
        fig.add_vrect(x0=max(f.start, day.index.min()), x1=min(f.end, t), fillcolor=COLORS["failure"], opacity=0.18, line_width=0, row="all", col=1)
fig.update_annotations(font=dict(size=12, color=COLORS["muted"]), x=0, xanchor="left")
st.plotly_chart(base_layout(fig, 420), width="stretch")

c1, c2 = st.columns([1.25, 1])
with c1:
    section("Latest readings")
    typical = prev_day[SENSORS].median() if len(prev_day) else m[SENSORS].median()
    spread = prev_day[SENSORS].std().replace(0, pd.NA) if len(prev_day) else m[SENSORS].std()
    tbl = pd.DataFrame({
        "Sensor": SENSORS,
        "Now": [now[s] for s in SENSORS],
        "Unit": [SENSOR_INFO[s][0] for s in SENSORS],
        "24 h median": [typical[s] for s in SENSORS],
        "Deviation (σ)": [abs((now[s] - typical[s]) / spread[s]) if pd.notna(spread[s]) else 0 for s in SENSORS],
        "What it measures": [SENSOR_INFO[s][1] for s in SENSORS],
    })
    st.dataframe(tbl, hide_index=True, width="stretch", height=420,
                 column_config={"Now": st.column_config.NumberColumn(format="%.3f"),
                                "24 h median": st.column_config.NumberColumn(format="%.3f"),
                                "Deviation (σ)": st.column_config.ProgressColumn(min_value=0, max_value=4, format="%.1f σ")})
with c2:
    section("Whole period at a glance")
    daily = daily_summary()
    st.plotly_chart(timeline_strip(daily, f"{col}_max", marker=t), width="stretch", config={"displayModeBar": False})
    dd = daily["duty_cycle"]
    f2 = go.Figure(go.Scatter(x=dd.index, y=dd, line=dict(color=COLORS["accent2"], width=1.6), fill="tozeroy",
                              fillcolor="rgba(129,140,248,.12)", hovertemplate="%{x|%d %b}: %{y:.1%}<extra></extra>"))
    shade_failures(f2, label=False)
    f2.add_vline(x=t, line=dict(color=COLORS["text"], width=1.5))
    f2.update_yaxes(tickformat=".0%", title="Daily duty cycle")
    st.plotly_chart(base_layout(f2, 200, legend=False), width="stretch", config={"displayModeBar": False})
    st.caption("Red bands = reported air-leak failures. A leak makes the compressor work longer to hold pressure, so duty cycle tends to rise around them.")

footer()
