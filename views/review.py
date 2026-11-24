import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from metrocare.charts import MODEL_COLORS, base_layout, risk_chart, signals_chart
from metrocare.config import FAILURES, LEAD_LOOKBACK_HOURS, MODEL_NAMES, SPLIT_DATE
from metrocare.data import load_minutes, load_scores, with_gaps
from metrocare.scoring import lead_times
from metrocare.ui import COLORS, card, footer, hero, note, section

m, scores = load_minutes(), load_scores()
model, watch, alert = st.session_state.model, st.session_state.watch, st.session_state.alert
col = f"risk_{model}"

hero("Investigate", "Past Failure Review",
     "Replay each of the four air leaks reported by Metro do Porto and check whether the risk score rose "
     "before maintenance staff logged the failure.")

c1, c2 = st.columns([1.3, 1])
with c1:
    fid = st.segmented_control("Failure report", FAILURES.id.tolist(), default=4, format_func=lambda i: f"Failure {i}") or 4
with c2:
    hours = st.slider("Hours before the failure to show", 2, 72, 24, 2)
f = FAILURES.iloc[fid - 1]
start, end = f.start - pd.Timedelta(hours=hours), f.end + pd.Timedelta(hours=2)
sv, mv = with_gaps(scores.loc[start:end]), with_gaps(m.loc[start:end])

win = scores.loc[f.start - pd.Timedelta(hours=LEAD_LOOKBACK_HOURS): f.end, col]
hit = win[win >= alert]
first = hit.index.min() if len(hit) else None
lead = (f.start - first).total_seconds() / 3600 if first is not None else None
w_hit = win[win >= watch]
w_first = w_hit.index.min() if len(w_hit) else None
w_lead = (f.start - w_first).total_seconds() / 3600 if w_first is not None else None

k = st.columns(4)
with k[0]: card("Failure window", f"{f.start:%d %b, %H:%M}", f"to {f.end:%d %b %H:%M} · {(f.end - f.start).total_seconds() / 3600:.1f} h")
with k[1]: card("Type / severity", f.type, f.severity)
with k[2]:
    if lead is None:
        card("Early warning", "Missed", f"risk never reached {alert:.2f}")
    elif lead > 0:
        card("Early warning", f"{lead:.1f} h", f"first alert {first:%d %b %H:%M}, before the report")
    else:
        card("Detected", f"{-lead:.1f} h after start", f"first alert {first:%d %b %H:%M}")
with k[3]:
    if w_lead is not None and w_lead > 0:
        card("Watch-level warning", f"{w_lead:.1f} h", f"risk passed {watch:.2f} at {w_first:%d %b %H:%M} · peak {win.max():.2f}")
    else:
        card("Peak risk", f"{win.max():.2f}" if len(win) else "–", f"{MODEL_NAMES[model]}")

fig = risk_chart(sv, [model], watch, alert, height=300, title="Risk score around the failure")
fig.add_vrect(x0=f.start - pd.Timedelta(hours=2), x1=f.start, fillcolor=COLORS["watch"], opacity=0.10, line_width=0,
              annotation_text="2 h warning label", annotation_position="bottom left", annotation_font=dict(size=10, color=COLORS["watch"]))
if first is not None:
    fig.add_vline(x=first, line=dict(color=COLORS["alert"], width=2, dash="dot"))
    fig.add_annotation(x=first, y=1, text="First alert", showarrow=False, yanchor="bottom", font=dict(color=COLORS["alert"]))
st.plotly_chart(fig, width="stretch")

sigs = st.multiselect("Sensors to show", ["duty_cycle", "Motor_current", "TP2", "TP3", "Reservoirs", "Oil_temperature", "DV_pressure", "H1", "COMP", "MPG", "LPS"],
                      default=["duty_cycle", "Motor_current", "TP2", "Oil_temperature"])
if sigs:
    plot = mv[sigs].resample("5min").mean() if hours > 24 else mv[sigs]
    st.plotly_chart(signals_chart(plot, sigs, 130, marker=first), width="stretch")

if f.start < SPLIT_DATE:
    note(f"Failure {fid} is in Feb–May, so it is scored by the model trained on Jun–Sep (it never saw this failure).", warn=True)

section("All four failures, all models")
rows = []
for k_ in MODEL_NAMES:
    for r in lead_times(scores[f"risk_{k_}"], alert):
        rows.append({"Model": MODEL_NAMES[k_], "Failure": f"Failure {r['failure']}", "Detected": r["detected"],
                     "Lead time (h)": r["lead_h"], "Peak risk": r["peak_risk"]})
lt = pd.DataFrame(rows)
fig2 = go.Figure()
for k_ in MODEL_NAMES:
    d = lt[lt.Model == MODEL_NAMES[k_]]
    fig2.add_trace(go.Bar(x=d.Failure, y=d["Lead time (h)"].fillna(0), name=MODEL_NAMES[k_], marker_color=MODEL_COLORS[k_],
                          text=[f"{v:.1f} h" if pd.notna(v) else "✕" for v in d["Lead time (h)"]], textposition="outside"))
fig2.add_hline(y=0, line=dict(color=COLORS["muted"], width=1))
fig2.update_yaxes(title="Hours of warning before the report")
st.plotly_chart(base_layout(fig2, 340, f"Early-warning lead time at alert threshold {alert:.2f} (negative = detected after the report started, ✕ = missed)").update_layout(barmode="group", hovermode="closest"),
                width="stretch")
st.dataframe(lt.pivot(index="Model", columns="Failure", values="Lead time (h)").round(1), width="stretch")

footer()
