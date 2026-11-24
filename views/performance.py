import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from metrocare.charts import MODEL_COLORS, base_layout
from metrocare.config import MODEL_NAMES, PRE_FAILURE_HOURS
from metrocare.data import load_metrics, load_scores
from metrocare.ui import COLORS, card, footer, hero, note, section

mt = load_metrics()
M = mt["models"]
best = mt["best_model"]

hero("Objective 2 · Evaluate", "Model Performance",
     f"Four models compared on a strict chronological split: trained on Feb–May 2020 (failures 1–2), tested on "
     f"Jun–Sep 2020 (failures 3–4) that they had never seen. Alert threshold {mt['threshold']}, risk smoothed over 15 minutes.")

b = M[best]
k = st.columns(5)
with k[0]: card("Best model", MODEL_NAMES[best], "most failures caught, then PR-AUC")
with k[1]: card("PR-AUC", f"{b['pr_auc']:.3f}", "1.0 = perfect separation")
with k[2]: card("F1-score", f"{b['f1']:.3f}", f"precision {b['precision']:.2f} · recall {b['recall']:.2f}")
with k[3]: card("False alarms / day", f"{b['false_alarms_per_day']:.2f}", f"{b['false_alarm_episodes']} episodes in {b['days']:.0f} test days")
with k[4]: card("Test failures caught", f"{b['failures_detected']} / {b['failures_total']}",
                f"mean lead {b['mean_lead_h']:.1f} h" if b["mean_lead_h"] is not None else "no early warning")

section("Side-by-side comparison (test period)")
rows = []
for kk, r in M.items():
    rows.append({"Model": ("★ " if kk == best else "") + MODEL_NAMES[kk], "Precision": r["precision"], "Recall": r["recall"],
                 "F1": r["f1"], "PR-AUC": r["pr_auc"], "False alarms/day": r["false_alarms_per_day"],
                 "Failures caught": f"{r['failures_detected']}/{r['failures_total']}",
                 "Mean lead (h)": r["mean_lead_h"], "Train time (s)": r["fit_seconds"]})
tbl = pd.DataFrame(rows).set_index("Model")
st.dataframe(tbl.style.format({"Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}", "PR-AUC": "{:.3f}",
                               "False alarms/day": "{:.2f}", "Mean lead (h)": lambda v: "–" if pd.isna(v) else f"{v:.1f}", "Train time (s)": "{:.0f}"})
             .background_gradient(subset=["Precision", "Recall", "F1", "PR-AUC"], cmap="Blues", vmin=0, vmax=1)
             .background_gradient(subset=["False alarms/day"], cmap="Reds"), width="stretch")

t1, t2, t3, t4 = st.tabs(["📊 Metrics", "📉 Precision–recall", "🧮 Confusion matrix", "🏷️ Labelling strategy"])
with t1:
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for kk in M:
            fig.add_trace(go.Bar(name=MODEL_NAMES[kk], x=["Precision", "Recall", "F1", "PR-AUC"],
                                 y=[M[kk]["precision"], M[kk]["recall"], M[kk]["f1"], M[kk]["pr_auc"]], marker_color=MODEL_COLORS[kk]))
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(base_layout(fig, 360, "Detection quality (higher is better)").update_layout(barmode="group", hovermode="closest"), width="stretch")
    with c2:
        fig = go.Figure(go.Bar(x=[MODEL_NAMES[k_] for k_ in M], y=[M[k_]["false_alarms_per_day"] for k_ in M],
                               marker_color=[MODEL_COLORS[k_] for k_ in M], text=[f"{M[k_]['false_alarms_per_day']:.2f}" for k_ in M], textposition="outside"))
        st.plotly_chart(base_layout(fig, 360, "False alarms per day (lower is better)", legend=False).update_layout(hovermode="closest"), width="stretch")
    lead_rows = [{"Model": MODEL_NAMES[k_], "Failure": f"Failure {l['failure']}", "Lead (h)": l["lead_h"]} for k_ in M for l in M[k_]["leads"]]
    if lead_rows:
        ld = pd.DataFrame(lead_rows)
        fig = go.Figure()
        for k_ in M:
            d = ld[ld.Model == MODEL_NAMES[k_]]
            fig.add_trace(go.Bar(x=d.Failure, y=d["Lead (h)"].fillna(0), name=MODEL_NAMES[k_], marker_color=MODEL_COLORS[k_],
                                 text=[f"{v:.1f} h" if pd.notna(v) else "✕" for v in d["Lead (h)"]], textposition="outside"))
        fig.update_yaxes(title="Hours before report")
        st.plotly_chart(base_layout(fig, 320, "Early-warning lead time on the unseen test failures (✕ = missed)").update_layout(barmode="group", hovermode="closest"), width="stretch")
with t2:
    fig = go.Figure()
    for kk, pr in mt["pr_curves"].items():
        fig.add_trace(go.Scatter(x=pr["recall"], y=pr["precision"], name=f"{MODEL_NAMES[kk]} (AP {M[kk]['pr_auc']:.3f})",
                                 line=dict(color=MODEL_COLORS[kk], width=2.2)))
    prev = mt["test_failure_minutes"] / mt["test_minutes"]
    fig.add_hline(y=prev, line=dict(color=COLORS["muted"], dash="dot"), annotation_text=f"Random guess ({prev:.3f})")
    fig.update_xaxes(title="Recall — share of failure minutes caught", range=[0, 1])
    fig.update_yaxes(title="Precision — share of alerts that were real", range=[0, 1.02])
    st.plotly_chart(base_layout(fig, 460).update_layout(hovermode="closest"), width="stretch")
    note("Plain accuracy is misleading here: a model that always says “normal” would be right "
         f"{1 - prev:.1%} of the time. PR-AUC focuses on the rare failure minutes instead.")
with t3:
    pick = st.segmented_control("Model", list(M), default=best, format_func=lambda k_: MODEL_NAMES[k_]) or best
    c = M[pick]["confusion"]
    z = [[c["tn"], c["fp"]], [c["fn"], c["tp"]]]
    tot = sum(c.values())
    fig = go.Figure(go.Heatmap(z=z, x=["Predicted normal", "Predicted failure"], y=["Actually normal", "Actually failure"],
                               colorscale=[[0, COLORS["panel2"]], [1, COLORS["accent"]]], showscale=False,
                               text=[[f"{v:,}<br>{v / tot:.2%}" for v in row] for row in z], texttemplate="%{text}", textfont=dict(size=16)))
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(base_layout(fig, 360, f"{MODEL_NAMES[pick]} · minutes in the test period", legend=False).update_layout(hovermode="closest"), width="stretch")
    st.caption("Missed failures = bottom-left. False alarms = top-right. Counted per minute; the KPI cards count alert episodes.")
with t4:
    lc = mt["label_comparison"]
    rows = []
    for kk in M:
        rows.append({"Model": MODEL_NAMES[kk], "Labels": f"Failure window + {PRE_FAILURE_HOURS} h before", "F1": M[kk]["f1"], "PR-AUC": M[kk]["pr_auc"],
                     "False alarms/day": M[kk]["false_alarms_per_day"], "Mean lead (h)": M[kk]["mean_lead_h"]})
        rows.append({"Model": MODEL_NAMES[kk], "Labels": "Failure window only", "F1": lc[kk]["f1"], "PR-AUC": lc[kk]["pr_auc"],
                     "False alarms/day": lc[kk]["false_alarms_per_day"], "Mean lead (h)": lc[kk]["mean_lead_h"]})
    lab = pd.DataFrame(rows)
    fig = go.Figure()
    for lbl, colr in ((f"Failure window + {PRE_FAILURE_HOURS} h before", COLORS["accent"]), ("Failure window only", COLORS["accent2"])):
        d = lab[lab.Labels == lbl]
        fig.add_trace(go.Bar(x=d.Model, y=d["PR-AUC"], name=lbl, marker_color=colr, text=[f"{v:.3f}" for v in d["PR-AUC"]], textposition="outside"))
    st.plotly_chart(base_layout(fig, 340, "PR-AUC by labelling strategy").update_layout(barmode="group", hovermode="closest"), width="stretch")
    st.dataframe(lab, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.3f") for c in ["F1", "PR-AUC", "False alarms/day"]} |
                               {"Mean lead (h)": st.column_config.NumberColumn(format="%.1f")})
    note(f"Adding a {PRE_FAILURE_HOURS}-hour warning label teaches the model to react <i>before</i> a leak is reported, "
         "trading some precision for earlier warnings. Each strategy is scored against its own labels.")

with st.expander("Reverse check: train on Jun–Sep, test on Feb–May"):
    rr = pd.DataFrame({MODEL_NAMES[k_]: {"PR-AUC": M[k_]["backward"]["pr_auc"], "F1": M[k_]["backward"]["f1"],
                                         "False alarms/day": M[k_]["backward"]["false_alarms_per_day"],
                                         "Failures caught": f"{M[k_]['backward']['failures_detected']}/{M[k_]['backward']['failures_total']}"} for k_ in M}).T
    st.dataframe(rr, width="stretch")
    st.caption("Not a fair forecast (it trains on the future), but it shows whether results hold when the failures swap roles.")

if "hybrid" in M:
    section("Why a hybrid model?")
    note("The three supervised models only know what failures 1–2 looked like: DV_pressure jumping to about 2 bar while the "
         "compressor runs nonstop. <b>Failure 4 was different</b> — DV_pressure stayed near zero and the low-pressure switch fired — "
         "so they missed it. Isolation Forest catches it because it flags anything unusual, but it raises more false alarms. "
         "The hybrid takes the higher of Logistic Regression and Isolation Forest so a known leak signature <i>or</i> a new, "
         "unusual pattern can trigger an alert.")
    note(f"Honesty note: the hybrid was added <b>after</b> we saw the single models' test results, so its test scores are "
         f"slightly optimistic. In the reverse check (train Jun–Sep, test Feb–May) it caught "
         f"{M['hybrid']['backward']['failures_detected']}/{M['hybrid']['backward']['failures_total']} failures with "
         f"PR-AUC {M['hybrid']['backward']['pr_auc']:.3f}.", warn=True)

section("Limits of this evaluation")
note("MetroPT-3 contains only <b>four failures, all air leaks, from a single train</b>. The test period holds just two of them, "
     "so every metric here rests on very few events and could change a lot with more data. Results are a <b>proof of concept</b>: "
     "the method could be retrained on Malaysian LRT/MRT sensor logs, but it is not ready to make safety decisions on its own.", warn=True)
st.caption(f"Train: {mt['train_minutes']:,} minutes ({mt['train_failure_minutes']:,} labelled failure) · Test: {mt['test_minutes']:,} minutes "
           f"({mt['test_failure_minutes']:,} labelled failure) · {mt['n_features']} engineered features · split date {mt['split_date']}")
footer()
