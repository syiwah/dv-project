import pandas as pd
import streamlit as st

from metrocare.config import ANALOG, DIGITAL, FAILURES, SENSOR_INFO
from metrocare.data import load_metrics, load_quality
from metrocare.ui import COLORS, card, footer, hero, note, section

q, mt = load_quality(), load_metrics()

hero("About", "MetroCare — predicting air compressor failures in urban metro trains",
     "A Survivors.py project for TFB3133/TEB3133 Data Visualization, Universiti Teknologi PETRONAS, September 2026 semester.")

t1, t2, t3, t4 = st.tabs(["🎯 Project", "🛠️ Method", "🧪 Data quality", "👥 Team & references"])

with t1:
    c1, c2 = st.columns([1.3, 1])
    with c1:
        section("Why it matters")
        st.markdown(
            "- The Kelana Jaya LRT line carries around **250,000 passengers a day**; Prasarana recorded **71 rail disruptions in 2024** and 31 in 2025.\n"
            "- A metro train's **Air Production Unit (APU)** feeds the brakes and secondary suspension. At Metro do Porto it has no backup — "
            "when it fails the train leaves service, and **170+ trips were cancelled** for this reason in 2017.\n"
            "- Air leaks build up slowly and slip past fixed-threshold checks, so the train looks fine until it breaks.")
        section("Objectives and where to find them")
        st.markdown(
            "| # | Objective | In the app |\n|---|---|---|\n"
            "| 1 | Analyse & visualise historical sensor data — trends, relationships, abnormal patterns | **Sensor Explorer**, Overview |\n"
            "| 2 | Develop & evaluate ML models on detection, false alarms and lead time | **Model Performance**, Failure Risk, Past Failure Review |\n"
            "| 3 | Interactive dashboard for conditions, risk, past failures and explanations | **Every page** · Why this alert? · Upload & Predict |")
    with c2:
        section("Reported failures (ground truth)")
        f = FAILURES.copy()
        f["duration (h)"] = ((f.end - f.start).dt.total_seconds() / 3600).round(1)
        st.dataframe(f.rename(columns={"id": "#"}), hide_index=True, width="stretch")
        section("Sensors")
        st.dataframe(pd.DataFrame([{"Sensor": s, "Type": "Analogue" if s in ANALOG else ("Digital" if s in DIGITAL else "Derived"),
                                    "Unit": SENSOR_INFO[s][0], "Meaning": SENSOR_INFO[s][1]} for s in ANALOG + DIGITAL + ["duty_cycle"]]),
                     hide_index=True, width="stretch", height=350)

with t2:
    section("Pipeline")
    st.graphviz_chart(f"""
    digraph {{
      rankdir=LR; bgcolor="transparent";
      node [shape=box, style="rounded,filled", fillcolor="{COLORS['panel2']}", color="{COLORS['accent']}", fontcolor="{COLORS['text']}", fontname="Helvetica", fontsize=11];
      edge [color="{COLORS['muted']}"];
      a [label="MetroPT-3\\n1.5 M readings\\n15 sensors"]; b [label="Check & clean\\nduplicates, gaps,\\nspikes flagged"];
      c [label="1-minute averages\\n+ duty cycle"]; d [label="Label\\nfailure window\\n+ 2 h warning"];
      e [label="Features\\n10 min & 1 h\\nmean·std·min·max·trend"]; f [label="Models\\nLogReg · RF\\nGB · IsolationForest"];
      g [label="Chronological test\\nPR-AUC · F1 · FA/day\\nlead time"]; h [label="SHAP\\nexplanations"]; i [label="Streamlit\\ndashboard", fillcolor="{COLORS['accent']}", fontcolor="#0B1220"];
      a->b->c->d->e->f->g->i; f->h->i;
    }}""", width="stretch")
    st.markdown(
        f"- **Resampling:** readings are averaged per minute (short gaps ≤ 5 min forward-filled, longer gaps treated as the train being offline).\n"
        f"- **Duty cycle:** share of each minute the motor draws more than 1 A — a leak should push this up.\n"
        f"- **Features:** {mt['n_features']} rolling statistics over 10-minute and 1-hour windows for every sensor.\n"
        f"- **Class imbalance:** failures are rare, so models use balanced class weights; normal minutes are thinned 1-in-3 for training.\n"
        f"- **Fair testing:** train on data before **{mt['split_date']}**, test after it — never shuffled, so no peeking at the future. "
        "Scaling is fitted on training data only.\n"
        "- **Isolation Forest** learns only from normal minutes and needs no labels.\n"
        "- **Alerts:** risk is smoothed over 15 minutes; alerts less than 30 minutes apart count as one episode; alerts up to 24 h before a failure count as early warnings.\n"
        "- **Reproduce:** `python prepare_data.py && python train_models.py && streamlit run app.py`")

with t3:
    if q:
        k = st.columns(4)
        with k[0]: card("Raw readings", f"{q['raw_rows']:,}", q["file"])
        with k[1]: card("Missing values", f"{q['missing_values']:,}", f"{q['duplicate_rows']:,} duplicate rows removed")
        with k[2]: card("Sampling interval", f"{q['median_interval_s']:.0f} s", f"median gap · {q['gaps_over_5min']} gaps > 5 min")
        with k[3]: card("Minutes kept", f"{q['minutes_kept']:,}", f"{q['minutes_offline']:,} offline minutes dropped")
        st.markdown(f"**Period:** {q['start']} → {q['end']} · **Longest gap:** {q['longest_gap_h']} h · "
                    f"**Constant columns:** {', '.join(q['constant_columns']) or 'none'}")
        if q["negative_values"]:
            neg = pd.DataFrame(q["negative_values"]).T.rename(columns={"count": "negative readings", "min": "lowest value", "impossible": "below −0.1 (impossible)"})
            st.markdown("**Negative readings** — pressures slightly below zero while the compressor is idle are a sensor offset, not real values:")
            st.dataframe(neg, width="stretch")
        if q["median_interval_s"] > 2:
            note(f"The proposal describes the data as 1 Hz, but the file's median spacing between readings is "
                 f"<b>{q['median_interval_s']:.0f} seconds</b>. All features are built on 1-minute averages, so this does not affect the models.", warn=True)
        section("Spikes flagged (kept, not deleted — a spike can be a real warning sign)")
        st.bar_chart(pd.Series(q["spikes_flagged"]), color=COLORS["accent"], height=240)
        section("Sensor statistics")
        st.dataframe(pd.DataFrame(q["sensor_stats"]).set_index("sensor"), width="stretch")
    else:
        st.info("Run prepare_data.py to generate the data quality report.")

with t4:
    section("Survivors.py")
    st.dataframe(pd.DataFrame({
        "Name": ["Nur Fatihah binti Mohd Noor", "Nurin Akma Alya Muhammad Akmal", "Zulaikha binti Mohd Azhar",
                 "Insyirah binti Hamzah", "Dania Adriana binti Mohd Faizal"],
        "Student ID": ["24000227", "24006493", "24000918", "24000157", "22006373"], "Course": ["CS"] * 5}),
        hide_index=True, width="stretch")
    section("References")
    st.markdown(
        "- Davari, N., Veloso, B., Ribeiro, R., & Gama, J. (2021). *MetroPT-3 dataset* [Data set]. UCI Machine Learning Repository. https://doi.org/10.24432/C5VW3R\n"
        "- Malay Mail. (2026, May 18). *LRT disruption hits Kelana Jaya Line during peak hour after train breakdown at Kelana Jaya station.*\n"
        "- Newswav. (2026, January 7). *Prasarana sees 50% drop in service disruptions in 2025, committed to improving reliability.*\n"
        "- Veloso, B., Ribeiro, R. P., Gama, J., & Pereira, P. M. (2022). The MetroPT dataset for predictive maintenance. *Scientific Data, 9*, 764. https://doi.org/10.1038/s41597-022-01877-3")
footer()
