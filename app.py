"""MetroCare — predictive maintenance dashboard for metro train air compressors.

Run:  streamlit run app.py
"""
import streamlit as st

from metrocare.config import MODEL_NAMES, ROOT
from metrocare.data import artifacts_ready, load_metrics
from metrocare.ui import COLORS, inject_css

st.set_page_config(page_title="MetroCare · APU Predictive Maintenance", page_icon="🚇", layout="wide",
                   initial_sidebar_state="expanded")
inject_css()

if not artifacts_ready():
    st.error("Model artifacts not found. Run the pipeline first:")
    st.code("python prepare_data.py\npython train_models.py", language="bash")
    st.stop()

metrics = load_metrics()

st.logo(str(ROOT / "assets" / "logo.svg"), size="large")

pages = {
    "Monitor": [
        st.Page("views/overview.py", title="Overview", icon=":material/dashboard:", default=True),
        st.Page("views/explorer.py", title="Sensor Explorer", icon=":material/show_chart:"),
        st.Page("views/risk.py", title="Failure Risk", icon=":material/warning:"),
    ],
    "Investigate": [
        st.Page("views/review.py", title="Past Failure Review", icon=":material/history:"),
        st.Page("views/explain.py", title="Why this alert?", icon=":material/psychology:"),
        st.Page("views/performance.py", title="Model Performance", icon=":material/leaderboard:"),
    ],
    "Tools": [
        st.Page("views/upload.py", title="Upload & Predict", icon=":material/upload_file:"),
        st.Page("views/about.py", title="About & Data", icon=":material/info:"),
    ],
}
nav = st.navigation(pages)

with st.sidebar:
    st.divider()
    st.markdown(f"<div style='color:{COLORS['muted']};font-size:.8rem;margin-bottom:4px'>Air compressor health monitor · MetroPT-3</div>",
                unsafe_allow_html=True)
    st.markdown("##### Alert settings")
    keys = list(MODEL_NAMES)
    best = metrics.get("best_model", "rf")
    st.selectbox("Prediction model", keys, index=keys.index(best), key="model",
                 format_func=lambda k: MODEL_NAMES[k] + ("  ★ best" if k == best else ""),
                 help="★ marks the model that caught the most test failures with the highest PR-AUC.")
    st.slider("Watch threshold", 0.05, 0.95, 0.30, 0.05, key="watch", help="Risk at which the status turns amber.")
    st.slider("Alert threshold", 0.05, 0.95, 0.50, 0.05, key="alert", help="Risk at which the status turns red and an alert is raised.")
    if st.session_state.watch >= st.session_state.alert:
        st.warning("Watch threshold should be below the alert threshold.")
    st.caption("Risk scores are smoothed over 15 minutes. Each minute is scored by a model that never saw it during training.")

nav.run()
