import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from metrocare.charts import base_layout, contribution_bar, waterfall
from metrocare.config import FAILURES, MODEL_DIR, MODEL_NAMES
from metrocare.data import features_at, fold_for, load_metrics, load_model, load_scores
from metrocare.features import feature_signal, pretty_feature
from metrocare.ui import COLORS, footer, hero, note, section

scores, metrics = load_scores(), load_metrics()
sel = st.session_state.model
col = f"risk_{sel}"

hero("Investigate", "Why this alert?",
     "SHAP breaks a single prediction into the push each sensor feature gave it, so maintenance staff can see the reason "
     "behind a warning instead of trusting a black box.")

opts = {}
for f in FAILURES.itertuples():
    w = scores.loc[f.start - pd.Timedelta(hours=24): f.end, col]
    opts[f"Highest risk around failure {f.id}"] = w.idxmax()
normal = scores.loc[(scores.index.hour >= 10) & (scores.index.hour < 16), col]
opts["A calm, normal moment"] = normal.idxmin()
opts["Pick a moment myself"] = None

c1, c2 = st.columns([1.3, 1])
choice = c1.selectbox("Moment to explain", list(opts), index=3)
if opts[choice] is None:
    t = c2.slider("Moment", scores.index.min().to_pydatetime(), scores.index.max().to_pydatetime(),
                  value=FAILURES.iloc[3].start.to_pydatetime(), step=pd.Timedelta(minutes=10).to_pytimedelta(), format="DD MMM YYYY HH:mm")
    t = pd.Timestamp(t)
    t = scores.index[max(scores.index.get_indexer([t], method="ffill")[0], 0)]
else:
    t = opts[choice]
    c2.metric("Moment", f"{t:%d %b %Y, %H:%M}", f"{MODEL_NAMES[sel]} risk {scores.at[t, col]:.2f}", delta_color="off")

# Which model's reasoning to show: the selected one, or for the hybrid, whichever part produced the risk
if sel == "hybrid":
    lr_r, if_r = float(scores.at[t, "risk_logreg"]), float(scores.at[t, "risk_iforest"])
    xkey = "logreg" if lr_r >= if_r else "iforest"
    driver = ("the <b>supervised</b> part — the pattern matches a known leak signature" if xkey == "logreg" else
              "the <b>anomaly</b> part — the pattern is unusual even though it does not match past leaks")
    note(f"Hybrid risk is the higher of two parts: Logistic Regression <b>{lr_r:.2f}</b> and Isolation Forest <b>{if_r:.2f}</b>. "
         f"Here it comes from {driver}, so SHAP below explains <b>{MODEL_NAMES[xkey]}</b>.")
else:
    xkey = sel
if xkey == "iforest":
    st.caption("Isolation Forest output = how easy this minute is to isolate from normal data. Red bars made the minute look more unusual.")

X = features_at(t)
if X.empty:
    st.warning("Not enough history before this moment to build features (needs about an hour of data).")
    st.stop()
fold = fold_for(t)
model = load_model(xkey, fold)


@st.cache_resource(show_spinner=False)
def explainer(key, fold_):
    import shap
    m_ = load_model(key, fold_)
    return shap.TreeExplainer(m_.pipe[-1] if key == "iforest" else m_)


def explain_row(X):
    if xkey == "logreg":
        sc, lr = model[0], model[-1]
        z = (X.to_numpy()[0] - sc.mean_) / sc.scale_
        contrib = pd.Series(lr.coef_[0] * z, index=X.columns)
        return float(lr.intercept_[0]), contrib, "log-odds"
    try:
        ex = explainer(xkey, fold)
        if xkey == "iforest":  # SHAP explains average isolation depth; shorter depth = more unusual, so flip the sign
            Z = pd.DataFrame(model.pipe[0].transform(X), columns=X.columns)
            sv = ex.shap_values(Z)
            return -float(np.atleast_1d(ex.expected_value)[0]), pd.Series(-np.asarray(sv)[0], index=X.columns), "unusualness (negative isolation depth)"
        sv = ex.shap_values(X)
        ev = ex.expected_value
        if isinstance(sv, list):
            sv, ev = sv[1], ev[1]
        elif np.ndim(sv) == 3:
            sv, ev = sv[..., 1], np.atleast_1d(ev)[1]
        return float(np.atleast_1d(ev)[0]), pd.Series(np.asarray(sv)[0], index=X.columns), ("probability" if xkey == "rf" else "log-odds")
    except Exception as e:  # SHAP missing → fall back to occlusion
        st.caption(f"SHAP unavailable ({e}); using a feature-occlusion approximation.")
        base_row = pd.read_parquet(MODEL_DIR / "shap_sample_X.parquet").median() if (MODEL_DIR / "shap_sample_X.parquet").exists() else X.iloc[0]
        f_ = (lambda d: model.risk(d)[0]) if xkey == "iforest" else (lambda d: model.predict_proba(d)[0, 1])
        p0 = f_(X)
        contrib = {}
        for c in X.columns:
            Xc = X.copy(); Xc[c] = base_row[c]
            contrib[c] = p0 - f_(Xc)
        return float(p0 - sum(contrib.values())), pd.Series(contrib), "risk"


with st.spinner("Explaining prediction…"):
    base, contrib, space = explain_row(X)
final = base + contrib.sum()
contrib_pretty = contrib.rename(pretty_feature)
by_signal = contrib.groupby(feature_signal).sum().sort_values(key=abs, ascending=False)

# Plain-language summary
typical = pd.read_parquet(MODEL_DIR / "shap_sample_X.parquet").median() if (MODEL_DIR / "shap_sample_X.parquet").exists() else None
ups = contrib.sort_values(ascending=False)
ups = ups[ups > 0][:3]
risk_now = float(scores.at[t, col])
if len(ups) and risk_now >= st.session_state.watch:
    parts = []
    for feat in ups.index:
        v = X.iloc[0][feat]
        tv = f" (typically {typical[feat]:.2f})" if typical is not None else ""
        parts.append(f"<b>{pretty_feature(feat)}</b> = {v:.2f}{tv}")
    note(f"🔎 At {t:%d %b %H:%M} the risk is <b>{risk_now:.2f}</b>. The biggest reasons: " + "; ".join(parts) + ".")
else:
    note(f"✅ At {t:%d %b %H:%M} the risk is <b>{risk_now:.2f}</b> — the features look like normal operation, "
         f"and most pushes are towards <i>lower</i> risk (green).")

c1, c2 = st.columns([1.5, 1])
with c1:
    st.plotly_chart(waterfall(base, contrib_pretty, final, top=10), width="stretch")
    st.caption(f"Model output in {space} space. Red bars push risk up, green bars pull it down. "
               f"Scored by the {'forward (Feb–May)' if fold == 'fwd' else 'backward (Jun–Sep)'}-trained model.")
with c2:
    st.plotly_chart(contribution_bar(by_signal, "Total push per sensor", top=16), width="stretch")

@st.cache_data(show_spinner=False)
def global_importance(key):
    if key in metrics["importance"]:
        return metrics["importance"][key]
    # Isolation Forest: mean |SHAP| on a sample of test minutes
    import shap
    sx = pd.read_parquet(MODEL_DIR / "shap_sample_X.parquet").sample(300, random_state=0)
    ad = load_model(key, "fwd")
    sv = np.abs(shap.TreeExplainer(ad.pipe[-1]).shap_values(ad.pipe[0].transform(sx))).mean(0)
    return dict(zip(sx.columns, (sv / sv.sum()).tolist()))


section("What the model relies on overall")
c1, c2 = st.columns(2)
with c1:
    imp = pd.Series(global_importance(xkey)).groupby(feature_signal).sum().sort_values()
    fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=COLORS["accent"],
                           hovertemplate="%{y}: %{x:.1%}<extra></extra>"))
    fig.update_xaxes(tickformat=".0%")
    st.plotly_chart(base_layout(fig, 460, f"Feature importance by sensor · {MODEL_NAMES[xkey]}", legend=False).update_layout(hovermode="closest"), width="stretch")
with c2:
    sp = MODEL_DIR / "shap_sample.parquet"
    if sp.exists() and xkey == metrics.get("explain_model"):
        sv = pd.read_parquet(sp)
        sx = pd.read_parquet(MODEL_DIR / "shap_sample_X.parquet")
        top = sv.abs().mean().sort_values(ascending=False).index[:12][::-1]
        fig = go.Figure()
        for i, feat in enumerate(top):
            xv = sx[feat]
            norm = (xv - xv.min()) / (xv.max() - xv.min() + 1e-9)
            jitter = np.random.default_rng(i).uniform(-0.3, 0.3, len(xv))
            fig.add_trace(go.Scattergl(x=sv[feat], y=i + jitter, mode="markers", showlegend=False,
                                       marker=dict(size=3.5, color=norm, colorscale="Plasma", opacity=0.6,
                                                   colorbar=dict(title="feature<br>value", thickness=8, tickvals=[0, 1], ticktext=["low", "high"]) if i == 0 else None),
                                       hovertemplate=f"{pretty_feature(feat)}<br>SHAP %{{x:.3f}}<extra></extra>"))
        fig.update_yaxes(tickvals=list(range(len(top))), ticktext=[pretty_feature(f) for f in top])
        fig.add_vline(x=0, line=dict(color=COLORS["muted"], width=1))
        st.plotly_chart(base_layout(fig, 460, "SHAP summary (test-period sample)", legend=False).update_layout(hovermode="closest"), width="stretch")
        st.caption("Each dot is one minute. Dots to the right raised the risk; bright colours mean a high feature value.")
    else:
        top = pd.Series(global_importance(xkey)).sort_values(ascending=False)[:15][::-1]
        fig = go.Figure(go.Bar(x=top.values, y=[pretty_feature(f) for f in top.index], orientation="h", marker_color=COLORS["accent2"]))
        st.plotly_chart(base_layout(fig, 460, "Top individual features", legend=False), width="stretch")

with st.expander("Raw feature values at this moment"):
    st.dataframe(pd.DataFrame({"feature": [pretty_feature(c) for c in X.columns], "value": X.iloc[0].values,
                               "SHAP": contrib.values}).sort_values("SHAP", key=abs, ascending=False),
                 hide_index=True, width="stretch")
footer()
