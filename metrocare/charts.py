"""Plotly chart helpers with one consistent look."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import FAILURES, MODEL_NAMES, SENSOR_INFO
from .ui import COLORS

MODEL_COLORS = {"logreg": "#94A3B8", "rf": "#22D3EE", "gb": "#818CF8", "iforest": "#F59E0B", "hybrid": "#F472B6"}
SIGNAL_PALETTE = ["#22D3EE", "#818CF8", "#34D399", "#F472B6", "#FBBF24", "#60A5FA", "#A78BFA", "#2DD4BF"]


def base_layout(fig: go.Figure, height: int = 380, title: str | None = None, legend: bool = True) -> go.Figure:
    if title:
        fig.update_layout(title=dict(text=title, x=0, y=0.985, yref="container", yanchor="top",
                                     font=dict(size=15, color=COLORS["text"])))
    top = (34 if title else 8) + (30 if legend else 0)
    fig.update_layout(
        height=height + (30 if title and legend else 0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=COLORS["text"], size=12),
        margin=dict(l=10, r=10, t=top, b=10), hovermode="x unified",
        showlegend=legend, legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=COLORS["panel2"], bordercolor=COLORS["border"], font=dict(color=COLORS["text"])),
    )
    fig.update_xaxes(gridcolor="rgba(139,155,184,.12)", zeroline=False, linecolor=COLORS["border"])
    fig.update_yaxes(gridcolor="rgba(139,155,184,.12)", zeroline=False, linecolor=COLORS["border"])
    return fig


def shade_failures(fig: go.Figure, start=None, end=None, label: bool = True, pre_hours: float = 0):
    for f in FAILURES.itertuples():
        if start is not None and (f.end < pd.Timestamp(start) or f.start > pd.Timestamp(end)):
            continue
        if pre_hours:
            fig.add_vrect(x0=f.start - pd.Timedelta(hours=pre_hours), x1=f.start, fillcolor=COLORS["watch"], opacity=0.10, line_width=0, layer="below")
        extra = dict(annotation_text=f"Failure {f.id}", annotation_position="top left",
                     annotation_font=dict(color=COLORS["failure"], size=11)) if label else {}
        fig.add_vrect(x0=f.start, x1=f.end, fillcolor=COLORS["failure"], opacity=0.18, line_width=0, layer="below", **extra)
    return fig


def gauge(risk: float, watch: float, alert: float) -> go.Figure:
    color = COLORS["alert"] if risk >= alert else COLORS["watch"] if risk >= watch else COLORS["normal"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=risk * 100, number=dict(suffix="%", font=dict(size=40, color=COLORS["text"])),
        title=dict(text="Failure risk", font=dict(size=13, color=COLORS["muted"])),
        gauge=dict(axis=dict(range=[0, 100], tickcolor=COLORS["muted"], tickfont=dict(size=10)),
                   bar=dict(color=color, thickness=0.28), bgcolor="rgba(0,0,0,0)", borderwidth=0,
                   steps=[dict(range=[0, watch * 100], color="rgba(34,197,94,.14)"),
                          dict(range=[watch * 100, alert * 100], color="rgba(245,158,11,.16)"),
                          dict(range=[alert * 100, 100], color="rgba(239,68,68,.18)")],
                   threshold=dict(line=dict(color=COLORS["text"], width=2), value=alert * 100))))
    return base_layout(fig, height=250, legend=False).update_layout(margin=dict(l=30, r=30, t=50, b=20))


def risk_chart(scores: pd.DataFrame, models: list[str], watch: float, alert: float, height: int = 360,
               marker: pd.Timestamp | None = None, title: str | None = None) -> go.Figure:
    fig = go.Figure()
    for k in models:
        col = f"risk_{k}"
        if col not in scores:
            continue
        fig.add_trace(go.Scattergl(x=scores.index, y=scores[col], name=MODEL_NAMES[k], mode="lines",
                                   line=dict(width=1.8 if len(models) == 1 else 1.4, color=MODEL_COLORS[k]),
                                   fill="tozeroy" if len(models) == 1 else None, fillcolor="rgba(34,211,238,.08)"))
    fig.add_hline(y=alert, line=dict(color=COLORS["alert"], dash="dash", width=1.2), annotation_text="Alert", annotation_font_color=COLORS["alert"])
    fig.add_hline(y=watch, line=dict(color=COLORS["watch"], dash="dot", width=1.2), annotation_text="Watch", annotation_font_color=COLORS["watch"])
    if len(scores):
        shade_failures(fig, scores.index.min(), scores.index.max())
    if marker is not None:
        fig.add_vline(x=marker, line=dict(color=COLORS["text"], width=1.5))
    fig.update_yaxes(range=[0, 1.02], title="Risk score")
    return base_layout(fig, height, title, legend=len(models) > 1)


def signals_chart(df: pd.DataFrame, signals: list[str], height_per: int = 150, marker=None) -> go.Figure:
    fig = make_subplots(rows=len(signals), cols=1, shared_xaxes=True, vertical_spacing=0.025)
    for i, s in enumerate(signals, 1):
        unit = SENSOR_INFO.get(s, ("", ""))[0]
        fig.add_trace(go.Scattergl(x=df.index, y=df[s], name=s, mode="lines",
                                   line=dict(width=1.3, color=SIGNAL_PALETTE[(i - 1) % len(SIGNAL_PALETTE)]),
                                   hovertemplate=f"{s}: %{{y:.3f}} {unit}<extra></extra>"), row=i, col=1)
        fig.update_yaxes(title_text=f"{s}<br><span style='font-size:10px'>{unit}</span>", title_font=dict(size=11), row=i, col=1)
    if len(df):
        for f in FAILURES.itertuples():
            if f.end < df.index.min() or f.start > df.index.max():
                continue
            fig.add_vrect(x0=f.start, x1=f.end, fillcolor=COLORS["failure"], opacity=0.18, line_width=0, layer="below", row="all", col=1)
    if marker is not None:
        fig.add_vline(x=marker, line=dict(color=COLORS["text"], width=1.2))
    return base_layout(fig, max(260, height_per * len(signals)), legend=False)


def timeline_strip(daily: pd.DataFrame, col: str, marker=None, title: str = "") -> go.Figure:
    fig = go.Figure(go.Bar(x=daily.index, y=daily[col], marker=dict(color=daily[col], colorscale=[[0, "#14532D"], [0.4, "#22C55E"], [0.6, "#F59E0B"], [1, "#EF4444"]], cmin=0, cmax=1),
                           hovertemplate="%{x|%d %b %Y}<br>Peak risk %{y:.2f}<extra></extra>"))
    shade_failures(fig, label=True)
    if marker is not None:
        fig.add_vline(x=marker, line=dict(color=COLORS["text"], width=1.5))
    fig.update_yaxes(range=[0, 1.05], title="Daily peak risk")
    return base_layout(fig, 200, title, legend=False)


def contribution_bar(contrib: pd.Series, title: str = "", top: int = 12) -> go.Figure:
    c = contrib.reindex(contrib.abs().sort_values(ascending=False).index)[:top][::-1]
    colors = [COLORS["alert"] if v > 0 else COLORS["normal"] for v in c]
    fig = go.Figure(go.Bar(x=c.values, y=c.index, orientation="h", marker_color=colors,
                           hovertemplate="%{y}: %{x:+.4f}<extra></extra>", text=[f"{v:+.3f}" for v in c.values], textposition="outside"))
    fig.add_vline(x=0, line=dict(color=COLORS["muted"], width=1))
    return base_layout(fig, max(300, 28 * len(c) + 60), title, legend=False).update_layout(hovermode="closest")


def waterfall(base: float, contrib: pd.Series, final: float, top: int = 10) -> go.Figure:
    c = contrib.reindex(contrib.abs().sort_values(ascending=False).index)
    head, rest = c[:top], c[top:].sum()
    labels = ["Typical (baseline)"] + list(head.index) + ([f"{len(c) - top} other features"] if len(c) > top else []) + ["This moment"]
    vals = [base] + list(head.values) + ([rest] if len(c) > top else []) + [final]
    measure = ["absolute"] + ["relative"] * (len(vals) - 2) + ["total"]
    fig = go.Figure(go.Waterfall(orientation="v", x=labels, y=vals, measure=measure,
                                 increasing=dict(marker=dict(color=COLORS["alert"])), decreasing=dict(marker=dict(color=COLORS["normal"])),
                                 totals=dict(marker=dict(color=COLORS["accent2"])), connector=dict(line=dict(color=COLORS["border"])),
                                 text=[f"{v:+.3f}" if m == "relative" else f"{v:.3f}" for v, m in zip(vals, measure)], textposition="outside"))
    fig.update_xaxes(tickangle=-35)
    return base_layout(fig, 430, "How each feature moved the model output (SHAP)", legend=False).update_layout(hovermode="closest")


def heatmap(z: pd.DataFrame, title: str = "", zmin=-1, zmax=1, colorscale="RdBu_r", fmt=".2f", height=460) -> go.Figure:
    fig = go.Figure(go.Heatmap(z=z.values, x=list(z.columns), y=list(z.index), zmin=zmin, zmax=zmax, colorscale=colorscale,
                               text=np.vectorize(lambda v: format(v, fmt))(z.values), texttemplate="%{text}", textfont=dict(size=9),
                               hovertemplate="%{y} × %{x}: %{z:.3f}<extra></extra>", colorbar=dict(thickness=10)))
    return base_layout(fig, height, title, legend=False).update_layout(hovermode="closest")
