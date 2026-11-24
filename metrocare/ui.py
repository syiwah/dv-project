"""Visual building blocks: theme CSS, header, KPI cards, status pill, callouts."""
from __future__ import annotations

import html

import streamlit as st

COLORS = {
    "bg": "#0B1220", "panel": "#111A2E", "panel2": "#162238", "border": "#22314F",
    "text": "#E6EBF5", "muted": "#8B9BB8", "accent": "#22D3EE", "accent2": "#818CF8",
    "normal": "#22C55E", "watch": "#F59E0B", "alert": "#EF4444", "failure": "#F43F5E",
}
STATUS_COLOR = {"NORMAL": COLORS["normal"], "WATCH": COLORS["watch"], "ALERT": COLORS["alert"]}
STATUS_TEXT = {
    "NORMAL": "Compressor behaving normally — no action needed.",
    "WATCH": "Unusual pattern building — check at the next maintenance window.",
    "ALERT": "High failure risk — inspect the APU for air leaks as soon as possible.",
}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500&display=swap');
html, body, [class*="css"], .stMarkdown, .stText {{ font-family: 'Inter', sans-serif; }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1400px; }}
[data-testid="stSidebar"] {{ background: linear-gradient(180deg, #0E1730 0%, #0B1220 100%); border-right: 1px solid {COLORS['border']}; }}
h1, h2, h3 {{ letter-spacing: -0.02em; }}
.mc-hero {{ border: 1px solid {COLORS['border']}; border-radius: 18px; padding: 22px 26px; margin-bottom: 18px;
  background: radial-gradient(1200px 200px at 0% 0%, rgba(34,211,238,.16), transparent 60%),
              radial-gradient(800px 200px at 100% 0%, rgba(129,140,248,.14), transparent 60%), {COLORS['panel']}; }}
.mc-hero .eyebrow {{ color: {COLORS['accent']}; font-weight: 700; font-size: .78rem; letter-spacing: .14em; text-transform: uppercase; }}
.mc-hero h1 {{ margin: .2rem 0 .35rem 0; font-size: 2.0rem; font-weight: 800; color: {COLORS['text']}; }}
.mc-hero p {{ color: {COLORS['muted']}; margin: 0; font-size: 1rem; max-width: 900px; }}
.mc-card {{ border: 1px solid {COLORS['border']}; border-radius: 14px; padding: 14px 16px; background: {COLORS['panel']}; height: 100%; }}
.mc-card .label {{ color: {COLORS['muted']}; font-size: .78rem; font-weight: 600; text-transform: uppercase; letter-spacing: .08em; }}
.mc-card .value {{ color: {COLORS['text']}; font-size: 1.65rem; font-weight: 800; margin-top: 4px; font-variant-numeric: tabular-nums; }}
.mc-card .sub {{ color: {COLORS['muted']}; font-size: .82rem; margin-top: 2px; }}
.mc-card .up {{ color: {COLORS['alert']}; }} .mc-card .down {{ color: {COLORS['normal']}; }}
.mc-status {{ border-radius: 16px; padding: 18px 20px; border: 1px solid; }}
.mc-status .big {{ font-size: 2.1rem; font-weight: 800; letter-spacing: .02em; }}
.mc-status .msg {{ font-size: .95rem; margin-top: 4px; color: {COLORS['text']}; }}
.mc-pill {{ display:inline-block; padding: 3px 10px; border-radius: 999px; font-weight: 700; font-size: .75rem; letter-spacing: .06em; }}
.mc-note {{ border-left: 3px solid {COLORS['accent']}; background: rgba(34,211,238,.06); padding: 10px 14px; border-radius: 0 10px 10px 0; color: {COLORS['text']}; font-size: .92rem; margin: 6px 0 14px 0; }}
.mc-note.warn {{ border-left-color: {COLORS['watch']}; background: rgba(245,158,11,.07); }}
.mc-section {{ font-weight: 700; font-size: 1.05rem; color: {COLORS['text']}; margin: 18px 0 6px 0; }}
.mc-mono {{ font-family: 'JetBrains Mono', monospace; }}
.mc-footer {{ color: {COLORS['muted']}; font-size: .78rem; text-align: center; margin-top: 40px; }}
div[data-testid="stMetric"] {{ background: {COLORS['panel']}; border: 1px solid {COLORS['border']}; border-radius: 14px; padding: 12px 16px; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 6px; }}
.stTabs [data-baseweb="tab"] {{ background: {COLORS['panel']}; border-radius: 10px 10px 0 0; padding: 8px 16px; }}
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def hero(eyebrow: str, title: str, text: str):
    st.markdown(f"<div class='mc-hero'><div class='eyebrow'>{eyebrow}</div><h1>{title}</h1><p>{text}</p></div>",
                unsafe_allow_html=True)


def card(label: str, value: str, sub: str = "", delta: float | None = None, bad_when_up: bool = True, fmt: str = "{:+.2f}"):
    d = ""
    if delta is not None:
        cls = ("up" if delta > 0 else "down") if bad_when_up else ("down" if delta > 0 else "up")
        d = f" <span class='{cls}'>{fmt.format(delta)}</span>"
    st.markdown(f"<div class='mc-card'><div class='label'>{label}</div><div class='value'>{value}</div>"
                f"<div class='sub'>{sub}{d}</div></div>", unsafe_allow_html=True)


def status_box(state: str, risk: float):
    c = STATUS_COLOR[state]
    st.markdown(
        f"<div class='mc-status' style='border-color:{c}; background: linear-gradient(135deg, {c}26, transparent 70%);'>"
        f"<div class='label' style='color:{c};font-weight:700;font-size:.78rem;letter-spacing:.12em'>COMPRESSOR STATUS</div>"
        f"<div class='big' style='color:{c}'>● {state}</div>"
        f"<div class='msg'>{STATUS_TEXT[state]}</div>"
        f"<div class='msg' style='color:{COLORS['muted']}'>Risk score <b class='mc-mono' style='color:{COLORS['text']}'>{risk:.2f}</b></div></div>",
        unsafe_allow_html=True)


def pill(state: str) -> str:
    c = STATUS_COLOR.get(state, COLORS["muted"])
    return f"<span class='mc-pill' style='background:{c}22;color:{c};border:1px solid {c}55'>{html.escape(state)}</span>"


def note(text: str, warn: bool = False):
    st.markdown(f"<div class='mc-note{' warn' if warn else ''}'>{text}</div>", unsafe_allow_html=True)


def section(text: str):
    st.markdown(f"<div class='mc-section'>{text}</div>", unsafe_allow_html=True)


def footer():
    st.markdown("<div class='mc-footer'>MetroCare · Survivors.py · TFB3133/TEB3133 Data Visualization, UTP · "
                "Data: MetroPT-3 (Davari et al., 2021), CC BY 4.0 · Proof of concept, not a certified safety system</div>",
                unsafe_allow_html=True)
