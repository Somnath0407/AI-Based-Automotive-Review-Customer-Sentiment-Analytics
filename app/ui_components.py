"""
app/ui_components.py
─────────────────────
Reusable Streamlit UI components and helpers.
Keeps the page files clean and focused on page logic.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import streamlit as st

# ── Sentiment badge renderer ──────────────────────────────────────────────────

SENTIMENT_CONFIG = {
    "Positive": {"emoji": "✅", "color": "#22c55e", "bg": "#052e16"},
    "Negative": {"emoji": "❌", "color": "#ef4444", "bg": "#2d0a0a"},
    "Neutral":  {"emoji": "⚖️", "color": "#f59e0b", "bg": "#2d1f00"},
}


def sentiment_badge(sentiment: str) -> str:
    """Return an HTML sentiment badge string."""
    cfg = SENTIMENT_CONFIG.get(sentiment, {"emoji": "❓", "color": "#64748b", "bg": "#1e293b"})
    return (
        f'<span style="background:{cfg["bg"]};color:{cfg["color"]};'
        f'border:1px solid {cfg["color"]};border-radius:20px;'
        f'padding:2px 12px;font-weight:600;font-size:0.85rem;">'
        f'{cfg["emoji"]} {sentiment}</span>'
    )


def confidence_bar_html(confidence: float, sentiment: str) -> str:
    """Return an HTML confidence progress bar."""
    pct = int(confidence * 100)
    color = SENTIMENT_CONFIG.get(sentiment, {}).get("color", "#64748b")
    return (
        f'<div style="background:#1e293b;border-radius:6px;height:8px;width:100%;">'
        f'<div style="width:{pct}%;background:{color};height:100%;border-radius:6px;'
        f'transition:width 0.4s ease;"></div></div>'
        f'<small style="color:#94a3b8;">{pct}% confidence</small>'
    )


def metric_card(label: str, value: str, delta: Optional[str] = None,
                color: str = "#3b82f6") -> str:
    """Return an HTML metric card."""
    delta_html = ""
    if delta:
        delta_html = f'<div style="color:#94a3b8;font-size:0.75rem;margin-top:4px;">{delta}</div>'
    return (
        f'<div style="background:#1e293b;border:1px solid #334155;border-radius:12px;'
        f'padding:16px;text-align:center;">'
        f'<div style="color:#94a3b8;font-size:0.8rem;margin-bottom:4px;">{label}</div>'
        f'<div style="color:{color};font-size:1.8rem;font-weight:700;">{value}</div>'
        f'{delta_html}</div>'
    )


def section_header(title: str, subtitle: str = "") -> None:
    """Render a stylised section header."""
    st.markdown(
        f'<div style="margin-bottom:1rem;">'
        f'<h3 style="color:#e2e8f0;margin:0;font-weight:700;">{title}</h3>'
        + (f'<p style="color:#94a3b8;margin:4px 0 0 0;font-size:0.9rem;">{subtitle}</p>'
           if subtitle else "")
        + "</div>",
        unsafe_allow_html=True,
    )


def divider() -> None:
    st.markdown(
        '<hr style="border:0;border-top:1px solid #334155;margin:1.5rem 0;">',
        unsafe_allow_html=True,
    )


def info_banner(message: str, kind: str = "info") -> None:
    """Render an info/warning/error banner."""
    colors = {
        "info":    ("#1d4ed8", "#1e3a5f"),
        "warning": ("#d97706", "#3d2500"),
        "error":   ("#dc2626", "#2d0a0a"),
        "success": ("#16a34a", "#052e16"),
    }
    border_color, bg_color = colors.get(kind, colors["info"])
    icons = {"info": "ℹ️", "warning": "⚠️", "error": "🚫", "success": "✅"}
    icon = icons.get(kind, "ℹ️")
    st.markdown(
        f'<div style="background:{bg_color};border-left:4px solid {border_color};'
        f'border-radius:8px;padding:12px 16px;margin:8px 0;">'
        f'{icon} <span style="color:#e2e8f0;">{message}</span></div>',
        unsafe_allow_html=True,
    )


def prediction_card(aspect: str, sentiment: str, confidence: float,
                    probs: Dict[str, float], keywords: List[str]) -> None:
    """Render a full prediction result card."""
    cfg = SENTIMENT_CONFIG.get(sentiment, {"emoji": "❓", "color": "#64748b", "bg": "#1e293b"})
    pct = int(confidence * 100)

    kw_html = ""
    if keywords:
        kw_chips = " ".join(
            f'<span style="background:#0f172a;color:#94a3b8;border:1px solid #334155;'
            f'border-radius:12px;padding:2px 8px;font-size:0.75rem;">{kw}</span>'
            for kw in keywords
        )
        kw_html = f'<div style="margin-top:8px;">{kw_chips}</div>'

    prob_bars = ""
    for label, prob in [("Positive", probs.get("Positive", 0)),
                         ("Negative", probs.get("Negative", 0)),
                         ("Neutral", probs.get("Neutral", 0))]:
        lc = SENTIMENT_CONFIG.get(label, {}).get("color", "#64748b")
        w = int(prob * 100)
        prob_bars += (
            f'<div style="display:flex;align-items:center;gap:8px;margin-top:4px;">'
            f'<span style="color:#94a3b8;font-size:0.75rem;min-width:60px;">{label}</span>'
            f'<div style="flex:1;background:#0f172a;border-radius:4px;height:6px;">'
            f'<div style="width:{w}%;background:{lc};height:100%;border-radius:4px;"></div></div>'
            f'<span style="color:#94a3b8;font-size:0.75rem;min-width:35px;">{prob:.1%}</span>'
            f'</div>'
        )

    st.markdown(
        f'<div style="background:{cfg["bg"]};border:1px solid {cfg["color"]}40;'
        f'border-left:4px solid {cfg["color"]};border-radius:12px;padding:16px;margin:8px 0;">'
        f'<div style="display:flex;justify-content:space-between;align-items:center;">'
        f'<span style="color:#e2e8f0;font-weight:700;font-size:1.05rem;">🔍 {aspect}</span>'
        f'<span style="background:{cfg["bg"]};color:{cfg["color"]};border:1px solid {cfg["color"]};'
        f'border-radius:20px;padding:3px 14px;font-weight:600;">{cfg["emoji"]} {sentiment}</span>'
        f'</div>'
        f'<div style="margin-top:10px;">'
        f'<div style="background:#0f172a;border-radius:6px;height:8px;">'
        f'<div style="width:{pct}%;background:{cfg["color"]};height:100%;border-radius:6px;"></div></div>'
        f'<small style="color:#94a3b8;">{pct}% confidence</small>'
        f'</div>'
        f'<div style="margin-top:10px;">{prob_bars}</div>'
        f'{kw_html}'
        f'</div>',
        unsafe_allow_html=True,
    )
