"""
app/pages/page_home.py
───────────────────────
Vehicle review aspect sentiment page.

Responsibilities
----------------
- Accept a review from the user (text area or example buttons).
- Run ABSA through the inference engine.
- Display structured prediction cards with confidence and probability bars.
- Show a confidence bar chart.
"""

from __future__ import annotations

import streamlit as st

from app.ui_components import (
    section_header, divider, info_banner, prediction_card, metric_card
)
from src.data.validator import validate_review_text


# ── Example reviews ───────────────────────────────────────────────────────────
EXAMPLES = [
    "The battery range is excellent, but the charging time is too long.",
    "The mileage is impressive and the engine is very responsive, though the service at the dealership was disappointing.",
    "Comfortable seats and a smooth ride, but the infotainment system keeps lagging.",
    "Great value for money — this car offers safety features you'd normally only find in luxury vehicles.",
    "The vehicle looks stunning and the fuel efficiency is remarkable. However, the rear legroom is too cramped for a family.",
]


def render():
    # ── Header ──────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="padding:20px 0 10px;">
        <h1 style="color:#38bdf8;font-weight:800;font-size:2rem;margin:0;">
            Analyze vehicle reviews
        </h1>
        <p style="color:#94a3b8;margin:6px 0 0 0;">
            Identify sentiment for individual aspects of a vehicle review.
        </p>
    </div>
    """, unsafe_allow_html=True)

    divider()

    # ── Model check ──────────────────────────────────────────────────────────
    if not st.session_state.get("model_loaded"):
        info_banner(
            "No model is loaded. Use the sidebar to select and load a model before analysing reviews.",
            kind="warning",
        )

    # ── Input section ────────────────────────────────────────────────────────
    # Apply any pending example text BEFORE the widget is instantiated
    if "_review_input_pending" in st.session_state:
        st.session_state["single_review_input"] = st.session_state.pop("_review_input_pending")

    col_input, col_examples = st.columns([3, 1])

    with col_input:
        section_header("Vehicle review", "Type or paste a customer review below")
        review_text = st.text_area(
            "Review",
            height=140,
            placeholder="e.g. The battery range is excellent, but the charging time is too long.",
            label_visibility="collapsed",
            key="single_review_input",
        )

    with col_examples:
        section_header("Examples", "Click to load a sample review")
        for i, example in enumerate(EXAMPLES):
            short = example[:55] + "…" if len(example) > 55 else example
            if st.button(short, key=f"example_{i}", width="stretch"):
                # Store in a staging key — applied on the NEXT run before widget renders
                st.session_state["_review_input_pending"] = example
                st.rerun()

    # ── Analyse button ───────────────────────────────────────────────────────
    col_btn, _ = st.columns([1, 3])
    with col_btn:
        analyse_clicked = st.button(
            "🚀 Analyze review",
            width="stretch",
            disabled=not st.session_state.get("model_loaded"),
        )

    # ── Analysis ─────────────────────────────────────────────────────────────
    if analyse_clicked:
        text = st.session_state.get("single_review_input", "").strip()

        # Validate
        val = validate_review_text(text)
        if not val.is_valid:
            for err in val.errors:
                info_banner(err, kind="error")
            return

        for warn in val.warnings:
            info_banner(warn, kind="warning")

        engine = st.session_state.get("engine")
        if engine is None:
            info_banner("Engine not initialised. Please reload the model.", kind="error")
            return

        with st.spinner("Analysing review…"):
            try:
                result = engine.analyse_review(text)
                st.session_state["single_result"] = result
            except Exception as exc:
                info_banner(f"Analysis failed: {exc}", kind="error")
                return

    # ── Display results ───────────────────────────────────────────────────────
    result = st.session_state.get("single_result")
    if result is None:
        return

    divider()
    section_header("Aspect sentiment results")

    if result.no_aspects_detected:
        info_banner(
            "No recognisable automotive aspects were detected in this review. "
            "Try rephrasing or including more specific automotive terms.",
            kind="warning",
        )
        return

    # Summary metrics row
    preds = result.predictions
    sentiments = [p.sentiment for p in preds]
    pos_count = sentiments.count("Positive")
    neg_count = sentiments.count("Negative")
    neu_count = sentiments.count("Neutral")
    avg_conf = sum(p.confidence for p in preds) / len(preds) if preds else 0

    cols = st.columns(4)
    metric_data = [
        ("Aspects Detected", str(len(preds)), None, "#38bdf8"),
        ("Positive", str(pos_count), None, "#22c55e"),
        ("Negative", str(neg_count), None, "#ef4444"),
        ("Avg Confidence", f"{avg_conf:.1%}", None, "#f59e0b"),
    ]
    for col, (label, value, delta, color) in zip(cols, metric_data):
        with col:
            st.markdown(metric_card(label, value, delta, color), unsafe_allow_html=True)

    st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)

    # Prediction cards
    section_header("Aspect-Sentiment Breakdown")
    cols_per_row = 2
    rows = [preds[i:i+cols_per_row] for i in range(0, len(preds), cols_per_row)]
    for row in rows:
        cols = st.columns(len(row))
        for col, pred in zip(cols, row):
            with col:
                prediction_card(
                    aspect=pred.aspect,
                    sentiment=pred.sentiment,
                    confidence=pred.confidence,
                    probs=pred.probabilities,
                    keywords=pred.triggered_keywords,
                )

    # Confidence chart
    divider()
    section_header("Confidence Overview")
    from src.analytics.visualizer import plot_confidence_bar
    conf_df = result.as_dataframe
    fig = plot_confidence_bar(conf_df)
    st.plotly_chart(fig, width="stretch")

    # Raw results table
    with st.expander("📋 Raw Results Table"):
        st.dataframe(conf_df, width="stretch")
