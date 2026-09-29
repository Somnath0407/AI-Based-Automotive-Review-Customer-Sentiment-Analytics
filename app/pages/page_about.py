"""
app/pages/page_about.py
────────────────────────
About / documentation page for AutoReview.
"""

from __future__ import annotations

import streamlit as st

from app.ui_components import section_header, divider


def render():
    st.markdown("""
    <div style="padding:20px 0 10px;">
        <h1 style="color:#64748b;font-weight:800;font-size:2rem;margin:0;">
            ℹ️ About AutoReview
        </h1>
        <p style="color:#94a3b8;margin:6px 0 0 0;">
            AI-Based Automotive Review &amp; Customer Sentiment Analytics
        </p>
    </div>
    """, unsafe_allow_html=True)

    divider()

    # ── Overview card ─────────────────────────────────────────────────────
    st.markdown("""
    <div style="background:#1e293b;border:1px solid #334155;border-radius:16px;padding:24px;margin-bottom:20px;">
        <h3 style="color:#38bdf8;margin:0 0 12px 0;">🚗 AutoReview</h3>
        <p style="color:#cbd5e1;margin:0;line-height:1.7;">
            AutoReview is an <b>Aspect-Based Sentiment Analysis (ABSA)</b> system
            purpose-built for automotive customer reviews. It uses fine-tuned
            <b>BERT</b> and <b>XLM-RoBERTa</b> transformer models to detect
            sentiment for individual vehicle aspects such as engine, battery,
            comfort, and service — all within a single review.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Feature table ─────────────────────────────────────────────────────
    section_header("Features")
    features = [
        ("🔍", "Single Review Analysis", "Enter any review → detect aspects → classify sentiment per aspect with confidence scores."),
        ("📊", "Batch Analytics", "Upload CSV/JSON datasets or use synthetic data. Visualise sentiment distribution, aspect frequency, and heatmaps."),
        ("🎓", "Model Training", "Fine-tune BERT or XLM-RoBERTa on your own annotated data with configurable hyperparameters."),
        ("📈", "Model Evaluation", "Accuracy, Precision, Recall, F1, Confusion Matrix, and radar comparison across models."),
        ("⬇️", "Export Results", "Download analysis results as CSV, JSON, or styled Excel files."),
        ("🌍", "Multilingual Support", "XLM-RoBERTa supports multilingual reviews out of the box."),
    ]
    cols = st.columns(2)
    for i, (icon, title, desc) in enumerate(features):
        with cols[i % 2]:
            st.markdown(
                f'<div style="background:#0f172a;border:1px solid #1e293b;border-radius:12px;'
                f'padding:16px;margin-bottom:12px;">'
                f'<span style="font-size:1.5rem;">{icon}</span>'
                f'<h4 style="color:#e2e8f0;margin:6px 0 4px 0;">{title}</h4>'
                f'<p style="color:#94a3b8;font-size:0.88rem;margin:0;">{desc}</p>'
                f'</div>',
                unsafe_allow_html=True,
            )

    divider()

    # ── Supported Aspects ─────────────────────────────────────────────────
    section_header("Supported Automotive Aspects")
    from src.config_loader import get_aspects
    aspects = get_aspects()

    aspect_icons = {
        "Vehicle": "🚗", "Engine": "⚙️", "Battery": "🔋", "Mileage": "⛽",
        "Safety": "🛡️", "Comfort": "🛋️", "Service": "🔧",
        "Infotainment": "📱", "Price": "💰",
    }

    cols3 = st.columns(3)
    for i, asp in enumerate(aspects):
        with cols3[i % 3]:
            icon = aspect_icons.get(asp, "🔹")
            st.markdown(
                f'<div style="background:#1e293b;border:1px solid #334155;border-radius:10px;'
                f'padding:12px 16px;margin-bottom:10px;display:flex;align-items:center;gap:10px;">'
                f'<span style="font-size:1.4rem;">{icon}</span>'
                f'<span style="color:#e2e8f0;font-weight:600;">{asp}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    divider()

    # ── Models table ──────────────────────────────────────────────────────
    section_header("Models")
    st.markdown("""
    <div style="overflow-x:auto;">
    <table style="width:100%;border-collapse:collapse;">
        <thead>
            <tr style="background:#1e293b;border-bottom:2px solid #334155;">
                <th style="padding:12px 16px;color:#94a3b8;text-align:left;font-weight:600;">Model</th>
                <th style="padding:12px 16px;color:#94a3b8;text-align:left;font-weight:600;">HuggingFace ID</th>
                <th style="padding:12px 16px;color:#94a3b8;text-align:left;font-weight:600;">Best For</th>
                <th style="padding:12px 16px;color:#94a3b8;text-align:left;font-weight:600;">Max Length</th>
            </tr>
        </thead>
        <tbody>
            <tr style="border-bottom:1px solid #1e293b;">
                <td style="padding:12px 16px;color:#38bdf8;font-weight:600;">BERT</td>
                <td style="padding:12px 16px;color:#e2e8f0;"><code>bert-base-uncased</code></td>
                <td style="padding:12px 16px;color:#cbd5e1;">English reviews, fast inference</td>
                <td style="padding:12px 16px;color:#cbd5e1;">256 tokens</td>
            </tr>
            <tr>
                <td style="padding:12px 16px;color:#a855f7;font-weight:600;">XLM-RoBERTa</td>
                <td style="padding:12px 16px;color:#e2e8f0;"><code>xlm-roberta-base</code></td>
                <td style="padding:12px 16px;color:#cbd5e1;">Multilingual, better generalisation</td>
                <td style="padding:12px 16px;color:#cbd5e1;">256 tokens</td>
            </tr>
        </tbody>
    </table>
    </div>
    """, unsafe_allow_html=True)

    divider()

    # ── Dataset format ────────────────────────────────────────────────────
    section_header("Dataset Format")
    st.markdown("""
    <div style="background:#0f172a;border:1px solid #334155;border-radius:10px;padding:16px;">
        <p style="color:#94a3b8;margin:0 0 8px 0;font-size:0.9rem;">Each row represents one aspect-sentiment annotation for a review:</p>
        <pre style="color:#22c55e;margin:0;font-size:0.85rem;">Review,Aspect,Sentiment
"The mileage is great but service is poor.",Mileage,Positive
"The mileage is great but service is poor.",Service,Negative
"Excellent engine performance.",Engine,Positive</pre>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#1e293b;border:1px solid #334155;border-radius:10px;padding:16px;">
        <p style="color:#94a3b8;margin:0 0 4px 0;font-size:0.85rem;font-weight:600;">SPLIT RATIOS (configurable in config.yaml)</p>
        <p style="color:#e2e8f0;margin:0;font-size:0.9rem;">70% Train · 15% Validation · 15% Test</p>
    </div>
    """, unsafe_allow_html=True)

    divider()

    # ── Constraints ───────────────────────────────────────────────────────
    section_header("Known Constraints")
    constraints = [
        "Aspect extraction is <b>keyword-based</b> — implicit aspects are not detected.",
        "Sarcasm and figurative language may reduce prediction accuracy.",
        "Confidence scores are model probability estimates, not guaranteed correctness.",
        "This system is an <b>analytical tool</b>, not a substitute for human judgment.",
        "Training without a GPU will be significantly slower.",
    ]
    for c in constraints:
        st.markdown(
            f'<div style="background:#1c0a0a;border-left:3px solid #ef4444;border-radius:6px;'
            f'padding:10px 14px;margin-bottom:8px;color:#cbd5e1;font-size:0.9rem;">⚠️ {c}</div>',
            unsafe_allow_html=True,
        )

    divider()
    st.markdown(
        '<p style="color:#334155;font-size:0.8rem;text-align:center;">'
        'AutoReview v1.0.0 · ABSA System · Academic & Research Use Only'
        '</p>',
        unsafe_allow_html=True,
    )
