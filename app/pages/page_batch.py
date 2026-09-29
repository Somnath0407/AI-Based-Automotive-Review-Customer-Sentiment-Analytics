"""
app/pages/page_batch.py
────────────────────────
Batch analytics page — upload or generate a dataset and run ABSA.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd

from app.ui_components import section_header, divider, info_banner, metric_card
from src.data.validator import validate_uploaded_file
from src.data.dataset_generator import SyntheticDatasetGenerator
from src.analytics.analytics_engine import AnalyticsEngine
from src.analytics.visualizer import (
    plot_sentiment_donut,
    plot_aspect_frequency,
    plot_aspect_sentiment_stacked,
    plot_aspect_sentiment_heatmap,
)
from src.export.export_service import ExportService


def render():
    # ── Header ──────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="padding:20px 0 10px;">
        <h1 style="color:#38bdf8;font-weight:800;font-size:2rem;margin:0;">
            📊 Batch Analytics
        </h1>
        <p style="color:#94a3b8;margin:6px 0 0 0;">
            Upload a dataset or use synthetic data to run ABSA on multiple reviews.
        </p>
    </div>
    """, unsafe_allow_html=True)

    divider()

    if not st.session_state.get("model_loaded"):
        info_banner(
            "No model loaded. Please load a model from the sidebar before running batch analysis.",
            kind="warning",
        )

    # ── Data source ──────────────────────────────────────────────────────────
    section_header("Data Source", "Choose how to provide your review data")

    tab_upload, tab_synthetic = st.tabs(["📁 Upload Dataset", "🔬 Synthetic Dataset"])

    with tab_upload:
        st.markdown('<p style="color:#94a3b8;font-size:0.9rem;margin-bottom:8px;">Upload a CSV or JSON file with Review, Aspect, Sentiment columns.</p>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Upload dataset",
            type=["csv", "json"],
            label_visibility="collapsed",
            key="batch_file_uploader",
        )
        if uploaded:
            val_result, df_upload = validate_uploaded_file(uploaded.name, uploaded.read())
            if not val_result.is_valid:
                for err in val_result.errors:
                    info_banner(err, kind="error")
            else:
                for warn in val_result.warnings:
                    info_banner(warn, kind="warning")
                st.session_state["batch_df_raw"] = df_upload
                info_banner(f"✅ Loaded {len(df_upload):,} rows from **{uploaded.name}**", kind="success")

    with tab_synthetic:
        col_n, col_btn = st.columns([2, 1])
        with col_n:
            n_reviews = st.slider("Number of synthetic reviews", 50, 500, 200, 50)
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔬 Generate Dataset", width="stretch"):
                gen = SyntheticDatasetGenerator()
                df_synth = gen.generate(n_reviews)
                st.session_state["batch_df_raw"] = df_synth
                info_banner(f"✅ Generated {len(df_synth):,} synthetic review annotations.", kind="success")

    # ── Preview uploaded / generated data ─────────────────────────────────
    df_raw: pd.DataFrame | None = st.session_state.get("batch_df_raw")
    if df_raw is not None:
        divider()
        section_header("Dataset Preview", f"{len(df_raw):,} rows · {df_raw['Review'].nunique():,} unique reviews")
        st.dataframe(df_raw.head(20), use_container_width=True)

    divider()

    # ── Run analysis ──────────────────────────────────────────────────────
    col_btn2, _ = st.columns([1, 3])
    with col_btn2:
        run_clicked = st.button(
            "🚀 Run Batch Analysis",
            width="stretch",
            disabled=(not st.session_state.get("model_loaded") or df_raw is None),
        )

    if run_clicked:
        engine = st.session_state.get("engine")
        if engine is None:
            info_banner("Engine not initialised. Please reload the model.", kind="error")
            return

        reviews = df_raw["Review"].dropna().unique().tolist()
        progress_bar = st.progress(0, text="Analysing reviews…")
        status_text = st.empty()

        def _cb(current, total):
            frac = current / total
            progress_bar.progress(frac, text=f"Analysing review {current}/{total}…")
            status_text.markdown(
                f'<p style="color:#94a3b8;font-size:0.8rem;">{current}/{total} reviews processed</p>',
                unsafe_allow_html=True,
            )

        try:
            results_df = engine.analyse_dataframe(
                pd.DataFrame({"Review": reviews}),
                progress_callback=_cb,
            )
            st.session_state["batch_results_df"] = results_df
            progress_bar.progress(1.0, text="Complete!")
            status_text.empty()
            info_banner(f"✅ Analysed {len(reviews):,} reviews — {len(results_df):,} predictions generated.", kind="success")
        except Exception as exc:
            info_banner(f"Batch analysis failed: {exc}", kind="error")
            return

    # ── Display results ────────────────────────────────────────────────────
    results_df: pd.DataFrame | None = st.session_state.get("batch_results_df")
    if results_df is None or results_df.empty:
        return

    divider()
    section_header("Batch Analysis Results")

    # ── Summary metrics ────────────────────────────────────────────────────
    engine_analytics = AnalyticsEngine()
    metrics = engine_analytics.summary_metrics(results_df)

    cols = st.columns(4)
    with cols[0]:
        st.markdown(metric_card("Total Reviews", str(metrics["total_reviews"]), color="#38bdf8"), unsafe_allow_html=True)
    with cols[1]:
        st.markdown(metric_card("Positive", f"{metrics['positive_pct']}%", color="#22c55e"), unsafe_allow_html=True)
    with cols[2]:
        st.markdown(metric_card("Negative", f"{metrics['negative_pct']}%", color="#ef4444"), unsafe_allow_html=True)
    with cols[3]:
        st.markdown(metric_card("Neutral", f"{metrics['neutral_pct']}%", color="#f59e0b"), unsafe_allow_html=True)

    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)

    # ── Charts row 1 ──────────────────────────────────────────────────────
    col_l, col_r = st.columns(2)
    with col_l:
        dist_df = engine_analytics.sentiment_distribution(results_df)
        if not dist_df.empty:
            fig_donut = plot_sentiment_donut(dist_df)
            st.plotly_chart(fig_donut, use_container_width=True)

    with col_r:
        freq_df = engine_analytics.aspect_frequency(results_df)
        if not freq_df.empty:
            fig_freq = plot_aspect_frequency(freq_df)
            st.plotly_chart(fig_freq, use_container_width=True)

    # ── Charts row 2 ──────────────────────────────────────────────────────
    pivot_df = engine_analytics.aspect_sentiment_pivot(results_df)
    if not pivot_df.empty:
        fig_stacked = plot_aspect_sentiment_stacked(pivot_df)
        st.plotly_chart(fig_stacked, use_container_width=True)

    asp_sent_df = engine_analytics.aspect_sentiment_distribution(results_df)
    if not asp_sent_df.empty:
        fig_heatmap = plot_aspect_sentiment_heatmap(asp_sent_df)
        st.plotly_chart(fig_heatmap, use_container_width=True)

    # ── Raw data table ─────────────────────────────────────────────────────
    with st.expander("📋 Full Results Table"):
        st.dataframe(results_df, use_container_width=True)

    # ── Export ─────────────────────────────────────────────────────────────
    divider()
    section_header("Export Results", "Download the analysis results in your preferred format")
    exporter = ExportService()

    col_csv, col_json, col_xlsx = st.columns(3)
    with col_csv:
        st.download_button(
            "⬇️ Download CSV",
            data=exporter.to_csv(results_df),
            file_name="autoreview_batch_results.csv",
            mime=exporter.get_mime_type("csv"),
            use_container_width=True,
        )
    with col_json:
        st.download_button(
            "⬇️ Download JSON",
            data=exporter.to_json(results_df),
            file_name="autoreview_batch_results.json",
            mime=exporter.get_mime_type("json"),
            use_container_width=True,
        )
    with col_xlsx:
        st.download_button(
            "⬇️ Download Excel",
            data=exporter.to_excel(results_df),
            file_name="autoreview_batch_results.xlsx",
            mime=exporter.get_mime_type("excel"),
            use_container_width=True,
        )
