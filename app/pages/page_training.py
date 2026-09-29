"""
app/pages/page_training.py
───────────────────────────
Model training page — fine-tune BERT or XLM-R on automotive ABSA data.
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
import pandas as pd

from app.ui_components import section_header, divider, info_banner, metric_card
from src.data.validator import validate_uploaded_file
from src.data.dataset_generator import SyntheticDatasetGenerator
from src.config_loader import load_config, get_project_root


def render():
    # ── Header ──────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="padding:20px 0 10px;">
        <h1 style="color:#a855f7;font-weight:800;font-size:2rem;margin:0;">
            🎓 Model Training
        </h1>
        <p style="color:#94a3b8;margin:6px 0 0 0;">
            Fine-tune BERT or XLM-RoBERTa on your automotive review dataset.
        </p>
    </div>
    """, unsafe_allow_html=True)

    divider()

    cfg = load_config()
    model_options = {v["display_name"]: k for k, v in cfg["models"].items()}

    # ── Warning banner ────────────────────────────────────────────────────
    info_banner(
        "Training may take several minutes to hours depending on dataset size and available hardware. "
        "A CUDA GPU is strongly recommended for training.",
        kind="info",
    )

    # ── Training configuration ────────────────────────────────────────────
    section_header("Training Configuration")

    col_model, col_epochs = st.columns(2)
    with col_model:
        selected_display = st.selectbox("Model to train", list(model_options.keys()), key="train_model_select")
        train_model_key = model_options[selected_display]

    with col_epochs:
        epochs = st.slider("Training Epochs", 1, 20,
                           cfg["training"].get("epochs", 5), key="train_epochs")

    col_lr, col_bs = st.columns(2)
    with col_lr:
        lr = st.select_slider(
            "Learning Rate",
            options=[1e-5, 2e-5, 3e-5, 5e-5, 1e-4],
            value=cfg["training"].get("learning_rate", 2e-5),
            format_func=lambda x: f"{x:.0e}",
            key="train_lr",
        )
    with col_bs:
        batch_size = st.selectbox("Batch Size", [8, 16, 32, 64],
                                  index=1, key="train_batch_size")

    divider()

    # ── Data source ───────────────────────────────────────────────────────
    section_header("Training Data", "Upload your dataset or use the built-in synthetic generator")

    tab_upload, tab_synth = st.tabs(["📁 Upload Dataset", "🔬 Synthetic Generator"])

    with tab_upload:
        st.markdown(
            '<p style="color:#94a3b8;font-size:0.9rem;">Upload a CSV/JSON with Review, Aspect, Sentiment columns.</p>',
            unsafe_allow_html=True,
        )
        up_file = st.file_uploader(
            "Training dataset",
            type=["csv", "json"],
            label_visibility="collapsed",
            key="train_file_uploader",
        )
        if up_file:
            val_res, df_up = validate_uploaded_file(up_file.name, up_file.read())
            if not val_res.is_valid:
                for err in val_res.errors:
                    info_banner(err, kind="error")
            else:
                for warn in val_res.warnings:
                    info_banner(warn, kind="warning")
                st.session_state["train_df"] = df_up
                info_banner(f"✅ Loaded {len(df_up):,} rows from **{up_file.name}**", kind="success")

    with tab_synth:
        col_n2, col_gen = st.columns([2, 1])
        with col_n2:
            n_synth = st.slider("Number of synthetic samples", 100, 2000, 500, 50, key="train_n_synth")
        with col_gen:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔬 Generate", width="stretch", key="train_gen_btn"):
                gen = SyntheticDatasetGenerator()
                df_synth = gen.generate_balanced(samples_per_class=max(10, n_synth // (9 * 3)))
                st.session_state["train_df"] = df_synth
                info_banner(f"✅ Generated {len(df_synth):,} balanced training samples.", kind="success")

    # Preview
    train_df: pd.DataFrame | None = st.session_state.get("train_df")
    if train_df is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.markdown(metric_card("Total Samples", str(len(train_df)), color="#38bdf8"), unsafe_allow_html=True)
        with col_info2:
            st.markdown(metric_card("Unique Reviews", str(train_df["Review"].nunique()), color="#a855f7"), unsafe_allow_html=True)
        with col_info3:
            st.markdown(metric_card("Aspects", str(train_df["Aspect"].nunique()), color="#22c55e"), unsafe_allow_html=True)

        with st.expander("Preview Dataset (first 20 rows)"):
            st.dataframe(train_df.head(20), use_container_width=True)

    divider()

    # ── Launch training ───────────────────────────────────────────────────
    col_start, _ = st.columns([1, 2])
    with col_start:
        start_training = st.button(
            "🚀 Start Training",
            width="stretch",
            disabled=(train_df is None),
            key="start_training_btn",
        )

    if start_training:
        if train_df is None:
            info_banner("Please provide a training dataset first.", kind="error")
            return

        # Update config values for this run
        cfg["training"]["epochs"] = epochs
        cfg["training"]["learning_rate"] = lr
        cfg["training"]["batch_size"] = batch_size

        output_dir = str(get_project_root() / "models" / "checkpoints" / train_model_key)
        os.makedirs(output_dir, exist_ok=True)

        progress_bar = st.progress(0.0, text="Initialising training…")
        status_placeholder = st.empty()

        def _progress_cb(stage: str, fraction: float):
            progress_bar.progress(min(fraction, 1.0), text=stage)
            status_placeholder.markdown(
                f'<p style="color:#94a3b8;font-size:0.85rem;">⚙️ {stage}</p>',
                unsafe_allow_html=True,
            )

        try:
            from src.models.trainer import ABSATrainer
            trainer_obj = ABSATrainer(train_model_key)
            result = trainer_obj.train(
                train_df,
                output_dir=output_dir,
                progress_callback=_progress_cb,
            )
            status_placeholder.empty()
            progress_bar.progress(1.0, text="Training complete!")

            divider()
            section_header("Training Results")

            r_cols = st.columns(4)
            metrics_to_show = [
                ("Train Samples", str(result.get("train_samples", "—")), "#38bdf8"),
                ("Test Samples", str(result.get("test_samples", "—")), "#a855f7"),
                ("Test Accuracy", f"{result.get('test_accuracy', 0):.1%}", "#22c55e"),
                ("Test F1 (macro)", f"{result.get('test_f1', 0):.1%}", "#f59e0b"),
            ]
            for col, (label, value, color) in zip(r_cols, metrics_to_show):
                with col:
                    st.markdown(metric_card(label, value, color=color), unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            best_dir = result.get("best_model_dir", output_dir)
            info_banner(
                f"✅ Training complete! Best model saved to: `{best_dir}`  \n"
                f"Reload the model in the sidebar to start using the fine-tuned weights.",
                kind="success",
            )

        except Exception as exc:
            status_placeholder.empty()
            info_banner(f"Training failed: {exc}", kind="error")
            st.exception(exc)

    # ── Existing checkpoints ──────────────────────────────────────────────
    divider()
    section_header("Available Checkpoints", "Fine-tuned model checkpoints found on disk")

    ckpt_root = get_project_root() / "models" / "checkpoints"
    found_any = False

    for model_key in cfg["models"]:
        model_ckpt_dir = ckpt_root / model_key
        if not model_ckpt_dir.is_dir():
            continue
        subdirs = [d for d in model_ckpt_dir.iterdir() if d.is_dir()]
        if not subdirs:
            continue
        found_any = True
        display = cfg["models"][model_key].get("display_name", model_key)
        st.markdown(
            f'<div style="background:#1e293b;border:1px solid #334155;border-radius:10px;'
            f'padding:14px;margin-bottom:10px;">'
            f'<span style="color:#38bdf8;font-weight:700;">{display}</span>'
            f'<ul style="color:#94a3b8;margin:6px 0 0 0;">',
            unsafe_allow_html=True,
        )
        for sub in sorted(subdirs):
            config_exists = (sub / "config.json").exists()
            icon = "✅" if config_exists else "📁"
            st.markdown(
                f'<li style="font-size:0.85rem;">{icon} <code>{sub.name}</code></li>',
                unsafe_allow_html=True,
            )
        st.markdown("</ul></div>", unsafe_allow_html=True)

    if not found_any:
        info_banner(
            "No checkpoints found yet. Train a model above to create one.",
            kind="info",
        )
