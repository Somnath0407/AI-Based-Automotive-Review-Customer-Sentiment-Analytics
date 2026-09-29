"""
app/pages/page_evaluation.py
──────────────────────────────
Model evaluation page — displays all training results and charts
directly from saved checkpoint files. No dataset upload required.
"""

from __future__ import annotations

import json

import streamlit as st
import pandas as pd

from app.ui_components import section_header, divider, info_banner, metric_card
from src.analytics.visualizer import (
    plot_confusion_matrix,
    plot_training_loss_curve,
    plot_accuracy_f1_curve,
    plot_model_comparison,
)
from src.config_loader import load_config, get_project_root


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_training_metadata(model_key: str) -> dict | None:
    path = get_project_root() / "models" / "checkpoints" / model_key / "training_metadata.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _load_log_history(model_key: str) -> list:
    ckpt_root = get_project_root() / "models" / "checkpoints" / model_key
    if not ckpt_root.exists():
        return []
    candidates = sorted(
        [d for d in ckpt_root.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0,
        reverse=True,
    )
    for ckpt_dir in candidates:
        state_file = ckpt_dir / "trainer_state.json"
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("log_history", [])
    return []


def _fig_to_png(fig) -> bytes:
    try:
        return fig.to_image(format="png", width=1400, height=800, scale=2)
    except Exception as exc:
        st.warning(f"PNG export unavailable: {exc}")
        return b""


def _has_checkpoint(model_key: str) -> bool:
    ckpt_root = get_project_root() / "models" / "checkpoints" / model_key
    if not ckpt_root.exists():
        return False
    return any(
        d.is_dir() and (d / "config.json").exists()
        for d in ckpt_root.rglob("*")
    )


# ── Main render ───────────────────────────────────────────────────────────────

def render():
    st.markdown("""
    <div style="padding:20px 0 10px;">
        <h1 style="color:#f59e0b;font-weight:800;font-size:2rem;margin:0;">
            📈 Model Evaluation
        </h1>
        <p style="color:#94a3b8;margin:6px 0 0 0;">
            Training results, performance metrics and charts loaded automatically
            from saved checkpoints — no file upload needed.
        </p>
    </div>
    """, unsafe_allow_html=True)

    divider()

    cfg = load_config()
    model_keys = list(cfg["models"].keys())
    model_displays = {k: cfg["models"][k]["display_name"] for k in model_keys}

    # ── Model selector ────────────────────────────────────────────────────
    section_header("Select Model Checkpoint")
    col_sel, col_status = st.columns([1, 2])
    with col_sel:
        selected_model = st.selectbox(
            "Model",
            options=model_keys,
            format_func=lambda k: model_displays[k],
            key="eval_model_select",
            label_visibility="collapsed",
        )

    with col_status:
        if _has_checkpoint(selected_model):
            info_banner(f"Checkpoint found for **{model_displays[selected_model]}**", kind="success")
        else:
            info_banner(
                f"No checkpoint found for **{model_displays[selected_model]}**. "
                "Train the model first from the Model Training page.",
                kind="warning",
            )
            return

    divider()

    meta = _load_training_metadata(selected_model)
    log_hist = _load_log_history(selected_model)

    # ── Training summary ──────────────────────────────────────────────────
    if meta:
        train_info = meta.get("training", {})
        data_info  = meta.get("data", {})
        timestamp  = meta.get("timestamp", "N/A")
        device     = meta.get("device", "N/A").upper()
        epochs_run = train_info.get("epochs_run", "N/A")

        section_header(
            "Training Summary",
            f"Trained: {timestamp}  |  Device: {device}  |  Epochs: {epochs_run}",
        )

        c1, c2, c3, c4 = st.columns(4)
        for col, (label, val, color) in zip(
            [c1, c2, c3, c4],
            [
                ("Train Samples", str(data_info.get("train_rows", "N/A")), "#38bdf8"),
                ("Val Samples",   str(data_info.get("devel_rows", "N/A")), "#818cf8"),
                ("Test Samples",  str(data_info.get("test_rows",  "N/A")), "#a855f7"),
                ("Training Time", f"{train_info.get('training_secs', 0):.0f}s", "#f59e0b"),
            ],
        ):
            with col:
                st.markdown(metric_card(label, val, color=color), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        divider()

        # ── Per-split results ─────────────────────────────────────────────
        eval_metrics = meta.get("eval_metrics", {})

        for split_name in ("test", "devel"):
            split = eval_metrics.get(split_name)
            if not split:
                continue

            split_label = "Test Set Results" if split_name == "test" else "Validation Set Results"
            section_header(split_label)

            mc1, mc2, mc3, mc4 = st.columns(4)
            for col, (label, val, color) in zip(
                [mc1, mc2, mc3, mc4],
                [
                    ("Accuracy",          f"{split.get('accuracy', 0):.2%}",         "#38bdf8"),
                    ("Precision (macro)", f"{split.get('precision_macro', 0):.2%}",  "#22c55e"),
                    ("Recall (macro)",    f"{split.get('recall_macro', 0):.2%}",     "#a855f7"),
                    ("F1 (macro)",        f"{split.get('f1_macro', 0):.2%}",         "#f59e0b"),
                ],
            ):
                with col:
                    st.markdown(metric_card(label, val, color=color), unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            clf_report = split.get("classification_report", {})
            if clf_report:
                rows = []
                for cls in ["Positive", "Negative", "Neutral"]:
                    if cls in clf_report:
                        r = clf_report[cls]
                        rows.append({
                            "Class":     cls,
                            "Precision": f"{r.get('precision', 0):.3f}",
                            "Recall":    f"{r.get('recall', 0):.3f}",
                            "F1-Score":  f"{r.get('f1-score', 0):.3f}",
                            "Support":   int(r.get("support", 0)),
                        })
                if rows:
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            st.markdown("<br>", unsafe_allow_html=True)

            cm_data = split.get("confusion_matrix")
            if cm_data:
                acc_val = split.get("accuracy", 0)
                cm_fig = plot_confusion_matrix(cm_data)
                cm_fig.update_layout(
                    title=dict(
                        text=f"Confusion Matrix — {model_displays[selected_model]} ({split_name.capitalize()} | Accuracy {acc_val:.1%})"
                    )
                )
                st.plotly_chart(cm_fig, use_container_width=True, key=f"cm_{split_name}")
                cm_png = _fig_to_png(cm_fig)
                col_dl, _ = st.columns([1, 2])
                with col_dl:
                    st.download_button(
                        label=f"Download {split_name.capitalize()} Confusion Matrix (PNG)",
                        data=cm_png,
                        file_name=f"{selected_model}_{split_name}_confusion_matrix.png",
                        mime="image/png",
                        disabled=(not cm_png),
                        use_container_width=True,
                        key=f"dl_cm_{split_name}",
                    )

            divider()

    else:
        info_banner("No training_metadata.json found. Training curves may still be available below.", kind="info")

    # ── Training curves ───────────────────────────────────────────────────
    if log_hist:
        n_train = len([e for e in log_hist if "loss" in e])
        n_eval  = len([e for e in log_hist if "eval_loss" in e])
        section_header("Training Curves", f"{n_train} training steps  |  {n_eval} eval checkpoints")

        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown('<p style="color:#94a3b8;font-size:0.85rem;font-weight:600;margin-bottom:4px;">Training & Validation Loss</p>', unsafe_allow_html=True)
            loss_fig = plot_training_loss_curve(log_hist)
            st.plotly_chart(loss_fig, use_container_width=True, key="loss_chart")
            loss_png = _fig_to_png(loss_fig)
            st.download_button(
                label="Download Loss Graph (PNG)",
                data=loss_png,
                file_name=f"{selected_model}_training_loss.png",
                mime="image/png",
                disabled=(not loss_png),
                use_container_width=True,
                key="dl_loss",
            )

        with col_r:
            st.markdown('<p style="color:#94a3b8;font-size:0.85rem;font-weight:600;margin-bottom:4px;">Validation Accuracy & F1 per Epoch</p>', unsafe_allow_html=True)
            acc_fig = plot_accuracy_f1_curve(log_hist)
            st.plotly_chart(acc_fig, use_container_width=True, key="acc_chart")
            acc_png = _fig_to_png(acc_fig)
            st.download_button(
                label="Download Accuracy & F1 Graph (PNG)",
                data=acc_png,
                file_name=f"{selected_model}_accuracy_f1.png",
                mime="image/png",
                disabled=(not acc_png),
                use_container_width=True,
                key="dl_acc",
            )

        divider()

    # ── Multi-model comparison ────────────────────────────────────────────
    available_metas = {k: _load_training_metadata(k) for k in model_keys if _load_training_metadata(k)}
    if len(available_metas) >= 2:
        section_header("Model Comparison", "Radar chart comparing all trained models on test-set metrics")
        comparison_data = {}
        for mk, m in available_metas.items():
            ts = m.get("eval_metrics", {}).get("test", {})
            if ts:
                comparison_data[mk] = {
                    "accuracy":        ts.get("accuracy", 0),
                    "precision_macro": ts.get("precision_macro", 0),
                    "recall_macro":    ts.get("recall_macro", 0),
                    "f1_macro":        ts.get("f1_macro", 0),
                }
        if len(comparison_data) >= 2:
            radar_fig = plot_model_comparison(comparison_data)
            st.plotly_chart(radar_fig, use_container_width=True, key="radar_chart")
            radar_png = _fig_to_png(radar_fig)
            col_r2, _ = st.columns([1, 2])
            with col_r2:
                st.download_button(
                    label="Download Model Comparison Chart (PNG)",
                    data=radar_png,
                    file_name="model_comparison_radar.png",
                    mime="image/png",
                    disabled=(not radar_png),
                    use_container_width=True,
                    key="dl_radar",
                )
