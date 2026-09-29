"""
src/analytics/visualizer.py
─────────────────────────────
Plotly-based visualisation helpers for the analytics dashboard.

Each function returns a plotly Figure object.
The presentation layer calls these and passes the figure to st.plotly_chart().

All charts follow a consistent dark-themed design system.
"""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Design tokens ─────────────────────────────────────────────────────────────
THEME = dict(
    bg_color="#0f172a",
    paper_color="#1e293b",
    font_color="#e2e8f0",
    grid_color="#334155",
    font_family="Inter, Roboto, sans-serif",
)

SENTIMENT_COLORS = {
    "Positive": "#22c55e",
    "Negative": "#ef4444",
    "Neutral": "#f59e0b",
}

ASPECT_PALETTE = px.colors.qualitative.Vivid


def _apply_dark_theme(fig: go.Figure, title: str = "") -> go.Figure:
    """Apply the shared dark theme to any figure."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=THEME["font_color"]), x=0.02),
        paper_bgcolor=THEME["paper_color"],
        plot_bgcolor=THEME["bg_color"],
        font=dict(color=THEME["font_color"], family=THEME["font_family"]),
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor=THEME["grid_color"],
            font=dict(color=THEME["font_color"]),
        ),
    )
    fig.update_xaxes(
        gridcolor=THEME["grid_color"],
        zerolinecolor=THEME["grid_color"],
        tickfont=dict(color=THEME["font_color"]),
    )
    fig.update_yaxes(
        gridcolor=THEME["grid_color"],
        zerolinecolor=THEME["grid_color"],
        tickfont=dict(color=THEME["font_color"]),
    )
    return fig


# ── Chart functions ───────────────────────────────────────────────────────────

def plot_sentiment_donut(dist_df: pd.DataFrame) -> go.Figure:
    """
    Donut chart of overall sentiment distribution.

    Parameters
    ----------
    dist_df : DataFrame with columns Sentiment, Count.
    """
    labels = dist_df["Sentiment"].tolist()
    values = dist_df["Count"].tolist()
    colors = [SENTIMENT_COLORS.get(l, "#64748b") for l in labels]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=colors, line=dict(color="#0f172a", width=2)),
        textinfo="label+percent",
        textfont=dict(color=THEME["font_color"], size=13),
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>",
    ))
    fig.update_layout(
        showlegend=True,
        annotations=[dict(
            text="Sentiment",
            x=0.5, y=0.5,
            font_size=14,
            font_color=THEME["font_color"],
            showarrow=False,
        )],
    )
    return _apply_dark_theme(fig, "Overall Sentiment Distribution")


def plot_aspect_frequency(freq_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart showing how often each aspect appears."""
    if freq_df.empty:
        return _empty_chart("Aspect Frequency — No Data")

    fig = go.Figure(go.Bar(
        y=freq_df["Aspect"],
        x=freq_df["Count"],
        orientation="h",
        marker=dict(
            color=freq_df["Count"],
            colorscale="Blues",
            showscale=False,
            line=dict(color="#0f172a", width=0.5),
        ),
        text=freq_df["Count"],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Mentions: %{x}<extra></extra>",
    ))
    fig.update_layout(
        yaxis=dict(categoryorder="total ascending"),
        xaxis_title="Number of Predictions",
    )
    return _apply_dark_theme(fig, "Aspect Mention Frequency")


def plot_aspect_sentiment_stacked(pivot_df: pd.DataFrame) -> go.Figure:
    """
    Grouped stacked bar chart: one group per aspect,
    bars coloured by Positive / Negative / Neutral.
    """
    if pivot_df.empty:
        return _empty_chart("Aspect-wise Sentiment — No Data")

    fig = go.Figure()
    for sentiment in ["Positive", "Negative", "Neutral"]:
        if sentiment not in pivot_df.columns:
            continue
        fig.add_trace(go.Bar(
            name=sentiment,
            x=pivot_df["Aspect"],
            y=pivot_df[sentiment],
            marker_color=SENTIMENT_COLORS[sentiment],
            hovertemplate="<b>%{x}</b><br>" + sentiment + ": %{y}<extra></extra>",
        ))
    fig.update_layout(barmode="stack", xaxis_title="Aspect", yaxis_title="Count")
    return _apply_dark_theme(fig, "Aspect-wise Sentiment Distribution")


def plot_aspect_sentiment_heatmap(dist_df: pd.DataFrame) -> go.Figure:
    """
    Heatmap of sentiment percentages per aspect.
    Rows = Aspects, Columns = Sentiment labels.
    """
    if dist_df.empty:
        return _empty_chart("Sentiment Heatmap — No Data")

    pivot = dist_df.pivot_table(
        index="Aspect", columns="Sentiment", values="Percentage", fill_value=0
    )
    for col in ["Positive", "Negative", "Neutral"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot[["Positive", "Negative", "Neutral"]]

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=[
            [0.0, "#0f172a"],
            [0.5, "#1d4ed8"],
            [1.0, "#22c55e"],
        ],
        text=[[f"{v:.1f}%" for v in row] for row in pivot.values],
        texttemplate="%{text}",
        hovertemplate="Aspect: %{y}<br>Sentiment: %{x}<br>Percentage: %{text}<extra></extra>",
        showscale=True,
        colorbar=dict(tickfont=dict(color=THEME["font_color"])),
    ))
    return _apply_dark_theme(fig, "Sentiment Percentage Heatmap (per Aspect)")


def plot_confidence_gauge(confidence: float, aspect: str, sentiment: str) -> go.Figure:
    """Gauge chart showing confidence for a single prediction."""
    color = SENTIMENT_COLORS.get(sentiment, "#64748b")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(confidence * 100, 1),
        number=dict(suffix="%", font=dict(color=THEME["font_color"], size=28)),
        title=dict(
            text=f"{aspect} → {sentiment}",
            font=dict(color=THEME["font_color"], size=14),
        ),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor=THEME["font_color"]),
            bar=dict(color=color),
            bgcolor=THEME["bg_color"],
            bordercolor=THEME["grid_color"],
            steps=[
                dict(range=[0, 50], color="#1e293b"),
                dict(range=[50, 75], color="#1e3a5f"),
                dict(range=[75, 100], color="#1e4d2b"),
            ],
        ),
    ))
    fig.update_layout(paper_bgcolor=THEME["paper_color"], height=200)
    return fig


def plot_confidence_bar(predictions_df: pd.DataFrame) -> go.Figure:
    """
    Horizontal confidence bar for each predicted aspect in a single review.
    """
    if predictions_df.empty:
        return _empty_chart("Confidence — No Data")

    df = predictions_df.copy()

    # Robustly convert Confidence to a float in [0, 1]
    def _to_float(val):
        try:
            s = str(val).strip().rstrip("%")
            f = float(s)
            # If value came as e.g. "87.3" (percent string without dividing)
            return f / 100.0 if f > 1.0 else f
        except (ValueError, TypeError):
            return 0.0

    df["Confidence_num"] = df["Confidence"].apply(_to_float).astype(float)

    colors = [SENTIMENT_COLORS.get(s, "#64748b") for s in df["Sentiment"]]
    labels = [f"{row.Aspect} ({row.Sentiment})" for row in df.itertuples()]
    text_labels = [f"{float(v):.1%}" for v in df["Confidence_num"].tolist()]

    fig = go.Figure(go.Bar(
        x=df["Confidence_num"].tolist(),
        y=labels,
        orientation="h",
        marker_color=colors,
        text=text_labels,
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Confidence: %{text}<extra></extra>",
    ))
    fig.update_layout(
        xaxis=dict(range=[0, 1.15], tickformat=".0%"),
        yaxis=dict(categoryorder="total ascending"),
    )
    return _apply_dark_theme(fig, "Prediction Confidence")



def plot_model_comparison(eval_results: dict) -> go.Figure:
    """
    Radar / spider chart comparing two models across accuracy, precision, recall, F1.
    eval_results: {"bert": {metrics...}, "xlmr": {metrics...}}
    """
    metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    labels = ["Accuracy", "Precision", "Recall", "F1"]

    fig = go.Figure()
    colors = {"bert": "#3b82f6", "xlmr": "#a855f7"}

    for model_key, result in eval_results.items():
        values = [result.get(m, 0) for m in metrics]
        values_closed = values + [values[0]]  # close the shape
        labels_closed = labels + [labels[0]]

        fig.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=labels_closed,
            fill="toself",
            name=model_key.upper(),
            line=dict(color=colors.get(model_key, "#64748b"), width=2),
            fillcolor=colors.get(model_key, "#64748b").replace(")", ",0.15)").replace("rgb", "rgba"),
        ))

    fig.update_layout(
        polar=dict(
            bgcolor=THEME["bg_color"],
            radialaxis=dict(
                visible=True, range=[0, 1],
                tickformat=".0%",
                tickfont=dict(color=THEME["font_color"]),
                gridcolor=THEME["grid_color"],
            ),
            angularaxis=dict(
                tickfont=dict(color=THEME["font_color"]),
                gridcolor=THEME["grid_color"],
            ),
        ),
    )
    return _apply_dark_theme(fig, "Model Performance Comparison")


def plot_confusion_matrix(cm: list, labels=None) -> go.Figure:
    """Annotated confusion matrix heatmap."""
    if labels is None:
        labels = ["Positive", "Negative", "Neutral"]
    fig = go.Figure(go.Heatmap(
        z=cm,
        x=labels,
        y=labels,
        colorscale="Blues",
        text=cm,
        texttemplate="%{text}",
        hovertemplate="True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
        showscale=True,
    ))
    fig.update_layout(
        xaxis_title="Predicted",
        yaxis_title="Actual",
        yaxis=dict(autorange="reversed"),
    )
    return _apply_dark_theme(fig, "Confusion Matrix")


# ── Training curve charts ─────────────────────────────────────────────────────

def plot_training_loss_curve(log_history: list) -> go.Figure:
    """
    Line chart of training loss over steps, parsed from HuggingFace
    trainer_state.json ``log_history`` list.

    Parameters
    ----------
    log_history : list of dicts from trainer_state["log_history"]
    """
    train_steps, train_loss = [], []
    eval_steps, eval_loss = [], []

    for entry in log_history:
        if "loss" in entry:
            train_steps.append(entry.get("step", 0))
            train_loss.append(entry["loss"])
        if "eval_loss" in entry:
            eval_steps.append(entry.get("step", 0))
            eval_loss.append(entry["eval_loss"])

    if not train_steps:
        return _empty_chart("Training Loss — No Data")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=train_steps, y=train_loss,
        mode="lines+markers",
        name="Train Loss",
        line=dict(color="#38bdf8", width=2),
        marker=dict(size=4, color="#38bdf8"),
        hovertemplate="Step: %{x}<br>Loss: %{y:.4f}<extra>Train</extra>",
    ))
    if eval_steps:
        fig.add_trace(go.Scatter(
            x=eval_steps, y=eval_loss,
            mode="lines+markers",
            name="Eval Loss",
            line=dict(color="#f59e0b", width=2, dash="dot"),
            marker=dict(size=7, color="#f59e0b", symbol="diamond"),
            hovertemplate="Step: %{x}<br>Loss: %{y:.4f}<extra>Eval</extra>",
        ))
    fig.update_layout(
        xaxis_title="Training Step",
        yaxis_title="Loss",
        legend=dict(orientation="h", y=1.08, x=0),
    )
    return _apply_dark_theme(fig, "Training & Validation Loss Curve")


def plot_accuracy_f1_curve(log_history: list) -> go.Figure:
    """
    Dual-axis line chart of eval accuracy and F1 over epochs,
    parsed from HuggingFace trainer_state.json ``log_history`` list.

    Parameters
    ----------
    log_history : list of dicts from trainer_state["log_history"]
    """
    steps, accuracy, f1 = [], [], []

    for entry in log_history:
        if "eval_accuracy" in entry and "eval_f1" in entry:
            steps.append(entry.get("step", 0))
            accuracy.append(entry["eval_accuracy"])
            f1.append(entry["eval_f1"])

    if not steps:
        return _empty_chart("Accuracy & F1 Curve — No Eval Data")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=steps, y=accuracy,
        mode="lines+markers",
        name="Accuracy",
        line=dict(color="#22c55e", width=2),
        marker=dict(size=8, color="#22c55e"),
        hovertemplate="Step: %{x}<br>Accuracy: %{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=steps, y=f1,
        mode="lines+markers",
        name="F1 (macro)",
        line=dict(color="#a855f7", width=2, dash="dot"),
        marker=dict(size=8, color="#a855f7", symbol="diamond"),
        hovertemplate="Step: %{x}<br>F1: %{y:.3f}<extra></extra>",
    ))
    fig.update_layout(
        xaxis_title="Training Step (end of epoch)",
        yaxis_title="Score",
        yaxis=dict(range=[0, 1], tickformat=".0%"),
        legend=dict(orientation="h", y=1.08, x=0),
    )
    return _apply_dark_theme(fig, "Validation Accuracy & F1 per Epoch")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _empty_chart(title: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text="No data available",
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(color=THEME["font_color"], size=16),
    )
    return _apply_dark_theme(fig, title)
