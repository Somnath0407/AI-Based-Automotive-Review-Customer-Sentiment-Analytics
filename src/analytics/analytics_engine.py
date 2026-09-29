"""
src/analytics/analytics_engine.py
───────────────────────────────────
Aggregate metrics computation for batch analysis results.

Takes a list of ReviewAnalysisResult objects (or a flat DataFrame) and
produces summary DataFrames ready for visualisation.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd


class AnalyticsEngine:
    """
    Computes aggregate statistics from batch ABSA results.

    All methods are pure functions of the input — no state is maintained.
    """

    # ── Public API ────────────────────────────────────────────────────────────

    def sentiment_distribution(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Overall sentiment distribution across all predictions.

        Parameters
        ----------
        df : Flat results DataFrame with a 'Sentiment' column.

        Returns
        -------
        DataFrame with columns: Sentiment, Count, Percentage
        """
        if df.empty or "Sentiment" not in df.columns:
            return pd.DataFrame(columns=["Sentiment", "Count", "Percentage"])

        counts = (
            df["Sentiment"]
            .value_counts()
            .reset_index()
            .rename(columns={"index": "Sentiment", "Sentiment": "Count"})
        )
        # Ensure consistent column names regardless of pandas version
        if "count" in counts.columns:
            counts = counts.rename(columns={"Sentiment": "Sentiment", "count": "Count"})

        total = counts["Count"].sum()
        counts["Percentage"] = (counts["Count"] / total * 100).round(1)
        return counts.sort_values("Count", ascending=False).reset_index(drop=True)

    def aspect_frequency(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        How many times each aspect appears in the results.

        Returns
        -------
        DataFrame with columns: Aspect, Count
        """
        if df.empty or "Aspect" not in df.columns:
            return pd.DataFrame(columns=["Aspect", "Count"])

        freq = (
            df["Aspect"]
            .value_counts()
            .reset_index()
            .rename(columns={"index": "Aspect", "Aspect": "Count"})
        )
        if "count" in freq.columns:
            freq = freq.rename(columns={"Aspect": "Aspect", "count": "Count"})
        return freq.sort_values("Count", ascending=False).reset_index(drop=True)

    def aspect_sentiment_pivot(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pivot table: rows = aspects, columns = sentiment labels, values = count.

        Returns
        -------
        DataFrame with columns: Aspect, Positive, Negative, Neutral
        """
        if df.empty or not {"Aspect", "Sentiment"}.issubset(df.columns):
            return pd.DataFrame(columns=["Aspect", "Positive", "Negative", "Neutral"])

        pivot = (
            df.groupby(["Aspect", "Sentiment"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
        )
        for col in ["Positive", "Negative", "Neutral"]:
            if col not in pivot.columns:
                pivot[col] = 0
        return pivot[["Aspect", "Positive", "Negative", "Neutral"]]

    def aspect_sentiment_distribution(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Percentage breakdown of sentiments per aspect.

        Returns
        -------
        DataFrame with columns: Aspect, Sentiment, Count, Percentage
        """
        if df.empty or not {"Aspect", "Sentiment"}.issubset(df.columns):
            return pd.DataFrame(columns=["Aspect", "Sentiment", "Count", "Percentage"])

        grouped = df.groupby(["Aspect", "Sentiment"]).size().reset_index(name="Count")
        totals = grouped.groupby("Aspect")["Count"].transform("sum")
        grouped["Percentage"] = (grouped["Count"] / totals * 100).round(1)
        return grouped.sort_values(["Aspect", "Count"], ascending=[True, False])

    def summary_metrics(self, df: pd.DataFrame) -> Dict[str, object]:
        """
        High-level summary statistics for the batch.

        Returns
        -------
        Dict with keys: total_reviews, total_predictions, aspects_detected,
                        positive_pct, negative_pct, neutral_pct, avg_confidence
        """
        if df.empty:
            return {
                "total_reviews": 0,
                "total_predictions": 0,
                "aspects_detected": 0,
                "positive_pct": 0.0,
                "negative_pct": 0.0,
                "neutral_pct": 0.0,
                "avg_confidence": 0.0,
            }

        total = len(df)
        sentiments = df["Sentiment"] if "Sentiment" in df.columns else pd.Series(dtype=str)
        valid = sentiments[sentiments.isin(["Positive", "Negative", "Neutral"])]
        n = len(valid)

        pos_pct = round((valid == "Positive").sum() / n * 100, 1) if n else 0.0
        neg_pct = round((valid == "Negative").sum() / n * 100, 1) if n else 0.0
        neu_pct = round((valid == "Neutral").sum() / n * 100, 1) if n else 0.0

        # Average confidence — handle both numeric and string formats
        avg_conf = 0.0
        if "Confidence" in df.columns:
            def _to_float(v):
                try:
                    s = str(v).strip().rstrip("%")
                    f = float(s)
                    return f / 100.0 if f > 1.0 else f
                except Exception:
                    return None

            conf_vals = df["Confidence"].apply(_to_float).dropna()
            avg_conf = round(conf_vals.mean() * 100, 1) if len(conf_vals) else 0.0

        n_reviews = df["Review"].nunique() if "Review" in df.columns else total
        n_aspects = df["Aspect"].nunique() if "Aspect" in df.columns else 0

        return {
            "total_reviews": n_reviews,
            "total_predictions": total,
            "aspects_detected": n_aspects,
            "positive_pct": pos_pct,
            "negative_pct": neg_pct,
            "neutral_pct": neu_pct,
            "avg_confidence": avg_conf,
        }
