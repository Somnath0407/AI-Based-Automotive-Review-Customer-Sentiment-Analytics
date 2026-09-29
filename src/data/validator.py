"""
src/data/validator.py
──────────────────────
Input validation layer — sits between the UI and the processing pipeline.

Responsibilities
────────────────
- Validate single review text for inference.
- Validate uploaded files (type, size, columns).
- Validate DataFrame contents before preprocessing.

Returns structured ValidationResult objects so the UI can render
user-friendly messages without exposing internal exceptions.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from src.config_loader import get_dataset_config, get_aspects, get_sentiment_labels


@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.is_valid = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    @property
    def summary(self) -> str:
        parts = []
        if self.errors:
            parts.append("Errors:\n" + "\n".join(f"  • {e}" for e in self.errors))
        if self.warnings:
            parts.append("Warnings:\n" + "\n".join(f"  • {w}" for w in self.warnings))
        return "\n".join(parts) if parts else "Validation passed."


# ── Single review validation ──────────────────────────────────────────────────

def validate_review_text(text: str,
                          min_length: int = 5,
                          max_length: int = 1000) -> ValidationResult:
    """Validate a single review string submitted for inference."""
    result = ValidationResult(is_valid=True)

    if not isinstance(text, str):
        result.add_error("Review must be a text string.")
        return result

    stripped = text.strip()

    if not stripped:
        result.add_error("Review text cannot be empty.")
        return result

    if len(stripped) < min_length:
        result.add_error(
            f"Review is too short ({len(stripped)} chars). "
            f"Please provide at least {min_length} characters."
        )

    if len(stripped) > max_length:
        result.add_warning(
            f"Review is very long ({len(stripped)} chars). "
            f"Only the first {max_length} characters will be analysed."
        )

    return result


# ── File upload validation ────────────────────────────────────────────────────

def validate_uploaded_file(
    filename: str,
    file_content: bytes,
) -> Tuple[ValidationResult, Optional[pd.DataFrame]]:
    """
    Validate an uploaded file and, if valid, return the parsed DataFrame.

    Parameters
    ----------
    filename   : original filename (used to determine extension)
    file_content : raw bytes of the uploaded file

    Returns
    -------
    (ValidationResult, DataFrame | None)
    """
    cfg = get_dataset_config()
    max_bytes = cfg.get("max_file_size_mb", 50) * 1024 * 1024
    accepted_ext = cfg.get("accepted_extensions", [".csv", ".json"])

    result = ValidationResult(is_valid=True)

    # --- Extension check
    ext = Path(filename).suffix.lower()
    if ext not in accepted_ext:
        result.add_error(
            f"Unsupported file type '{ext}'. "
            f"Accepted types: {', '.join(accepted_ext)}"
        )
        return result, None

    # --- Size check
    if len(file_content) > max_bytes:
        result.add_error(
            f"File size ({len(file_content) / 1_048_576:.1f} MB) exceeds the "
            f"maximum allowed size ({cfg['max_file_size_mb']} MB)."
        )
        return result, None

    # --- Parse
    try:
        if ext == ".csv":
            df = pd.read_csv(io.BytesIO(file_content))
        elif ext == ".json":
            df = pd.read_json(io.BytesIO(file_content))
        else:
            result.add_error(f"Cannot parse file with extension '{ext}'.")
            return result, None
    except Exception as exc:
        result.add_error(f"Could not parse file: {exc}")
        return result, None

    # --- Column check
    required = cfg.get("required_columns", ["Review", "Aspect", "Sentiment"])
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        result.add_error(
            f"Dataset is missing required columns: {missing_cols}. "
            f"Found columns: {list(df.columns)}"
        )
        return result, None

    # --- Empty check
    if len(df) == 0:
        result.add_error("The uploaded file contains no data rows.")
        return result, None

    # --- Content warnings
    known_aspects = set(get_aspects())
    unknown_aspects = set(df["Aspect"].dropna().unique()) - known_aspects
    if unknown_aspects:
        result.add_warning(
            f"The following aspects are not in the configured taxonomy and will be ignored: "
            f"{sorted(unknown_aspects)}"
        )

    known_sentiments = set(get_sentiment_labels())
    unknown_sentiments = set(df["Sentiment"].dropna().unique()) - known_sentiments
    if unknown_sentiments:
        result.add_warning(
            f"The following sentiment labels are unrecognised: {sorted(unknown_sentiments)}"
        )

    missing_review_count = df["Review"].isna().sum()
    if missing_review_count:
        result.add_warning(
            f"{missing_review_count} rows have missing review text and will be dropped."
        )

    return result, df
