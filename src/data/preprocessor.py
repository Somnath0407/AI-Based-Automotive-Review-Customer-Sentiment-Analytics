"""
src/data/preprocessor.py
─────────────────────────
Text preprocessing pipeline for vehicle reviews.

Responsibilities
----------------
- Clean and normalise raw review text before it reaches the transformer.
- Handle common noise: extra whitespace, HTML entities, control characters.
- Preserve automotive terminology while removing irrelevant artefacts.
"""

from __future__ import annotations

import re
import unicodedata
from typing import List


class ReviewPreprocessor:
    """
    Lightweight text preprocessor for automotive review strings.

    The preprocessing is intentionally conservative — transformers are robust
    to moderate noise, so we only strip artefacts that would harm tokenisation.
    """

    # HTML entity table for quick replacement
    _HTML_ENTITIES = {
        "&amp;": "&", "&lt;": "<", "&gt;": ">",
        "&quot;": '"', "&#39;": "'", "&nbsp;": " ",
    }

    def preprocess_single(self, text: str) -> str:
        """
        Preprocess a single review string.

        Parameters
        ----------
        text : Raw review text.

        Returns
        -------
        Cleaned text string. Raises ValueError if the result is empty.
        """
        if not isinstance(text, str):
            raise ValueError(f"Expected str, got {type(text).__name__}")

        # 1. Decode common HTML entities
        for entity, char in self._HTML_ENTITIES.items():
            text = text.replace(entity, char)

        # 2. Normalise unicode (NFKC: canonical + compatibility decomposition)
        text = unicodedata.normalize("NFKC", text)

        # 3. Remove control characters (but keep newlines → spaces)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

        # 4. Convert newlines / tabs to spaces
        text = re.sub(r"[\n\r\t]+", " ", text)

        # 5. Collapse multiple spaces
        text = re.sub(r" {2,}", " ", text)

        # 6. Strip leading / trailing whitespace
        text = text.strip()

        if not text:
            raise ValueError("Review is empty after preprocessing.")

        return text

    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """
        Preprocess a list of reviews.

        Rows that fail preprocessing are replaced with an empty string and
        a warning is logged; they should be filtered out before training.
        """
        import logging
        logger = logging.getLogger(__name__)

        results: List[str] = []
        for i, text in enumerate(texts):
            try:
                results.append(self.preprocess_single(text))
            except ValueError as exc:
                logger.warning("Row %d skipped during preprocessing: %s", i, exc)
                results.append("")
        return results
