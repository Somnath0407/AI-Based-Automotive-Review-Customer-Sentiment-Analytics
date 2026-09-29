"""
src/export/export_service.py
─────────────────────────────
CSV / JSON / Excel export service.

All export methods return bytes so they can be streamed directly
through Streamlit's st.download_button without touching the filesystem.
"""

from __future__ import annotations

import io
import json
from typing import Optional

import pandas as pd


class ExportService:
    """
    Converts a results DataFrame to downloadable byte payloads.

    Supported formats
    -----------------
    - CSV  : pandas to_csv (UTF-8 with BOM for Excel compatibility)
    - JSON : records-oriented JSON
    - Excel: openpyxl-based xlsx with auto-column widths and header styling
    """

    # ── Public API ────────────────────────────────────────────────────────────

    def to_csv(self, df: pd.DataFrame) -> bytes:
        """Return the DataFrame as UTF-8 CSV bytes (BOM-prefixed for Excel)."""
        buffer = io.StringIO()
        df.to_csv(buffer, index=False, encoding="utf-8")
        return ("\ufeff" + buffer.getvalue()).encode("utf-8")

    def to_json(self, df: pd.DataFrame, indent: int = 2) -> bytes:
        """Return the DataFrame as pretty-printed JSON bytes (records format)."""
        records = df.to_dict(orient="records")
        return json.dumps(records, ensure_ascii=False, indent=indent).encode("utf-8")

    def to_excel(self, df: pd.DataFrame, sheet_name: str = "Results") -> bytes:
        """
        Return the DataFrame as an xlsx file bytes with styled headers.

        Requires openpyxl (listed in requirements.txt).
        """
        try:
            import openpyxl
            from openpyxl.styles import Font, PatternFill, Alignment
        except ImportError as exc:
            raise ImportError("openpyxl is required for Excel export.") from exc

        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name=sheet_name)

            ws = writer.sheets[sheet_name]

            # Style the header row
            header_fill = PatternFill("solid", fgColor="1D4ED8")
            header_font = Font(bold=True, color="FFFFFF", name="Calibri", size=11)
            for cell in ws[1]:
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = Alignment(horizontal="center", vertical="center")

            # Auto-fit column widths
            for column_cells in ws.columns:
                max_len = max(
                    (len(str(c.value)) if c.value is not None else 0)
                    for c in column_cells
                )
                col_letter = column_cells[0].column_letter
                ws.column_dimensions[col_letter].width = min(max_len + 4, 60)

            # Freeze the header row
            ws.freeze_panes = "A2"

        return buffer.getvalue()

    # ── Convenience helpers ───────────────────────────────────────────────────

    def get_mime_type(self, fmt: str) -> str:
        """Return the MIME type for a format string ('csv', 'json', 'excel')."""
        return {
            "csv": "text/csv",
            "json": "application/json",
            "excel": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        }.get(fmt.lower(), "application/octet-stream")

    def get_extension(self, fmt: str) -> str:
        """Return the file extension for a format string."""
        return {
            "csv": ".csv",
            "json": ".json",
            "excel": ".xlsx",
        }.get(fmt.lower(), ".bin")
