"""
InputParser — accepts Excel, CSV, JSON, plain text, unstructured data.
Returns a list of raw complaint strings for the agent pipeline.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import BinaryIO


def parse_input(source: str | Path | BinaryIO, fmt: str = "auto") -> list[str]:
    """
    Parse any input format into a list of complaint strings.
    fmt: "auto" | "text" | "json" | "csv" | "excel"
    """
    if isinstance(source, (str, Path)) and Path(source).exists():
        source = Path(source)
        fmt = fmt if fmt != "auto" else _detect_format(source)
        return _parse_file(source, fmt)
    # Plain string fallback
    return [str(source)]


def _detect_format(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".csv": "csv",
        ".json": "json",
        ".xlsx": "excel",
        ".xls": "excel",
        ".txt": "text",
    }.get(suffix, "text")


def _parse_file(path: Path, fmt: str) -> list[str]:
    if fmt == "text":
        return [path.read_text(encoding="utf-8")]

    if fmt == "json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [json.dumps(item) if isinstance(item, dict) else str(item) for item in data]
        return [json.dumps(data)]

    if fmt == "csv":
        results = []
        with path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Concatenate all fields into a readable string
                results.append(" | ".join(f"{k}: {v}" for k, v in row.items() if v))
        return results

    if fmt == "excel":
        try:
            import openpyxl
        except ImportError:
            raise ImportError("pip install openpyxl for Excel support")
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        ws = wb.active
        rows = list(ws.iter_rows(values_only=True))
        if not rows:
            return []
        headers = [str(h) for h in rows[0]]
        results = []
        for row in rows[1:]:
            parts = [f"{headers[i]}: {v}" for i, v in enumerate(row) if v is not None]
            if parts:
                results.append(" | ".join(parts))
        return results

    return [path.read_text(encoding="utf-8")]
