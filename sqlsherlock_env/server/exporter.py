# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Exporter for SQLSherlock-Env.

Writes the cleaned dataset in the SAME FORMAT as the original input.
Supported output formats: csv, json, jsonl, parquet, hf_dataset (→ csv).

Returns a file descriptor dict that the environment embeds in the
observation and that the /download/{file_id} endpoint serves.
"""

import csv
import io
import json
import os
import tempfile
import uuid
from typing import Any


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def export_cleaned(
    cleaned_rows: list[dict],
    source_format: str,
    dataset_name: str,
) -> dict:
    """Write cleaned rows to a temp file matching the original format.

    Args:
        cleaned_rows:  List of cleaned row dicts (no _source_format key).
        source_format: One of csv | json | jsonl | parquet | hf_dataset.
        dataset_name:  Original dataset name/path (used to derive filename).

    Returns:
        Dict with keys:
            file_id      — UUID string (used in /download/{file_id})
            filename     — human-readable filename
            format       — detected output format
            download_url — relative URL path
            row_count    — number of rows written
    """
    if not cleaned_rows:
        raise ValueError("Cannot export empty cleaned_rows list.")

    # Strip internal metadata column before writing
    rows = _strip_meta(cleaned_rows)

    file_id  = str(uuid.uuid4())
    stem     = _stem_from_name(dataset_name)
    fmt      = source_format if source_format in _WRITERS else "csv"

    filename, filepath = _make_temp_path(file_id, stem, fmt)

    _WRITERS[fmt](rows, filepath)

    return {
        "file_id":      file_id,
        "filename":     filename,
        "format":       fmt,
        "download_url": f"/download/{file_id}",
        "row_count":    len(rows),
        "filepath":     filepath,   # kept server-side for FileResponse
    }


# ---------------------------------------------------------------------------
# Format writers
# ---------------------------------------------------------------------------

def _write_csv(rows: list[dict], path: str) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(rows: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, default=str)


def _write_jsonl(rows: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")


def _write_parquet(rows: list[dict], path: str) -> None:
    try:
        import pandas as pd
    except ImportError:
        raise ValueError(
            "pandas is required to export Parquet files. "
            "pip install pandas pyarrow"
        )
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)


_WRITERS = {
    "csv":        _write_csv,
    "json":       _write_json,
    "jsonl":      _write_jsonl,
    "parquet":    _write_parquet,
    "hf_dataset": _write_csv,   # HF datasets exported as CSV
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_meta(rows: list[dict]) -> list[dict]:
    """Remove _source_format from every row."""
    return [
        {k: v for k, v in row.items() if k != "_source_format"}
        for row in rows
    ]


def _stem_from_name(dataset_name: str) -> str:
    """Derive a clean file stem from the dataset name."""
    if not dataset_name:
        return "cleaned"
    # HF dataset: "owner/name" or "owner/name:split"
    # For raw CSV text, take only the first line (header) to avoid huge filenames.
    first_line = dataset_name.strip().split("\n")[0]
    base = first_line.split(":")[0].split("/")[-1]
    safe = "".join(c if c.isalnum() or c == "_" else "_" for c in base.lower())
    # Truncate to 40 chars to stay well under filesystem path length limits.
    safe = (safe or "cleaned")[:40].rstrip("_")
    return (safe or "cleaned") + "_cleaned"


def _ext_for_format(fmt: str) -> str:
    return {
        "csv":        ".csv",
        "json":       ".json",
        "jsonl":      ".jsonl",
        "parquet":    ".parquet",
        "hf_dataset": ".csv",
    }.get(fmt, ".csv")


def _make_temp_path(file_id: str, stem: str, fmt: str) -> tuple[str, str]:
    """Return (filename, full_filepath) in the system temp directory."""
    ext      = _ext_for_format(fmt)
    filename = f"{stem}{ext}"
    filepath = os.path.join(tempfile.gettempdir(), f"{file_id}_{filename}")
    return filename, filepath
