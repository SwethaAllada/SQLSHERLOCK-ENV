# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Dataset loader for SQLSherlock-Env.

Supports: local CSV/JSON/JSONL/Parquet, HuggingFace dataset names, raw CSV text.
ZERO defaults — raises ValueError if source is empty or unrecognisable.
"""

import csv
import io
import json
import math
import os
import sqlite3
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load(source: str, max_rows: int = 500) -> dict[str, list[dict]]:
    """Load a dataset from *source* and return a table-name → records mapping.

    Args:
        source:   One of:
                    - Absolute/relative path ending in .csv/.json/.jsonl/.parquet
                    - HuggingFace dataset name  "owner/name" or "owner/name:split"
                    - Raw CSV text (multi-line string with comma-separated header)
        max_rows: Maximum rows to keep per table.

    Returns:
        Dict mapping table name (str) to list of row dicts.
        Each dict has an "id" key added if not already present.
        A ``_source_format`` key is injected into each record for the
        exporter to reconstruct the original format.

    Raises:
        ValueError: On empty source, auth failure, not found, too few rows,
                    no columns, or unrecognised format.
    """
    if not source or not source.strip():
        raise ValueError("Dataset source must not be empty.")

    source = source.strip()

    # Dispatch to loader
    if _is_local_file(source):
        records, fmt = _load_local(source, max_rows)
    elif _is_hf_dataset(source):
        records, fmt = _load_hf(source, max_rows)
    elif _looks_like_csv_text(source):
        records, fmt = _load_raw_csv(source, max_rows)
    else:
        raise ValueError(
            f"Unrecognised source '{source}'. "
            "Provide a file path (.csv/.json/.jsonl/.parquet), "
            "a HuggingFace dataset name (owner/name), "
            "or raw CSV text."
        )

    _validate_records(records)
    records = _ensure_id_column(records)
    records = coerce(records)

    # Inject source format so exporter can match output format
    for row in records:
        row["_source_format"] = fmt

    table_name = _table_name_from_source(source)
    return {table_name: records}


def coerce(records: list[dict]) -> list[dict]:
    """Auto-detect and coerce int/float values per column.

    For each column, if ALL non-null values can be cast to int → cast to int.
    Else if ALL non-null values can be cast to float → cast to float.
    Otherwise leave as string.

    The ``_source_format`` and ``id`` columns are never coerced.
    """
    if not records:
        return records

    columns = [c for c in records[0].keys() if c not in ("_source_format",)]

    for col in columns:
        values = [r.get(col) for r in records]
        non_null = [v for v in values if not _is_null(v)]
        if not non_null:
            continue

        target_type = _detect_target_type(non_null)
        if target_type is None:
            continue

        for row in records:
            v = row.get(col)
            if _is_null(v):
                row[col] = None
                continue
            try:
                fval = float(str(v))
                if target_type == "int":
                    # Only cast to int if value is genuinely whole-number
                    # (avoids silently truncating 3.7 → 3)
                    row[col] = int(fval) if fval == int(fval) else fval
                else:
                    row[col] = fval
            except (ValueError, TypeError):
                pass  # leave as-is if cast fails (type_error issue will detect it)

    return records


def records_to_sqlite(
    conn: sqlite3.Connection,
    table: str,
    records: list[dict],
) -> None:
    """Write *records* into an in-memory SQLite table.

    Creates the table fresh (DROP IF EXISTS then CREATE).
    Column types are inferred from the records.

    The ``_source_format`` column is NOT written to SQLite
    (it is preserved in the Python records only).
    """
    if not records:
        raise ValueError(f"Cannot create table '{table}' from empty records.")

    # Filter out the internal metadata column
    columns = [c for c in records[0].keys() if c != "_source_format"]

    # Infer SQLite column types
    col_types = {}
    for col in columns:
        vals = [r.get(col) for r in records if not _is_null(r.get(col))]
        col_types[col] = _sqlite_type(vals)

    col_defs = ", ".join(
        f'"{col}" {col_types[col]}' for col in columns
    )

    conn.execute(f'DROP TABLE IF EXISTS "{table}"')
    conn.execute(f'CREATE TABLE "{table}" ({col_defs})')

    placeholders = ", ".join("?" for _ in columns)
    rows_to_insert = [
        tuple(_sqlite_val(r.get(col)) for col in columns)
        for r in records
    ]
    conn.executemany(
        f'INSERT INTO "{table}" VALUES ({placeholders})',
        rows_to_insert,
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Local file loaders
# ---------------------------------------------------------------------------

def _load_local(path: str, max_rows: int) -> tuple[list[dict], str]:
    p = Path(path)
    if not p.exists():
        raise ValueError(f"File not found: {path}")

    suffix = p.suffix.lower()
    if suffix == ".csv":
        return _load_csv_file(p, max_rows), "csv"
    elif suffix == ".json":
        return _load_json_file(p, max_rows), "json"
    elif suffix == ".jsonl":
        return _load_jsonl_file(p, max_rows), "jsonl"
    elif suffix == ".parquet":
        return _load_parquet_file(p, max_rows), "parquet"
    else:
        raise ValueError(
            f"Unsupported file extension '{suffix}'. "
            "Use .csv, .json, .jsonl, or .parquet."
        )


def _load_csv_file(path: Path, max_rows: int) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = []
        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            rows.append(dict(row))
    return rows


def _load_json_file(path: Path, max_rows: int) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        # Might be {records: [...]} or similar
        for key in ("records", "data", "rows", "items"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
        else:
            raise ValueError("JSON file must contain a list of records.")
    if not isinstance(data, list):
        raise ValueError("JSON file must contain a list of records.")
    return [dict(r) for r in data[:max_rows]]


def _load_jsonl_file(path: Path, max_rows: int) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_rows:
                break
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_parquet_file(path: Path, max_rows: int) -> list[dict]:
    try:
        import pandas as pd
    except ImportError:
        raise ValueError("pandas is required to load Parquet files. pip install pandas pyarrow")
    df = pd.read_parquet(path)
    df = df.head(max_rows)
    return _df_to_records(df)


# ---------------------------------------------------------------------------
# HuggingFace dataset loader
# ---------------------------------------------------------------------------

def _load_hf(source: str, max_rows: int) -> tuple[list[dict], str]:
    """Load a dataset from HuggingFace Hub.

    source format: "owner/name" or "owner/name:split"
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ValueError(
            "The 'datasets' package is required for HuggingFace datasets. "
            "pip install datasets"
        )

    # Parse split
    split = "train"
    name = source
    if ":" in source:
        name, split = source.rsplit(":", 1)

    hf_token = os.environ.get("HF_TOKEN")

    try:
        ds = load_dataset(name, split=split, token=hf_token)
    except Exception as exc:
        msg = str(exc).lower()
        if "401" in msg or "unauthorized" in msg or "authentication" in msg:
            raise ValueError(
                f"Dataset '{name}' requires authentication. "
                "Use a public dataset or set the HF_TOKEN environment variable."
            ) from exc
        if "404" in msg or "not found" in msg or "doesn't exist" in msg:
            raise ValueError(
                f"Dataset '{name}' not found. "
                "Check the owner/name format (e.g. 'mstz/titanic')."
            ) from exc
        raise ValueError(f"Failed to load HuggingFace dataset '{source}': {exc}") from exc

    # Convert to list of dicts
    try:
        import pandas as pd
        df = ds.to_pandas().head(max_rows)
        records = _df_to_records(df)
    except Exception:
        records = [dict(row) for row in ds.select(range(min(max_rows, len(ds))))]

    return records, "hf_dataset"


# ---------------------------------------------------------------------------
# Raw CSV text loader
# ---------------------------------------------------------------------------

def _load_raw_csv(source: str, max_rows: int) -> tuple[list[dict], str]:
    reader = csv.DictReader(io.StringIO(source))
    rows = []
    for i, row in enumerate(reader):
        if i >= max_rows:
            break
        rows.append(dict(row))
    return rows, "csv"


# ---------------------------------------------------------------------------
# Validation & helpers
# ---------------------------------------------------------------------------

def _validate_records(records: list[dict]) -> None:
    if not records:
        raise ValueError("Dataset loaded 0 rows. Need at least 5.")
    if len(records) < 5:
        raise ValueError(
            f"Dataset has only {len(records)} rows. Need at least 5."
        )
    if not records[0]:
        raise ValueError("Dataset has no columns.")


def _ensure_id_column(records: list[dict]) -> list[dict]:
    """Guarantee every record has an integer 'id' column as the FIRST field."""
    if not records:
        return records

    # Check all columns for a PK-like column (not just the first)
    all_cols = list(records[0].keys())
    pk_col = None
    for col in all_cols:
        if col.lower() in ("id", "passengerid", "index", "passengerId"):
            pk_col = col
            break

    if pk_col is not None:
        # Rename to 'id' and reorder to put it first
        for i, row in enumerate(records):
            pk_val = row.pop(pk_col) if pk_col != "id" else row.pop("id")
            try:
                pk_val = int(pk_val)
            except (ValueError, TypeError):
                pk_val = i + 1
            # Rebuild dict with 'id' first
            records[i] = {"id": pk_val, **row}
        return records

    # No obvious PK — inject sequential id as first field
    for i, row in enumerate(records):
        records[i] = {"id": i + 1, **row}

    return records


def _table_name_from_source(source: str) -> str:
    """Derive a clean table name from the source string."""
    if _is_local_file(source):
        stem = Path(source).stem
        return _sanitise_name(stem)
    if _is_hf_dataset(source):
        base = source.split(":")[0]          # strip split
        parts = base.split("/")
        return _sanitise_name(parts[-1])     # e.g. "titanic"
    return "dataset"


def _sanitise_name(name: str) -> str:
    """Return a SQLite-safe lowercase identifier."""
    safe = "".join(c if c.isalnum() or c == "_" else "_" for c in name.lower())
    if safe and safe[0].isdigit():
        safe = "t_" + safe
    return safe or "dataset"


def _is_local_file(source: str) -> bool:
    return any(source.lower().endswith(ext) for ext in (".csv", ".json", ".jsonl", ".parquet"))


def _is_hf_dataset(source: str) -> bool:
    """Heuristic: 'owner/name' with no spaces and not a file path."""
    if "/" not in source:
        return False
    if any(source.lower().endswith(ext) for ext in (".csv", ".json", ".jsonl", ".parquet")):
        return False
    if "\n" in source or "," not in source.split("\n")[0]:
        # Might still be HF if no comma in first line
        parts = source.split("/")
        return len(parts) == 2 or (len(parts) == 2 and ":" in parts[-1])
    return "/" in source and "\n" not in source and len(source.split("/")) == 2


def _looks_like_csv_text(source: str) -> bool:
    """Return True if source looks like raw CSV text (has newlines and commas)."""
    lines = source.strip().splitlines()
    return len(lines) >= 2 and "," in lines[0]


def _detect_target_type(non_null: list[Any]) -> Optional[str]:
    """Return 'int' or 'float' if all values are numeric, else None."""
    # Try int
    try:
        for v in non_null:
            f = float(str(v))
            if f != int(f):
                raise ValueError
        return "int"
    except (ValueError, TypeError):
        pass
    # Try float
    try:
        for v in non_null:
            float(str(v))
        return "float"
    except (ValueError, TypeError):
        pass
    return None


def _is_null(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def _sqlite_type(non_null_vals: list[Any]) -> str:
    if not non_null_vals:
        return "TEXT"
    target = _detect_target_type(non_null_vals)
    if target == "int":
        return "INTEGER"
    if target == "float":
        return "REAL"
    return "TEXT"


def _sqlite_val(value: Any) -> Any:
    """Convert a Python value to a SQLite-compatible scalar."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (int, float, str, bytes)):
        return value
    return str(value)


def _df_to_records(df) -> list[dict]:
    """Convert a pandas DataFrame to a list of plain Python dicts."""
    import math as _math
    records = []
    for _, row in df.iterrows():
        d = {}
        for col, val in row.items():
            # Convert numpy/pandas scalars to Python natives
            if hasattr(val, "item"):
                try:
                    val = val.item()
                except Exception:
                    val = str(val)
            if isinstance(val, float) and _math.isnan(val):
                val = None
            d[str(col)] = val
        records.append(d)
    return records
