# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Schema profiler for SQLSherlock-Env.

Computes per-column statistical profiles from raw records.
Used by DatabaseEngine at load time and by issue_detector / validator.
"""

import math
import sqlite3
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def profile_table(
    table: str,
    records: list[dict],
    conn: Optional[sqlite3.Connection] = None,
) -> dict[str, dict]:
    """Return a statistical profile for every column in *records*.

    Args:
        table:   Table name (stored in the profile for reference).
        records: List of row dicts (already coerced to Python types).
        conn:    Optional SQLite connection (unused currently; reserved for
                 future SQL-based profiling).

    Returns:
        Dict keyed by column name.  Each value is a column-profile dict::

            {
                "table":            str,
                "column":           str,
                "dtype":            "int" | "float" | "str" | "bool" | "unknown",
                "row_count":        int,
                "null_count":       int,
                "null_rate":        float,          # 0.0 – 1.0
                "unique_count":     int,
                "all_unique":       bool,
                "mean":             float | None,   # numeric only
                "std":              float | None,   # numeric only
                "min":              float | None,   # numeric only
                "max":              float | None,   # numeric only
                "must_be_positive": bool,           # numeric only
                "z_scores":         dict[int, float],  # row_id → z
                "sample_values":    list[Any],      # up to 5 non-null values
            }
    """
    if not records:
        return {}

    columns = list(records[0].keys())
    profile: dict[str, dict] = {}

    for col in columns:
        values = [row.get(col) for row in records]
        col_profile = _profile_column(table, col, values, records)
        profile[col] = col_profile

    return profile


def _profile_column(
    table: str,
    col: str,
    values: list[Any],
    records: list[dict],
) -> dict:
    """Compute statistics for a single column."""
    row_count = len(values)
    null_count = sum(1 for v in values if _is_null(v))
    null_rate = null_count / row_count if row_count > 0 else 0.0

    non_null = [v for v in values if not _is_null(v)]
    unique_count = len(set(str(v) for v in non_null))
    # all_unique: every non-null value is distinct AND covers all rows
    # Compare against row_count so that a column with 1 null among unique values
    # is NOT considered all-unique (the null breaks the uniqueness guarantee)
    all_unique = (unique_count == row_count) and row_count > 0 and null_count == 0

    dtype = _infer_dtype(non_null)

    # Numeric statistics
    mean = std = mn = mx = None
    must_be_positive = False
    z_scores: dict[int, float] = {}

    if dtype in ("int", "float") and non_null:
        numeric_vals = []
        for v in non_null:
            try:
                numeric_vals.append(float(v))
            except (ValueError, TypeError):
                pass

        if numeric_vals:
            mean = sum(numeric_vals) / len(numeric_vals)
            variance = sum((x - mean) ** 2 for x in numeric_vals) / len(numeric_vals)
            std = math.sqrt(variance)
            mn = min(numeric_vals)
            mx = max(numeric_vals)

            # must_be_positive: all non-null values are >= 0 and at least one > 0
            # Handles columns like age/fare that should never be negative
            must_be_positive = len(numeric_vals) > 0 and all(v >= 0 for v in numeric_vals) and any(v > 0 for v in numeric_vals)

            # z-scores per row keyed by primary key value
            # Use find_primary_key() for accuracy; fall back to first column
            pk_col = find_primary_key(records) if records else None
            if pk_col is None and records:
                pk_col = list(records[0].keys())[0]
            for row in records:
                raw = row.get(col)
                if _is_null(raw):
                    continue
                try:
                    fval = float(raw)
                except (ValueError, TypeError):
                    continue
                rid = row.get(pk_col) if pk_col else None
                if rid is not None and std > 0:
                    z = (fval - mean) / std
                    z_scores[int(rid)] = round(z, 4)
                elif rid is not None:
                    z_scores[int(rid)] = 0.0

    # Sample values: up to 5 non-null
    sample_values = non_null[:5]

    return {
        "table": table,
        "column": col,
        "dtype": dtype,
        "row_count": row_count,
        "null_count": null_count,
        "null_rate": round(null_rate, 4),
        "unique_count": unique_count,
        "all_unique": all_unique,
        "mean": round(mean, 6) if mean is not None else None,
        "std": round(std, 6) if std is not None else None,
        "min": mn,
        "max": mx,
        "must_be_positive": must_be_positive,
        "z_scores": z_scores,
        "sample_values": sample_values,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_null(value: Any) -> bool:
    """Return True if *value* represents a missing / null entry."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def _infer_dtype(non_null_values: list[Any]) -> str:
    """Infer column dtype from a list of non-null values.

    Priority: bool > int > float > str > unknown.
    """
    if not non_null_values:
        return "unknown"

    # Bool check first (Python bool is subclass of int)
    if all(isinstance(v, bool) for v in non_null_values):
        return "bool"

    # Try int
    int_ok = True
    for v in non_null_values:
        if isinstance(v, bool):
            int_ok = False
            break
        if isinstance(v, int):
            continue
        try:
            f = float(v)
            if f != int(f):
                int_ok = False
                break
        except (ValueError, TypeError):
            int_ok = False
            break
    if int_ok:
        return "int"

    # Try float
    float_ok = True
    for v in non_null_values:
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            continue
        try:
            float(v)
        except (ValueError, TypeError):
            float_ok = False
            break
    if float_ok:
        return "float"

    # Default to str
    if all(isinstance(v, str) for v in non_null_values):
        return "str"

    return "unknown"


def find_primary_key(records: list[dict]) -> Optional[str]:
    """Return the name of the primary-key column.

    Convention: the first column whose name is 'id' or ends with '_id',
    OR simply the first column if all values are unique integers.
    Falls back to the first column name.
    """
    if not records:
        return None

    columns = list(records[0].keys())
    if not columns:
        return None

    # Explicit id column
    for col in columns:
        if col.lower() == "id" or col.lower().endswith("_id"):
            vals = [row.get(col) for row in records]
            if len(set(str(v) for v in vals)) == len(vals):
                return col

    # First column with all-unique integer-like values
    first = columns[0]
    vals = [row.get(first) for row in records]
    try:
        int_vals = [int(v) for v in vals if v is not None]
        if len(int_vals) == len(records) and len(set(int_vals)) == len(int_vals):
            return first
    except (ValueError, TypeError):
        pass

    # Last resort: first column
    return first
