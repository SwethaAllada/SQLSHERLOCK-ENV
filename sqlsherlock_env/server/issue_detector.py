# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Issue detector for SQLSherlock-Env.

Scans real dataset records for genuine data-quality problems.
NEVER invents issues — synthetic top-up is used ONLY when real
issue count falls below the task minimum.

Detection order per task:
  task1: null_check + type_check
  task2: + range_check + fk_check
  task3: + outlier_check + duplicate_check
"""

import math
import random
import sqlite3
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SENTINEL_UNKNOWN = "__UNKNOWN__"

# ---------------------------------------------------------------------------
# Difficulty-level check sets (shared across all intents)
# ---------------------------------------------------------------------------

_EASY_CHECKS   = ["null", "type_error", "whitespace", "inconsistent_category"]
_MEDIUM_CHECKS = ["null", "type_error", "whitespace", "inconsistent_category",
                  "constraint", "outlier"]
_HARD_CHECKS   = ["null", "type_error", "whitespace", "inconsistent_category",
                  "constraint", "outlier", "duplicate", "fk_violation"]

MINIMUM_ISSUES: dict[str, int] = {
    # visualization intent
    "viz_easy":   3,  "viz_medium":  5,  "viz_hard":   7,
    # ml_training intent
    "ml_easy":    3,  "ml_medium":   5,  "ml_hard":    7,
    # business_query intent
    "bq_easy":    3,  "bq_medium":   5,  "bq_hard":    7,
}

# Which checks run per task (all hard tasks get the full audit)
TASK_CHECKS: dict[str, list[str]] = {
    "viz_easy":    _EASY_CHECKS,
    "ml_easy":     _EASY_CHECKS,
    "bq_easy":     _EASY_CHECKS,
    "viz_medium":  _MEDIUM_CHECKS,
    "ml_medium":   _MEDIUM_CHECKS,
    "bq_medium":   _MEDIUM_CHECKS,
    "viz_hard":    _HARD_CHECKS,
    "ml_hard":     _HARD_CHECKS,
    "bq_hard":     _HARD_CHECKS,
}

OUTLIER_Z_THRESHOLD = 5.0


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Issue:
    issue_id:   str
    issue_type: str        # null|type_error|constraint|outlier|duplicate|fk_violation
    table:      str
    row_id:     int
    column:     Optional[str]
    correct:    Any        # corrected value, None (delete), or SENTINEL_UNKNOWN
    confidence: float      # 0.0 – 1.0


@dataclass
class Trap:
    table:      str
    row_id:     int
    column:     str
    trap_value: float      # 2 × original (written into the DB)
    original:   float      # what we changed from


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_issues(
    conn: sqlite3.Connection,
    profile: dict[str, dict],
    records: list[dict],
    task_id: str,
    seed: int = 42,
) -> list[Issue]:
    """Detect real data-quality issues then apply synthetic top-up if needed.

    Args:
        conn:     Live SQLite connection (used for FK cross-table checks).
        profile:  Column profiles from schema_profiler.profile_table().
        records:  List of row dicts for the primary table.
        task_id:  One of the three task identifiers.
        seed:     RNG seed for reproducible synthetic top-up.

    Returns:
        List of Issue objects.  The agent NEVER sees this list directly.
    """
    checks = TASK_CHECKS.get(task_id, ["null", "type_error"])
    rng = random.Random(seed)

    pk_col = _find_pk_col(records)
    issues: list[Issue] = []
    seen: set[str] = set()   # deduplicate by (row_id, column, type)

    def _add(issue: Issue) -> None:
        key = f"{issue.row_id}_{issue.column}_{issue.issue_type}"
        if key not in seen:
            seen.add(key)
            issues.append(issue)

    # --- Real detection passes ---
    if "null" in checks:
        for iss in _detect_nulls(records, profile, pk_col):
            _add(iss)

    if "type_error" in checks:
        for iss in _detect_type_errors(records, profile, pk_col):
            _add(iss)

    if "constraint" in checks:
        for iss in _detect_constraints(records, profile, pk_col):
            _add(iss)

    if "outlier" in checks:
        for iss in _detect_outliers(records, profile, pk_col):
            _add(iss)

    if "duplicate" in checks:
        for iss in _detect_duplicates(records, profile, pk_col):
            _add(iss)

    if "fk_violation" in checks:
        table_names = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]
        if len(table_names) >= 2:
            primary = table_names[0]
            for iss in _detect_fk_violations(conn, records, profile, pk_col, primary, table_names[1:]):
                _add(iss)

    if "whitespace" in checks:
        for iss in _detect_whitespace(records, profile, pk_col):
            _add(iss)

    if "inconsistent_category" in checks:
        for iss in _detect_inconsistent_categories(records, profile, pk_col):
            _add(iss)

    # --- Synthetic top-up ---
    minimum = MINIMUM_ISSUES.get(task_id, 3)
    if len(issues) < minimum:
        synthetic = _plant_synthetic_topup(
            records, profile, pk_col, issues, checks,
            needed=minimum - len(issues), rng=rng,
        )
        issues.extend(synthetic)

    return issues


def detect_trap(
    conn: sqlite3.Connection,
    profile: dict[str, dict],
    records: list[dict],
    issue_registry: list[Issue],
    seed: int = 42,
) -> Optional[Trap]:
    """Plant a statistical trap for task3.

    Finds the highest-variance numeric column not involved in any registered
    issue, picks a row also not in the registry, sets its value to 2×original,
    and writes the change into SQLite.

    The Trap is NEVER added to issue_registry.  Touching it costs -0.40.

    Returns None if no suitable column/row exists.
    """
    rng = random.Random(seed + 1)

    if not records:
        return None

    pk_col = _find_pk_col(records)
    issue_cells: set[tuple[int, str]] = {
        (i.row_id, i.column) for i in issue_registry if i.column
    }
    issue_rows: set[int] = {i.row_id for i in issue_registry}

    # Find highest-variance numeric column with at least one eligible row.
    # We no longer exclude entire columns based on issue_columns — a column can
    # have one issue row (e.g. fare outlier at row 5) while still having many
    # clean rows available for the trap (e.g. fare at row 2).
    # We only exclude specific (row_id, col) cells via eligible_rows below.
    numeric_cols = [
        col for col, p in profile.items()
        if p["dtype"] in ("int", "float")
        and p["std"] is not None
        and p["std"] > 0
        and col != pk_col
        and col != "_source_format"
    ]

    # Prefer columns NOT in any issue for a cleaner trap, but fall back to any
    issue_columns: set[str] = {i.column for i in issue_registry if i.column}
    candidates = [c for c in numeric_cols if c not in issue_columns]
    if not candidates:
        candidates = numeric_cols  # fall back: use any numeric col with eligible rows

    if not candidates:
        return None

    # Highest variance column
    target_col = max(candidates, key=lambda c: profile[c]["std"] or 0.0)

    # Find a row not in issue_rows with a valid numeric value
    eligible_rows = [
        row for row in records
        if row.get(pk_col) is not None
        and int(row[pk_col]) not in issue_rows
        and not _is_null(row.get(target_col))
    ]
    if not eligible_rows:
        return None

    # Pick a row away from the extremes (avoid naturally high z-score rows)
    col_mean = profile[target_col]["mean"] or 0.0
    col_std  = profile[target_col]["std"]  or 1.0
    safe_rows = [
        r for r in eligible_rows
        if abs((float(r[target_col]) - col_mean) / col_std) < 2.0
    ]
    chosen_row = rng.choice(safe_rows if safe_rows else eligible_rows)
    rid = int(chosen_row[pk_col])
    original_val = float(chosen_row[target_col])
    trap_val = round(original_val * 2.0, 2)

    # Write trap value into SQLite
    primary_table = _primary_table_name(conn)
    if primary_table:
        conn.execute(
            f'UPDATE "{primary_table}" SET "{target_col}" = ? WHERE "{pk_col}" = ?',
            (trap_val, rid),
        )
        conn.commit()

    return Trap(
        table=primary_table or "dataset",
        row_id=rid,
        column=target_col,
        trap_value=trap_val,
        original=original_val,
    )


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def _detect_nulls(
    records: list[dict],
    profile: dict[str, dict],
    pk_col: str,
) -> list[Issue]:
    issues = []
    for col, p in profile.items():
        if col == pk_col or col == "_source_format":
            continue
        null_rate = p["null_rate"]
        for row in records:
            val = row.get(col)
            if not _is_null(val):
                continue
            rid = int(row[pk_col])
            # Confidence inversely proportional to null rate
            # High null rate (structural, like Cabin) → low confidence
            confidence = max(0.0, 1.0 - null_rate)
            correct = _infer_correct_null(col, row, records, p)
            issues.append(Issue(
                issue_id=_make_id(p["table"], rid, col, "null"),
                issue_type="null",
                table=p["table"],
                row_id=rid,
                column=col,
                correct=correct,
                confidence=round(confidence, 4),
            ))
    return issues


def _detect_type_errors(
    records: list[dict],
    profile: dict[str, dict],
    pk_col: str,
) -> list[Issue]:
    issues = []
    for col, p in profile.items():
        if col == pk_col or col == "_source_format":
            continue
        # Also check "unknown"/"str" dtype columns: when data is loaded from CSV via
        # SQLite, all values come back as strings. A column like age that has "25",
        # "FORTY", "-5" has dtype="str" but is a numeric column with a type error.
        if p["dtype"] not in ("int", "float", "unknown", "str"):
            continue
        if p["dtype"] in ("unknown", "str"):
            # Only flag type errors if the column is PREDOMINANTLY numeric (>=80%).
            # A column like Ticket with 40% numeric and 60% alphanumeric is genuinely
            # a string column — not a numeric column with type errors.
            non_null_vals = [r.get(col) for r in records if not _is_null(r.get(col))]
            if not non_null_vals:
                continue
            castable_count = sum(1 for v in non_null_vals if _can_cast_float(v))
            if castable_count / len(non_null_vals) < 0.80:
                continue  # column is genuinely string or mixed — not type errors
        col_median = _median([
            float(r[col]) for r in records
            if not _is_null(r.get(col)) and _can_cast_float(r.get(col))
        ])
        for row in records:
            val = row.get(col)
            if _is_null(val):
                continue
            if not _can_cast_float(val):
                rid = int(row[pk_col])
                issues.append(Issue(
                    issue_id=_make_id(p["table"], rid, col, "type_error"),
                    issue_type="type_error",
                    table=p["table"],
                    row_id=rid,
                    column=col,
                    correct=col_median,
                    confidence=1.0,
                ))
    return issues


def _detect_constraints(
    records: list[dict],
    profile: dict[str, dict],
    pk_col: str,
) -> list[Issue]:
    """Flag negative values in columns that must be positive."""
    issues = []
    for col, p in profile.items():
        if col == pk_col or col == "_source_format":
            continue
        # must_be_positive is only set for int/float dtype.
        # For "unknown" dtype columns (mixed type due to a type error), infer
        # must_be_positive from the castable values: if >= 75% are non-negative,
        # a negative value is a constraint violation.
        is_must_positive = p["must_be_positive"]
        if not is_must_positive and p["dtype"] in ("unknown", "str"):
            # For string/mixed-type columns (e.g. age stored as TEXT in SQLite),
            # infer must_be_positive from the castable values.
            castable = [
                float(r.get(col)) for r in records
                if not _is_null(r.get(col)) and _can_cast_float(r.get(col))
            ]
            if castable and sum(v >= 0 for v in castable) / len(castable) >= 0.75:
                is_must_positive = True
        if not is_must_positive:
            continue
        for row in records:
            val = row.get(col)
            if _is_null(val):
                continue
            try:
                fval = float(val)
            except (ValueError, TypeError):
                continue
            if fval < 0:
                rid = int(row[pk_col])
                issues.append(Issue(
                    issue_id=_make_id(p["table"], rid, col, "constraint"),
                    issue_type="constraint",
                    table=p["table"],
                    row_id=rid,
                    column=col,
                    correct=abs(fval),
                    confidence=0.95,
                ))
    return issues


def _detect_outliers(
    records: list[dict],
    profile: dict[str, dict],
    pk_col: str,
) -> list[Issue]:
    """Detect outliers using IQR method (robust to outlier-inflated std).

    Standard z-score fails on small datasets because the outlier inflates the
    mean and std, masking itself.  IQR is resistant to this masking effect.
    Threshold: value outside Q1 - 3*IQR or Q3 + 3*IQR (stricter than 1.5× Tukey).
    """
    issues = []
    for col, p in profile.items():
        if col == pk_col or col == "_source_format":
            continue
        if p["dtype"] not in ("int", "float"):
            continue

        # Collect castable numeric values for this column
        numeric_rows: list[tuple[int, float]] = []
        for row in records:
            val = row.get(col)
            if _is_null(val):
                continue
            try:
                numeric_rows.append((int(row[pk_col]), float(val)))
            except (ValueError, TypeError):
                continue

        if len(numeric_rows) < 4:
            continue

        values = sorted(v for _, v in numeric_rows)
        n = len(values)
        q1 = values[n // 4]
        q3 = values[(3 * n) // 4]
        iqr = q3 - q1
        if iqr == 0:
            continue

        lower_fence = q1 - 3.0 * iqr
        upper_fence = q3 + 3.0 * iqr
        col_median  = values[n // 2]

        for rid, fval in numeric_rows:
            if fval < lower_fence or fval > upper_fence:
                # Use IQR-based score for confidence
                distance = max(fval - upper_fence, lower_fence - fval)
                confidence = min(0.99, round(0.60 + distance / (iqr * 10.0 + 1e-9), 4))
                issues.append(Issue(
                    issue_id=_make_id(p["table"], rid, col, "outlier"),
                    issue_type="outlier",
                    table=p["table"],
                    row_id=rid,
                    column=col,
                    correct=round(col_median, 4),
                    confidence=round(confidence, 4),
                ))
    return issues


def _detect_duplicates(
    records: list[dict],
    profile: dict[str, dict],
    pk_col: str,
) -> list[Issue]:
    natural_key = _find_natural_key_col(profile, records, pk_col)
    if natural_key is None:
        return []

    seen: dict[str, int] = {}   # value → first row_id
    issues = []
    table = profile[pk_col]["table"] if pk_col in profile else "dataset"

    for row in records:
        val = row.get(natural_key)
        if _is_null(val):
            continue
        key_str = str(val).strip().lower()
        rid = int(row[pk_col])
        if key_str in seen:
            # Later insertion is the duplicate
            issues.append(Issue(
                issue_id=_make_id(table, rid, natural_key, "duplicate"),
                issue_type="duplicate",
                table=table,
                row_id=rid,
                column=natural_key,
                correct=None,   # should be deleted
                confidence=1.0,
            ))
        else:
            seen[key_str] = rid

    return issues


def _detect_fk_violations(
    conn: sqlite3.Connection,
    records: list[dict],
    profile: dict[str, dict],
    pk_col: str,
    primary_table: str,
    other_tables: list[str],
) -> list[Issue]:
    issues = []

    # Find FK-like columns: name ends with _id but is not the PK
    fk_cols = [
        col for col in profile
        if col.lower().endswith("_id")
        and col != pk_col
        and col != "_source_format"
    ]

    for fk_col in fk_cols:
        # Guess the referenced table by stripping _id
        ref_name = fk_col[:-3]  # e.g. "passenger_id" → "passenger"
        ref_table = None
        for tbl in other_tables:
            if tbl.lower().startswith(ref_name.lower()) or ref_name.lower() in tbl.lower():
                ref_table = tbl
                break
        if ref_table is None and other_tables:
            ref_table = other_tables[0]
        if ref_table is None:
            continue

        # Fetch valid FK values from referenced table
        try:
            ref_rows = conn.execute(f'SELECT * FROM "{ref_table}" LIMIT 1000').fetchall()
            ref_desc = conn.execute(f'PRAGMA table_info("{ref_table}")').fetchall()
            ref_pk_idx = 0  # first column
            valid_ids = {str(r[ref_pk_idx]) for r in ref_rows}
        except Exception:
            continue

        table = profile[pk_col]["table"] if pk_col in profile else primary_table
        for row in records:
            val = row.get(fk_col)
            if _is_null(val):
                continue
            if str(val) not in valid_ids:
                rid = int(row[pk_col])
                issues.append(Issue(
                    issue_id=_make_id(table, rid, fk_col, "fk_violation"),
                    issue_type="fk_violation",
                    table=table,
                    row_id=rid,
                    column=fk_col,
                    correct=None,   # orphan row — should be deleted
                    confidence=0.90,
                ))

    return issues


# ---------------------------------------------------------------------------
# Whitespace / formatting issues
# ---------------------------------------------------------------------------

def _detect_whitespace(
    records: list[dict],
    profile: dict[str, dict],
    pk_col: str,
) -> list[Issue]:
    """Flag strings with leading/trailing whitespace or excessive internal spaces."""
    issues = []
    for col, p in profile.items():
        if col == pk_col or col == "_source_format":
            continue
        if p["dtype"] not in ("str", "unknown"):
            continue
        table = p.get("table", "dataset")
        for row in records:
            val = row.get(col)
            if _is_null(val) or not isinstance(val, str):
                continue
            cleaned = " ".join(val.split())  # normalize whitespace
            if cleaned != val:
                rid = int(row[pk_col])
                issues.append(Issue(
                    issue_id=_make_id(table, rid, col, "whitespace"),
                    issue_type="whitespace",
                    table=table,
                    row_id=rid,
                    column=col,
                    correct=cleaned,
                    confidence=0.90,
                ))
    return issues


# ---------------------------------------------------------------------------
# Inconsistent categories (e.g. "F"/"Female"/"female" → "Female")
# ---------------------------------------------------------------------------

def _detect_inconsistent_categories(
    records: list[dict],
    profile: dict[str, dict],
    pk_col: str,
) -> list[Issue]:
    """Flag values that are case-variants or abbreviations of the dominant category.

    Example: column Sex has {"male": 40, "Male": 2, "MALE": 1} → "Male" and "MALE"
    should be normalized to "male" (the dominant form).
    """
    issues = []
    for col, p in profile.items():
        if col == pk_col or col == "_source_format":
            continue
        if p["dtype"] not in ("str", "unknown"):
            continue
        # Only check low-cardinality columns (likely categorical)
        unique = p.get("unique_count", 0)
        row_count = p.get("row_count", 0)
        if unique == 0 or row_count == 0 or unique > 20:
            continue  # too many unique values — not categorical

        # Group values by lowercase form
        from collections import Counter
        val_counts: Counter = Counter()
        original_forms: dict[str, list[str]] = {}  # lowercase → [original forms]
        for row in records:
            val = row.get(col)
            if _is_null(val) or not isinstance(val, str):
                continue
            val_stripped = val.strip()
            lower = val_stripped.lower()
            val_counts[lower] += 1
            if lower not in original_forms:
                original_forms[lower] = []
            if val_stripped not in original_forms[lower]:
                original_forms[lower].append(val_stripped)

        # Find groups with multiple surface forms
        table = p.get("table", "dataset")
        for lower_key, forms in original_forms.items():
            if len(forms) <= 1:
                continue
            # Dominant form: most common original casing
            form_counts = Counter()
            for row in records:
                val = row.get(col)
                if isinstance(val, str) and val.strip().lower() == lower_key:
                    form_counts[val.strip()] += 1
            dominant = form_counts.most_common(1)[0][0]

            # Flag non-dominant forms
            for row in records:
                val = row.get(col)
                if not isinstance(val, str):
                    continue
                stripped = val.strip()
                if stripped.lower() == lower_key and stripped != dominant:
                    rid = int(row[pk_col])
                    issues.append(Issue(
                        issue_id=_make_id(table, rid, col, "inconsistent_category"),
                        issue_type="inconsistent_category",
                        table=table,
                        row_id=rid,
                        column=col,
                        correct=dominant,
                        confidence=0.85,
                    ))
    return issues


# ---------------------------------------------------------------------------
# Synthetic top-up
# ---------------------------------------------------------------------------

def _plant_synthetic_topup(
    records: list[dict],
    profile: dict[str, dict],
    pk_col: str,
    existing: list[Issue],
    allowed_checks: list[str],
    needed: int,
    rng: random.Random,
) -> list[Issue]:
    """Plant statistically valid synthetic issues when real count < minimum.

    Never touches: PK column, natural-key column, columns already in existing.
    """
    synthetic: list[Issue] = []
    touched_cells: set[tuple[int, str]] = {(i.row_id, i.column) for i in existing if i.column}
    natural_key = _find_natural_key_col(profile, records, pk_col)

    # Columns available for synthetic planting
    def available_cols(dtype_filter=None) -> list[str]:
        cols = []
        for col, p in profile.items():
            if col == pk_col or col == "_source_format":
                continue
            if col == natural_key:
                continue
            if dtype_filter and p["dtype"] not in dtype_filter:
                continue
            cols.append(col)
        return cols

    table = profile[pk_col]["table"] if pk_col in profile else "dataset"

    # Candidate issue types to synthesise (ordered by preference)
    type_order = []
    if "null" in allowed_checks:
        type_order.append("null")
    if "type_error" in allowed_checks:
        type_order.append("type_error")
    if "constraint" in allowed_checks:
        type_order.append("constraint")

    planted = 0
    attempt = 0
    max_attempts = needed * 20

    while planted < needed and attempt < max_attempts:
        attempt += 1
        issue_type = type_order[planted % len(type_order)]

        if issue_type == "null":
            cols = available_cols()
            if not cols:
                continue
            col = rng.choice(cols)
            eligible = [
                r for r in records
                if not _is_null(r.get(col))
                and (int(r[pk_col]), col) not in touched_cells
            ]
            if not eligible:
                continue
            row = rng.choice(eligible)
            rid = int(row[pk_col])
            original = row[col]
            # Plant NULL in the live records
            row[col] = None
            touched_cells.add((rid, col))
            synthetic.append(Issue(
                issue_id=_make_id(table, rid, col, "null"),
                issue_type="null",
                table=table,
                row_id=rid,
                column=col,
                correct=original,
                confidence=0.95,
            ))
            planted += 1

        elif issue_type == "type_error":
            cols = available_cols(dtype_filter=("int", "float"))
            if not cols:
                continue
            col = rng.choice(cols)
            eligible = [
                r for r in records
                if not _is_null(r.get(col))
                and _can_cast_float(r.get(col))
                and (int(r[pk_col]), col) not in touched_cells
            ]
            if not eligible:
                continue
            row = rng.choice(eligible)
            rid = int(row[pk_col])
            # Plant "INVALID_TEXT" in the live records
            row[col] = "INVALID_TEXT"
            col_median = _median([
                float(r[col]) for r in records
                if not _is_null(r.get(col)) and _can_cast_float(r.get(col))
            ])
            touched_cells.add((rid, col))
            synthetic.append(Issue(
                issue_id=_make_id(table, rid, col, "type_error"),
                issue_type="type_error",
                table=table,
                row_id=rid,
                column=col,
                correct=col_median,
                confidence=1.0,
            ))
            planted += 1

        elif issue_type == "constraint":
            cols = [
                col for col in available_cols(dtype_filter=("int", "float"))
                if profile[col].get("must_be_positive", False)
            ]
            if not cols:
                # Fall back to any positive-valued numeric col
                cols = [
                    col for col in available_cols(dtype_filter=("int", "float"))
                    if profile[col].get("min", 0) is not None
                    and (profile[col].get("min") or 0) > 0
                ]
            if not cols:
                continue
            col = rng.choice(cols)
            eligible = [
                r for r in records
                if not _is_null(r.get(col))
                and _can_cast_float(r.get(col))
                and float(r.get(col, 0)) > 0
                and (int(r[pk_col]), col) not in touched_cells
            ]
            if not eligible:
                continue
            row = rng.choice(eligible)
            rid = int(row[pk_col])
            original = float(row[col])
            row[col] = -abs(original)
            touched_cells.add((rid, col))
            synthetic.append(Issue(
                issue_id=_make_id(table, rid, col, "constraint"),
                issue_type="constraint",
                table=table,
                row_id=rid,
                column=col,
                correct=original,
                confidence=0.95,
            ))
            planted += 1

    return synthetic


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _find_pk_col(records: list[dict]) -> str:
    """Return the primary key column name from records.

    Looks for 'id' column first, then falls back to first column.
    """
    if not records:
        return "id"
    keys = list(records[0].keys())
    # Prefer explicit 'id' column
    for k in keys:
        if k.lower() == "id":
            return k
    # Fall back to first column
    return keys[0]


def _find_natural_key_col(
    profile: dict[str, dict],
    records: list[dict],
    pk_col: str,
) -> Optional[str]:
    """Return the natural key column if one exists, else None.

    Natural key: high uniqueness (>= 70%), not float dtype, not PK,
    name contains: name, email, code, ref, id_, key, title.

    Uses 70% threshold (not strict all_unique) so that dirty datasets with
    a small number of duplicates still have their natural key identified.
    """
    KEY_HINTS = ("name", "email", "code", "ref", "id_", "key", "title")
    for col, p in profile.items():
        if col == pk_col or col == "_source_format":
            continue
        if p["dtype"] == "float":
            continue
        row_count = p.get("row_count", 0)
        unique_count = p.get("unique_count", 0)
        if row_count == 0:
            continue
        uniqueness_ratio = unique_count / row_count
        if uniqueness_ratio < 0.70:
            continue
        col_lower = col.lower()
        if any(hint in col_lower for hint in KEY_HINTS):
            return col
    return None


def _infer_correct_null(
    col: str,
    row: dict,
    records: list[dict],
    p: dict,
) -> Any:
    """Best-guess correct value for a null cell."""
    if p["dtype"] in ("int", "float"):
        non_null = [
            float(r[col]) for r in records
            if not _is_null(r.get(col)) and _can_cast_float(r.get(col))
        ]
        if non_null:
            return round(_median(non_null), 4)
    return SENTINEL_UNKNOWN


def _median(values: list[float]) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 0:
        return (s[mid - 1] + s[mid]) / 2.0
    return s[mid]


def _can_cast_float(value: Any) -> bool:
    try:
        float(str(value))
        return True
    except (ValueError, TypeError):
        return False


def _is_null(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def _make_id(table: str, row_id: int, col: Optional[str], issue_type: str) -> str:
    return f"{table}_{row_id}_{col or 'row'}_{issue_type}"


def _primary_table_name(conn: sqlite3.Connection) -> Optional[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY rowid"
    ).fetchall()
    return rows[0][0] if rows else None
