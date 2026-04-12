# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Universal grader for SQLSherlock-Env.

Implements the 7-step scoring pipeline shared by all task graders.
Task graders (task1/task2/task3) call grade() with an issue_filter
to restrict which issue types count toward the score.
"""

import math
from typing import Any, Optional

from server.issue_detector import SENTINEL_UNKNOWN


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def grade(
    db: Any,
    cleaned_rows: list[dict],
    removed_ids: list[int],
    task_id: str,
    validation_was_called: bool,
    issue_filter: Optional[set[str]] = None,
) -> float:
    """Score an agent's submitted solution in [0.0, 1.0].

    Args:
        db:                    DatabaseEngine for this episode.
        cleaned_rows:          Rows the agent claims are clean.
        removed_ids:           Row PKs the agent deleted.
        task_id:               Task identifier (used for trap / reasoning checks).
        validation_was_called: Whether validate() was called during the episode.
        issue_filter:          If set, only issues whose type is in this set
                               contribute to resolution_score.  None = all types.

    Returns:
        Float in [0.0, 1.0].
    """
    issue_registry = db.issue_registry
    pk_col = db.pk_col
    primary_table = db.primary_table

    # Filter issues by type if requested
    if issue_filter:
        scored_issues = [i for i in issue_registry if i.issue_type in issue_filter]
    else:
        scored_issues = list(issue_registry)

    # --- STEP 1: Zero-change check ---
    # Compare against the ORIGINAL dirty state (before any fixes), not the current state.
    # db.rows() returns the current (post-fix) state, so it would always match cleaned_rows.
    dirty_rows = db.original_state()

    if not removed_ids and _rows_identical(cleaned_rows, dirty_rows, pk_col):
        if db.total_issues > 0:
            return 0.0

    # --- STEP 2: Resolution score ---
    resolution_score, total_weight = _resolution_score(
        scored_issues, cleaned_rows, removed_ids, pk_col, db
    )

    # --- STEP 3: False positive penalty ---
    fp_penalty = _false_positive_penalty(
        db, cleaned_rows, removed_ids, pk_col, primary_table
    )

    # --- STEP 4: Trap penalty (task3 only) ---
    trap_penalty = _trap_penalty(db, cleaned_rows, removed_ids, pk_col, task_id)

    # --- STEP 5: Validation score ---
    validation_score = _validation_score(
        db, cleaned_rows, validation_was_called
    )

    # --- STEP 6: Reasoning bonus (task3 only) ---
    reasoning_bonus = _reasoning_bonus(db, task_id, validation_was_called)

    # --- STEP 7: Final score ---
    raw = (
        resolution_score * 0.60
        + validation_score  * 0.30
        + reasoning_bonus   * 0.10
        - fp_penalty
        - trap_penalty
    )
    return max(0.0, min(1.0, round(raw, 4)))


# ---------------------------------------------------------------------------
# Step implementations
# ---------------------------------------------------------------------------

def _resolution_score(
    issues: list,
    cleaned_rows: list[dict],
    removed_ids: list[int],
    pk_col: str,
    db: Any,
) -> tuple[float, float]:
    """Return (weighted_resolution_score, total_weight)."""
    if not issues:
        return 1.0, 1.0   # No issues to resolve → full resolution score

    cleaned_map = {row.get(pk_col): row for row in cleaned_rows if row.get(pk_col) is not None}
    removed_set  = set(removed_ids)
    total_weight = sum(i.confidence for i in issues)

    if total_weight == 0:
        return 0.0, 0.0

    # Per-column stats for outlier z-score recheck
    col_stats: dict[str, dict] = {}
    profile = db._profiles.get(db.primary_table, {})

    weighted_sum = 0.0

    for iss in issues:
        C = iss.confidence
        col = iss.column
        rid = iss.row_id

        p = profile.get(col, {}) if col else {}
        col_mean = p.get("mean")
        col_std  = p.get("std")

        resolved = _resolve_issue(
            iss, cleaned_map, removed_set, col_mean, col_std
        )
        weighted_sum += resolved * C

    return weighted_sum / total_weight, total_weight


def _resolve_issue(
    iss: Any,
    cleaned_map: dict,
    removed_set: set,
    col_mean: Optional[float],
    col_std: Optional[float],
) -> float:
    """Return a resolution score in [0.0, 1.0] for one issue."""
    C   = iss.confidence
    col = iss.column
    rid = iss.row_id

    itype = iss.issue_type

    # --- duplicate / fk_violation ---
    if itype in ("duplicate", "fk_violation"):
        if rid in removed_set:
            return 1.0
        if rid not in cleaned_map:
            return 1.0   # row absent from cleaned output = deleted
        return 0.0

    # --- null ---
    if itype == "null":
        row = cleaned_map.get(rid)
        if row is None:
            return 0.5 * C   # deleted instead of fixed
        val = row.get(col)
        if _is_null(val):
            return 0.0
        if iss.correct == SENTINEL_UNKNOWN:
            # Any non-null value of correct type accepted
            col_dtype = _guess_dtype(val)
            return C if col_dtype != "unknown" else C * 0.5
        return C if _values_match(val, iss.correct) else 0.0

    # --- type_error ---
    if itype == "type_error":
        row = cleaned_map.get(rid)
        if row is None:
            return 0.5
        val = row.get(col)
        if _is_null(val):
            return 0.0
        try:
            float(str(val))
            return 1.0
        except (ValueError, TypeError):
            return 0.0

    # --- constraint ---
    if itype == "constraint":
        row = cleaned_map.get(rid)
        if row is None:
            return 0.5 * C
        val = row.get(col)
        if _is_null(val):
            return 0.0
        try:
            fval = float(str(val))
        except (ValueError, TypeError):
            return 0.0
        if fval >= 0:
            correct = iss.correct
            if correct is not None and correct != SENTINEL_UNKNOWN:
                if fval <= abs(float(correct)) * 5:
                    return C           # positive and close to original
                return C * 0.7         # positive but far from original
            return C                   # unknown correct — any non-negative OK
        return 0.0                     # still negative

    # --- outlier ---
    if itype == "outlier":
        row = cleaned_map.get(rid)
        if row is None:
            return 0.5 * C
        val = row.get(col)
        if _is_null(val):
            return 0.0
        try:
            fval = float(str(val))
        except (ValueError, TypeError):
            return 0.0

        # Primary check: new value is close to the stored correct (column median).
        # The median is set at detection time from clean values — not contaminated.
        correct = iss.correct
        if correct is not None and correct != SENTINEL_UNKNOWN:
            try:
                cf = float(str(correct))
                if _values_match(fval, cf):
                    return C   # exact match to median → fully resolved
                # Accept any value within 2× std of the stored median
                if col_std is not None and col_std > 0:
                    z_from_median = abs(fval - cf) / col_std
                    if z_from_median <= 2.0:
                        return C        # close to median
                    if z_from_median <= 4.0:
                        return C * 0.7  # reasonable but not ideal
            except (ValueError, TypeError):
                pass

        # Fallback: z-score from profile mean (may be contaminated; best-effort)
        if col_mean is not None and col_std is not None and col_std > 0:
            try:
                z = abs(fval - col_mean) / col_std
                if z <= 3.0:
                    return C
                if z <= 5.0:
                    return C * 0.5
            except (ValueError, TypeError):
                pass
        return 0.0

    # --- whitespace ---
    if itype == "whitespace":
        row = cleaned_map.get(rid)
        if row is None:
            return 0.0
        val = row.get(col)
        if _is_null(val):
            return 0.0
        s = str(val)
        if s == " ".join(s.split()):
            return C  # whitespace cleaned
        return 0.0

    # --- inconsistent_category ---
    if itype == "inconsistent_category":
        row = cleaned_map.get(rid)
        if row is None:
            return 0.0
        val = row.get(col)
        if _is_null(val):
            return 0.0
        if _values_match(val, iss.correct):
            return C  # normalized to dominant form
        # Accept if same lowercase (partially resolved)
        if str(val).strip().lower() == str(iss.correct).strip().lower():
            return C * 0.8
        return 0.0

    return 0.0


def _false_positive_penalty(
    db: Any,
    cleaned_rows: list[dict],
    removed_ids: list[int],
    pk_col: str,
    primary_table: str,
) -> float:
    """Penalise changes to cells that were not in the issue registry."""
    originals = db._originals.get(primary_table, [])
    orig_map   = {row.get(pk_col): row for row in originals if row.get(pk_col) is not None}
    issue_cells = {
        (i.row_id, i.column)
        for i in db.issue_registry
        if i.column
    }
    issue_rows = {i.row_id for i in db.issue_registry}
    removed_set = set(removed_ids)

    fp_count = 0

    # Changed cells that are not in issue_registry
    for row in cleaned_rows:
        rid  = row.get(pk_col)
        orig = orig_map.get(rid)
        if orig is None:
            continue
        for col, val in row.items():
            if col == pk_col or col == "_source_format":
                continue
            if (rid, col) in issue_cells:
                continue
            if not _values_match(val, orig.get(col)):
                fp_count += 1

    # Rows that were deleted but should NOT have been
    orig_ids    = set(orig_map.keys())
    cleaned_ids = {row.get(pk_col) for row in cleaned_rows if row.get(pk_col) is not None} | removed_set
    wrongly_removed = orig_ids - cleaned_ids - issue_rows
    fp_count += len(wrongly_removed)

    return min(fp_count * 0.05, 0.35)


def _trap_penalty(
    db: Any,
    cleaned_rows: list[dict],
    removed_ids: list[int],
    pk_col: str,
    task_id: str,
) -> float:
    """Return 0.40 if the agent touched the trap cell, else 0.0."""
    if not task_id.endswith("_hard"):
        return 0.0
    trap = db.trap
    if trap is None:
        return 0.0

    removed_set = set(removed_ids)
    cleaned_map = {row.get(pk_col): row for row in cleaned_rows if row.get(pk_col) is not None}

    # Trap hit if row was deleted OR value was changed from trap_value
    if trap.row_id in removed_set:
        return 0.40

    trap_row = cleaned_map.get(trap.row_id)
    if trap_row is None:
        # Row not in cleaned output and not in removed_ids → deleted implicitly
        return 0.40

    current_val = trap_row.get(trap.column)
    # The trap value was written into the DB — if agent left it unchanged,
    # current_val should still equal trap.trap_value → no penalty.
    # If agent changed it (even to original) → penalty.
    if _values_match(current_val, trap.trap_value):
        return 0.0   # agent correctly left the trap alone
    return 0.40


def _validation_score(
    db: Any,
    cleaned_rows: list[dict],
    validation_was_called: bool,
) -> float:
    """Run all 6 validator checks on cleaned_rows and return pass ratio."""
    try:
        result = db._validator.validate(
            conn=db._conn,
            current_records=cleaned_rows,
            touched_columns=db._touched_columns,
        )
        score = result.checks_passed / result.total_checks
    except Exception:
        score = 0.0

    if not validation_was_called and db.total_issues > 0:
        score *= 0.50   # penalty for skipping validate()

    return round(score, 4)


def _reasoning_bonus(
    db: Any,
    task_id: str,
    validation_was_called: bool,
) -> float:
    """Return 0.05 if hard-level agent used statistical reasoning, else 0.0."""
    if not task_id.endswith("_hard"):
        return 0.0
    if not validation_was_called:
        return 0.0

    stat_terms = {
        "z-score", "z_score", "zscore", "mean", "std",
        "standard dev", "average", "distribution",
        "statistical", "outlier", "sigma",
    }
    all_reasons = " ".join(
        (a.reason or "") for a in db._action_log if hasattr(a, "reason")
    ).lower()

    return 0.05 if any(term in all_reasons for term in stat_terms) else 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rows_identical(
    cleaned_rows: list[dict],
    dirty_rows: list[dict],
    pk_col: str,
) -> bool:
    """Return True if cleaned_rows has the same values as dirty_rows."""
    if len(cleaned_rows) != len(dirty_rows):
        return False
    dirty_map = {row.get(pk_col): row for row in dirty_rows if row.get(pk_col) is not None}
    for row in cleaned_rows:
        rid  = row.get(pk_col)
        orig = dirty_map.get(rid)
        if orig is None:
            return False
        for col, val in row.items():
            if col == "_source_format":
                continue
            if not _values_match(val, orig.get(col)):
                return False
    return True


def _values_match(a: Any, b: Any) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    try:
        return math.isclose(float(str(a)), float(str(b)), rel_tol=1e-4)
    except (ValueError, TypeError):
        return str(a).strip().lower() == str(b).strip().lower()


def _is_null(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def _guess_dtype(value: Any) -> str:
    if value is None:
        return "unknown"
    try:
        f = float(str(value))
        return "int" if f == int(f) else "float"
    except (ValueError, TypeError):
        return "str"
