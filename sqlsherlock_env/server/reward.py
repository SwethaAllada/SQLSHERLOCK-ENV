# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Reward calculator for SQLSherlock-Env.

Dense per-step rewards with hard caps on investigation bonuses.
Every action produces a reward signal so the RL agent gets
continuous feedback throughout the episode.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Per-action reward magnitudes
# ---------------------------------------------------------------------------

INVEST_REWARDS: dict[str, float] = {
    "inspect":        0.02,
    "profile_column": 0.03,
    "run_sql":        0.03,
    "select_tables":  0.02,
}

INVEST_CAPS: dict[str, int] = {
    "inspect":        3,
    "profile_column": 3,
    "run_sql":        3,
    "validate":       2,
    "select_tables":  2,
}

FIX_CORRECT:        float =  0.15
FIX_FALSE_POSITIVE: float = -0.20
FIX_TRAP:           float = -0.40
FIX_WRONG_VALUE:    float = -0.10

DELETE_CORRECT:        float =  0.15
DELETE_FALSE_POSITIVE: float = -0.20

SUBMIT_ALL_RESOLVED:   float =  0.10
SUBMIT_ISSUES_OPEN:    float = -0.10

# --- Intent-aware and multi-table rewards ---
INTENT_ALIGNED:      float =  0.10   # fix action aligns with declared intent
INTENT_MISALIGNED:   float = -0.10   # fix action contradicts intent (e.g. delete on dashboard)
CLASSIFY_CORRECT:    float =  0.10   # agent correctly inferred the episode intent
CLASSIFY_INCORRECT:  float = -0.10   # agent misclassified the intent
JOIN_CORRECT:        float =  0.20   # valid join with correct key
JOIN_INCORRECT:      float = -0.20   # invalid join (wrong key / cartesian product)

# Issue types that align with each intent
_INTENT_ISSUE_TYPES: dict[str, set[str]] = {
    "visualization":    {"null", "type_error", "whitespace", "inconsistent_category"},
    "ml_training":      {"null", "type_error", "constraint", "outlier", "duplicate"},
    "business_query":   {"null", "whitespace", "inconsistent_category", "type_error",
                         "fk_violation", "constraint"},
}


# ---------------------------------------------------------------------------
# InvestCounter — tracks capped investigation calls
# ---------------------------------------------------------------------------

class InvestCounter:
    """Tracks how many times each investigation action has been called.

    Once an action type hits its cap, further calls still execute
    but return 0 reward (no error raised).
    """

    def __init__(self) -> None:
        self._counts: dict[str, int] = {k: 0 for k in INVEST_CAPS}

    def record(self, action_type: str) -> float:
        """Record one call of *action_type* and return the reward earned.

        Returns 0.0 if the cap has already been reached.
        Always increments the counter so validate_reward() can detect over-cap.
        """
        if action_type not in INVEST_CAPS:
            return 0.0

        cap = INVEST_CAPS[action_type]
        current = self._counts.get(action_type, 0)

        # Always increment so validate_reward() can detect over-cap correctly.
        self._counts[action_type] = current + 1

        if current >= cap:
            return 0.0  # cap already hit before this call

        if action_type == "validate":
            # Reward computed externally (depends on checks_passed)
            return 0.0   # caller computes and adds the validate reward

        return INVEST_REWARDS.get(action_type, 0.0)

    def validate_reward(self, checks_passed: int, total_checks: int) -> float:
        """Return the validate reward if under cap, else 0.0.

        Must be called AFTER record("validate") so the count is incremented.
        """
        count = self._counts.get("validate", 0)
        if count > INVEST_CAPS["validate"]:  # count already incremented by record()
            return 0.0
        # count == cap means this IS the last rewarded call (e.g. cap=2, count=2 → reward)
        # count > cap means over the limit → 0 (checked above)
        if total_checks == 0:
            return 0.0
        return round(0.05 * (checks_passed / total_checks), 4)

    def count(self, action_type: str) -> int:
        return self._counts.get(action_type, 0)

    def to_dict(self) -> dict:
        return dict(self._counts)


# ---------------------------------------------------------------------------
# RB — per-step reward breakdown
# ---------------------------------------------------------------------------

@dataclass
class RB:
    """Reward breakdown for one step.

    Stored in reward_trace every step so judges (and the agent) can
    see exactly how reward was composed.
    """
    invest:     float = 0.0   # investigation bonus
    fix_delta:  float = 0.0   # fix / delete reward (positive or negative)
    validate_b: float = 0.0   # validate bonus
    penalty:    float = 0.0   # trap / fp / submit penalties (stored negative)
    intent_r:   float = 0.0   # intent alignment bonus/penalty
    join_r:     float = 0.0   # join quality reward

    @property
    def total(self) -> float:
        raw = (self.invest + self.fix_delta + self.validate_b
               + self.penalty + self.intent_r + self.join_r)
        return max(-1.0, min(1.0, round(raw, 4)))

    def to_dict(self) -> dict:
        return {
            "invest":     round(self.invest, 4),
            "fix_delta":  round(self.fix_delta, 4),
            "validate_b": round(self.validate_b, 4),
            "penalty":    round(self.penalty, 4),
            "intent_r":   round(self.intent_r, 4),
            "join_r":     round(self.join_r, 4),
            "total":      self.total,
        }


# ---------------------------------------------------------------------------
# calc — main reward function called from environment.py
# ---------------------------------------------------------------------------

def calc(
    action_type: str,
    db: Any,                          # DatabaseEngine (typed loosely to avoid circular)
    counter: InvestCounter,
    action: Any,                      # SQLSherlockAction
    validation_result: Optional[Any] = None,  # ValidationResult | None
    intent: Optional[str] = None,     # episode cleaning intent (dashboard|ml_training|reporting)
    correct_intent: Optional[str] = None,  # grader-known correct intent (for classify_intent)
    join_valid: Optional[bool] = None,     # result of join validation (for join_tables)
) -> RB:
    """Compute per-step reward for one action.

    Args:
        action_type:       The action type string.
        db:                Live DatabaseEngine instance.
        counter:           Shared InvestCounter for this episode.
        action:            The SQLSherlockAction taken.
        validation_result: Result from Validator.validate() if action_type=="validate".

    Returns:
        RB breakdown.  Caller adds rb.to_dict() to reward_trace.
    """
    rb = RB()

    # ------------------------------------------------------------------
    # Investigation actions
    # ------------------------------------------------------------------
    if action_type in ("inspect", "profile_column", "run_sql"):
        rb.invest = counter.record(action_type)
        return rb

    if action_type == "select_tables":
        # Only reward if there are actually multiple tables to explore
        if len(db.table_names()) >= 2:
            rb.invest = counter.record(action_type)
        # else: single-table episode — no reward for pointless select_tables
        return rb

    # ------------------------------------------------------------------
    # classify_intent — agent declares inferred cleaning intent
    # ------------------------------------------------------------------
    if action_type == "classify_intent":
        guessed = str(getattr(action, "value", "") or "").strip().lower()
        valid_intents = {"visualization", "ml_training", "business_query"}
        if guessed not in valid_intents:
            # Unknown intent string — small penalty
            rb.intent_r = CLASSIFY_INCORRECT * 0.5
        elif correct_intent is not None:
            rb.intent_r = CLASSIFY_CORRECT if guessed == correct_intent else CLASSIFY_INCORRECT
        # If correct_intent is None, no intent reward (episode has no hidden target)
        return rb

    # ------------------------------------------------------------------
    # join_tables — validate the join quality
    # ------------------------------------------------------------------
    if action_type == "join_tables":
        if join_valid is True:
            rb.join_r = JOIN_CORRECT
        elif join_valid is False:
            rb.join_r = JOIN_INCORRECT
        # join_valid=None means we couldn't determine → neutral
        return rb

    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------
    if action_type == "validate":
        counter.record("validate")   # increment count (may be over cap)
        if validation_result is not None:
            rb.validate_b = counter.validate_reward(
                validation_result.checks_passed,
                validation_result.total_checks,
            )
        return rb

    # ------------------------------------------------------------------
    # fix_cell
    # ------------------------------------------------------------------
    if action_type == "fix_cell":
        table  = action.table or db.primary_table
        row_id = action.row_id
        column = action.column

        if row_id is None or column is None:
            rb.penalty = FIX_FALSE_POSITIVE
            return rb

        # Trap check (task3 only — highest priority)
        trap = db.trap
        if trap and trap.row_id == row_id and trap.column == column:
            rb.penalty = FIX_TRAP
            return rb

        # Is this cell in the issue registry?
        issue_match = _find_issue(db, row_id, column)

        if issue_match is None:
            # Not a known issue — check if we changed a clean original cell
            orig = _original_val(db, table, row_id, column)
            current_val = action.value
            if orig is not None and not _values_match(current_val, orig):
                rb.penalty = FIX_FALSE_POSITIVE
            # If we can't find original (row may not exist), small FP penalty
            elif orig is None:
                rb.penalty = FIX_FALSE_POSITIVE
            return rb

        # Issue exists — check if the fix actually resolves it
        if _fix_resolves(issue_match, action.value, db):
            rb.fix_delta = FIX_CORRECT
        else:
            rb.fix_delta = FIX_WRONG_VALUE

        # Intent alignment bonus: fixing an intent-aligned issue type
        if intent and rb.fix_delta > 0:
            aligned_types = _INTENT_ISSUE_TYPES.get(intent, set())
            if issue_match.issue_type in aligned_types:
                rb.intent_r = INTENT_ALIGNED * 0.5   # half-weight per individual cell

        return rb

    # ------------------------------------------------------------------
    # delete_row
    # ------------------------------------------------------------------
    if action_type == "delete_row":
        table  = action.table or db.primary_table
        row_id = action.row_id

        if row_id is None:
            rb.penalty = DELETE_FALSE_POSITIVE
            return rb

        # Valid delete: row must be a duplicate or fk_violation issue
        valid_issue = any(
            iss.row_id == row_id and iss.issue_type in ("duplicate", "fk_violation")
            for iss in db.issue_registry
        )
        if valid_issue:
            rb.fix_delta = DELETE_CORRECT
        else:
            rb.penalty = DELETE_FALSE_POSITIVE

        # Intent misalignment: visualization prefers row preservation over deletion
        if intent == "visualization" and rb.fix_delta > 0:
            rb.intent_r = INTENT_MISALIGNED * 0.3  # mild penalty for visualization + delete

        return rb

    # ------------------------------------------------------------------
    # fix_column (bulk fix)
    # ------------------------------------------------------------------
    if action_type == "fix_column":
        column = action.column
        if column is None:
            rb.penalty = FIX_FALSE_POSITIVE
            return rb

        # Count how many registered issues in this column were null-type
        column_issues = [
            iss for iss in db.issue_registry
            if iss.column == column and iss.issue_type in ("null", "type_error", "whitespace")
        ]
        if column_issues:
            # Reward scales with what fraction of all issues this bulk fix covers.
            # FIX_CORRECT * (1.0 + 2.0 * fraction) → range +0.15 to +0.45
            resolved_fraction = min(len(column_issues) / max(db.total_issues, 1), 1.0)
            rb.fix_delta = round(FIX_CORRECT * (1.0 + 2.0 * resolved_fraction), 4)
            # Intent alignment: fixing null/whitespace aligns with dashboard/reporting
            if intent:
                aligned_types = _INTENT_ISSUE_TYPES.get(intent, set())
                aligned_issues = [i for i in column_issues if i.issue_type in aligned_types]
                if aligned_issues:
                    rb.intent_r = INTENT_ALIGNED * 0.5
        else:
            # No registered issues in this column — possible false positive
            rb.penalty = FIX_FALSE_POSITIVE * 0.5  # lighter penalty for bulk ops
        return rb

    # ------------------------------------------------------------------
    # submit
    # ------------------------------------------------------------------
    if action_type == "submit":
        if db.issues_remaining() == 0:
            rb.fix_delta = SUBMIT_ALL_RESOLVED
        else:
            rb.penalty = SUBMIT_ISSUES_OPEN
        return rb

    # ------------------------------------------------------------------
    # export  (no direct step reward; grader scores the file)
    # ------------------------------------------------------------------
    if action_type == "export":
        return rb

    return rb


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_issue(db: Any, row_id: int, column: str):
    """Return the matching Issue from the registry using O(1) dict lookup.

    The issue index is lazily built and cached on the db object.
    """
    if not hasattr(db, "_issue_index"):
        db._issue_index = {
            (iss.row_id, iss.column): iss
            for iss in db.issue_registry
            if iss.column is not None
        }
    return db._issue_index.get((row_id, column))


def _original_val(db: Any, table: str, row_id: int, column: str) -> Any:
    """Return the original (pre-episode) value for a cell using O(1) dict lookup.

    The originals index is lazily built and cached on the db object.
    """
    cache_key = f"_orig_index_{table}"
    if not hasattr(db, cache_key):
        originals = db._originals.get(table, [])
        pk = db.pk_col
        setattr(db, cache_key, {row.get(pk): row for row in originals})
    orig_map = getattr(db, cache_key)
    row = orig_map.get(row_id)
    return row.get(column) if row is not None else None


def _fix_resolves(issue: Any, new_value: Any, db: Any) -> bool:
    """Return True if *new_value* resolves *issue*."""
    from server.issue_detector import SENTINEL_UNKNOWN

    itype = issue.issue_type

    if itype == "null":
        if _is_null(new_value):
            return False
        if issue.correct == SENTINEL_UNKNOWN:
            return True   # any non-null value accepted
        # Accept the fix if the value matches OR is the same type.
        # For numeric nulls: any valid numeric value is a reasonable fix
        # (the agent imputes from column statistics, not from our stored correct).
        if _values_match(new_value, issue.correct):
            return True
        # Type-compatible acceptance: if correct is numeric, accept any numeric
        if _can_cast_float(issue.correct) and _can_cast_float(new_value):
            return True
        # If correct is string, accept any non-null string
        if isinstance(issue.correct, str) and isinstance(new_value, str):
            return True
        return False

    if itype == "type_error":
        return _can_cast_float(new_value)

    if itype == "constraint":
        try:
            return float(str(new_value)) >= 0
        except (ValueError, TypeError):
            return False

    if itype == "outlier":
        # Resolves if new value is within 2 std of the stored correct (column median).
        # Using the stored correct avoids relying on contaminated profile statistics.
        if not _can_cast_float(new_value):
            return False
        fval = float(str(new_value))
        correct = issue.correct
        if correct is not None and correct != SENTINEL_UNKNOWN and _can_cast_float(correct):
            cf = float(str(correct))
            profile = db._profiles.get(db.primary_table, {})
            p = profile.get(issue.column, {})
            std = p.get("std")
            if std and std > 0:
                return abs(fval - cf) / std <= 2.0
            # No std available — any value close to the median counts
            return abs(fval - cf) <= abs(cf) * 0.20 if cf != 0 else fval == 0.0
        # Fallback: contaminated profile mean (best effort)
        profile = db._profiles.get(db.primary_table, {})
        p = profile.get(issue.column, {})
        mean = p.get("mean")
        std  = p.get("std")
        if mean is None or not std or std == 0:
            return True   # can't compute — assume resolved
        try:
            return abs(float(str(new_value)) - mean) / std <= 3.0
        except (ValueError, TypeError):
            return False

    if itype == "whitespace":
        # Resolved if the new value has no leading/trailing/excessive whitespace
        if _is_null(new_value):
            return False
        s = str(new_value)
        return s == " ".join(s.split())

    if itype == "inconsistent_category":
        # Resolved if new value matches the correct (dominant) form
        if _is_null(new_value):
            return False
        return _values_match(new_value, issue.correct)

    return False


def _values_match(a: Any, b: Any) -> bool:
    """Loose equality: handles numeric vs string comparisons."""
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


def _can_cast_float(value: Any) -> bool:
    try:
        float(str(value))
        return True
    except (ValueError, TypeError):
        return False
