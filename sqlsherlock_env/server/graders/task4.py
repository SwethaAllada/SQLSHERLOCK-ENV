# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Task 4 grader — Context-aware multi-table analysis.

Scoring formula:
    task4_score = base_cleaning_score  × 0.40
               + join_quality_score    × 0.20
               + intent_alignment      × 0.20
               + downstream_score      × 0.20
               - over_cleaning_penalty

base_cleaning_score uses the task3 grader (all issue types including outlier/duplicate).
join_quality_score rewards agents that performed a correct join on the context table.
intent_alignment scores how well fixes matched the declared/true intent.
downstream_score measures post-cleaning data quality via lightweight sklearn evaluation.
over_cleaning_penalty applies if >20% of rows are deleted without matching issues.
"""

import math
from typing import Any, Optional

from server.database import DatabaseEngine
from server.graders.task3 import grade as task3_grade
from server.graders.universal import (
    _resolution_score,
    _rows_identical,
    _reasoning_bonus,
    _validation_score,
    _is_null,
    _values_match,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def grade(
    db: DatabaseEngine,
    cleaned_rows: list[dict],
    removed_ids: list[int],
    validation_was_called: bool,
) -> float:
    """Score a task4 submission.

    Args:
        db:                    DatabaseEngine for this episode.
        cleaned_rows:          Agent-provided cleaned rows.
        removed_ids:           Agent-provided deleted row PKs.
        validation_was_called: Whether validate() was called.

    Returns:
        Float score in [0.0, 1.0].
    """
    pk_col = db.pk_col

    # Zero-change guard — compare against ORIGINAL dirty state
    dirty_rows = db.original_state()
    if not removed_ids and _rows_identical(cleaned_rows, dirty_rows, pk_col):
        if db.total_issues > 0:
            return 0.0

    # --- Base cleaning quality (reuse task3 grader for all issue types) ---
    base_score = task3_grade(
        db=db,
        cleaned_rows=cleaned_rows,
        removed_ids=removed_ids,
        validation_was_called=validation_was_called,
    )

    # --- Join quality ---
    join_score = _join_quality_score(db)

    # --- Intent alignment ---
    intent_score = _intent_alignment_score(db, cleaned_rows, removed_ids)

    # --- Downstream data quality ---
    downstream_score = _downstream_score(db, cleaned_rows)

    # --- Over-cleaning penalty ---
    overclean_pen = _over_cleaning_penalty(db, cleaned_rows, removed_ids)

    raw = (
        base_score         * 0.40
        + join_score       * 0.20
        + intent_score     * 0.20
        + downstream_score * 0.20
        - overclean_pen
    )
    return max(0.0, min(1.0, round(raw, 4)))


# ---------------------------------------------------------------------------
# Sub-score implementations
# ---------------------------------------------------------------------------

def _join_quality_score(db: DatabaseEngine) -> float:
    """Score the agent's join attempts.

    Returns 1.0 if a valid join was performed on the context table,
    0.5 if no join was attempted (partial credit for other cleaning),
    0.3 if only incorrect joins were attempted.
    """
    context_table = getattr(db, "_context_table", None)
    join_attempts = getattr(db, "_join_attempts", [])

    if context_table is None:
        return 1.0  # No context table → joins not required

    if not join_attempts:
        return 0.5  # Agent didn't join — loses multi-table marks

    # Check if any attempt was a valid join involving the context table
    correct_key = getattr(db, "_context_join_key", ("", ""))[0]  # primary FK col
    for attempt in join_attempts:
        if attempt.get("table2") == context_table and attempt.get("valid"):
            return 1.0
        if attempt.get("table1") == context_table and attempt.get("valid"):
            return 1.0

    # Attempts were made but none were correct
    return 0.3


def _intent_alignment_score(
    db: DatabaseEngine,
    cleaned_rows: list[dict],
    removed_ids: list[int],
) -> float:
    """Score how well the cleaning strategy aligned with the intent.

    Evaluates the *outcome* (what was cleaned) relative to the intent,
    independent of the reward given during the episode.
    """
    intent = getattr(db, "_intent", None)
    if intent is None:
        return 0.8  # No intent set — neutral

    pk_col = db.pk_col
    original_count = len(db.original_state())
    cleaned_count  = len(cleaned_rows)
    preserved_ratio = cleaned_count / max(original_count, 1)

    if intent == "dashboard":
        # Dashboard intent: preserve rows (>80% = full score)
        if preserved_ratio >= 0.90:
            return 1.0
        elif preserved_ratio >= 0.80:
            return 0.85
        elif preserved_ratio >= 0.70:
            return 0.60
        else:
            return max(0.0, preserved_ratio)

    elif intent == "ml_training":
        # ML intent: outliers and type errors should be resolved
        ml_issues = [
            i for i in db.issue_registry
            if i.issue_type in ("outlier", "type_error", "constraint")
        ]
        if not ml_issues:
            return 1.0
        cleaned_map  = {r[pk_col]: r for r in cleaned_rows}
        removed_set  = set(removed_ids)
        profile      = db._profiles.get(db.primary_table, {})
        resolved = sum(
            1 for iss in ml_issues
            if _issue_resolved(iss, cleaned_map, removed_set, profile)
        )
        return resolved / len(ml_issues)

    elif intent == "reporting":
        # Reporting intent: whitespace and inconsistent categories should be resolved
        rep_issues = [
            i for i in db.issue_registry
            if i.issue_type in ("whitespace", "inconsistent_category", "null")
        ]
        if not rep_issues:
            return 1.0
        cleaned_map  = {r[pk_col]: r for r in cleaned_rows}
        removed_set  = set(removed_ids)
        profile      = db._profiles.get(db.primary_table, {})
        resolved = sum(
            1 for iss in rep_issues
            if _issue_resolved(iss, cleaned_map, removed_set, profile)
        )
        return resolved / len(rep_issues)

    return 0.8  # Unknown intent — neutral


def _issue_resolved(
    iss: Any,
    cleaned_map: dict,
    removed_set: set,
    profile: dict,
) -> bool:
    """Return True if the issue appears resolved in cleaned_map."""
    rid = iss.row_id
    col = iss.column

    if rid in removed_set or rid not in cleaned_map:
        return iss.issue_type in ("duplicate", "fk_violation")

    row = cleaned_map[rid]
    val = row.get(col)

    if iss.issue_type == "null":
        return not _is_null(val)
    if iss.issue_type == "type_error":
        if _is_null(val):
            return False
        try:
            float(str(val))
            return True
        except (ValueError, TypeError):
            return False
    if iss.issue_type == "constraint":
        if _is_null(val):
            return False
        try:
            return float(str(val)) >= 0
        except (ValueError, TypeError):
            return False
    if iss.issue_type == "outlier":
        p = profile.get(col, {})
        mean, std = p.get("mean"), p.get("std")
        if mean is None or not std or std == 0:
            return True
        try:
            return abs(float(str(val)) - mean) / std <= 3.0
        except (ValueError, TypeError):
            return False
    if iss.issue_type == "whitespace":
        if _is_null(val):
            return False
        s = str(val)
        return s == " ".join(s.split())
    if iss.issue_type == "inconsistent_category":
        return _values_match(val, iss.correct)

    return False


def _downstream_score(db: DatabaseEngine, cleaned_rows: list[dict]) -> float:
    """Evaluate downstream data quality of the cleaned dataset.

    For ml_training intent: trains a lightweight LogisticRegression and
    compares accuracy on dirty vs. clean data.
    For all other intents: measures null-ratio improvement as a proxy.
    """
    intent = getattr(db, "_intent", None)
    if intent == "ml_training":
        return _ml_downstream_score(db, cleaned_rows)
    return _analytics_downstream_score(db, cleaned_rows)


def _ml_downstream_score(db: DatabaseEngine, cleaned_rows: list[dict]) -> float:
    """Train LogisticRegression on dirty vs. clean data; reward if accuracy improves."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import LabelEncoder
        import numpy as np
    except ImportError:
        return 0.75  # sklearn not installed — neutral

    profile = db._profiles.get(db.primary_table, {})
    pk_col  = db.pk_col

    # Find target column (binary/low-cardinality) and numeric feature columns
    target_col  = None
    feature_cols: list[str] = []
    for col, p in profile.items():
        if col in (pk_col, "_source_format"):
            continue
        unique = p.get("unique_count", 0)
        if unique == 2 and target_col is None:
            target_col = col
        elif p["dtype"] in ("int", "float"):
            feature_cols.append(col)

    if target_col is None or len(feature_cols) < 2:
        return 0.75  # insufficient structure for ML evaluation

    def _vectorize(rows: list[dict]):
        X, y = [], []
        for row in rows:
            t = row.get(target_col)
            if t is None:
                continue
            feats = []
            for fc in feature_cols:
                v = row.get(fc)
                try:
                    feats.append(float(str(v)) if v is not None else 0.0)
                except (ValueError, TypeError):
                    feats.append(0.0)
            X.append(feats)
            y.append(str(t))
        return np.array(X, dtype=float), np.array(y)

    try:
        dirty_rows = db.original_state()
        X_d, y_d = _vectorize(dirty_rows)
        X_c, y_c = _vectorize(cleaned_rows)

        if len(X_d) < 10 or len(X_c) < 10:
            return 0.75

        le = LabelEncoder()
        y_d_enc = le.fit_transform(y_d)
        clf_d = LogisticRegression(max_iter=300, random_state=42, n_jobs=1)
        clf_d.fit(X_d, y_d_enc)
        dirty_acc = clf_d.score(X_d, y_d_enc)

        try:
            y_c_enc = le.transform(y_c)
        except ValueError:
            y_c_enc = LabelEncoder().fit_transform(y_c)
        clf_c = LogisticRegression(max_iter=300, random_state=42, n_jobs=1)
        clf_c.fit(X_c, y_c_enc)
        clean_acc = clf_c.score(X_c, y_c_enc)

        delta = clean_acc - dirty_acc
        if delta > 0.02:
            return min(1.0, 0.75 + delta * 2.5)   # meaningful improvement
        elif delta > -0.02:
            return 0.75                             # roughly equal
        else:
            return max(0.30, 0.75 + delta * 2.5)   # degraded
    except Exception:
        return 0.75


def _analytics_downstream_score(db: DatabaseEngine, cleaned_rows: list[dict]) -> float:
    """Measure post-cleaning null ratio in numeric columns as a downstream proxy."""
    if not cleaned_rows:
        return 0.0

    profile = db._profiles.get(db.primary_table, {})
    pk_col  = db.pk_col
    numeric_cols = [
        col for col, p in profile.items()
        if p["dtype"] in ("int", "float") and col != pk_col
    ]

    if not numeric_cols:
        return 0.80  # No numeric columns — can't measure

    total = 0
    null_count = 0
    for row in cleaned_rows:
        for col in numeric_cols:
            total += 1
            val = row.get(col)
            if val is None or (isinstance(val, float) and math.isnan(val)):
                null_count += 1

    if total == 0:
        return 0.80

    null_ratio = null_count / total
    if null_ratio == 0.0:
        return 1.0
    elif null_ratio <= 0.05:
        return 0.90
    elif null_ratio <= 0.15:
        return 0.70
    elif null_ratio <= 0.30:
        return 0.50
    else:
        return max(0.20, 1.0 - null_ratio)


def _over_cleaning_penalty(
    db: DatabaseEngine,
    cleaned_rows: list[dict],
    removed_ids: list[int],
) -> float:
    """Penalise excessive row deletion beyond what issues warranted.

    Penalty applies when:
      - More than 20% of original rows are missing from cleaned output.
      - The excess removals exceed the count of duplicate/fk issues.
    """
    original = db.original_state()
    orig_count = len(original)
    if orig_count == 0:
        return 0.0

    preserved_count = len(cleaned_rows)
    total_removed   = orig_count - preserved_count  # includes rows not in cleaned_rows

    # How many removals were legitimately required?
    legitimate_removals = sum(
        1 for iss in db.issue_registry
        if iss.issue_type in ("duplicate", "fk_violation")
    )

    removal_ratio = total_removed / orig_count
    if removal_ratio <= 0.20:
        return 0.0  # within acceptable range

    excess_ratio  = removal_ratio - 0.20
    excess_rows   = max(0, total_removed - legitimate_removals)
    excess_factor = excess_rows / max(orig_count, 1)

    return min(0.20, excess_factor * 0.5)
