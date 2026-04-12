# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Task 2 grader — ML Model Development (Medium).

Intent: ml_training
Issues scored: all task1 + constraint, outlier, duplicate

Scoring formula:
    task2_score = task1_score × 0.40
                + (constraint_resolved + outlier_resolved + duplicate_resolved) / 3 × 0.60
"""

from server.database import DatabaseEngine
from server.graders.task1 import grade as task1_grade
from server.graders.universal import (
    _resolution_score,
    _false_positive_penalty,
    _rows_identical,
    _validation_score,
)

_CONSTRAINT_FILTER = {"constraint"}
_OUTLIER_FILTER    = {"outlier"}
_DUPLICATE_FILTER  = {"duplicate"}


def grade(
    db: DatabaseEngine,
    cleaned_rows: list[dict],
    removed_ids: list[int],
    validation_was_called: bool,
) -> float:
    """Score a task2 submission.

    Args:
        db:                    DatabaseEngine for this episode.
        cleaned_rows:          Agent-provided cleaned rows.
        removed_ids:           Agent-provided deleted row PKs.
        validation_was_called: Whether validate() was called.

    Returns:
        Float score in [0.0, 1.0].
    """
    pk_col = db.pk_col

    # Zero-change guard — compare against ORIGINAL dirty state, not current state
    dirty_rows = db.original_state()
    if not removed_ids and _rows_identical(cleaned_rows, dirty_rows, pk_col):
        if db.total_issues > 0:
            return 0.0

    # task1 component (null + type + whitespace + inconsistent_category)
    t1 = task1_grade(
        db=db,
        cleaned_rows=cleaned_rows,
        removed_ids=removed_ids,
        validation_was_called=validation_was_called,
    )

    # Constraint resolution score
    constraint_issues = [i for i in db.issue_registry if i.issue_type in _CONSTRAINT_FILTER]
    if constraint_issues:
        c_score, _ = _resolution_score(constraint_issues, cleaned_rows, removed_ids, pk_col, db)
    else:
        c_score = 1.0  # No constraint issues → full credit

    # Outlier resolution score
    outlier_issues = [i for i in db.issue_registry if i.issue_type in _OUTLIER_FILTER]
    if outlier_issues:
        o_score, _ = _resolution_score(outlier_issues, cleaned_rows, removed_ids, pk_col, db)
    else:
        o_score = 1.0  # No outlier issues → full credit

    # Duplicate resolution score
    duplicate_issues = [i for i in db.issue_registry if i.issue_type in _DUPLICATE_FILTER]
    if duplicate_issues:
        d_score, _ = _resolution_score(duplicate_issues, cleaned_rows, removed_ids, pk_col, db)
    else:
        d_score = 1.0  # No duplicate issues → full credit

    fp_penalty = _false_positive_penalty(
        db, cleaned_rows, removed_ids, pk_col, db.primary_table
    )

    combined = (c_score + o_score + d_score) / 3.0
    raw = t1 * 0.40 + combined * 0.60 - fp_penalty
    return max(0.0, min(1.0, round(raw, 4)))
