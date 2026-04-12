# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Task 1 grader — Visualization Prep (Easy).

Intent: visualization
Issues scored: null, type_error, whitespace, inconsistent_category

Scoring formula:
    task1_score = resolution_score × 0.70 + validation_score × 0.30
"""

from server.database import DatabaseEngine
from server.graders.universal import (
    _resolution_score,
    _false_positive_penalty,
    _validation_score,
    _rows_identical,
)

_ISSUE_FILTER = {"null", "type_error", "whitespace", "inconsistent_category"}


def grade(
    db: DatabaseEngine,
    cleaned_rows: list[dict],
    removed_ids: list[int],
    validation_was_called: bool,
) -> float:
    """Score a task1 submission.

    Args:
        db:                    DatabaseEngine for this episode.
        cleaned_rows:          Agent-provided cleaned rows.
        removed_ids:           Agent-provided deleted row PKs.
        validation_was_called: Whether validate() was called.

    Returns:
        Float score in [0.0, 1.0].
    """
    issue_registry = db.issue_registry
    scored_issues = [i for i in issue_registry if i.issue_type in _ISSUE_FILTER]
    pk_col = db.pk_col

    # Zero-change guard — compare against ORIGINAL dirty state, not current state
    dirty_rows = db.original_state()
    if not removed_ids and _rows_identical(cleaned_rows, dirty_rows, pk_col):
        if db.total_issues > 0:
            return 0.0

    res_score, _ = _resolution_score(
        scored_issues, cleaned_rows, removed_ids, pk_col, db
    )

    fp_penalty = _false_positive_penalty(
        db, cleaned_rows, removed_ids, pk_col, db.primary_table
    )

    val_score = _validation_score(db, cleaned_rows, validation_was_called)

    raw = res_score * 0.70 + val_score * 0.30 - fp_penalty
    return max(0.0, min(1.0, round(raw, 4)))
