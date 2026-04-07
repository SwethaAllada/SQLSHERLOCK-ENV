# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Task 3 grader — Full statistical audit with trap.

Scoring formula:
    task3_score = task2_score × 0.50
                + audit_issues_resolved × 0.50
                + reasoning_bonus (0.05)
                - trap_penalty (0.40 if trap hit)

audit_issues_resolved = weighted resolution score for
outlier + duplicate issue types.
"""

from server.database import DatabaseEngine
from server.graders.task2 import grade as task2_grade
from server.graders.universal import (
    _resolution_score,
    _trap_penalty,
    _rows_identical,
    _reasoning_bonus,
)

_AUDIT_FILTER = {"outlier", "duplicate"}


def grade(
    db: DatabaseEngine,
    cleaned_rows: list[dict],
    removed_ids: list[int],
    validation_was_called: bool,
) -> float:
    """Score a task3 submission.

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

    # task2 component (null + type + constraint + fk)
    t2 = task2_grade(
        db=db,
        cleaned_rows=cleaned_rows,
        removed_ids=removed_ids,
        validation_was_called=validation_was_called,
    )

    # Audit issues: outlier + duplicate
    audit_issues = [
        i for i in db.issue_registry if i.issue_type in _AUDIT_FILTER
    ]
    if audit_issues:
        audit_score, _ = _resolution_score(
            audit_issues, cleaned_rows, removed_ids, pk_col, db
        )
    else:
        audit_score = 1.0   # No audit issues → full credit

    # Trap penalty
    trap_pen = _trap_penalty(
        db, cleaned_rows, removed_ids, pk_col,
        task_id="task3_full_audit_with_trap",
    )

    # Reasoning bonus
    r_bonus = _reasoning_bonus(db, "task3_full_audit_with_trap", validation_was_called)

    # NOTE: FP penalty is already applied inside t2 (task2_grade) — not applied
    # again here to avoid double-counting.

    raw = (
        t2          * 0.50
        + audit_score * 0.50
        + r_bonus
        - trap_pen
    )
    return max(0.0, min(1.0, round(raw, 4)))
