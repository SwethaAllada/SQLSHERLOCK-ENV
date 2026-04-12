# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Task 3 grader — Business Analytics Query (Hard).

Intent: business_query
Issues scored: all task2 + fk_violation + trap

Scoring formula:
    task3_score = task2_score × 0.50
                + fk_resolved × 0.50
                + reasoning_bonus (0.05)
                - trap_penalty (0.40 if trap hit)

fk_resolved is the weighted resolution score for fk_violation issue type.
The trap tests whether the agent checks z-scores before "fixing" values.
"""

from server.database import DatabaseEngine
from server.graders.task2 import grade as task2_grade
from server.graders.universal import (
    _resolution_score,
    _trap_penalty,
    _rows_identical,
    _reasoning_bonus,
)

_FK_FILTER = {"fk_violation"}


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

    # task2 component (null + type + whitespace + inconsistent_category +
    #                   constraint + outlier + duplicate)
    t2 = task2_grade(
        db=db,
        cleaned_rows=cleaned_rows,
        removed_ids=removed_ids,
        validation_was_called=validation_was_called,
    )

    # FK violation resolution score
    fk_issues = [i for i in db.issue_registry if i.issue_type in _FK_FILTER]

    # Trap penalty (−0.40 if trap value was changed; applies to all hard tasks)
    trap_pen = _trap_penalty(
        db, cleaned_rows, removed_ids, pk_col,
        task_id=db.task_id,
    )

    # Reasoning bonus (+0.05 if statistical reasoning terms were used in reasons)
    r_bonus = _reasoning_bonus(db, db.task_id, validation_was_called)

    # NOTE: FP penalty is already applied inside t2 (task2_grade) — not applied
    # again here to avoid double-counting.

    if fk_issues:
        # Dataset has FK violations — score FK resolution as 50% of hard grade
        fk_score, _ = _resolution_score(fk_issues, cleaned_rows, removed_ids, pk_col, db)
        raw = (
            t2        * 0.50
            + fk_score  * 0.50
            + r_bonus
            - trap_pen
        )
    else:
        # Single-table dataset — no FK issues to resolve.
        # Hard difficulty is demonstrated entirely through trap avoidance and reasoning.
        # Grade purely on medium performance + trap + reasoning (no free FK credit).
        raw = t2 + r_bonus - trap_pen

    return max(0.0, min(1.0, round(raw, 4)))
