# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Graders package for SQLSherlock-Env.

Each task has a dedicated grader that delegates to universal.grade()
with task-appropriate filters.

Usage (from environment.py)::

    from server import graders

    score = graders.grade(
        db=db,
        cleaned_rows=cleaned_rows,
        removed_ids=removed_ids,
        task_id=task_id,
        validation_was_called=validation_was_called,
    )
"""

from server.graders.task1 import grade as _grade_task1
from server.graders.task2 import grade as _grade_task2
from server.graders.task3 import grade as _grade_task3

# 9 task IDs: 3 intents × 3 difficulty levels
_GRADERS = {
    # Easy tasks (any intent) → easy grader
    "viz_easy":    _grade_task1,
    "ml_easy":     _grade_task1,
    "bq_easy":     _grade_task1,
    # Medium tasks (any intent) → medium grader
    "viz_medium":  _grade_task2,
    "ml_medium":   _grade_task2,
    "bq_medium":   _grade_task2,
    # Hard tasks (any intent) → hard grader
    "viz_hard":    _grade_task3,
    "ml_hard":     _grade_task3,
    "bq_hard":     _grade_task3,
}


def grade(
    db,
    cleaned_rows: list[dict],
    removed_ids: list[int],
    task_id: str,
    validation_was_called: bool,
) -> float:
    """Dispatch to the correct task grader and return a score in [0.0, 1.0].

    Args:
        db:                    DatabaseEngine instance for this episode.
        cleaned_rows:          Agent-provided cleaned row list.
        removed_ids:           Agent-provided list of deleted row PKs.
        task_id:               Task identifier string.
        validation_was_called: Whether the agent called validate() at least once.

    Returns:
        Float score in [0.0, 1.0].

    Raises:
        ValueError: If task_id is not recognised.
    """
    grader_fn = _GRADERS.get(task_id)
    if grader_fn is None:
        raise ValueError(
            f"Unknown task_id '{task_id}'. "
            f"Valid tasks: {sorted(_GRADERS.keys())}"
        )
    raw = grader_fn(
        db=db,
        cleaned_rows=cleaned_rows,
        removed_ids=removed_ids,
        validation_was_called=validation_was_called,
    )
    # Scale internal [0.0, 1.0] → (0.01, 0.99) so the score is strictly between
    # 0 and 1 as required by the OpenEnv hackathon validation.
    # 0.0 (zero-change) → 0.01;  1.0 (perfect) → 0.99.
    return round(0.01 + raw * 0.98, 4)


__all__ = ["grade"]
