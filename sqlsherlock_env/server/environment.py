# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
SQLSherlock RL environment — server-side implementation.

Implements the OpenEnv Environment interface.  One instance per
WebSocket session; each reset() creates a fresh DatabaseEngine.
"""

import uuid
from typing import Any, Optional

from openenv.core.env_server import Environment

from models import SQLSherlockAction, SQLSherlockObservation, SQLSherlockState
from server.database import DatabaseEngine
from server.reward import calc, RB, InvestCounter
from server import graders
from server.exporter import export_cleaned
from server.validator import Validator


# ---------------------------------------------------------------------------
# Task catalogue
# ---------------------------------------------------------------------------

TASKS: list[dict] = [
    {
        "id":          "task1_null_and_types",
        "name":        "Null and type error repair",
        "difficulty":  "easy",
        "max_steps":   20,
        "description": (
            "Find and fix null values and type errors in the primary table. "
            "Profile columns, identify anomalies, fix with reasoning, "
            "validate your work, and export the cleaned dataset."
        ),
    },
    {
        "id":          "task2_constraints_and_fk",
        "name":        "Constraint and FK integrity",
        "difficulty":  "medium",
        "max_steps":   25,
        "description": (
            "Everything in Task 1 plus constraint violations "
            "(negative values in must-be-positive columns) and FK "
            "violations (orphan references in related tables)."
        ),
    },
    {
        "id":          "task3_full_audit_with_trap",
        "name":        "Full statistical audit with trap",
        "difficulty":  "hard",
        "max_steps":   30,
        "description": (
            "Full audit including statistical outliers. TRAP WARNING: "
            "one numeric value looks suspicious but is legitimate. "
            "You MUST check z-scores before fixing any numeric value. "
            "z > 5 = real outlier. z < 3 = leave alone."
        ),
    },
]

_TASK_MAP: dict[str, dict] = {t["id"]: t for t in TASKS}


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class SQLSherlockEnvironment(Environment):
    """One episode of the SQLSherlock RL environment."""

    # Called by create_app() as a factory — __init__ must be zero-arg.
    def __init__(self) -> None:
        self._db: Optional[DatabaseEngine] = None
        self._state: Optional[SQLSherlockState] = None
        self._counter: Optional[InvestCounter] = None
        self._reward_trace: list[dict] = []
        self._validation_called: bool = False
        self._export_result: Optional[dict] = None

    # ------------------------------------------------------------------
    # reset()
    # ------------------------------------------------------------------

    def reset(self, **kwargs) -> SQLSherlockObservation:
        """Start a new episode.

        Keyword Args:
            dataset (str):  Dataset source — required, no default.
            task_id (str):  Task identifier — required, no default.
            seed    (int):  RNG seed (default 42).
            max_rows(int):  Row limit (default 500).

        Raises:
            ValueError: If dataset or task_id is missing/invalid.
        """
        dataset = kwargs.get("dataset", "")
        task_id = kwargs.get("task_id", "")
        seed    = int(kwargs.get("seed", 42))
        max_rows = int(kwargs.get("max_rows", 500))

        if not dataset or not dataset.strip():
            raise ValueError(
                "reset() requires 'dataset' keyword argument. "
                "Provide a file path, HuggingFace dataset name, or raw CSV text."
            )
        if not task_id or not task_id.strip():
            raise ValueError(
                "reset() requires 'task_id' keyword argument. "
                f"Valid tasks: {sorted(_TASK_MAP.keys())}"
            )
        if task_id not in _TASK_MAP:
            raise ValueError(
                f"Unknown task_id '{task_id}'. "
                f"Valid tasks: {sorted(_TASK_MAP.keys())}"
            )

        task_cfg = _TASK_MAP[task_id]

        # Fresh database for this episode
        self._db = DatabaseEngine(
            task_id=task_id,
            seed=seed,
            dataset_source=dataset,
            max_rows=max_rows,
        )

        self._state = SQLSherlockState(
            episode_id=str(uuid.uuid4()),
            task_id=task_id,
            step_count=0,
            grader_score=0.0,
            done=False,
            dataset_name=dataset,
            source_format=self._db.source_format,
            investigation_count=0,
            validation_called=False,
        )

        self._counter = InvestCounter()
        self._reward_trace = []
        self._validation_called = False
        self._export_result = None
        self._deleted_row_ids: list[int] = []   # track deletes for grader

        return self._make_obs(
            last_feedback=(
                f"Episode started. Dataset loaded: {self._db.primary_table} "
                f"({len(self._db.rows(self._db.primary_table))} rows). "
                f"Task: {task_cfg['name']}. Max steps: {task_cfg['max_steps']}. "
                "Begin by inspecting the table or profiling columns."
            ),
            query_result=None,
            validation_result=None,
        )

    # ------------------------------------------------------------------
    # step()
    # ------------------------------------------------------------------

    def step(
        self, action: SQLSherlockAction, **kwargs
    ) -> SQLSherlockObservation:
        """Execute one agent action.

        Returns the observation with reward and done set on it.
        The openenv framework extracts reward/done from the observation.
        """
        if self._db is None or self._state is None:
            raise RuntimeError("Call reset() before step().")

        task_cfg  = _TASK_MAP[self._state.task_id]
        max_steps = task_cfg["max_steps"]

        self._state.step_count += 1
        step = self._state.step_count

        # Log action for reasoning bonus check
        self._db.log_action(action)

        query_result      = None
        validation_result = None
        feedback          = ""
        done              = False

        atype = action.action_type

        # ------------------------------------------------------------------
        # Dispatch
        # ------------------------------------------------------------------
        try:
            if atype == "inspect":
                table = action.table or self._db.primary_table
                rows  = self._db.rows(table)
                query_result = rows
                feedback = f"inspect: returned {len(rows)} rows from '{table}'."

            elif atype == "profile_column":
                table  = action.table or self._db.primary_table
                column = action.column
                if not column:
                    raise ValueError("profile_column requires 'column' field.")
                profile = self._db.profile_col(table, column)
                query_result = [profile]
                feedback = (
                    f"profile_column '{column}': "
                    f"mean={profile.get('mean')}, std={profile.get('std')}, "
                    f"null_count={profile.get('null_count')}, "
                    f"must_be_positive={profile.get('must_be_positive')}."
                )

            elif atype == "run_sql":
                sql = action.sql
                if not sql:
                    raise ValueError("run_sql requires 'sql' field.")
                rows = self._db.query(sql)
                query_result = rows
                feedback = f"run_sql: returned {len(rows)} rows."

            elif atype == "fix_cell":
                table  = action.table or self._db.primary_table
                row_id = action.row_id
                column = action.column
                value  = action.value
                if row_id is None or column is None:
                    raise ValueError("fix_cell requires 'row_id' and 'column'.")
                self._db.fix_cell(table, row_id, column, value)
                feedback = (
                    f"fix_cell: set [{table}].{column}[id={row_id}] = {value!r}. "
                    f"Reason: {action.reason or '(none provided)'}."
                )

            elif atype == "fix_column":
                table  = action.table or self._db.primary_table
                column = action.column
                value  = action.value
                if column is None:
                    raise ValueError("fix_column requires 'column'.")
                result = self._db.fix_column(table, column, value)
                parts = []
                if result["nulls_fixed"]:
                    parts.append(f"{result['nulls_fixed']} nulls")
                if result["type_errors_fixed"]:
                    parts.append(f"{result['type_errors_fixed']} type errors")
                if result["negatives_fixed"]:
                    parts.append(f"{result['negatives_fixed']} negatives")
                detail = ", ".join(parts) if parts else "0 issues"
                feedback = (
                    f"fix_column '{column}': fixed {detail} "
                    f"(total {result['total_fixed']} rows) with value={value!r}. "
                    f"Reason: {action.reason or '(none provided)'}."
                )

            elif atype == "delete_row":
                table  = action.table or self._db.primary_table
                row_id = action.row_id
                if row_id is None:
                    raise ValueError("delete_row requires 'row_id'.")
                self._db.delete_row(table, row_id)
                if row_id not in self._deleted_row_ids:
                    self._deleted_row_ids.append(row_id)
                feedback = (
                    f"delete_row: removed row id={row_id} from '{table}'. "
                    f"Reason: {action.reason or '(none provided)'}."
                )

            elif atype == "validate":
                vr = self._db.validate()
                validation_result = vr.to_dict()
                self._validation_called = True
                self._state.validation_called = True
                self._last_vr = vr          # cache — avoid second validate() call
                feedback = (
                    f"validate: {vr.overall} — "
                    f"{vr.checks_passed}/{vr.total_checks} checks passed. "
                    + (f"Warnings: {vr.warnings}" if vr.warnings else "")
                )

            elif atype == "submit":
                current = self._db.current_state()
                score = graders.grade(
                    db=self._db,
                    cleaned_rows=current,
                    removed_ids=list(self._deleted_row_ids),
                    task_id=self._state.task_id,
                    validation_was_called=self._validation_called,
                )
                self._state.grader_score = score
                done = True
                feedback = (
                    f"submit: episode complete. "
                    f"Grader score = {score:.4f}. "
                    f"Issues remaining: {self._db.issues_remaining()}."
                )

            elif atype == "export":
                cleaned_rows  = action.cleaned_rows or self._db.current_state()
                removed_ids   = action.removed_ids or []
                score = graders.grade(
                    db=self._db,
                    cleaned_rows=cleaned_rows,
                    removed_ids=removed_ids,
                    task_id=self._state.task_id,
                    validation_was_called=self._validation_called,
                )
                self._state.grader_score = score
                export_info = export_cleaned(
                    cleaned_rows=cleaned_rows,
                    source_format=self._db.source_format,
                    dataset_name=self._db.dataset_name,
                )
                self._export_result = export_info
                done = True
                feedback = (
                    f"export: {export_info['row_count']} rows written to "
                    f"{export_info['download_url']} ({export_info['format']}). "
                    f"Grader score = {score:.4f}."
                )

            else:
                feedback = f"Unknown action_type '{atype}'. No-op."

        except ValueError as exc:
            feedback = f"Action error: {exc}"

        # ------------------------------------------------------------------
        # Reward
        # ------------------------------------------------------------------
        rb: RB = calc(
            action_type=atype,
            db=self._db,
            counter=self._counter,
            action=action,
            validation_result=(
                getattr(self, "_last_vr", None) if atype == "validate" else None
            ),
        )

        step_reward = rb.total
        rb_dict = rb.to_dict()
        rb_dict["step"] = step
        rb_dict["action_type"] = atype
        self._reward_trace.append(rb_dict)

        # Update investigation count
        if atype in ("inspect", "profile_column", "run_sql"):
            self._state.investigation_count += 1

        # Max-steps termination
        if step >= max_steps and not done:
            done = True
            feedback += f" [max_steps={max_steps} reached]"

        self._state.done = done

        obs = self._make_obs(
            last_feedback=feedback,
            query_result=query_result,
            validation_result=validation_result,
        )
        obs.done = done
        obs.reward = step_reward

        return obs

    # ------------------------------------------------------------------
    # get_state()
    # ------------------------------------------------------------------

    @property
    def state(self) -> SQLSherlockState:
        """Required by openenv-core Environment base class."""
        return self.get_state()

    def get_state(self) -> SQLSherlockState:
        if self._state is None:
            return SQLSherlockState()
        return self._state

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_obs(
        self,
        last_feedback: str,
        query_result: Optional[list],
        validation_result: Optional[dict],
    ) -> SQLSherlockObservation:
        task_cfg = _TASK_MAP.get(self._state.task_id, TASKS[0]) if self._state else TASKS[0]
        return SQLSherlockObservation(
            task_id=self._state.task_id if self._state else "",
            task_description=task_cfg["description"],
            step=self._state.step_count if self._state else 0,
            max_steps=task_cfg["max_steps"],
            tables_summary=self._db.tables_summary() if self._db else {},
            query_result=query_result,
            validation_result=validation_result,
            last_feedback=last_feedback,
            reward_trace=list(self._reward_trace),
            done=self._state.done if self._state else False,
        )
