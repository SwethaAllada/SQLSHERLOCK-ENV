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

def _make_tasks() -> list[dict]:
    """Build the 9-task catalogue: 3 intents × 3 difficulty levels."""

    _TRAP_NOTE = (
        " TRAP WARNING: one numeric value looks suspicious but has z < 3 "
        "(statistically legitimate). ALWAYS verify z-scores before fixing numeric "
        "values. z > 5 = real outlier to fix. z < 3 = leave alone."
    )

    return [
        # ── Visualization ──────────────────────────────────────────────────
        {
            "id": "viz_easy", "intent": "visualization",
            "difficulty": "easy", "max_steps": 30,
            "name": "Visualization Prep — Easy",
            "correct_intent": "visualization",
            "description": (
                "Prepare this dataset for dashboard visualization (Easy). "
                "Fix null values, type errors, whitespace noise, and inconsistent "
                "category labels. Goal: every chart cell has a clean, consistent value."
            ),
        },
        {
            "id": "viz_medium", "intent": "visualization",
            "difficulty": "medium", "max_steps": 40,
            "name": "Visualization Prep — Medium",
            "correct_intent": "visualization",
            "description": (
                "Visualization prep — Medium. Fix all Easy issues plus constraint "
                "violations (negative values where impossible) and statistical "
                "outliers (z > 5). Goal: no outliers distorting chart scales or axes."
            ),
        },
        {
            "id": "viz_hard", "intent": "visualization",
            "difficulty": "hard", "max_steps": 50,
            "name": "Visualization Prep — Hard",
            "correct_intent": "visualization",
            "description": (
                "Visualization prep — Hard. Full audit: all Medium issues plus "
                "duplicate rows and FK violations." + _TRAP_NOTE
            ),
        },
        # ── ML Training ────────────────────────────────────────────────────
        {
            "id": "ml_easy", "intent": "ml_training",
            "difficulty": "easy", "max_steps": 30,
            "name": "ML Model Development — Easy",
            "correct_intent": "ml_training",
            "description": (
                "Prepare this dataset for ML model training (Easy). "
                "Fix null values, type errors, whitespace, and inconsistent "
                "category labels. Goal: every feature has a valid, consistent value."
            ),
        },
        {
            "id": "ml_medium", "intent": "ml_training",
            "difficulty": "medium", "max_steps": 40,
            "name": "ML Model Development — Medium",
            "correct_intent": "ml_training",
            "description": (
                "ML training prep — Medium. Fix all Easy issues plus constraint "
                "violations (negative features) and statistical outliers (z > 5). "
                "Goal: no corrupted or extreme values biasing model weights."
            ),
        },
        {
            "id": "ml_hard", "intent": "ml_training",
            "difficulty": "hard", "max_steps": 50,
            "name": "ML Model Development — Hard",
            "correct_intent": "ml_training",
            "description": (
                "ML training prep — Hard. Full audit: all Medium issues plus "
                "duplicate rows and FK violations." + _TRAP_NOTE
            ),
        },
        # ── Business Query ─────────────────────────────────────────────────
        {
            "id": "bq_easy", "intent": "business_query",
            "difficulty": "easy", "max_steps": 30,
            "name": "Business Analytics Query — Easy",
            "correct_intent": "business_query",
            "description": (
                "Prepare this dataset for business SQL analytics (Easy). "
                "Fix null values, type errors, whitespace, and inconsistent "
                "category labels. Goal: clean data for accurate GROUP BY queries."
            ),
        },
        {
            "id": "bq_medium", "intent": "business_query",
            "difficulty": "medium", "max_steps": 40,
            "name": "Business Analytics Query — Medium",
            "correct_intent": "business_query",
            "description": (
                "Business analytics prep — Medium. Fix all Easy issues plus "
                "constraint violations and statistical outliers. "
                "Goal: consistent, valid values for JOIN and aggregation."
            ),
        },
        {
            "id": "bq_hard", "intent": "business_query",
            "difficulty": "hard", "max_steps": 50,
            "name": "Business Analytics Query — Hard",
            "correct_intent": "business_query",
            "description": (
                "Business analytics prep — Hard. Full audit: all Medium issues "
                "plus FK violations and duplicate rows." + _TRAP_NOTE
            ),
        },
    ]


TASKS: list[dict] = _make_tasks()

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
        self._intent: Optional[str] = None           # user-provided or task-hidden intent
        self._correct_intent: Optional[str] = None  # grader-known correct intent (task4)

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
        dataset       = kwargs.get("dataset", "") or "phihung/titanic"
        task_id       = kwargs.get("task_id", "") or "viz_easy"
        seed          = int(kwargs.get("seed", 42))
        max_rows      = int(kwargs.get("max_rows", 500))
        output_format = kwargs.get("output_format", None)
        if task_id not in _TASK_MAP:
            raise ValueError(
                f"Unknown task_id '{task_id}'. "
                f"Valid tasks: {sorted(_TASK_MAP.keys())}"
            )

        task_cfg = _TASK_MAP[task_id]

        # Intent: explicit kwarg overrides task default; task config defines correct intent
        raw_intent = kwargs.get("intent", None)
        task_intent = task_cfg.get("intent")   # always set for our 3 tasks
        self._intent: Optional[str] = (
            raw_intent.strip().lower() if isinstance(raw_intent, str) and raw_intent.strip()
            else task_intent
        )
        self._correct_intent: Optional[str] = task_cfg.get("correct_intent")
        # For task4, if no explicit intent was given, we use the hidden correct intent
        # as the internal ground truth (not shown in obs) to score classify_intent.

        # Fresh database for this episode
        self._db = DatabaseEngine(
            task_id=task_id,
            seed=seed,
            dataset_source=dataset,
            max_rows=max_rows,
        )
        # Share intent with database engine so graders can access it
        self._db._intent = self._intent or self._correct_intent

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
            intent=self._intent,
            tables_selected=[],
            joins_performed=0,
            output_format=output_format,
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

            # ------------------------------------------------------------------
            # select_tables — declare active tables for multi-table analysis
            # ------------------------------------------------------------------
            elif atype == "select_tables":
                requested = list(action.tables or [])
                available = self._db.table_names()
                valid   = [t for t in requested if t in available]
                invalid = [t for t in requested if t not in available]
                self._state.tables_selected = valid
                if invalid:
                    feedback = (
                        f"select_tables: WARNING — tables not found: {invalid}. "
                        f"Selected valid tables: {valid}. "
                        f"All available: {available}."
                    )
                elif not valid:
                    feedback = (
                        f"select_tables: no tables specified. "
                        f"Available tables: {available}."
                    )
                else:
                    feedback = (
                        f"select_tables: selected {len(valid)} table(s): {valid}. "
                        f"Use join_tables to combine them."
                    )

            # ------------------------------------------------------------------
            # join_tables — join two tables on a matching key
            # ------------------------------------------------------------------
            elif atype == "join_tables":
                t1  = action.table or self._db.primary_table
                t2  = action.table2
                key = action.key
                if not t2:
                    raise ValueError("join_tables requires 'table2' field.")
                if not key:
                    raise ValueError("join_tables requires 'key' field.")
                join_result = self._db.join_tables(t1, t2, key)
                query_result = join_result["rows"]
                self._state.joins_performed += 1
                status = "VALID" if join_result["valid"] else "INVALID"
                feedback = (
                    f"join_tables: {t1} LEFT JOIN {t2} ON '{key}' [{status}]. "
                    f"Returned {len(query_result)} rows. "
                    f"Match rate: {join_result['match_rate']:.1%}."
                    + (f" Error: {join_result['error']}" if join_result["error"] else "")
                )

            # ------------------------------------------------------------------
            # classify_intent — agent declares the inferred cleaning intent
            # ------------------------------------------------------------------
            elif atype == "classify_intent":
                guessed = str(action.value or "").strip().lower()
                valid_intents = {"visualization", "ml_training", "business_query"}
                if guessed not in valid_intents:
                    feedback = (
                        f"classify_intent: unknown intent '{guessed}'. "
                        f"Valid values: {sorted(valid_intents)}."
                    )
                else:
                    # Store as the active intent for this episode
                    if self._intent is None:
                        self._intent = guessed
                        self._state.intent = guessed
                        self._db._intent = guessed
                    correct = self._correct_intent
                    if correct is None:
                        feedback = f"classify_intent: intent set to '{guessed}'."
                    elif guessed == correct:
                        feedback = f"classify_intent: correct! Cleaning strategy set to '{guessed}'."
                    else:
                        feedback = (
                            f"classify_intent: classified as '{guessed}'. "
                            "Consider reviewing the task description for more clues."
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
                    output_format=self._state.output_format if self._state else None,
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
        # Determine join_valid for join_tables reward
        _join_valid: Optional[bool] = None
        if atype == "join_tables" and self._db._join_attempts:
            _join_valid = self._db._join_attempts[-1].get("valid")

        rb: RB = calc(
            action_type=atype,
            db=self._db,
            counter=self._counter,
            action=action,
            validation_result=(
                getattr(self, "_last_vr", None) if atype == "validate" else None
            ),
            intent=self._intent,
            correct_intent=self._correct_intent,
            join_valid=_join_valid,
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
            intent=self._intent,  # None for task4 (agent must infer via classify_intent)
        )
