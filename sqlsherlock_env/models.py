# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the SQLSherlock-Env RL environment.

An AI agent acts as a data scientist investigating a dirty dataset,
discovering real data quality issues through statistical investigation,
fixing them with reasoning, validating fixes, and exporting cleaned output.
"""

from typing import Any, Literal, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field

ActionType = Literal[
    "inspect",          # view all rows in a table
    "profile_column",   # stats: mean/std/min/max/nulls/z_scores per col
    "run_sql",          # SELECT query only
    "fix_cell",         # correct one cell value with reason
    "fix_column",       # fix ALL nulls in a column with one value (bulk operation)
    "delete_row",       # remove a row with reason
    "validate",         # run all 6 checks: before vs after
    "submit",           # end episode and score
    "export",           # terminal: write cleaned file, return URL
    "select_tables",    # declare which tables to include in multi-table analysis
    "join_tables",      # join two tables on a matching key column
    "classify_intent",  # declare the inferred cleaning intent for this dataset
]


class SQLSherlockAction(Action):
    """Action for the SQLSherlock-Env environment.

    The agent issues one of 8 action types per step.
    Every fix action MUST include a reason field with statistical justification.
    """

    action_type: ActionType = Field(
        ...,
        description="Type of action to perform.",
    )
    table: Optional[str] = Field(
        default=None,
        description="Target table name (required for inspect, profile_column, fix_cell, delete_row).",
    )
    row_id: Optional[int] = Field(
        default=None,
        description="Row primary key (required for fix_cell, delete_row).",
    )
    column: Optional[str] = Field(
        default=None,
        description="Column name (required for profile_column, fix_cell).",
    )
    value: Optional[Any] = Field(
        default=None,
        description="Corrected value to write (required for fix_cell).",
    )
    sql: Optional[str] = Field(
        default=None,
        description="SELECT SQL query string (required for run_sql).",
    )
    cleaned_rows: Optional[list[dict]] = Field(
        default=None,
        description="Full list of cleaned rows for export action.",
    )
    removed_ids: Optional[list[int]] = Field(
        default=None,
        description="List of deleted row primary keys for export action.",
    )
    reason: Optional[str] = Field(
        default=None,
        description="Statistical justification for this action (required for fix_cell, delete_row).",
    )
    # --- Multi-table reasoning fields ---
    tables: Optional[list[str]] = Field(
        default=None,
        description="List of table names to select (for select_tables action).",
    )
    table2: Optional[str] = Field(
        default=None,
        description="Second table name for join_tables action.",
    )
    key: Optional[str] = Field(
        default=None,
        description="Join key column in the primary table (for join_tables action).",
    )


class SQLSherlockObservation(Observation):
    """Observation returned to the agent after each step.

    Contains the current environment state the agent can see.
    The issue_registry is NEVER included here — the agent must discover issues.
    """

    task_id: str = Field(
        default="",
        description="Current task identifier.",
    )
    task_description: str = Field(
        default="",
        description="Human-readable task description for the agent.",
    )
    step: int = Field(
        default=0,
        description="Current step number (1-indexed).",
    )
    max_steps: int = Field(
        default=20,
        description="Maximum steps allowed for this task.",
    )
    tables_summary: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Summary of all loaded tables: "
            "{table_name: {row_count: int, columns: list[str], dtypes: dict}}"
        ),
    )
    query_result: Optional[list[dict]] = Field(
        default=None,
        description="Result rows from inspect or run_sql actions.",
    )
    validation_result: Optional[dict] = Field(
        default=None,
        description="Detailed validation results after a validate action.",
    )
    last_feedback: str = Field(
        default="",
        description="Human-readable feedback about the last action taken.",
    )
    reward_trace: list[dict] = Field(
        default_factory=list,
        description="Cumulative reward log — grows every step; judges review this.",
    )
    done: bool = Field(
        default=False,
        description="True when the episode has ended.",
    )
    intent: Optional[str] = Field(
        default=None,
        description=(
            "Cleaning intent for this episode if explicitly provided: "
            "dashboard | ml_training | reporting. "
            "None means the agent must infer it via classify_intent."
        ),
    )


class SQLSherlockState(State):
    """Internal server-side state for one SQLSherlock episode.

    Not exposed to the agent. Used by the environment and graders.
    """

    episode_id: str = Field(
        default="",
        description="Unique identifier for this episode.",
    )
    task_id: str = Field(
        default="",
        description="Task identifier for this episode.",
    )
    step_count: int = Field(
        default=0,
        description="Number of steps taken so far.",
    )
    grader_score: float = Field(
        default=0.0,
        description="Most recent grader score (0.0–1.0).",
    )
    done: bool = Field(
        default=False,
        description="Whether the episode has ended.",
    )
    dataset_name: str = Field(
        default="",
        description="Name or path of the loaded dataset.",
    )
    source_format: str = Field(
        default="",
        description="Detected source format: csv|json|jsonl|parquet|hf_dataset.",
    )
    investigation_count: int = Field(
        default=0,
        description="Number of investigation actions taken (inspect + profile + sql).",
    )
    validation_called: bool = Field(
        default=False,
        description="Whether the agent called validate() at least once.",
    )
    intent: Optional[str] = Field(
        default=None,
        description="Cleaning intent for this episode (dashboard|ml_training|reporting|None).",
    )
    tables_selected: list[str] = Field(
        default_factory=list,
        description="Tables the agent explicitly selected for multi-table analysis.",
    )
    joins_performed: int = Field(
        default=0,
        description="Number of join_tables operations performed.",
    )
    output_format: Optional[str] = Field(
        default=None,
        description="User-requested output format override (csv|json|parquet|jsonl).",
    )
