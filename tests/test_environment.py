# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for server/environment.py

Covers: reset validation, step dispatch for all 8 action types,
        reward accumulation, done flag, max-steps termination,
        and WebSocket minimal-action compatibility (Nemotron Phase 2).
"""

import pytest

from server.environment import SQLSherlockEnvironment, TASKS
from models import SQLSherlockAction, SQLSherlockObservation, SQLSherlockState
from tests.conftest import RAW_CSV_TEXT


def _step(env, action):
    """Call env.step() and unpack the observation into (obs, reward, done, info).

    The openenv-core Environment.step() returns an Observation with reward/done
    set on it. This helper provides the classic RL tuple interface for tests.
    """
    obs = env.step(action)
    return obs, float(obs.reward or 0.0), obs.done, {}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def env():
    return SQLSherlockEnvironment()


@pytest.fixture
def env_task1(env):
    env.reset(dataset=RAW_CSV_TEXT, task_id="task1_null_and_types")
    return env


@pytest.fixture
def env_task3(env):
    env.reset(dataset=RAW_CSV_TEXT, task_id="task3_full_audit_with_trap")
    return env


# ---------------------------------------------------------------------------
# TASKS catalogue
# ---------------------------------------------------------------------------

class TestTasksCatalogue:
    def test_three_tasks_defined(self):
        assert len(TASKS) == 3

    def test_task_ids_correct(self):
        ids = {t["id"] for t in TASKS}
        assert ids == {
            "task1_null_and_types",
            "task2_constraints_and_fk",
            "task3_full_audit_with_trap",
        }

    def test_tasks_have_required_fields(self):
        for t in TASKS:
            for field in ("id", "name", "difficulty", "max_steps", "description"):
                assert field in t, f"Task missing field '{field}': {t}"

    def test_max_steps_values(self):
        step_map = {t["id"]: t["max_steps"] for t in TASKS}
        assert step_map["task1_null_and_types"]       == 20
        assert step_map["task2_constraints_and_fk"]   == 25
        assert step_map["task3_full_audit_with_trap"] == 30


# ---------------------------------------------------------------------------
# reset() validation
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_returns_observation(self, env):
        obs = env.reset(dataset=RAW_CSV_TEXT, task_id="task1_null_and_types")
        assert isinstance(obs, SQLSherlockObservation)

    def test_reset_populates_tables_summary(self, env):
        obs = env.reset(dataset=RAW_CSV_TEXT, task_id="task1_null_and_types")
        assert len(obs.tables_summary) > 0

    def test_reset_task_description_set(self, env):
        obs = env.reset(dataset=RAW_CSV_TEXT, task_id="task2_constraints_and_fk")
        assert "Task" in obs.task_description or len(obs.task_description) > 0

    def test_reset_step_zero(self, env):
        obs = env.reset(dataset=RAW_CSV_TEXT, task_id="task1_null_and_types")
        assert obs.step == 0

    def test_reset_no_dataset_raises(self, env):
        with pytest.raises(ValueError, match="dataset"):
            env.reset(dataset="", task_id="task1_null_and_types")

    def test_reset_no_task_raises(self, env):
        with pytest.raises(ValueError, match="task_id"):
            env.reset(dataset=RAW_CSV_TEXT, task_id="")

    def test_reset_invalid_task_raises(self, env):
        with pytest.raises(ValueError, match="Unknown task_id"):
            env.reset(dataset=RAW_CSV_TEXT, task_id="task99_bad")

    def test_reset_clears_reward_trace(self, env):
        env.reset(dataset=RAW_CSV_TEXT, task_id="task1_null_and_types")
        env.step(SQLSherlockAction(action_type="inspect",
                                   table=list(env._db.table_names())[0]))
        # Second reset should clear trace
        obs = env.reset(dataset=RAW_CSV_TEXT, task_id="task1_null_and_types")
        assert obs.reward_trace == []

    def test_reset_before_step_raises(self, env):
        with pytest.raises(RuntimeError):
            env.step(SQLSherlockAction(action_type="inspect"))


# ---------------------------------------------------------------------------
# step() — inspect
# ---------------------------------------------------------------------------

class TestStepInspect:
    def test_inspect_returns_rows(self, env_task1):
        table = list(env_task1._db.table_names())[0]
        obs, reward, done, info = _step(env_task1,
            SQLSherlockAction(action_type="inspect", table=table)
        )
        assert obs.query_result is not None
        assert len(obs.query_result) > 0

    def test_inspect_positive_reward(self, env_task1):
        table = list(env_task1._db.table_names())[0]
        _, reward, _, _ = _step(env_task1,
            SQLSherlockAction(action_type="inspect", table=table)
        )
        assert reward > 0

    def test_inspect_capped_at_3(self, env_task1):
        table = list(env_task1._db.table_names())[0]
        rewards = []
        for _ in range(5):
            _, r, _, _ = _step(env_task1,
                SQLSherlockAction(action_type="inspect", table=table)
            )
            rewards.append(r)
        # First 3 positive, after that 0
        assert rewards[0] > 0
        assert rewards[1] > 0
        assert rewards[2] > 0
        assert rewards[3] == 0.0
        assert rewards[4] == 0.0


# ---------------------------------------------------------------------------
# step() — profile_column
# ---------------------------------------------------------------------------

class TestStepProfileColumn:
    def test_profile_returns_stats(self, env_task1):
        table = list(env_task1._db.table_names())[0]
        obs, reward, done, _ = _step(env_task1,
            SQLSherlockAction(action_type="profile_column",
                              table=table, column="fare")
        )
        assert obs.query_result is not None
        profile = obs.query_result[0]
        assert "mean" in profile
        assert "std"  in profile
        assert "z_scores" in profile

    def test_profile_missing_column_gives_feedback(self, env_task1):
        table = list(env_task1._db.table_names())[0]
        obs, _, _, _ = _step(env_task1,
            SQLSherlockAction(action_type="profile_column",
                              table=table, column="nonexistent_col")
        )
        assert "error" in obs.last_feedback.lower() or "not found" in obs.last_feedback.lower()


# ---------------------------------------------------------------------------
# step() — run_sql
# ---------------------------------------------------------------------------

class TestStepRunSQL:
    def test_select_query_works(self, env_task1):
        table = list(env_task1._db.table_names())[0]
        obs, reward, done, _ = _step(env_task1,
            SQLSherlockAction(
                action_type="run_sql",
                sql=f'SELECT * FROM "{table}" LIMIT 3',
            )
        )
        assert obs.query_result is not None
        assert len(obs.query_result) <= 3

    def test_blocked_keyword_gives_error_feedback(self, env_task1):
        obs, _, _, _ = _step(env_task1,
            SQLSherlockAction(
                action_type="run_sql",
                sql="DROP TABLE passengers",
            )
        )
        assert "error" in obs.last_feedback.lower() or "blocked" in obs.last_feedback.lower()

    def test_non_select_gives_error_feedback(self, env_task1):
        obs, _, _, _ = _step(env_task1,
            SQLSherlockAction(
                action_type="run_sql",
                sql="UPDATE passengers SET age=0",
            )
        )
        assert "error" in obs.last_feedback.lower() or "select" in obs.last_feedback.lower()


# ---------------------------------------------------------------------------
# step() — fix_cell
# ---------------------------------------------------------------------------

class TestStepFixCell:
    def test_fix_real_issue_positive_reward(self, env_task1):
        # Find a null issue
        null_issue = next(
            (i for i in env_task1._db.issue_registry if i.issue_type == "null"),
            None,
        )
        if null_issue is None:
            pytest.skip("No null issues in registry")
        _, reward, _, _ = _step(env_task1,
            SQLSherlockAction(
                action_type="fix_cell",
                table=null_issue.table,
                row_id=null_issue.row_id,
                column=null_issue.column,
                value=30,
                reason="median imputation",
            )
        )
        assert reward > 0

    def test_fix_clean_cell_negative_reward(self, env_task1):
        # Fix a cell not in the issue registry
        table = env_task1._db.primary_table
        pk = env_task1._db.pk_col
        issue_cells = {(i.row_id, i.column) for i in env_task1._db.issue_registry}
        rows = env_task1._db.rows(table)
        target = None
        for row in rows:
            rid = row[pk]
            for col in row:
                if col not in (pk, "_source_format") and (rid, col) not in issue_cells:
                    target = (rid, col)
                    break
            if target:
                break
        if target is None:
            pytest.skip("No clean cell available to test FP")
        _, reward, _, _ = _step(env_task1,
            SQLSherlockAction(
                action_type="fix_cell",
                table=table,
                row_id=target[0],
                column=target[1],
                value="TAMPERED",
                reason="test",
            )
        )
        assert reward < 0

    def test_fix_trap_negative_reward(self, env_task3):
        trap = env_task3._db.trap
        if trap is None:
            pytest.skip("No trap in this episode")
        _, reward, _, _ = _step(env_task3,
            SQLSherlockAction(
                action_type="fix_cell",
                table=trap.table,
                row_id=trap.row_id,
                column=trap.column,
                value=trap.original,
                reason="looks like outlier",
            )
        )
        assert reward <= -0.39


# ---------------------------------------------------------------------------
# step() — validate
# ---------------------------------------------------------------------------

class TestStepValidate:
    def test_validate_returns_result(self, env_task1):
        obs, _, _, _ = _step(env_task1,
            SQLSherlockAction(action_type="validate")
        )
        assert obs.validation_result is not None
        assert "checks_passed" in obs.validation_result
        assert "overall" in obs.validation_result

    def test_validate_reward_capped_at_2(self, env_task1):
        rewards = []
        for _ in range(4):
            _, r, _, _ = _step(env_task1,
                SQLSherlockAction(action_type="validate")
            )
            rewards.append(r)
        # Reward only for first 2 calls
        assert rewards[2] == 0.0
        assert rewards[3] == 0.0

    def test_validate_sets_validation_called(self, env_task1):
        assert env_task1._validation_called is False
        env_task1.step(SQLSherlockAction(action_type="validate"))
        assert env_task1._validation_called is True


# ---------------------------------------------------------------------------
# step() — submit
# ---------------------------------------------------------------------------

class TestStepSubmit:
    def test_submit_ends_episode(self, env_task1):
        _, _, done, _ = _step(env_task1,
            SQLSherlockAction(action_type="submit")
        )
        assert done is True

    def test_submit_with_open_issues_negative_reward(self, env_task1):
        _, reward, _, _ = _step(env_task1,
            SQLSherlockAction(action_type="submit")
        )
        # Issues still open -> negative reward
        assert reward < 0


# ---------------------------------------------------------------------------
# step() — export
# ---------------------------------------------------------------------------

class TestStepExport:
    def test_export_ends_episode(self, env_task1):
        _, _, done, _ = _step(env_task1,
            SQLSherlockAction(action_type="export")
        )
        assert done is True

    def test_export_feedback_contains_download(self, env_task1):
        obs, _, _, _ = _step(env_task1,
            SQLSherlockAction(action_type="export")
        )
        assert "download" in obs.last_feedback.lower() or "export" in obs.last_feedback.lower()


# ---------------------------------------------------------------------------
# Reward trace
# ---------------------------------------------------------------------------

class TestRewardTrace:
    def test_reward_trace_grows_each_step(self, env_task1):
        table = list(env_task1._db.table_names())[0]
        for i in range(3):
            obs, _, _, _ = _step(env_task1,
                SQLSherlockAction(action_type="inspect", table=table)
            )
        assert len(obs.reward_trace) == 3

    def test_reward_trace_has_required_keys(self, env_task1):
        table = list(env_task1._db.table_names())[0]
        obs, _, _, _ = _step(env_task1,
            SQLSherlockAction(action_type="inspect", table=table)
        )
        entry = obs.reward_trace[-1]
        for key in ("invest", "fix_delta", "validate_b", "penalty", "total", "step", "action_type"):
            assert key in entry, f"reward_trace entry missing key '{key}'"


# ---------------------------------------------------------------------------
# Max-steps termination
# ---------------------------------------------------------------------------

class TestMaxSteps:
    def test_done_at_max_steps(self, env):
        env.reset(dataset=RAW_CSV_TEXT, task_id="task1_null_and_types")
        table = list(env._db.table_names())[0]
        done = False
        for _ in range(25):   # more than max_steps=20
            _, _, done, _ = _step(env,
                SQLSherlockAction(action_type="inspect", table=table)
            )
            if done:
                break
        assert done is True


# ---------------------------------------------------------------------------
# get_state()
# ---------------------------------------------------------------------------

class TestGetState:
    def test_get_state_returns_state(self, env_task1):
        state = env_task1.get_state()
        assert isinstance(state, SQLSherlockState)

    def test_get_state_task_id(self, env_task1):
        state = env_task1.get_state()
        assert state.task_id == "task1_null_and_types"

    def test_get_state_step_count_increments(self, env_task1):
        table = list(env_task1._db.table_names())[0]
        env_task1.step(SQLSherlockAction(action_type="inspect", table=table))
        env_task1.step(SQLSherlockAction(action_type="inspect", table=table))
        state = env_task1.get_state()
        assert state.step_count == 2


# ---------------------------------------------------------------------------
# Nemotron Phase 2 — minimal action compatibility
# ---------------------------------------------------------------------------

class TestWebSocketActionMinimal:
    def test_action_with_only_action_type_accepted(self, env_task1):
        """A SQLSherlockAction with only action_type set must not crash the server."""
        action = SQLSherlockAction(action_type="validate")
        obs, reward, done, info = _step(env_task1, action)
        assert isinstance(obs, SQLSherlockObservation)
        assert isinstance(reward, float)
        assert isinstance(done, bool)

    def test_inspect_without_table_uses_primary(self, env_task1):
        """inspect with no table field defaults to the primary table."""
        action = SQLSherlockAction(action_type="inspect")
        obs, reward, done, _ = _step(env_task1, action)
        assert obs.query_result is not None

    def test_submit_without_extra_fields(self, env_task1):
        """submit with only action_type must terminate the episode."""
        action = SQLSherlockAction(action_type="submit")
        obs, reward, done, _ = _step(env_task1, action)
        assert done is True
