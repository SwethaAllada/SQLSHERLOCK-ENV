# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
SQLSherlock-Env client.

Wraps the OpenEnv EnvClient to provide a typed, synchronous interface for
SQLSherlockAction / SQLSherlockObservation / SQLSherlockState.

Usage::

    with SQLSherlockEnv(base_url="http://localhost:7860") as env:
        obs = env.reset(dataset="mstz/titanic", task_id="task1_null_and_types")
        obs, reward, done, info = env.step(
            SQLSherlockAction(action_type="inspect", table="titanic")
        )
"""

from typing import Any, Dict, Optional, Tuple

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from models import SQLSherlockAction, SQLSherlockObservation, SQLSherlockState


class _AsyncSQLSherlockClient(
    EnvClient[SQLSherlockAction, SQLSherlockObservation, SQLSherlockState]
):
    """Async EnvClient subclass with custom payload/parsing logic."""

    def _step_payload(self, action: SQLSherlockAction) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"action_type": action.action_type}

        if action.table is not None:
            payload["table"] = action.table
        if action.row_id is not None:
            payload["row_id"] = action.row_id
        if action.column is not None:
            payload["column"] = action.column
        if action.value is not None:
            payload["value"] = action.value
        if action.sql is not None:
            payload["sql"] = action.sql
        if action.cleaned_rows is not None:
            payload["cleaned_rows"] = action.cleaned_rows
        if action.removed_ids is not None:
            payload["removed_ids"] = action.removed_ids
        if action.reason is not None:
            payload["reason"] = action.reason

        return payload

    def _parse_result(
        self, payload: Dict[str, Any]
    ) -> StepResult[SQLSherlockObservation]:
        obs_data = payload.get("observation", {})

        observation = SQLSherlockObservation(
            task_id=obs_data.get("task_id", ""),
            task_description=obs_data.get("task_description", ""),
            step=obs_data.get("step", 0),
            max_steps=obs_data.get("max_steps", 20),
            tables_summary=obs_data.get("tables_summary", {}),
            query_result=obs_data.get("query_result"),
            validation_result=obs_data.get("validation_result"),
            last_feedback=obs_data.get("last_feedback", ""),
            reward_trace=obs_data.get("reward_trace", []),
            done=payload.get("done", False),
            intent=obs_data.get("intent"),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> SQLSherlockState:
        return SQLSherlockState(
            episode_id=payload.get("episode_id", ""),
            task_id=payload.get("task_id", ""),
            step_count=payload.get("step_count", 0),
            grader_score=payload.get("grader_score", 0.0),
            done=payload.get("done", False),
            dataset_name=payload.get("dataset_name", ""),
            source_format=payload.get("source_format", ""),
            investigation_count=payload.get("investigation_count", 0),
            validation_called=payload.get("validation_called", False),
            intent=payload.get("intent"),
            tables_selected=payload.get("tables_selected", []),
            joins_performed=payload.get("joins_performed", 0),
            output_format=payload.get("output_format"),
        )


class SQLSherlockEnv:
    """Synchronous client for the SQLSherlock-Env RL environment.

    Provides the standard RL interface:
        obs = env.reset(dataset=..., task_id=...)
        obs, reward, done, info = env.step(action)

    Example::

        with SQLSherlockEnv(base_url="http://localhost:7860") as env:
            obs = env.reset(
                dataset="mstz/titanic",
                task_id="task1_null_and_types",
            )
            print(obs.tables_summary)

            obs, reward, done, info = env.step(
                SQLSherlockAction(action_type="inspect", table="titanic")
            )
            print(obs.last_feedback, reward)
    """

    def __init__(self, base_url: str = "http://localhost:7860") -> None:
        self._async_client = _AsyncSQLSherlockClient(base_url=base_url)
        self._sync = self._async_client.sync()

    def __enter__(self):
        self._sync.connect()
        return self

    def __exit__(self, *args):
        self.close()

    def reset(self, **kwargs) -> SQLSherlockObservation:
        """Reset the environment and return initial observation.

        Keyword Args:
            dataset (str):  Dataset source — required.
            task_id (str):  Task identifier — required.
            seed    (int):  RNG seed (default 42).
            max_rows(int):  Row limit (default 500).
        """
        result: StepResult = self._sync.reset(**kwargs)
        return result.observation

    def step(
        self, action: SQLSherlockAction
    ) -> Tuple[SQLSherlockObservation, float, bool, dict]:
        """Execute one action. Returns (obs, reward, done, info)."""
        result: StepResult = self._sync.step(action)
        return (
            result.observation,
            float(result.reward or 0.0),
            result.done,
            {},
        )

    def get_state(self) -> SQLSherlockState:
        """Return current episode state."""
        return self._sync.state()

    def close(self) -> None:
        """Close the connection."""
        try:
            self._sync.disconnect()
        except Exception:
            pass

    @classmethod
    def from_docker_image(cls, image: str, port: int = 7860) -> "SQLSherlockEnv":
        """Create client connected to a freshly launched Docker container."""
        import subprocess
        import time

        container_id = subprocess.check_output(
            ["docker", "run", "-d", "-p", f"{port}:{port}", image],
            text=True,
        ).strip()

        # Wait for server to be ready
        import urllib.request
        for _ in range(30):
            try:
                urllib.request.urlopen(f"http://localhost:{port}/health", timeout=2)
                break
            except Exception:
                time.sleep(1)

        client = cls(base_url=f"http://localhost:{port}")
        client._container_id = container_id
        return client
