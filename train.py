# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
SQLSherlock-Env — TRL GRPO Training Script.

Fine-tunes a language model via Group Relative Policy Optimisation (GRPO)
using the SQLSherlock RL environment as the reward signal.

The model learns the data-scientist investigation workflow:
  profile → hypothesise → fix → validate → export

Environment variables:
    SPACE_URL     — SQLSherlock server URL  (default: http://localhost:7860)
    MODEL_ID      — Base model to fine-tune (default: Qwen/Qwen2.5-1.5B-Instruct)
    DATASET_NAME  — Training dataset        (default: mstz/titanic)
    OUTPUT_DIR    — Checkpoint output dir   (default: ./grpo_output)
    NUM_STEPS     — Training steps          (default: 200)
    BATCH_SIZE    — Batch size              (default: 4)
    TASK_ID       — Task to train on        (default: task1_null_and_types)
"""

import os
import sys

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SPACE_URL    = os.environ.get("SPACE_URL",    "http://localhost:7860")
MODEL_ID     = os.environ.get("MODEL_ID",     "Qwen/Qwen2.5-1.5B-Instruct")
DATASET_NAME = os.environ.get("DATASET_NAME", "phihung/titanic")
OUTPUT_DIR   = os.environ.get("OUTPUT_DIR",   "./grpo_output")
NUM_STEPS    = int(os.environ.get("NUM_STEPS",  "200"))
BATCH_SIZE   = int(os.environ.get("BATCH_SIZE", "4"))
TASK_ID      = os.environ.get("TASK_ID",      "task1_null_and_types")

# ---------------------------------------------------------------------------
# GRPO Environment wrapper
# ---------------------------------------------------------------------------

class SQLSherlockGRPOEnv:
    """Thin wrapper around SQLSherlockEnv exposing tool-call methods.

    Each method corresponds to one action type.  TRL's GRPO trainer
    calls reset() to start an episode, then the model calls methods
    as tool calls.  The cumulative reward is read via reward_func().
    """

    def __init__(self) -> None:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "sqlsherlock_env"))
        from client import SQLSherlockEnv
        self._env_class = SQLSherlockEnv
        self._client = None
        self.reward = 0.0
        self._primary_table: str = "dataset"

    def _client_or_create(self):
        if self._client is None:
            self._client = self._env_class(base_url=SPACE_URL)
        return self._client

    def reset(self, **kwargs) -> str:
        """Reset the environment and return a string observation.

        Args:
            dataset (str): HuggingFace dataset name or file path.
            task_id (str): Task identifier string.
        """
        from client import SQLSherlockEnv
        # Fresh client each episode for isolation
        try:
            if self._client is not None:
                self._client.close()
        except Exception:
            pass
        self._client = SQLSherlockEnv(base_url=SPACE_URL)

        dataset = kwargs.get("dataset", DATASET_NAME)
        task_id = kwargs.get("task_id", TASK_ID)

        obs = self._client.reset(task_id=task_id, dataset=dataset)
        self._primary_table = list(obs.tables_summary.keys())[0]
        self.reward = 0.0

        return (
            f"Table: {self._primary_table}\n"
            f"Columns: {obs.tables_summary[self._primary_table]['columns']}\n"
            f"Rows: {obs.tables_summary[self._primary_table]['row_count']}\n"
            f"Task: {obs.task_description}"
        )

    def inspect_table(self, table: str) -> str:
        """View all rows in a database table.

        Args:
            table: Name of the table to inspect.
        """
        from models import SQLSherlockAction
        obs, r, done, _ = self._client_or_create().step(
            SQLSherlockAction(action_type="inspect", table=table)
        )
        self.reward += r
        return obs.last_feedback

    def profile_column(self, table: str, column: str) -> str:
        """Get statistical profile: mean, std, min, max, null_count, z-scores.

        Args:
            table:  Table name containing the column.
            column: Column name to profile statistically.
        """
        from models import SQLSherlockAction
        obs, r, done, _ = self._client_or_create().step(
            SQLSherlockAction(
                action_type="profile_column", table=table, column=column
            )
        )
        self.reward += r
        return obs.last_feedback

    def run_query(self, sql: str) -> str:
        """Execute a SELECT SQL query to find data quality issues.

        Args:
            sql: A SELECT SQL query string. No write operations allowed.
        """
        from models import SQLSherlockAction
        obs, r, done, _ = self._client_or_create().step(
            SQLSherlockAction(action_type="run_sql", sql=sql)
        )
        self.reward += r
        return obs.last_feedback

    def fix_cell(
        self,
        table: str,
        row_id: int,
        column: str,
        value: str,
        reason: str,
    ) -> str:
        """Fix a data quality issue in one cell.

        Args:
            table:   Table name.
            row_id:  Row primary key.
            column:  Column to fix.
            value:   The corrected value to write.
            reason:  Statistical justification for this fix (e.g. z-score, median).
        """
        from models import SQLSherlockAction
        obs, r, done, _ = self._client_or_create().step(
            SQLSherlockAction(
                action_type="fix_cell",
                table=table,
                row_id=row_id,
                column=column,
                value=value,
                reason=reason,
            )
        )
        self.reward += r
        return obs.last_feedback

    def delete_row(self, table: str, row_id: int, reason: str) -> str:
        """Delete a duplicate or FK-violation row.

        Args:
            table:   Table name.
            row_id:  Row primary key to delete.
            reason:  Why this row should be removed (e.g. duplicate key detected).
        """
        from models import SQLSherlockAction
        obs, r, done, _ = self._client_or_create().step(
            SQLSherlockAction(
                action_type="delete_row",
                table=table,
                row_id=row_id,
                reason=reason,
            )
        )
        self.reward += r
        return obs.last_feedback

    def validate(self) -> str:
        """Run all 6 validation checks comparing cleaned vs raw data.

        Call this after making fixes to verify your work is correct.
        Returns pass/fail status for each check.
        """
        from models import SQLSherlockAction
        obs, r, done, _ = self._client_or_create().step(
            SQLSherlockAction(action_type="validate")
        )
        self.reward += r
        return obs.last_feedback

    def submit(self) -> str:
        """Submit the investigation for final scoring.

        Call only when you have fixed all discovered issues and
        validate() shows improvement.
        """
        from models import SQLSherlockAction
        obs, r, done, _ = self._client_or_create().step(
            SQLSherlockAction(action_type="submit")
        )
        self.reward += r
        last = obs.reward_trace[-1] if obs.reward_trace else {}
        return f"Final reward: {last.get('total', 0.0):.4f}"


# ---------------------------------------------------------------------------
# GRPO reward function
# ---------------------------------------------------------------------------

def reward_func(environments: list, **kwargs) -> list[float]:
    """Return cumulative episode reward for each environment.

    Called by TRL's GRPOTrainer after each rollout batch.

    Args:
        environments: List of SQLSherlockGRPOEnv instances.

    Returns:
        List of float rewards, one per environment.
    """
    return [env.reward for env in environments]


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def main() -> None:
    try:
        from trl import GRPOConfig, GRPOTrainer
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError:
        print(
            "Training dependencies not installed.\n"
            "Install with:  pip install 'sqlsherlock-env[train]'\n"
            "  or:          pip install trl transformers torch"
        )
        sys.exit(1)

    print(f"SQLSherlock GRPO Training")
    print(f"  Model   : {MODEL_ID}")
    print(f"  Dataset : {DATASET_NAME}")
    print(f"  Task    : {TASK_ID}")
    print(f"  Steps   : {NUM_STEPS}")
    print(f"  Output  : {OUTPUT_DIR}")
    print(f"  Server  : {SPACE_URL}")
    print()

    # Load model and tokenizer
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model     = AutoModelForCausalLM.from_pretrained(MODEL_ID)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build a minimal training prompt dataset
    # The model generates tool calls; the environment provides rewards
    training_prompts = [
        {
            "prompt": (
                "You are a data scientist. Investigate the dataset for quality issues.\n"
                f"Dataset: {DATASET_NAME}\n"
                f"Task: {TASK_ID}\n"
                "Use the available tools: inspect_table, profile_column, run_query, "
                "fix_cell, delete_row, validate, submit.\n"
                "Start by inspecting the table."
            )
        }
        for _ in range(max(BATCH_SIZE * 4, 16))
    ]

    # GRPO configuration
    grpo_config = GRPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        max_steps=NUM_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
        logging_steps=10,
        save_steps=50,
        num_generations=BATCH_SIZE,
        max_new_tokens=256,
        temperature=0.7,
        report_to="none",
    )

    # Instantiate environments (one per generation slot)
    environments = [SQLSherlockGRPOEnv() for _ in range(BATCH_SIZE)]

    # Build tools list for the trainer
    tools = [
        environments[0].inspect_table,
        environments[0].profile_column,
        environments[0].run_query,
        environments[0].fix_cell,
        environments[0].delete_row,
        environments[0].validate,
        environments[0].submit,
    ]

    print("Starting GRPO training...")
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        tokenizer=tokenizer,
        train_dataset=training_prompts,
        reward_funcs=reward_func,
        env=environments,
        tools=tools,
    )

    trainer.train()

    print(f"\nTraining complete. Checkpoints saved to: {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Final model saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
