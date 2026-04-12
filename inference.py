# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
SQLSherlock-Env — Production Inference Script.

STDOUT FORMAT (mandatory — judges parse this exactly):

    [START] task=<task_name> env=sqlsherlock_env model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...>

Environment variables:
    API_BASE_URL   LLM endpoint  (default: https://api.openai.com/v1)
    MODEL_NAME     Model id       (default: gpt-4.1-mini)
    HF_TOKEN       API key        (required — no default)
    SPACE_URL      Server URL     (default: http://localhost:7860)
"""

import json
import os
import re
import sys
import time
from typing import Any, Optional

from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEMO_DATASET       = "phihung/titanic"
INFERENCE_MAX_ROWS = 500
ENV_NAME           = "sqlsherlock_env"

API_BASE_URL  = os.getenv("API_BASE_URL",  "https://api.openai.com/v1")
MODEL_NAME    = os.getenv("MODEL_NAME",    "gpt-4.1-mini")
HF_TOKEN      = os.getenv("HF_TOKEN")
SPACE_URL     = os.getenv("SPACE_URL",     "http://localhost:7860")
# DEMO_INTENT selects which intent's 3 difficulty tasks to run.
# Set via environment variable — defaults to "visualization".
DEMO_INTENT   = os.getenv("DEMO_INTENT",  "visualization")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# 9 tasks: 3 intents × 3 difficulty levels (easy / medium / hard)
# Task IDs follow the pattern: {intent_prefix}_{difficulty}
#   viz_easy, viz_medium, viz_hard
#   ml_easy,  ml_medium,  ml_hard
#   bq_easy,  bq_medium,  bq_hard

STEP_BUDGETS: dict[str, int] = {
    "viz_easy": 30, "viz_medium": 40, "viz_hard": 50,
    "ml_easy":  30, "ml_medium":  40, "ml_hard":  50,
    "bq_easy":  30, "bq_medium":  40, "bq_hard":  50,
}

# Map each intent to its 3 difficulty tasks (easy → medium → hard)
INTENT_TASK_MAP: dict[str, list[tuple[str, str]]] = {
    "visualization":  [("viz_easy", "easy"), ("viz_medium", "medium"), ("viz_hard", "hard")],
    "ml_training":    [("ml_easy",  "easy"), ("ml_medium",  "medium"), ("ml_hard",  "hard")],
    "business_query": [("bq_easy",  "easy"), ("bq_medium",  "medium"), ("bq_hard",  "hard")],
}

# Select tasks for the demo intent; fall back to visualization if unknown
TASKS: list[tuple[str, str, str]] = [
    (task_id, difficulty, DEMO_INTENT)
    for task_id, difficulty in INTENT_TASK_MAP.get(DEMO_INTENT, INTENT_TASK_MAP["visualization"])
]


# ---------------------------------------------------------------------------
# Mandatory log helpers
# ---------------------------------------------------------------------------

def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env={ENV_NAME} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool,
             error: Optional[str] = None) -> None:
    action_str = action.replace("\n", " ").replace("\r", " ").strip()[:120]
    error_str  = (
        error.replace("\n", " ").replace("\r", " ").strip()[:120]
        if error else "null"
    )
    print(
        f"[STEP] step={step} action={action_str} "
        f"reward={reward:.2f} done={str(done).lower()} "
        f"error={error_str}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"rewards={rewards_str}",
        flush=True,
    )


def _parse_score(feedback: str) -> Optional[float]:
    m = re.search(r"[Gg]rader\s+score\s*=?\s*(\d+\.\d+)", feedback)
    if m:
        try:
            return float(m.group(1))
        except (ValueError, TypeError):
            pass
    return None


def _label(d: dict) -> str:
    a = d.get("action_type", "?")
    if a == "fix_cell":
        return f"fix_cell(row={d.get('row_id')},col={d.get('column')},val={d.get('value')})"
    if a == "profile_column":
        return f"profile_column({d.get('column')})"
    if a == "run_sql":
        return f"run_sql({(d.get('sql') or '')[:40]})"
    if a == "delete_row":
        return f"delete_row(row={d.get('row_id')})"
    if a == "join_tables":
        return f"join_tables({d.get('table')},{d.get('table2')},key={d.get('key')})"
    if a == "classify_intent":
        return f"classify_intent({d.get('value')})"
    if a == "select_tables":
        return f"select_tables({d.get('tables')})"
    return a


# ---------------------------------------------------------------------------
# LLM call helper
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert data scientist performing a complete data quality audit.
You have inspected and profiled the data. Now decide the NEXT action.

Available actions (respond with ONE JSON object):
- {"action_type":"profile_column","table":"T","column":"C"}
- {"action_type":"run_sql","sql":"SELECT id FROM \\"T\\" WHERE \\"C\\" IS NULL LIMIT 10"}
- {"action_type":"fix_column","table":"T","column":"C","value":29.5,"reason":"bulk fix nulls, median=29.5"}
- {"action_type":"fix_cell","table":"T","row_id":6,"column":"C","value":29.5,"reason":"null, median=29.5"}
- {"action_type":"delete_row","table":"T","row_id":8,"reason":"duplicate Name"}
- {"action_type":"validate"}
- {"action_type":"submit"}

PREFER fix_column over fix_cell when fixing nulls — it fixes ALL nulls in one step.

RULES:
- Use EXACT row ids from SQL results. NEVER guess ids.
- Numeric nulls → column MEDIAN. String nulls → column MODE.
- Skip columns with null_rate >= 0.70 (structural nulls).
- OUTLIER detection uses IQR (Q1 - 3×IQR, Q3 + 3×IQR). Fix detected outliers to the column median.
- TRAP: if a value was flagged by the system as suspicious but is NOT in profile_column outlier list, DO NOT TOUCH it.

Respond with ONLY one JSON object. No markdown, no text."""


def _call_llm(client: OpenAI, messages: list[dict]) -> Optional[dict]:
    """Call LLM and parse JSON action. Returns None on failure."""
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME, messages=messages,
            max_tokens=300, temperature=0.0,
        )
        raw = (resp.choices[0].message.content or "").strip()
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```\s*$", "", raw)
        raw = raw.strip()
        if not raw.startswith("{"):
            start = raw.find("{")
            end   = raw.rfind("}")
            if start >= 0 and end > start:
                raw = raw[start:end + 1]
        return json.loads(raw)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------

def run_task(task_id: str, intent: str = "") -> float:
    """Run one task episode. Guarantees [START] … [END] on stdout regardless of errors."""
    pkg_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sqlsherlock_env")
    if pkg_dir not in sys.path:
        sys.path.insert(0, pkg_dir)

    from client import SQLSherlockEnv
    from models import SQLSherlockAction

    budget      = STEP_BUDGETS[task_id]
    rewards: list[float] = []
    step_num    = 0        # initialised here so except/finally can always read it
    score       = 0.0
    success     = False
    env         = None     # initialised inside try so finally can check

    log_start(task=task_id, model=MODEL_NAME)

    try:
        llm = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
        env = SQLSherlockEnv(base_url=SPACE_URL)

        obs = env.reset(
            dataset=DEMO_DATASET, task_id=task_id,
            max_rows=INFERENCE_MAX_ROWS,
            intent=intent or None,
        )

        table        = list(obs.tables_summary.keys())[0] if obs.tables_summary else (DEMO_DATASET.split("/")[-1])
        columns      = obs.tables_summary.get(table, {}).get("columns", [])
        visible_cols = [c for c in columns if c not in ("id", "_source_format")]

        done         = False
        col_profiles: dict[str, dict] = {}

        llm_messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        def _do_step(action_dict: dict) -> tuple[float, bool]:
            nonlocal step_num, done, obs
            step_num += 1
            if step_num > budget or done:
                return 0.0, True
            filtered = {k: v for k, v in action_dict.items() if v is not None}
            action = SQLSherlockAction(**filtered)
            reward = 0.0
            error_msg = None
            try:
                obs, reward, done, _ = env.step(action)
                reward = float(reward or 0.0)
            except Exception as exc:
                error_msg = str(exc)[:80]
            rewards.append(reward)
            log_step(step_num, _label(action_dict), reward, done, error_msg)
            return reward, done

        # Phase 1: Inspect
        _do_step({"action_type": "inspect", "table": table})

        # Phase 2: Profile + bulk fix interleaved
        for col in visible_cols:
            if done or step_num >= budget - 2:
                break
            _do_step({"action_type": "profile_column", "table": table, "column": col})
            if not obs.query_result:
                continue
            profile = obs.query_result[0]
            col_profiles[col] = profile

            null_count = profile.get("null_count", 0)
            null_rate  = profile.get("null_rate", 0.0)
            dtype      = profile.get("dtype", "unknown")
            median_val = profile.get("median")
            mode_val   = profile.get("mode")
            mean_val   = profile.get("mean")

            if null_count == 0:
                continue

            if dtype in ("int", "float"):
                fill_value = median_val or mean_val or 0
            elif null_rate >= 0.70:
                fill_value = "Unknown"
            else:
                fill_value = mode_val or "Unknown"

            strategy = "median" if dtype in ("int", "float") else "mode"
            _do_step({
                "action_type": "fix_column", "table": table,
                "column": col, "value": fill_value,
                "reason": f"bulk fix {null_count} nulls in {col}, {strategy}={fill_value}",
            })

        # Phase 3: LLM-assisted advanced cleaning
        if not done and step_num < budget - 3:
            fixed_summary    = f"Profiled {len(col_profiles)} columns. Bulk-fixed nulls."
            remaining_budget = budget - step_num - 2
            llm_messages.append({"role": "user", "content": (
                f"Table: \"{table}\", Columns: {visible_cols}\n"
                f"Done so far: {fixed_summary}\n"
                f"Remaining budget: {remaining_budget} actions.\n"
                "Find remaining issues: type errors, negatives, duplicates, whitespace. "
                "Respond with one JSON action, or {\"action_type\":\"validate\"} if done."
            )})

            for _ in range(min(remaining_budget, 6)):
                if done or step_num >= budget - 2:
                    break
                action_dict = _call_llm(llm, llm_messages)
                if action_dict is None or action_dict.get("action_type") in ("validate", "submit"):
                    break
                _, d = _do_step(action_dict)
                if d:
                    break
                feedback = (obs.last_feedback or "")[:300]
                if obs.query_result:
                    ids = [r2.get("id") for r2 in obs.query_result if r2.get("id") is not None]
                    if ids:
                        feedback += f"\nRow IDs: {ids[:15]}"
                llm_messages.append({"role": "assistant", "content": json.dumps(action_dict)})
                llm_messages.append({"role": "user", "content": feedback + "\nNext action?"})

        # Phase 4: Validate → Submit
        if not done and step_num < budget:
            _do_step({"action_type": "validate"})

        if not done:
            _do_step({"action_type": "submit"})
            if obs.last_feedback:
                parsed = _parse_score(obs.last_feedback)
                if parsed is not None:
                    score = max(0.0, min(1.0, parsed))

        # Fallback score from cumulative positive rewards
        if score == 0.0 and rewards:
            positive = sum(r for r in rewards if r > 0)
            score = max(0.0, min(1.0, positive / max(budget * 0.15, 0.01)))

        success = score >= 0.50

    except Exception as exc:
        # Catch-all — log a terminal error step so judges see what failed
        log_step(
            max(step_num + 1, 1), "error", 0.0, True,
            f"{type(exc).__name__}: {str(exc)[:60]}",
        )

    finally:
        # [END] is ALWAYS emitted here — even after early returns / exceptions
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        log_end(success=success, steps=step_num, rewards=rewards)

    return score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    wall_start   = time.time()
    all_scores: list[float] = []

    print(
        f"\n=== SQLSherlock-Env  intent={DEMO_INTENT}  model={MODEL_NAME} ===\n",
        file=sys.stderr,
    )

    for task_id, _, intent in TASKS:
        score = run_task(task_id, intent=intent)
        all_scores.append(score)
        time.sleep(1)

    avg   = sum(all_scores) / len(all_scores) if all_scores else 0.0
    total = time.time() - wall_start

    print(
        f"\n=== Results  intent={DEMO_INTENT}  avg={avg:.3f}  "
        f"runtime={total:.1f}s ===",
        file=sys.stderr,
    )
    for (tid, diff, _), sc in zip(TASKS, all_scores):
        bar = "\u2588" * int(sc * 20) + "\u2591" * (20 - int(sc * 20))
        print(f"  [{diff:<6}] {tid:<20} [{bar}] {sc:.3f}", file=sys.stderr)


if __name__ == "__main__":
    main()
