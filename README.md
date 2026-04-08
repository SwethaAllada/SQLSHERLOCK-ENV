---
title: SQLSherlock Env
emoji: 🔍
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
tags:
  - openenv
  - reinforcement-learning
  - data-quality
pinned: false
---

# SQLSherlock-Env

**An RL environment where an AI agent performs complete data quality audits on real-world datasets.**

Data cleaning consumes ~80% of a data scientist's time. This environment trains and evaluates AI agents to do it automatically: discover issues through statistical investigation, fix them with the right imputation strategy per column type, validate fixes, and export a clean dataset.

**Key design principles:**
- **No planted issues** — the environment scans real datasets at `reset()` and builds a ground-truth issue catalogue from what it actually finds
- **Any dataset** — pass any HuggingFace dataset URL, local CSV/JSON/Parquet file, or raw CSV text
- **Production data cleaning** — handles nulls, type errors, constraint violations, outliers, duplicates, whitespace issues, and inconsistent categories
- **Dense reward signal** — every action produces a training signal, not just end-of-episode binary feedback

---

## Architecture

### Episode Flow

```
reset(dataset, task_id)
        |
        v
+---------------------------------------------------------------+
|  DatabaseEngine                                                |
|                                                                |
|  1. load(source)     <-- CSV / JSON / JSONL / Parquet / HF    |
|  2. records_to_sqlite()  <-- In-memory SQLite per episode      |
|  3. deep_copy(originals) <-- Immutable snapshot before edits   |
|  4. profile_table()      <-- median/mode/std/z-scores          |
|  5. detect_issues()      <-- null/type/constraint/outlier/     |
|                              duplicate/whitespace/inconsistent  |
|  6. Validator(baseline)  <-- 6-check baseline captured         |
|  7. detect_trap()        <-- Task 3 only: plant 2x value       |
+---------------------------------------------------------------+
        |
        v
  Observation returned to agent
        |
        v
+-----------------------------------------------------+
|  Agent Step Loop                                     |
|                                                      |
|  investigate: inspect / profile_column / run_sql     |
|  bulk fix:    fix_column (all nulls in one step)     |
|  single fix:  fix_cell / delete_row                  |
|  check:       validate                               |
|  end:         submit / export                        |
|                                                      |
|  Each step -> reward signal -> observation            |
|  Repeat until submit/export or max_steps reached     |
+-----------------------------------------------------+
        |
        v
  Grader.score() -> final score [0.0 - 1.0]
```

### Grading Pipeline (7 steps)

```
1. Zero-change guard    -- if nothing changed -> 0.0
2. Resolution score     -- per issue: confidence-weighted
3. False-positive penalty -- -0.05 per clean cell touched
4. Trap penalty (task3) -- -0.40 if trap cell modified
5. Validation score     -- checks_passed / total * 0.30
6. Reasoning bonus      -- +0.05 for statistical reasoning
7. Final: res*0.60 + val*0.30 + bonus*0.10 - fp - trap
```

---

## Quick Start

### Docker (recommended)

```bash
docker build -t sqlsherlock-env:latest .
docker run -p 7860:7860 sqlsherlock-env:latest
curl http://localhost:7860/health
# -> {"status":"healthy"}
```

### Local Python

```bash
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r sqlsherlock_env/server/requirements.txt
cd sqlsherlock_env
PYTHONPATH=. uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Run Baseline Inference

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="hf_your_token"
export SPACE_URL="http://localhost:7860"
python inference.py
```

Output format (judges parse this exactly):

```
[START] task=task1_null_and_types env=sqlsherlock_env model=Qwen/Qwen2.5-72B-Instruct
[STEP]  step=1 action=inspect reward=0.02 done=false error=null
[STEP]  step=2 action=profile_column(Age) reward=0.03 done=false error=null
[STEP]  step=7 action=fix_column reward=0.18 done=false error=null
...
[END]   success=true steps=20 score=0.652 rewards=0.02,0.03,...
```

---

## Using Any Dataset

The environment works with **any** dataset, not just Titanic:

```python
from sqlsherlock_env.client import SQLSherlockEnv

env = SQLSherlockEnv(base_url="http://localhost:7860")

# HuggingFace dataset
obs = env.reset(dataset="scikit-learn/iris", task_id="task1_null_and_types")

# Local file
obs = env.reset(dataset="/path/to/data.csv", task_id="task2_constraints_and_fk")

# Raw CSV text
obs = env.reset(dataset="id,name,age\n1,Alice,\n2,Bob,25\n", task_id="task1_null_and_types")
```

Upload via API: `POST /upload_dataset` with a CSV/JSON/Parquet file.

Download cleaned output: use `export` action, then `GET /download/{file_id}`. Output format matches input (CSV in -> CSV out, Parquet in -> Parquet out).

---

## Data Cleaning Capabilities

The agent detects and fixes these real-world data quality issues:

| Issue Type | Detection Method | Fix Strategy | Tasks |
|---|---|---|---|
| **Null values** | `IS NULL` or empty string | Numeric: column **median**. String: column **mode** | All |
| **Type errors** | Text in predominantly numeric column (>=80% castable) | Column median | All |
| **Constraint violations** | Negative values in must-be-positive columns | `ABS(value)` | Task 2+ |
| **FK violations** | Orphan references across tables | `delete_row` | Task 2+ |
| **Whitespace** | Leading/trailing/extra spaces | Trimmed string | Task 2+ |
| **Inconsistent categories** | Case variants ("male"/"Male"/"MALE") | Dominant form (mode) | Task 2+ |
| **Statistical outliers** | IQR-based (robust to outlier-inflated std) | Column median | Task 3 |
| **Duplicates** | Same natural key appearing twice | `delete_row` on later row | Task 3 |
| **Trap** (task3 only) | Planted 2x value with z < 3 | **Do NOT touch** (-0.40 penalty) | Task 3 |

**Smart imputation per column type:**
- `profile_column` returns **median**, **mode**, **mean**, **null_rate**, **dtype**, **z_scores**
- Numeric nulls -> median (robust to outliers)
- String nulls -> mode (most frequent value)
- Structural nulls (>70% null) -> "Unknown" (low confidence)
- `fix_column` bulk-fixes ALL nulls + type errors + negatives in one step

---

## Action Space

| `action_type` | Required fields | Description |
|---|---|---|
| `inspect` | `table` | View all rows in the table |
| `profile_column` | `table`, `column` | Statistics: median, mode, mean, std, null_count, null_rate, z_scores, dtype |
| `run_sql` | `sql` | Read-only SELECT query (max 50 rows) |
| `fix_cell` | `table`, `row_id`, `column`, `value`, `reason` | Fix one cell with justification |
| `fix_column` | `table`, `column`, `value`, `reason` | Fix ALL nulls + type errors + negatives in a column (bulk) |
| `delete_row` | `table`, `row_id`, `reason` | Remove a duplicate or FK-violation row |
| `validate` | -- | Run all 6 before/after validation checks |
| `submit` | -- | Score and end episode |
| `export` | -- | Write cleaned file, score, and end episode |

---

## Reward System

Dense per-step rewards -- every action produces a training signal:

| Action | Reward | Cap |
|---|---|---|
| `inspect` | +0.02 | 3 rewarded |
| `profile_column` | +0.03 | 3 rewarded |
| `run_sql` | +0.03 | 3 rewarded |
| `validate` | +0.05 * (checks_passed / 6) | 2 rewarded |
| `fix_cell` -- correct | **+0.15** | -- |
| `fix_cell` -- false positive | **-0.20** | -- |
| `fix_cell` -- trap cell | **-0.40** | -- |
| `fix_column` -- has issues | **+0.15 to +0.30** (proportional to issues resolved) | -- |
| `fix_column` -- no issues | **-0.10** | -- |
| `delete_row` -- valid | **+0.15** | -- |
| `delete_row` -- false positive | **-0.20** | -- |
| `submit` -- all resolved | **+0.10** | -- |
| `submit` -- issues remain | **-0.10** | -- |

---

## Three Tasks

### Task 1 -- `task1_null_and_types` (Easy, max 30 steps)

Find and fix **null values** and **type errors** across all columns.

- Detects: nulls, empty strings, text in numeric columns
- Scoring: `resolution * 0.70 + validation * 0.30`
- Grader weights each issue by confidence (structural nulls = low confidence)

### Task 2 -- `task2_constraints_and_fk` (Medium, max 40 steps)

Everything in Task 1, plus:

- **Constraint violations**: negative values in must-be-positive columns
- **FK violations**: orphan references in related tables
- **Whitespace issues**: leading/trailing spaces
- **Inconsistent categories**: case variants normalized to dominant form

### Task 3 -- `task3_full_audit_with_trap` (Hard, max 50 steps)

Full statistical audit including:

- **Outliers**: IQR-based detection (z > 5)
- **Duplicates**: natural key collision
- **THE TRAP**: One numeric value set to 2x original. Looks suspicious but has z < 3. Touching it costs **-0.40**.

Rule: Always `profile_column` before fixing numeric values. z > 5 = outlier (fix). z < 3 = normal (leave alone).

---

## Validation (6 Checks)

| Check | Passes when |
|---|---|
| `null_check` | Nulls resolved (weighted by confidence) |
| `type_check` | Type errors castable to correct type |
| `range_check` | No negatives in must-be-positive columns |
| `distribution_check` | Column mean drift < 20% |
| `duplicate_check` | Duplicate count reduced |
| `outlier_check` | Flagged outlier rows no longer exceed z > 5 |

---

## API Reference

| Method | Path | Description |
|---|---|---|
| `WS` | `/ws` | Persistent WebSocket session |
| `POST` | `/reset` | Reset environment, load dataset |
| `POST` | `/step` | Execute one action |
| `GET` | `/state` | Current episode state |
| `GET` | `/health` | Health check |
| `GET` | `/tasks` | List all 3 tasks |
| `POST` | `/upload_dataset` | Upload CSV/JSON/Parquet file |
| `GET` | `/download/{file_id}` | Download cleaned output |
| `GET` | `/docs` | OpenAPI documentation (Swagger UI) |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `API_BASE_URL` | `https://router.huggingface.co/v1` | LLM API endpoint |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | Model identifier |
| `HF_TOKEN` | -- (required) | HuggingFace token for LLM access |
| `SPACE_URL` | `http://localhost:7860` | Environment server URL |

---

## Baseline Scores (phihung/titanic, 500 rows)

| Task | Difficulty | Baseline Score |
|---|---|---|
| `task1_null_and_types` | Easy | **0.652** |
| `task2_constraints_and_fk` | Medium | **0.861** |
| `task3_full_audit_with_trap` | Hard | **0.658** |
| **Average** | | **0.724** |

Runtime: ~50-90 seconds for all 3 tasks. Well within the 20-minute limit.

---

## Setup on a New Device

### Option A: Docker

```bash
git clone <repo-url> && cd SQLSherlock-env
docker build -t sqlsherlock-env:latest .
docker run -p 7860:7860 sqlsherlock-env:latest
# In another terminal:
export HF_TOKEN="hf_your_token"
export SPACE_URL="http://localhost:7860"
python inference.py
```

### Option B: Local Python (Linux/Mac)

```bash
git clone <repo-url> && cd SQLSherlock-env
python3 -m venv .venv && source .venv/bin/activate
pip install -r sqlsherlock_env/server/requirements.txt
# Terminal 1:
cd sqlsherlock_env && PYTHONPATH=. uvicorn server.app:app --host 0.0.0.0 --port 7860
# Terminal 2:
export HF_TOKEN="hf_your_token" SPACE_URL="http://localhost:7860"
python inference.py
```

### Option C: Local Python (Windows PowerShell)

```powershell
git clone <repo-url>; cd SQLSherlock-env
python -m venv .venv; .venv\Scripts\Activate.ps1
pip install -r sqlsherlock_env\server\requirements.txt
# Terminal 1:
cd sqlsherlock_env; $env:PYTHONPATH=(Get-Location).Path
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860
# Terminal 2:
$env:HF_TOKEN="hf_your_token"; $env:SPACE_URL="http://localhost:7860"
python inference.py
```

### Run Tests

```bash
PYTHONPATH=sqlsherlock_env pytest tests/ -v   # 98 tests, all pass
```

---

## GRPO Training

```bash
pip install trl transformers torch
export SPACE_URL="http://localhost:7860"
export MODEL_ID="Qwen/Qwen2.5-1.5B-Instruct"
python train.py
```

---

## Project Structure

```
SQLSherlock-env/
+-- Dockerfile                  <- repo root (HF Spaces)
+-- README.md                   <- this file
+-- openenv.yaml                <- OpenEnv manifest (3 tasks)
+-- inference.py                <- baseline inference ([START]/[STEP]/[END])
+-- train.py                    <- TRL GRPO training scaffold
+-- .gitignore
+-- .dockerignore
+-- sqlsherlock_env/
|   +-- __init__.py
|   +-- client.py               <- synchronous WebSocket/HTTP client
|   +-- models.py               <- Action / Observation / State (Pydantic)
|   +-- server/
|       +-- app.py              <- FastAPI + WebSocket handler
|       +-- environment.py      <- RL core: reset() / step() / state
|       +-- database.py         <- In-memory SQLite engine per episode
|       +-- dataset_loader.py   <- CSV / JSON / JSONL / Parquet / HF
|       +-- schema_profiler.py  <- Column statistics + z-scores
|       +-- issue_detector.py   <- 8 issue types + trap planting
|       +-- validator.py        <- 6-check before/after validator
|       +-- reward.py           <- Dense per-step reward
|       +-- exporter.py         <- Format-preserving output writer
|       +-- requirements.txt
|       +-- graders/
|           +-- universal.py    <- 7-step scoring pipeline
|           +-- task1.py
|           +-- task2.py
|           +-- task3.py
+-- tests/                      <- 98 tests
    +-- conftest.py
    +-- test_issue_detector.py
    +-- test_graders.py
    +-- test_environment.py
```
