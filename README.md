---
title: SQLSherlock Env
emoji: 🔍
colorFrom: indigo
colorTo: cyan
sdk: docker
app_port: 7860
tags:
  - openenv
  - reinforcement-learning
  - data-quality
pinned: false
---

# SQLSherlock-Env

An RL environment where an AI agent acts as a data scientist investigating a dirty dataset.

The agent discovers real data quality issues through statistical investigation — exactly like a human data scientist — fixes them with documented reasoning, validates fixes against the raw baseline, and exports the cleaned output in the same format as the input.

 Real datasets already have data quality problems. The issue detector scans the dataset at `reset()` time and builds a ground-truth catalogue from what it finds. The agent never sees this catalogue — it must discover everything through investigation.

---

## Architecture

### Episode Flow

```
reset(dataset, task_id)
        │
        ▼
┌───────────────────────────────────────────────────────────────────┐
│  DatabaseEngine.__init__                                          │
│                                                                   │
│  1. load(source)         ← CSV / JSON / JSONL / Parquet / HF     │
│  2. records_to_sqlite()  ← In-memory SQLite, isolated per episode│
│  3. deep_copy(originals) ← Immutable snapshot before any edits   │
│  4. profile_table()      ← mean/std/z-scores per column          │
│  5. detect_issues()      ← null / type / constraint / outlier    │
│                             duplicate / fk_violation             │
│  6. Validator(baseline)  ← 6-check baseline captured             │
│  7. detect_trap()        ← Task 3 only: plant 2x value in DB     │
└───────────────────────────────────────────────────────────────────┘
        │
        ▼
 SQLSherlockObservation returned to agent
        │
        ▼
┌─────────────────────────────────────────────────────┐
│              Agent Step Loop                        │
│                                                     │
│  ┌──────────────────────────────────────────────┐  │
│  │  Agent decides action (LLM call)             │  │
│  │                                              │  │
│  │  investigate:  inspect / profile / run_sql   │  │
│  │  fix:          fix_cell / delete_row         │  │
│  │  check:        validate                      │  │
│  │  end:          submit / export               │  │
│  └───────────────────┬──────────────────────────┘  │
│                      │                             │
│                      ▼                             │
│  ┌──────────────────────────────────────────────┐  │
│  │  Environment.step(action)                    │  │
│  │                                              │  │
│  │  1. dispatch action → DatabaseEngine        │  │
│  │  2. reward.calc() → RB breakdown            │  │
│  │  3. build observation (feedback + results)  │  │
│  │  4. return (obs, reward, done, info)        │  │
│  └──────────────────────────────────────────────┘  │
│                                                     │
│  Repeat until submit/export or budget exhausted     │
└─────────────────────────────────────────────────────┘
        │
        ▼
  Grader.score() → final score [0.0 – 1.0]
```

### Component Diagram

```
inference.py / train.py / custom agent
        │  HTTP + WebSocket
        ▼
┌─────────────────────────────────────────────────────────────┐
│  FastAPI App  (server/app.py)                               │
│  POST /reset   POST /step   GET /state   GET /health        │
│  WS /ws                                                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  SQLSherlockEnvironment  (server/environment.py)            │
│                                                             │
│  reset()  ─────────────────────────────────────────────►   │
│                                              DatabaseEngine │
│  step(action)  ─────►  dispatch  ──────────────────────►   │
│                              │                             │
│                              │                             │
│                         ┌────▼────┐                        │
│                         │ reward  │                        │
│                         │  .calc()│                        │
│                         └─────────┘                        │
│                                                             │
│  on submit/export  ─────►  Grader.score()                  │
└─────────────────────────────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────────────┐
        ▼              ▼                      ▼
┌─────────────┐ ┌─────────────────┐ ┌──────────────────┐
│  Database   │ │  IssueDetector  │ │    Validator      │
│  Engine     │ │                 │ │                   │
│             │ │  detect_issues()│ │  6-check before/  │
│  SQLite     │ │  detect_trap()  │ │  after comparison │
│  in-memory  │ │                 │ │                   │
│  per episode│ │  null           │ │  null_check       │
│             │ │  type_error     │ │  type_check       │
│  profile_   │ │  constraint     │ │  range_check      │
│  table()    │ │  outlier        │ │  distribution_    │
│             │ │  duplicate      │ │    check          │
│  z_scores   │ │  fk_violation   │ │  duplicate_check  │
│  per row    │ │                 │ │  outlier_check    │
└─────────────┘ └─────────────────┘ └──────────────────┘
```

### Grading Pipeline (7 steps)

```
submit / export triggered
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│  universal.py — 7-step grader                               │
│                                                             │
│  Step 1: Zero-change guard                                  │
│          └── if nothing changed → score = 0.0              │
│                                                             │
│  Step 2: Resolution score  (0.0 – 1.0)                     │
│          └── per issue: confidence-weighted correct/total   │
│              null: confidence 0.20 – 1.0 (structural=0.20)  │
│              type_error: always 1.0                         │
│              constraint / outlier: 0.80                     │
│              duplicate: 0.70                                │
│                                                             │
│  Step 3: False-positive penalty                             │
│          └── −0.15 per clean cell touched                   │
│                                                             │
│  Step 4: Trap penalty (Task 3 only)                         │
│          └── −0.40 if trap cell was modified                │
│                                                             │
│  Step 5: Validation score  (0.0 – 0.30)                    │
│          └── checks_passed / total_checks × 0.30           │
│                                                             │
│  Step 6: Reasoning bonus  (0.0 – 0.10)                     │
│          └── +0.02 per fix_cell/delete_row with reason str  │
│                                                             │
│  Step 7: Final clamp                                        │
│          raw = res×0.60 + val×0.30 + bonus×0.10 − fp − trap│
│          score = clamp(raw, 0.0, 1.0)                       │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Docker (recommended)

```bash
# Build from repo root
docker build -t sqlsherlock-env:latest .

# Run
docker run -p 7860:7860 sqlsherlock-env:latest

# Verify
curl http://localhost:7860/health
```

### 2. Local (without Docker)

```bash
cd sqlsherlock_env
pip install -r server/requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### 3. Run baseline inference

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="hf_..."
export SPACE_URL="http://localhost:7860"

python inference.py
```

Expected stdout (judges parse this exactly):

```
[START] task=task1_null_and_types env=sqlsherlock_env model=Qwen/Qwen2.5-72B-Instruct
[STEP]  step=1 action=inspect reward=0.02 done=false error=null
[STEP]  step=2 action=profile_column(age) reward=0.03 done=false error=null
...
[END]   success=true steps=8 score=0.820 rewards=0.02,0.03,0.15,0.15,0.05,0.15,0.10
```

---

## Using Your Own Dataset

`inference.py` uses `phihung/titanic` for hackathon validation. To use your own dataset, connect the client directly:

### HuggingFace dataset

```python
from sqlsherlock_env.client import SQLSherlockEnv

env = SQLSherlockEnv(base_url="http://localhost:7860")
obs = env.reset(
    dataset="your_org/your_dataset",         # any public HF dataset
    task_id="task1_null_and_types",
    max_rows=500,
)
```

### Local file (CSV / JSON / JSONL / Parquet)

```python
obs = env.reset(
    dataset="/absolute/path/to/data.csv",
    task_id="task2_constraints_and_fk",
)
```

### Raw CSV string

```python
csv_text = "id,name,age,fare\n1,Alice,,25.0\n2,Bob,FORTY,50.0\n..."
obs = env.reset(
    dataset=csv_text,
    task_id="task1_null_and_types",
)
```

### Upload via API

```bash
curl -X POST http://localhost:7860/upload_dataset \
  -F "file=@data.csv" \
  -F "task_id=task1_null_and_types"
```

**What the environment does with your dataset:**
1. Loads the data (any format above)
2. Auto-detects column types (int / float / str / bool)
3. Scans for real data quality issues — no injection
4. Builds a ground-truth issue catalogue the agent never sees
5. Plants a trap value in Task 3

The agent then investigates, fixes, validates, and exports. The exported file matches the input format (CSV in → CSV out, Parquet in → Parquet out).

---

## Action Space

| `action_type` | Required fields | Description |
|---|---|---|
| `inspect` | `table` | View all rows |
| `profile_column` | `table`, `column` | Stats: mean/std/min/max/nulls/z-scores |
| `run_sql` | `sql` | SELECT query (read-only, max 50 rows) |
| `fix_cell` | `table`, `row_id`, `column`, `value`, `reason` | Fix one cell with justification |
| `fix_column` | `table`, `column`, `value`, `reason` | Fix ALL nulls in a column at once (bulk) |
| `delete_row` | `table`, `row_id`, `reason` | Remove duplicate or FK row |
| `validate` | — | Run all 6 before/after checks |
| `submit` | — | Score and end episode |
| `export` | — | Write cleaned file, score and end episode |

---

## Reward System

| Action | Reward | Cap |
|---|---|---|
| `inspect` | +0.02 | 3 rewarded |
| `profile_column` | +0.03 | 3 rewarded |
| `run_sql` | +0.03 | 3 rewarded |
| `validate` | +0.05 × (checks_passed / 6) | 2 rewarded |
| `fix_cell` — correct | **+0.15** | — |
| `fix_cell` — false positive | **−0.20** | — |
| `fix_cell` — trap cell | **−0.40** | — |
| `fix_cell` — wrong value | **−0.10** | — |
| `delete_row` — valid | **+0.15** | — |
| `delete_row` — false positive | **−0.20** | — |
| `submit` — all resolved | +0.10 | — |
| `submit` — issues remain | −0.10 | — |

---

## Three Tasks

### Task 1 — `task1_null_and_types` (Easy, max 20 steps)

Find and fix **null values** and **type errors**.

- Null: `None` or empty string in any non-PK column
- Type error: text in a numeric column (e.g. `"FORTY"` in age)
- Score: `resolution × 0.70 + validation × 0.30`

### Task 2 — `task2_constraints_and_fk` (Medium, max 25 steps)

Everything in Task 1 plus:

- **Constraint violations**: negative values in must-be-positive columns (age, fare, price)
- **FK violations**: orphan references in related tables

### Task 3 — `task3_full_audit_with_trap` (Hard, max 30 steps)

Full audit including:

- **Statistical outliers**: z-score > 5 in any numeric column
- **Duplicates**: natural key appearing more than once

**THE TRAP**: One numeric value is set to 2x original — looks suspicious but has `z < 3`. Touching it costs **−0.40**.

> Rule: Always `profile_column` before fixing any numeric value.
> `z > 5` → real outlier → fix it. `z < 3` → legitimate → leave it.

---

## Validation (6 Checks)

Run with `validate` action. Compares current state against the baseline from `reset()`:

| Check | Passes when |
|---|---|
| `null_check` | High-confidence nulls resolved |
| `type_check` | All type errors castable to float |
| `range_check` | No negatives in must-be-positive columns |
| `distribution_check` | Column mean drift < 20% |
| `duplicate_check` | Duplicate count reduced |
| `outlier_check` | No previously-flagged rows still exceed z > 5 |

Returns `PASS` / `PARTIAL` / `FAIL` with per-check detail and drift warnings.

---

## API Reference

| Method | Path | Description |
|---|---|---|
| `WS` | `/ws` | Persistent WebSocket session |
| `POST` | `/reset` | Reset environment, load dataset |
| `POST` | `/step` | Execute one action |
| `GET` | `/state` | Current episode state |
| `GET` | `/health` | Health check (`{"status":"ok"}`) |
| `GET` | `/tasks` | List all 3 tasks |
| `POST` | `/upload_dataset` | Upload file, get session |
| `GET` | `/download/{file_id}` | Download cleaned output |
| `GET` | `/docs` | OpenAPI docs (Swagger UI) |

---

## Testing

### Run all tests

```bash
cd SQLSherlock-env
pip install pytest
pytest tests/ -v
```

### Test checklist

```
tests/test_issue_detector.py    ← null / type_error / constraint / outlier / duplicate
tests/test_graders.py           ← task1 / task2 / task3 scoring, trap penalty, FP penalty
tests/test_environment.py       ← reset → step → submit full episode
```

Expected: all tests pass. If any fail, check [tests/conftest.py](tests/conftest.py) — the `DIRTY_RECORDS` fixture must cover all issue types.

### Manual smoke test

```bash
# 1. Start server
docker run -p 7860:7860 sqlsherlock-env:latest

# 2. Health check
curl http://localhost:7860/health
# → {"status":"ok"}

# 3. List tasks
curl http://localhost:7860/tasks
# → [{id: task1_null_and_types, ...}, ...]

# 4. Run inference (requires HF_TOKEN for model access)
export HF_TOKEN="hf_..."
python inference.py 2>results.txt
# → check stdout for [START]/[STEP]/[END] lines
# → check stderr (results.txt) for score summary
```

---

## Submission Checklist

```
[ ] docker build -t sqlsherlock-env:latest .        ← must succeed from repo root
[ ] docker run -p 7860:7860 sqlsherlock-env:latest  ← must start, port 7860
[ ] curl http://localhost:7860/health                ← must return {"status":"ok"}
[ ] python inference.py                             ← must emit [START]/[STEP]/[END]
[ ] openenv validate                                 ← must pass (openenv.yaml at root)
[ ] Dockerfile is at repo root (not inside subdir)  ← validate-submission.sh checks this
[ ] openenv.yaml is at repo root                    ← openenv validate checks this
[ ] No hardcoded secrets in any file                ← use env vars only
[ ] All env vars documented (API_BASE_URL, MODEL_NAME, HF_TOKEN, SPACE_URL)
[ ] pytest tests/ -v                               ← all tests pass
```

---

## Setup on a New Device

### Option A: Docker (recommended for deployment)

```bash
# 1. Clone
git clone <your-repo-url>
cd SQLSherlock-env

# 2. Build and run
docker build -t sqlsherlock-env:latest .
docker run -p 7860:7860 sqlsherlock-env:latest

# 3. Verify (in another terminal)
curl http://localhost:7860/health
# → {"status":"healthy"}

# 4. Run inference
export HF_TOKEN="hf_your_token_here"
export SPACE_URL="http://localhost:7860"
python inference.py
```

### Option B: Local Python (for development)

```bash
# 1. Clone
git clone <your-repo-url>
cd SQLSherlock-env

# 2. Create virtual environment (Python 3.11+ required)
python -m venv .venv

# 3. Activate venv
# Linux/Mac:
source .venv/bin/activate
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Windows CMD:
.venv\Scripts\activate.bat

# 4. Install dependencies
pip install -r sqlsherlock_env/server/requirements.txt
pip install pytest   # for tests

# 5. Start the server (Terminal 1)
cd sqlsherlock_env
# Linux/Mac:
PYTHONPATH=. uvicorn server.app:app --host 0.0.0.0 --port 7860
# Windows PowerShell:
$env:PYTHONPATH = (Get-Location).Path
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860

# 6. Run inference (Terminal 2)
cd SQLSherlock-env
# Linux/Mac:
export HF_TOKEN="hf_your_token_here"
export SPACE_URL="http://localhost:7860"
python inference.py
# Windows PowerShell:
$env:HF_TOKEN = "hf_your_token_here"
$env:SPACE_URL = "http://localhost:7860"
python inference.py

# 7. Run tests (server not needed for tests)
cd SQLSherlock-env
# Linux/Mac:
PYTHONPATH=sqlsherlock_env pytest tests/ -v
# Windows PowerShell:
$env:PYTHONPATH = "sqlsherlock_env"
python -m pytest tests/ -v
```

**Python version**: 3.11+ required. Dependencies: `fastapi`, `uvicorn`, `openai`, `datasets`, `pandas`, `pyarrow`.

---

## GRPO Training

```bash
pip install trl transformers torch

export SPACE_URL="http://localhost:7860"
export MODEL_ID="Qwen/Qwen2.5-1.5B-Instruct"
python train.py
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `API_BASE_URL` | `https://router.huggingface.co/v1` | LLM endpoint |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | Model ID |
| `HF_TOKEN` | — | HuggingFace token (dataset access + LLM) |
| `SPACE_URL` | `http://localhost:7860` | Environment server URL |

---

## Baseline Scores (phihung/titanic, 150 rows)

| Task | Difficulty | Expected Score |
|---|---|---|
| `task1_null_and_types` | Easy | 0.70 – 0.88 |
| `task2_constraints_and_fk` | Medium | 0.55 – 0.76 |
| `task3_full_audit_with_trap` | Hard | 0.40 – 0.65 |

---

## Project Structure

```
SQLSherlock-env/
├── Dockerfile                  ← repo root (required for HF Spaces)
├── README.md                   ← this file
├── openenv.yaml                ← OpenEnv + HF Spaces manifest (repo root)
├── inference.py                ← baseline agent ([START]/[STEP]/[END] format)
├── train.py                    ← TRL GRPO training loop
├── sqlsherlock_env/
│   ├── __init__.py
│   ├── client.py               ← SQLSherlockEnv WebSocket/HTTP client
│   ├── models.py               ← Action / Observation / State (Pydantic)
│   └── server/
│       ├── app.py              ← FastAPI application + WebSocket handler
│       ├── environment.py      ← RL core: reset() / step() / get_state()
│       ├── database.py         ← In-memory SQLite engine, per-episode
│       ├── dataset_loader.py   ← CSV / JSON / JSONL / Parquet / HF loader
│       ├── schema_profiler.py  ← Column statistics + z-scores
│       ├── issue_detector.py   ← Real issue detection + trap planting
│       ├── validator.py        ← 6-check before/after validator
│       ├── reward.py           ← Dense per-step reward with InvestCounter
│       ├── exporter.py         ← Format-fidelity output (CSV→CSV, etc.)
│       ├── requirements.txt
│       └── graders/
│           ├── universal.py    ← 7-step scoring pipeline
│           ├── task1.py        ← Task 1 grader
│           ├── task2.py        ← Task 2 grader
│           └── task3.py        ← Task 3 grader (trap-aware)
└── tests/
    ├── conftest.py             ← DIRTY_RECORDS fixture (all issue types)
    ├── test_issue_detector.py
    ├── test_graders.py
    └── test_environment.py
```
