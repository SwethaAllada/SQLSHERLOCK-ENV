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

Data cleaning consumes ~80% of a data scientist's time. SQLSherlock trains and evaluates AI agents to do it automatically — discover issues through statistical investigation, fix them with the right strategy per column type, validate the result, and export a clean dataset.

**Key design principles:**
- **Intent-driven cleaning** — 3 intents (Visualization / ML Training / Business Analytics), each with 3 escalating difficulty levels
- **Real dataset scanning** — environment scans the real dataset at `reset()` and builds a ground-truth issue catalogue dynamically
- **Any dataset** — HuggingFace datasets, local CSV / JSON / Parquet / XLSX, or raw CSV text
- **Dense reward signal** — every action produces a training signal, not just end-of-episode binary feedback
- **The Trap** — hard tasks plant a deliberately suspicious-looking but correct value; touching it costs −0.40

---

## Task Architecture: 3 Intents × 3 Difficulties = 9 Tasks

The user selects **one intent**; the agent automatically runs **Easy → Medium → Hard** for that intent.

| Intent | Easy | Medium | Hard |
|--------|------|--------|------|
| **Visualization** | `viz_easy` | `viz_medium` | `viz_hard` |
| **ML Training** | `ml_easy` | `ml_medium` | `ml_hard` |
| **Business Analytics** | `bq_easy` | `bq_medium` | `bq_hard` |

### Issue Coverage per Difficulty

| Difficulty | Issues Covered | Max Steps |
|------------|----------------|-----------|
| **Easy** | nulls, type errors, whitespace, inconsistent categories | 30 |
| **Medium** | + constraint violations, statistical outliers | 40 |
| **Hard** | + duplicate rows, FK violations — plus the **trap** | 50 |

---

## Episode Flow

```
User selects: dataset + intent + output format
                         │
                         ▼
          ┌──────────────────────────────┐
          │  Easy Task   (30 steps max)  │
          │  Medium Task (40 steps max)  │──► per-step reward signals
          │  Hard Task   (50 steps max)  │
          └──────────────────────────────┘
                         │
                         ▼
              Grader scores [0.0 – 1.0]
              Cleaned file exported
              User downloads result
```

### Inside Each Episode

```
reset(dataset, task_id, intent)
        │
        ▼
┌──────────────────────────────────────────┐
│  DatabaseEngine                          │
│                                          │
│  1. load(source)      CSV/JSON/Parquet   │
│  2. records_to_sqlite() in-memory SQLite │
│  3. deep_copy(originals) clean snapshot  │
│  4. profile_table()   stats per column   │
│  5. detect_issues()   8 issue types      │
│  6. Validator(baseline) 6 checks         │
│  7. detect_trap()     hard tasks only    │
└──────────────────────────────────────────┘
        │
        ▼
  Observation → Agent Step Loop
        │
  investigate → fix → validate → export
        │
        ▼
  Grader.score() → [0.0 – 1.0]
```

---

## Scoring Formulas

### Easy
```
score = resolution(null + type_error + whitespace + category) × 0.70
      + validation × 0.30
      − fp_penalty
```

### Medium
```
score = easy_score × 0.40
      + avg(constraint_resolved + outlier_resolved) × 0.60
      − fp_penalty
```

### Hard
```
# With FK violations:
score = medium_score × 0.50 + fk_resolved × 0.50 + reasoning_bonus − trap_penalty

# Single-table dataset (no FK violations to find):
score = medium_score + reasoning_bonus − trap_penalty
```

### Penalties & Bonuses
| Component | Value |
|-----------|-------|
| False-positive penalty | −0.05 per clean cell changed (capped at −0.35) |
| Skipping `validate()` | validation component × 0.50 |
| Trap hit | −0.40 |
| Reasoning bonus | +0.05 (hard tasks, statistical terms used in reasons) |

---

## Grading Pipeline (7 Steps)

```
1. Zero-change guard     — if nothing changed → 0.0
2. Resolution score      — per issue: confidence-weighted
3. False-positive penalty — −0.05 per clean cell touched (cap −0.35)
4. Trap penalty          — −0.40 if trap cell modified (hard only)
5. Validation score      — checks_passed / total × 0.30
6. Reasoning bonus       — +0.05 for statistical reasoning terms in reasons
7. Final weighted sum    — clamped to [0.0, 1.0]
```

---

## Data Cleaning Capabilities

| Issue Type | Detection | Fix Strategy | Difficulty |
|-----------|-----------|--------------|------------|
| **Null values** | `IS NULL` or empty string | Numeric → median. String → mode | All |
| **Type errors** | Text in ≥80% numeric column | Column median | All |
| **Whitespace** | Leading/trailing/extra spaces | Trimmed string | All |
| **Inconsistent categories** | Case variants ("male"/"Male"/"MALE") | Dominant form | All |
| **Constraint violations** | Negative values in must-be-positive columns | `ABS(value)` | Medium+ |
| **Statistical outliers** | IQR-based: outside Q1−3×IQR or Q3+3×IQR | Column median | Medium+ |
| **Duplicates** | Same natural key appearing twice | `delete_row` | Hard |
| **FK violations** | Orphan references across tables | `delete_row` | Hard |
| **Trap** (hard only) | Planted 2× value — z < 3 (looks normal) | **DO NOT TOUCH** (−0.40) | Hard |

**Smart imputation:** `profile_column` returns median, mode, mean, null_rate, dtype, z_scores. `fix_column` bulk-fixes ALL nulls + type errors + negatives in one step.

---

## Action Space

| `action_type` | Required fields | Description |
|---------------|----------------|-------------|
| `inspect` | `table` | View all current rows |
| `profile_column` | `table`, `column` | Stats: median, mode, mean, std, null_count, null_rate, dtype, z_scores |
| `run_sql` | `sql` | Read-only SELECT (max 50 rows) |
| `fix_cell` | `table`, `row_id`, `column`, `value`, `reason` | Fix one specific cell |
| `fix_column` | `table`, `column`, `value`, `reason` | Bulk fix: all nulls + type errors + negatives |
| `delete_row` | `table`, `row_id`, `reason` | Remove a duplicate or FK-violation row |
| `validate` | — | Run 6-check validator on current state |
| `submit` | — | Score and end episode |
| `export` | — | Write cleaned file, score, and end episode |
| `classify_intent` | `value` | Declare inferred intent (visualization / ml_training / business_query) |
| `select_tables` | `tables` | Declare active tables for multi-table analysis |
| `join_tables` | `table`, `table2`, `key` | LEFT JOIN two tables on a key column |

---

## Reward System

Dense per-step rewards — every action returns a signal:

| Action | Reward | Cap |
|--------|--------|-----|
| `inspect` | +0.02 | 3 rewarded |
| `profile_column` | +0.03 | 3 rewarded |
| `run_sql` | +0.03 | 3 rewarded |
| `select_tables` | +0.02 | 2 rewarded (only if ≥2 tables exist) |
| `validate` | +0.05 × (checks_passed / 6) | 2 rewarded |
| `fix_cell` — correct | **+0.15** | — |
| `fix_cell` — false positive | **−0.20** | — |
| `fix_cell` — trap cell | **−0.40** | — |
| `fix_column` — has issues | **+0.15 to +0.45** (scales with fraction of total issues) | — |
| `fix_column` — no issues | **−0.10** | — |
| `delete_row` — valid | **+0.15** | — |
| `delete_row` — false positive | **−0.20** | — |
| `classify_intent` — correct | **+0.10** | — |
| `classify_intent` — wrong | **−0.10** | — |
| `join_tables` — valid join | **+0.20** | — |
| `join_tables` — invalid | **−0.20** | — |
| `submit` — all resolved | **+0.10** | — |
| `submit` — issues remain | **−0.10** | — |

---

## Validation (6 Checks)

| Check | Passes when |
|-------|-------------|
| `null_check` | Null issues resolved (confidence-weighted) |
| `type_check` | Type errors castable to correct type |
| `range_check` | No negatives in must-be-positive columns |
| `distribution_check` | Column mean drift < 20% from baseline |
| `duplicate_check` | Duplicate count reduced |
| `outlier_check` | Flagged outlier rows within acceptable range of median |

---

## Quick Start

### Docker (recommended)

```bash
docker build -t sqlsherlock-env:latest .
docker run -p 7860:7860 -e HF_TOKEN=hf_your_token sqlsherlock-env:latest
curl http://localhost:7860/health
```

### Local Python

```bash
pip install -r sqlsherlock_env/server/requirements.txt
cd sqlsherlock_env
PYTHONPATH=. uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Run Baseline Inference

```bash
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4.1-mini"
export HF_TOKEN="hf_your_token"
export SPACE_URL="http://localhost:7860"
export DEMO_INTENT="visualization"   # or ml_training / business_query
python inference.py
```

Inference output format (judges parse this exactly):

```
=== SQLSherlock-Env  intent=visualization  model=gpt-4.1-mini ===

[START] task=viz_easy env=sqlsherlock_env model=gpt-4.1-mini
[STEP]  step=1 action=inspect reward=0.02 done=false error=null
[STEP]  step=2 action=profile_column(Age) reward=0.03 done=false error=null
[STEP]  step=7 action=fix_column(Age,29.5) reward=0.22 done=false error=null
...
[END]   success=true steps=18 rewards=0.02,0.03,...

[START] task=viz_medium env=sqlsherlock_env model=gpt-4.1-mini
...
[END]   success=true steps=28 rewards=...

[START] task=viz_hard env=sqlsherlock_env model=gpt-4.1-mini
...
[END]   success=true steps=35 rewards=...
```

---

## Using Any Dataset

```python
from sqlsherlock_env.client import SQLSherlockEnv

env = SQLSherlockEnv(base_url="http://localhost:7860")

# HuggingFace dataset
obs = env.reset(dataset="phihung/titanic", task_id="viz_easy", intent="visualization")

# Local file (CSV, JSON, Parquet, XLSX)
obs = env.reset(dataset="/path/to/data.csv", task_id="ml_medium", intent="ml_training")

# Upload via API then use
# POST /upload_dataset  →  returns {"dataset_id": "..."}
# obs = env.reset(dataset="upload://dataset_id", task_id="bq_hard")
```

---

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| `WS` | `/ws` | Persistent WebSocket session |
| `POST` | `/reset` | Start episode (`dataset`, `task_id`, `intent`, `output_format`) |
| `POST` | `/step` | Execute one action |
| `GET` | `/state` | Current episode state |
| `GET` | `/health` | Health check + task list |
| `GET` | `/tasks` | All 9 task definitions |
| `POST` | `/upload_dataset` | Upload CSV / JSON / Parquet / XLSX |
| `GET` | `/download/{file_id}` | Download cleaned output |
| `GET` | `/docs` | OpenAPI Swagger UI |

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | *(required)* | HuggingFace / OpenAI API key |
| `API_BASE_URL` | `https://api.openai.com/v1` | LLM API endpoint |
| `MODEL_NAME` | `gpt-4.1-mini` | Model identifier |
| `SPACE_URL` | `http://localhost:7860` | Environment server URL |
| `DEMO_INTENT` | `visualization` | Intent for inference.py (`visualization` / `ml_training` / `business_query`) |

---

## Project Structure

```
SQLSherlock-env/
├── Dockerfile                     ← HF Spaces Docker entrypoint
├── README.md                      ← this file
├── inference.py                   ← hackathon baseline ([START]/[STEP]/[END])
├── pyproject.toml
├── sqlsherlock_env/
│   ├── client.py                  ← sync WebSocket/HTTP client
│   ├── models.py                  ← Action / Observation / State (Pydantic)
│   └── server/
│       ├── app.py                 ← FastAPI + Gradio mount
│       ├── environment.py         ← RL core: reset() / step() — 9 tasks
│       ├── ui.py                  ← Gradio UI (intent selector, live log, download)
│       ├── database.py            ← In-memory SQLite engine per episode
│       ├── dataset_loader.py      ← CSV / JSON / JSONL / Parquet / XLSX / HF
│       ├── schema_profiler.py     ← Column stats: median, mode, std, IQR
│       ├── issue_detector.py      ← 8 issue types + trap planting
│       ├── validator.py           ← 6-check before/after validator
│       ├── reward.py              ← Dense per-step reward calculator
│       ├── exporter.py            ← Format-preserving output writer
│       ├── requirements.txt
│       └── graders/
│           ├── __init__.py        ← Routes 9 task IDs to 3 graders
│           ├── universal.py       ← 7-step scoring pipeline (shared)
│           ├── task1.py           ← Easy grader
│           ├── task2.py           ← Medium grader
│           └── task3.py           ← Hard grader (trap + FK + reasoning)
└── tests/
    └── test_environment.py        ← 58 tests
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

## Run Tests

```bash
PYTHONPATH=sqlsherlock_env pytest tests/test_environment.py -q
# 58 tests, all pass
```
