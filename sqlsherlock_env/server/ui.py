"""
SQLSherlock-Env dashboard UI.

User flow:
  1. Provide a dataset (file upload or HuggingFace name).
  2. Select ONE cleaning intent (visualization / ML training / business analytics).
  3. Select output format (CSV / JSON / Parquet).
  4. The agent runs 3 escalating difficulty tasks for that intent:
       Easy → Medium → Hard
  5. Live step log streams as each task runs.
  6. Download the cleaned file from the Hard task when complete.
"""

import os
import re
import tempfile
from typing import Generator

import gradio as gr


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Each intent maps to its 3 difficulty task IDs (easy → medium → hard)
INTENT_TASK_MAP: dict[str, list[str]] = {
    "visualization":  ["viz_easy",  "viz_medium",  "viz_hard"],
    "ml_training":    ["ml_easy",   "ml_medium",   "ml_hard"],
    "business_query": ["bq_easy",   "bq_medium",   "bq_hard"],
}

INTENT_LABELS: dict[str, str] = {
    "visualization":  "Visualization — clean for dashboards & charts",
    "ml_training":    "ML Training — clean for model training pipelines",
    "business_query": "Business Analytics — clean for SQL reporting & queries",
}

DIFFICULTY_LABELS: dict[str, str] = {
    "easy":   "Easy   (30 steps) — nulls, type errors, whitespace, categories",
    "medium": "Medium (40 steps) — + constraints, outliers",
    "hard":   "Hard   (50 steps) — + duplicates, FK violations, trap",
}

INTENT_CHOICES = [
    ("Visualization — dashboards & charts",          "visualization"),
    ("ML Training — model training pipelines",       "ml_training"),
    ("Business Analytics — SQL reporting & queries", "business_query"),
]

FORMAT_CHOICES = ["csv", "json", "parquet"]

_EXAMPLE_HF_DATASETS = [
    ["phihung/titanic"],
    ["scikit-learn/iris"],
]

_AGENT_EXAMPLES = [
    ["phihung/titanic", "visualization",  "csv"],
    ["phihung/titanic", "ml_training",    "csv"],
    ["phihung/titanic", "business_query", "csv"],
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_source(file_path: str | None, hf_name: str) -> tuple[str | None, str | None]:
    """Return (source, error_message). File upload takes precedence over HF name."""
    if file_path:
        return file_path, None
    name = (hf_name or "").strip()
    if name:
        return name, None
    return None, (
        "**No dataset provided.**  "
        "Upload a file (Option A) or enter a HuggingFace dataset name (Option B)."
    )


def _friendly_error(exc: Exception, source: str) -> str:
    msg = str(exc)
    if "404" in msg or "not found" in msg.lower() or "doesn't exist" in msg.lower():
        return f"Dataset not found: `{source}`. Check the name and try again."
    if "parse" in msg.lower() or "decode" in msg.lower() or "invalid" in msg.lower():
        return "Could not parse the file. Supported: CSV, JSON, JSONL, Parquet, XLSX."
    if "connection" in msg.lower() or "refused" in msg.lower():
        return "Cannot reach the server. Make sure the environment is running."
    return f"Failed to load dataset ({type(exc).__name__}). Check the name and try again."


# ---------------------------------------------------------------------------
# Tab 1: Data Quality Scanner
# ---------------------------------------------------------------------------

def scan_dataset(file_path: str | None, hf_name: str) -> tuple[str, list, list, list, list]:
    from server.database import DatabaseEngine

    source, err = _resolve_source(file_path, hf_name)
    if err:
        return err, [], [], [], []

    try:
        db = DatabaseEngine(task_id="bq_hard", seed=42, dataset_source=source, max_rows=500)
    except Exception as exc:
        return f"**Error:** {_friendly_error(exc, source)}", [], [], [], []

    table        = db.primary_table
    issues       = db.issue_registry
    profiles     = db._profiles.get(table, {})
    total_rows   = len(db.rows(table))
    total_cols   = sum(1 for c in profiles if c not in ("id", "_source_format"))
    total_issues = len(issues)
    cells        = max(total_rows * total_cols, 1)
    quality_pct  = max(0.0, min(1.0, 1.0 - total_issues / cells)) * 100
    grade        = ("Excellent" if quality_pct >= 90 else
                    "Good"      if quality_pct >= 70 else
                    "Fair"      if quality_pct >= 50 else "Poor")

    type_counts: dict[str, int] = {}
    for iss in issues:
        type_counts[iss.issue_type] = type_counts.get(iss.issue_type, 0) + 1

    breakdown_lines = "\n".join(
        f"| `{t}` | {c} |"
        for t, c in sorted(type_counts.items(), key=lambda x: -x[1])
    )
    summary = f"""## Data Quality Report

| Metric | Value |
|--------|-------|
| **Dataset** | `{source}` |
| **Table** | `{table}` |
| **Rows** | {total_rows:,} |
| **Columns** | {total_cols} |
| **Issues found** | {total_issues} |
| **Quality score** | **{quality_pct:.1f}% — {grade}** |

### Issue Breakdown

| Issue Type | Count |
|------------|-------|
{breakdown_lines}
"""

    col_rows = []
    for col, p in profiles.items():
        if col in ("id", "_source_format"):
            continue
        null_rate  = p.get("null_rate", 0.0)
        col_issues = sum(1 for i in issues if i.column == col)
        health = ("Good" if null_rate < 0.05 and col_issues == 0 else
                  "Fair" if null_rate < 0.30 and col_issues <= 2 else "Poor")
        col_rows.append([col, p.get("dtype", "?"), f"{null_rate*100:.1f}%", col_issues, health])

    issue_rows   = [[t, c] for t, c in sorted(type_counts.items(), key=lambda x: -x[1])]
    sample_rows  = [
        [iss.issue_type, iss.column or "-", str(iss.row_id),
         f"{iss.confidence:.0%}",
         "delete row" if iss.correct is None else str(iss.correct)[:40]]
        for iss in issues[:15]
    ]
    rows         = db.rows(table)[:10]
    preview_cols = [c for c in (rows[0].keys() if rows else []) if c != "_source_format"]
    preview_rows = [[row.get(c, "") for c in preview_cols] for row in rows]

    return summary, col_rows, issue_rows, sample_rows, preview_rows


# ---------------------------------------------------------------------------
# Tab 2: Agent Demo — one intent → 3 difficulty tasks streamed sequentially
# ---------------------------------------------------------------------------

def run_agent_streaming(
    file_path: str | None,
    hf_name: str,
    intent: str,
    output_format: str,
) -> Generator[tuple[str, str, str | None], None, None]:
    """
    Yields (log_text, summary_md, download_path) after every action.
    Runs 3 escalating tasks (easy → medium → hard) for the selected intent.
    """
    from server.environment import SQLSherlockEnvironment
    from models import SQLSherlockAction

    source, err = _resolve_source(file_path, hf_name)
    if err:
        yield err, "", None
        return

    task_ids     = INTENT_TASK_MAP.get(intent, INTENT_TASK_MAP["visualization"])
    intent_label = INTENT_LABELS.get(intent, intent)

    log_lines: list[str] = []
    all_scores: list[tuple[str, float]] = []   # [(task_id, score)]
    download_path: str | None = None
    summary_md = "_Running…_"

    def _log(*parts: str) -> None:
        log_lines.append(" ".join(parts))

    def _current_log() -> str:
        return "\n".join(log_lines)

    _log(f"Intent : {intent_label}")
    _log(f"Format : {output_format}")
    _log(f"Dataset: {source}")
    _log("")
    yield _current_log(), summary_md, None

    for task_id in task_ids:
        difficulty = task_id.split("_")[-1].upper()
        _log(f"{'─'*60}")
        _log(f"[{difficulty}] {task_id}")
        _log(f"{'─'*60}")
        yield _current_log(), summary_md, None

        env   = SQLSherlockEnvironment()
        score = 0.0
        step  = 0
        done  = False
        obs   = None
        rewards: list[float] = []
        fixed_cols: list[str] = []

        def _do(action_dict: dict) -> float:
            nonlocal step, done, obs
            if done:
                return 0.0
            step += 1
            action = SQLSherlockAction(
                **{k: v for k, v in action_dict.items() if v is not None}
            )
            reward = 0.0
            try:
                obs    = env.step(action)
                reward = float(getattr(obs, "reward", 0) or 0.0)
                done   = bool(getattr(obs, "done", False))
            except Exception:
                pass
            rewards.append(reward)
            return reward

        try:
            obs = env.reset(
                dataset=source,
                task_id=task_id,
                intent=intent,
                output_format=output_format,
            )
            table   = list(obs.tables_summary.keys())[0]
            summary = obs.tables_summary[table]
            visible = [c for c in summary.get("columns", [])
                       if c not in ("id", "_source_format")]

            _log(f"  Table: {table}  ({summary.get('row_count','?')} rows, {len(visible)} cols)")
            yield _current_log(), summary_md, None

            # Phase 1: Inspect
            r = _do({"action_type": "inspect", "table": table})
            _log(f"  [step {step:2d}] inspect         reward={r:+.2f}")
            yield _current_log(), summary_md, None

            # Phase 2: Profile + bulk-fix
            for col in visible[:12]:
                if done:
                    break
                r = _do({"action_type": "profile_column", "table": table, "column": col})
                if obs and obs.query_result:
                    p  = obs.query_result[0]
                    nc = p.get("null_count", 0)
                    dt = p.get("dtype", "?")
                    _log(f"  [step {step:2d}] profile({col:<12}) type={dt}  nulls={nc}  reward={r:+.2f}")
                    yield _current_log(), summary_md, None

                    if nc > 0 and not done:
                        fill     = (p.get("median") or p.get("mean") or 0) if dt in ("int", "float") else (p.get("mode") or "Unknown")
                        strategy = "median" if dt in ("int", "float") else "mode"
                        r2 = _do({
                            "action_type": "fix_column", "table": table,
                            "column": col, "value": fill,
                            "reason": f"bulk fix {nc} nulls, {strategy}={fill}",
                        })
                        _log(f"  [step {step:2d}] fix_column({col:<12}, {str(fill)[:8]!r:<10})  reward={r2:+.2f}")
                        if r2 > 0:
                            fixed_cols.append(col)
                        yield _current_log(), summary_md, None
                else:
                    _log(f"  [step {step:2d}] profile({col:<12})  reward={r:+.2f}")
                    yield _current_log(), summary_md, None

            # Phase 3: Validate
            if not done:
                r  = _do({"action_type": "validate"})
                fb = (obs.last_feedback if obs else "")[:100]
                _log(f"  [step {step:2d}] validate         reward={r:+.2f}  {fb}")
                yield _current_log(), summary_md, None

            # Phase 4: Export (hard tasks produce a downloadable file)
            if not done:
                r  = _do({"action_type": "export"})
                fb = (obs.last_feedback if obs else "")[:120]
                _log(f"  [step {step:2d}] export           reward={r:+.2f}")
                _log(f"         {fb}")
                m = re.search(r"[Gg]rader\s+score\s*=?\s*(\d+\.\d+)", fb)
                if m:
                    score = float(m.group(1))
                if task_id.endswith("_hard") and hasattr(env, "_export_result") and env._export_result:
                    download_path = env._export_result.get("filepath")
                yield _current_log(), summary_md, download_path

        except Exception as exc:
            _log(f"  [ERROR] {_friendly_error(exc, source)}")
        finally:
            try:
                env.close()
            except Exception:
                pass

        total_r = sum(rewards)
        success = score >= 0.50
        _log(f"  Score: {score:.2f}  Total reward: {total_r:+.3f}  "
             f"{'SUCCESS' if success else 'PARTIAL'}  Steps: {step}")
        _log("")
        all_scores.append((task_id, score))
        yield _current_log(), summary_md, download_path

    # Final summary table
    avg = sum(s for _, s in all_scores) / len(all_scores) if all_scores else 0.0
    rows_md = "\n".join(
        f"| `{tid}` | {s:.2f} | {'Pass' if s >= 0.50 else 'Fail'} |"
        for tid, s in all_scores
    )
    summary_md = f"""## Results — Intent: `{intent}`

| Task | Score | Result |
|------|-------|--------|
{rows_md}

**Average score: {avg:.2f}**

| Detail | Value |
|--------|-------|
| Dataset | `{source}` |
| Output format | `{output_format}` |
| Tasks run | {len(all_scores)} (easy → medium → hard) |
{"| **Download** | cleaned file available below |" if download_path else ""}
"""
    yield _current_log(), summary_md, download_path


# ---------------------------------------------------------------------------
# Build the Gradio Blocks app
# ---------------------------------------------------------------------------

def create_ui() -> gr.Blocks:
    with gr.Blocks(title="SQLSherlock — AI Data Quality Detective") as demo:

        gr.Markdown(
            "# SQLSherlock — AI Data Quality Detective\n"
            "Select your **cleaning intent** and let the agent clean your dataset "
            "across **3 escalating difficulty levels** — Easy, Medium, then Hard."
        )

        # ==================================================================
        # Tab 1: Data Quality Scanner
        # ==================================================================
        with gr.Tab("Data Quality Scanner"):
            gr.Markdown(
                "Provide a dataset, then click **Scan** to see a full quality report."
                "\n\n> If both file and name are filled, the uploaded file takes priority."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### Option A — Upload a file")
                    scan_file = gr.File(
                        label="CSV / JSON / JSONL / Parquet / Excel",
                        file_types=[".csv", ".json", ".jsonl", ".parquet", ".xlsx"],
                        type="filepath",
                    )
                    gr.Markdown("#### Option B — HuggingFace dataset")
                    scan_hf = gr.Textbox(
                        label="Dataset name",
                        placeholder="e.g. phihung/titanic",
                        value="phihung/titanic",
                    )
                    gr.Examples(examples=_EXAMPLE_HF_DATASETS, inputs=[scan_hf],
                                label="Quick examples")
                    scan_btn = gr.Button("Scan Dataset", variant="primary", size="lg")

                with gr.Column(scale=2):
                    summary_out = gr.Markdown(value="_Scan results will appear here._")

            gr.Markdown("---")
            col_health = gr.Dataframe(
                headers=["Column", "Type", "Null Rate", "Issues", "Health"],
                label="Column Health", interactive=False,
            )
            with gr.Row():
                issue_tbl = gr.Dataframe(
                    headers=["Issue Type", "Count"],
                    label="Issue Breakdown", interactive=False,
                )
                sample_tbl = gr.Dataframe(
                    headers=["Type", "Column", "Row", "Confidence", "Fix"],
                    label="Sample Issues (Top 15)", interactive=False,
                )
            with gr.Accordion("Data Preview (first 10 rows)", open=False):
                preview_tbl = gr.Dataframe(label="Raw Data", interactive=False)

            scan_btn.click(
                fn=scan_dataset,
                inputs=[scan_file, scan_hf],
                outputs=[summary_out, col_health, issue_tbl, sample_tbl, preview_tbl],
                show_progress="minimal",
            )

        # ==================================================================
        # Tab 2: Agent Demo — intent → 3 difficulty levels
        # ==================================================================
        with gr.Tab("Agent Demo"):
            gr.Markdown(
                "**Choose one cleaning intent** — the agent will automatically run "
                "**Easy → Medium → Hard** tasks for that intent.\n\n"
                "The cleaned file from the Hard task is available for download."
            )

            with gr.Row():
                # --- Dataset input ---
                with gr.Column(scale=2):
                    gr.Markdown("#### Dataset")
                    agent_file = gr.File(
                        label="Upload file (CSV / JSON / JSONL / Parquet / XLSX)",
                        file_types=[".csv", ".json", ".jsonl", ".parquet", ".xlsx"],
                        type="filepath",
                    )
                    agent_hf = gr.Textbox(
                        label="OR enter HuggingFace dataset name",
                        value="phihung/titanic",
                        placeholder="phihung/titanic",
                    )

                # --- Configuration ---
                with gr.Column(scale=1):
                    gr.Markdown("#### Cleaning Intent")
                    agent_intent = gr.Radio(
                        choices=INTENT_CHOICES,
                        label="What will you use this data for?",
                        value="visualization",
                    )
                    gr.Markdown("#### Output Format")
                    agent_format = gr.Radio(
                        choices=FORMAT_CHOICES,
                        label="Cleaned file format",
                        value="csv",
                    )

            gr.Markdown(
                "> The agent will run 3 tasks for the selected intent:\n"
                "> - **Easy** — basic cleaning (nulls, type errors, whitespace, categories)\n"
                "> - **Medium** — + constraints, outliers\n"
                "> - **Hard** — + duplicates, FK violations, trap detection"
            )

            gr.Examples(
                examples=_AGENT_EXAMPLES,
                inputs=[agent_hf, agent_intent, agent_format],
                label="Quick examples",
            )

            run_btn = gr.Button("Run Agent (3 Tasks)", variant="primary", size="lg")

            with gr.Row():
                with gr.Column(scale=3):
                    agent_log = gr.Textbox(
                        label="Live Step Log",
                        lines=30, max_lines=120,
                        interactive=False, show_copy_button=True,
                    )
                with gr.Column(scale=2):
                    agent_results = gr.Markdown(
                        value="_Results appear here after the agent finishes._"
                    )

            gr.Markdown("#### Download Cleaned File (Hard task output)")
            download_out = gr.File(
                label="Cleaned dataset — available after the Hard task completes",
                interactive=False,
            )

            def _run(file_path, hf_name, intent, output_format):
                for log, results, dl in run_agent_streaming(
                    file_path, hf_name, intent, output_format
                ):
                    yield log, results, dl

            run_btn.click(
                fn=_run,
                inputs=[agent_file, agent_hf, agent_intent, agent_format],
                outputs=[agent_log, agent_results, download_out],
                show_progress="minimal",
            )

        # ==================================================================
        # Tab 3: About
        # ==================================================================
        with gr.Tab("About"):
            gr.Markdown("""
## SQLSherlock Task Architecture

Each **cleaning intent** has **3 difficulty levels** — the agent always
runs Easy → Medium → Hard for the chosen intent.

### Intents & Their 3 Tasks

| Intent | Easy | Medium | Hard |
|--------|------|--------|------|
| **Visualization** | `viz_easy` | `viz_medium` | `viz_hard` |
| **ML Training** | `ml_easy` | `ml_medium` | `ml_hard` |
| **Business Analytics** | `bq_easy` | `bq_medium` | `bq_hard` |

### Issue Coverage per Difficulty

| Difficulty | Issues Covered | Max Steps |
|------------|----------------|-----------|
| **Easy** | nulls, type errors, whitespace, inconsistent categories | 30 |
| **Medium** | + constraint violations, statistical outliers (z > 5) | 40 |
| **Hard** | + duplicate rows, FK violations — plus the **trap** | 50 |

### Scoring Formulas

**Easy:**
```
score = resolution(null+type+whitespace+category) × 0.70 + validation × 0.30
```

**Medium:**
```
score = easy_score × 0.40 + avg(constraint + outlier) × 0.60
```

**Hard:**
```
score = medium_score × 0.50 + fk_resolved × 0.50 + reasoning_bonus − trap_penalty
```

### The Trap (Hard tasks only)

A numeric value is doubled — it *looks* like an outlier but z < 3.
Touching it costs **−0.40**. Always verify z-scores before fixing numeric values.

---

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| WebSocket | `/ws` | RL episode session |
| POST | `/reset` | Start episode (`task_id`, `intent`, `output_format`) |
| POST | `/step` | Execute one action |
| GET | `/health` | Health check + task list |
| GET | `/tasks` | All 9 task definitions |
| POST | `/upload_dataset` | Upload CSV/JSON/Parquet/XLSX |
| GET | `/download/{id}` | Download cleaned output |
""")

    return demo
