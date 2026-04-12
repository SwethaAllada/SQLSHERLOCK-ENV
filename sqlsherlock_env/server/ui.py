"""
SQLSherlock-Env dashboard UI.

User flow:
  1. Provide dataset(s) — upload up to 3 files OR enter a HuggingFace name.
  2. Select ONE cleaning intent.
  3. Describe your specific requirement (dashboard goal / model goal / business question).
  4. (Optional) enter a SQL query for direct business analytics.
  5. The agent runs 3 escalating tasks: Easy → Medium → Hard.
  6. Live step log, before/after comparison, and a final answer are shown.
  7. Download the cleaned file from the Hard task.
"""

import re
import tempfile
from pathlib import Path
from typing import Any, Generator

import gradio as gr
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

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

INTENT_CHOICES = [
    ("Visualization — dashboards & charts",          "visualization"),
    ("ML Training — model training pipelines",       "ml_training"),
    ("Business Analytics — SQL reporting & queries", "business_query"),
]

FORMAT_CHOICES = ["csv", "json", "parquet"]

_EXAMPLE_HF = [["phihung/titanic"], ["scikit-learn/iris"]]

_AGENT_EXAMPLES = [
    ["phihung/titanic", "visualization",
     "Build a dashboard showing survival rates by passenger class, gender and age group.", "csv", ""],
    ["phihung/titanic", "ml_training",
     "Train a survival prediction model using Age, Fare and Pclass as features.", "csv", ""],
    ["phihung/titanic", "business_query",
     "What is the average fare and survival rate for each passenger class?", "csv",
     'SELECT "Pclass", COUNT(*) AS passengers, ROUND(AVG("Fare"),2) AS avg_fare, ROUND(AVG("Survived"),2) AS survival_rate FROM dataset GROUP BY "Pclass" ORDER BY "Pclass"'],
]

# Requirement placeholders per intent
_REQ_CONFIG: dict[str, dict] = {
    "visualization": {
        "label": "Describe your dashboard (what do you want to visualise?)",
        "placeholder": (
            "e.g. I want to build a dashboard showing passenger survival rates "
            "by class, gender and age group. Key metrics: survival %, average fare, "
            "passenger count per category."
        ),
    },
    "ml_training": {
        "label": "Describe your model (goal, features, target column)",
        "placeholder": (
            "e.g. Building a binary classification model to predict passenger survival. "
            "Features: Age, Fare, Pclass, Sex. Target: Survived. "
            "Need clean numeric features with no nulls or extreme outliers."
        ),
    },
    "business_query": {
        "label": "What business question do you want answered?",
        "placeholder": (
            "e.g. What is the average ticket fare for each passenger class? "
            "Which class had the highest survival rate? "
            "How many passengers embarked from each port?"
        ),
    },
}

_SQL_PLACEHOLDER = (
    'SELECT "Pclass", COUNT(*) AS passengers,\n'
    '       ROUND(AVG("Fare"), 2) AS avg_fare,\n'
    '       ROUND(AVG("Survived"), 2) AS survival_rate\n'
    'FROM dataset\n'
    'GROUP BY "Pclass"\n'
    'ORDER BY "Pclass"'
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_source(file1: str | None, hf_name: str) -> tuple[str | None, str | None]:
    if file1:
        return file1, None
    name = (hf_name or "").strip()
    if name:
        return name, None
    return None, "**No dataset provided.** Upload a file or enter a HuggingFace dataset name."


def _friendly_error(exc: Exception, source: str) -> str:
    msg = str(exc)
    if "404" in msg or "not found" in msg.lower() or "doesn't exist" in msg.lower():
        return f"Dataset not found: `{source}`. Check the name and try again."
    if "parse" in msg.lower() or "decode" in msg.lower() or "invalid" in msg.lower():
        return "Could not parse the file. Supported: CSV, JSON, JSONL, Parquet, XLSX."
    if "connection" in msg.lower() or "refused" in msg.lower():
        return "Cannot reach the server. Make sure the environment is running."
    return f"Error ({type(exc).__name__}): {str(exc)[:150]}"


def _read_file(path: str) -> pd.DataFrame:
    ext = Path(path).suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in (".json", ".jsonl"):
        try:
            return pd.read_json(path)
        except Exception:
            return pd.read_json(path, lines=True)
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext == ".xlsx":
        return pd.read_excel(path, sheet_name=0)
    return pd.read_csv(path)


def _merge_to_xlsx(files: list[str]) -> str:
    """Merge 2–3 uploaded files into one multi-sheet XLSX.
    Each file → one sheet; sheet name = filename stem (≤31 chars).
    Returns path to the temp file.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
    tmp.close()
    with pd.ExcelWriter(tmp.name, engine="openpyxl") as writer:
        for i, fpath in enumerate(files):
            df = _read_file(fpath)
            sheet = (Path(fpath).stem[:31] or f"table{i+1}")
            df.to_excel(writer, sheet_name=sheet, index=False)
    return tmp.name


def _rows_to_df(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    exclude = {"_source_format"}
    cols = [c for c in rows[0].keys() if c not in exclude]
    return pd.DataFrame([{c: r.get(c, "") for c in cols} for r in rows])


def _df_to_md(df: pd.DataFrame, max_rows: int = 25) -> str:
    """Convert DataFrame to a markdown table (no external dependencies)."""
    if df.empty:
        return "_No results_"
    display = df.head(max_rows)
    cols = list(display.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep    = "| " + " | ".join("---" for _ in cols) + " |"
    body   = [
        "| " + " | ".join(str(display.iloc[i][c]) for c in cols) + " |"
        for i in range(len(display))
    ]
    tail = f"\n_{len(df) - max_rows} more rows…_" if len(df) > max_rows else ""
    return "\n".join([header, sep] + body) + tail


def _schema_md(tables_summary: dict) -> str:
    if not tables_summary or len(tables_summary) <= 1:
        return ""
    lines = ["### Loaded Tables\n"]
    for tname, info in tables_summary.items():
        cols   = [c for c in info.get("columns", []) if c != "_source_format"]
        dtypes = info.get("dtypes", {})
        n_rows = info.get("row_count", "?")
        lines.append(f"**`{tname}`** — {n_rows} rows, {len(cols)} columns\n")
        lines.append("| Column | Type |")
        lines.append("|--------|------|")
        for c in cols:
            lines.append(f"| `{c}` | {dtypes.get(c, '?')} |")
        lines.append("")
    return "\n".join(lines)


def _nl_to_sql(question: str, table: str, columns: list[str]) -> str | None:
    """Convert a natural-language business question to a best-effort SQL query.

    Returns None if the question is already SQL or no pattern is detected.
    """
    q = question.strip()
    if not q:
        return None
    if q.upper().startswith("SELECT"):
        return q  # already SQL

    q_low = q.lower()

    # Identify likely numeric and categorical columns
    num_cols = [c for c in columns if c not in ("id", "_source_format")
                and any(w in c.lower() for w in
                        ["fare", "age", "price", "amount", "rate", "score", "salary",
                         "revenue", "cost", "value", "petal", "sepal", "weight",
                         "height", "income", "fee", "count", "survived"])]
    cat_cols = [c for c in columns if c not in ("id", "_source_format")
                and any(w in c.lower() for w in
                        ["class", "sex", "gender", "type", "category", "status",
                         "species", "department", "region", "country", "city",
                         "embarked", "cabin", "name", "group", "tier"])]

    agg_words  = ["average", "avg", "mean", "total", "sum", "rate",
                  "how many", "count", "number of", "maximum", "minimum",
                  "highest", "lowest", "most", "least"]
    grp_words  = [" by ", " per ", " each ", " for each ", " across ", " group "]
    has_agg    = any(w in q_low for w in agg_words)
    has_group  = any(w in q_low for w in grp_words)

    # Detect which column names are mentioned in the question
    mentioned_num = [c for c in num_cols if c.lower() in q_low]
    mentioned_cat = [c for c in cat_cols if c.lower() in q_low]

    # Pattern: "average X by Y" / "count per Y"
    if has_group:
        grp_col = mentioned_cat[0] if mentioned_cat else (cat_cols[0] if cat_cols else None)
        agg_col = mentioned_num[0] if mentioned_num else (num_cols[0] if num_cols else None)
        if grp_col:
            if agg_col and has_agg:
                # Multiple aggregations for richness
                return (
                    f'SELECT "{grp_col}", COUNT(*) AS count, '
                    f'ROUND(AVG("{agg_col}"), 2) AS avg_{agg_col.lower()}, '
                    f'ROUND(MIN("{agg_col}"), 2) AS min_{agg_col.lower()}, '
                    f'ROUND(MAX("{agg_col}"), 2) AS max_{agg_col.lower()} '
                    f'FROM "{table}" GROUP BY "{grp_col}" ORDER BY count DESC LIMIT 20'
                )
            else:
                return (
                    f'SELECT "{grp_col}", COUNT(*) AS count '
                    f'FROM "{table}" GROUP BY "{grp_col}" ORDER BY count DESC LIMIT 20'
                )

    # Pattern: "top N …" or "highest / most"
    top_m = re.search(r'\btop\s+(\d+)\b', q_low)
    if top_m:
        n = top_m.group(1)
        sort_col = mentioned_num[0] if mentioned_num else (num_cols[0] if num_cols else "id")
        return f'SELECT * FROM "{table}" ORDER BY "{sort_col}" DESC LIMIT {n}'

    if any(w in q_low for w in ["highest", "most", "maximum", "largest"]):
        sort_col = mentioned_num[0] if mentioned_num else (num_cols[0] if num_cols else "id")
        return f'SELECT * FROM "{table}" ORDER BY "{sort_col}" DESC LIMIT 10'

    if any(w in q_low for w in ["lowest", "least", "minimum", "smallest"]):
        sort_col = mentioned_num[0] if mentioned_num else (num_cols[0] if num_cols else "id")
        return f'SELECT * FROM "{table}" ORDER BY "{sort_col}" ASC LIMIT 10'

    # Fallback: overall summary of numeric columns
    if cat_cols:
        c = cat_cols[0]
        return (
            f'SELECT "{c}", COUNT(*) AS count '
            f'FROM "{table}" GROUP BY "{c}" ORDER BY count DESC LIMIT 20'
        )

    return None


def _generate_final_answer(
    intent: str,
    user_requirement: str,
    issues_total: int,
    issues_remaining: int,
    columns_fixed: list[str],
    query_df: pd.DataFrame,
    auto_sql: str | None,
    table: str,
    columns: list[str],
) -> str:
    """Build the Final Answer markdown block shown at the end of all tasks."""

    resolved = max(0, issues_total - issues_remaining)
    fixed_cols_unique = sorted(set(columns_fixed))

    # --- Cleaning summary ---
    if issues_total == 0:
        clean_summary = "Dataset had no detectable issues — already clean."
    elif resolved == issues_total:
        clean_summary = (
            f"**All {issues_total} data quality issues resolved** "
            + (f"across columns: `{'`, `'.join(fixed_cols_unique)}`." if fixed_cols_unique else ".")
        )
    else:
        pct = round(100 * resolved / max(issues_total, 1))
        clean_summary = (
            f"**{resolved}/{issues_total} issues resolved ({pct}%)** "
            + (f"across columns: `{'`, `'.join(fixed_cols_unique)}`." if fixed_cols_unique else ".")
            + (f" {issues_remaining} issue(s) remain." if issues_remaining > 0 else "")
        )

    md = ["## Final Answer\n"]
    md.append(f"### Cleaning Summary\n{clean_summary}\n")

    safe_cols = [c for c in columns if c not in ("id", "_source_format")]

    # --- Intent-specific section ---
    if intent == "visualization":
        req = user_requirement.strip() if user_requirement else "No description provided."
        md.append(f"### Dashboard Requirement\n_{req}_\n")
        num_cols = [c for c in safe_cols
                    if any(w in c.lower() for w in
                           ["fare", "age", "price", "rate", "score", "count", "survived",
                            "amount", "value", "petal", "sepal"])]
        cat_cols = [c for c in safe_cols if c not in num_cols]
        if num_cols:
            md.append(f"**Recommended metric columns:** `{'`, `'.join(num_cols[:6])}`")
        if cat_cols:
            md.append(f"\n**Recommended dimension columns:** `{'`, `'.join(cat_cols[:6])}`")
        md.append(
            "\n\nThe dataset is now ready for your dashboard — "
            "nulls filled, types corrected, categories normalised."
        )

    elif intent == "ml_training":
        req = user_requirement.strip() if user_requirement else "No description provided."
        md.append(f"### Model Requirement\n_{req}_\n")
        feature_candidates = [c for c in safe_cols
                               if c.lower() not in ("name", "cabin", "ticket", "survived")]
        target_candidates  = [c for c in safe_cols if "survived" in c.lower() or "target" in c.lower()]
        if feature_candidates:
            md.append(f"**Suggested feature columns:** `{'`, `'.join(feature_candidates[:8])}`")
        if target_candidates:
            md.append(f"\n**Likely target column:** `{target_candidates[0]}`")
        md.append(
            "\n\nThe dataset is now ready for ML training — "
            "nulls imputed with column medians/modes, outliers corrected, "
            "negative constraint violations fixed."
        )

    elif intent == "business_query":
        req = user_requirement.strip() if user_requirement else "No question provided."
        md.append(f"### Business Question\n_{req}_\n")

        if not query_df.empty:
            md.append(f"### Answer\n")
            md.append(_df_to_md(query_df))
            md.append("\n")
            # Generate a brief narrative from the first row
            try:
                cols_list = list(query_df.columns)
                if len(cols_list) >= 2 and len(query_df) > 0:
                    top = query_df.iloc[0]
                    insights = []
                    for c in cols_list[1:4]:
                        val = top[c]
                        if isinstance(val, float):
                            val = round(val, 2)
                        insights.append(f"{c} = **{val}**")
                    md.append(
                        f"\n**Top result** — {cols_list[0]} = **{top[cols_list[0]]}**: "
                        + ", ".join(insights) + "\n"
                    )
            except Exception:
                pass

            if auto_sql:
                md.append(f"\n<details><summary>SQL used</summary>\n\n```sql\n{auto_sql}\n```\n</details>\n")
        else:
            md.append(
                "\n_No query results available. "
                "Add a SQL query in the **Business SQL Query** field and run again, "
                "or make sure your question contains group-by keywords like "
                "\"by class\", \"per gender\", \"average fare\"._\n"
            )

    md.append(
        f"\n---\n*Cleaned data available via the Download button above. "
        f"Table: `{table}` · Columns: {len(safe_cols)}*"
    )
    return "\n".join(md)


# ---------------------------------------------------------------------------
# Tab 1: Data Quality Scanner
# ---------------------------------------------------------------------------

def scan_dataset(file_path: str | None, hf_name: str):
    from server.database import DatabaseEngine

    source, err = _resolve_source(file_path, hf_name)
    if err:
        return err, [], [], [], []
    try:
        db = DatabaseEngine(task_id="bq_hard", seed=42,
                            dataset_source=source, max_rows=500)
    except Exception as exc:
        return f"**Error:** {_friendly_error(exc, source)}", [], [], [], []

    table       = db.primary_table
    issues      = db.issue_registry
    profiles    = db._profiles.get(table, {})
    total_rows  = len(db.rows(table))
    total_cols  = sum(1 for c in profiles if c not in ("id", "_source_format"))
    n_issues    = len(issues)
    cells       = max(total_rows * total_cols, 1)
    quality_pct = max(0.0, min(1.0, 1.0 - n_issues / cells)) * 100
    grade       = ("Excellent" if quality_pct >= 90 else
                   "Good"      if quality_pct >= 70 else
                   "Fair"      if quality_pct >= 50 else "Poor")

    type_counts: dict[str, int] = {}
    for iss in issues:
        type_counts[iss.issue_type] = type_counts.get(iss.issue_type, 0) + 1

    breakdown = "\n".join(
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
| **Issues found** | {n_issues} |
| **Quality score** | **{quality_pct:.1f}% — {grade}** |

### Issue Breakdown

| Issue Type | Count |
|------------|-------|
{breakdown}
"""

    col_rows = []
    for col, p in profiles.items():
        if col in ("id", "_source_format"):
            continue
        null_rate  = p.get("null_rate", 0.0)
        col_issues = sum(1 for i in issues if i.column == col)
        health = ("Good" if null_rate < 0.05 and col_issues == 0 else
                  "Fair" if null_rate < 0.30 and col_issues <= 2 else "Poor")
        col_rows.append([col, p.get("dtype", "?"),
                         f"{null_rate*100:.1f}%", col_issues, health])

    issue_rows  = [[t, c] for t, c in sorted(type_counts.items(), key=lambda x: -x[1])]
    sample_rows = [
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
# Tab 2: Agent Demo
# ---------------------------------------------------------------------------

def run_agent_streaming(
    file1: str | None,
    file2: str | None,
    file3: str | None,
    hf_name: str,
    intent: str,
    output_format: str,
    user_requirement: str,
    biz_sql: str,
) -> Generator[tuple[str, str, str | None, Any, Any, str, Any, str], None, None]:
    """Yields (log, summary_md, download_path, before_df, after_df,
               schema_md, query_df, final_answer_md)."""
    from server.environment import SQLSherlockEnvironment
    from models import SQLSherlockAction

    # --- Resolve source ---
    uploaded = [f for f in [file1, file2, file3] if f]
    if len(uploaded) >= 2:
        try:
            source = _merge_to_xlsx(uploaded)
            multi  = True
        except Exception as exc:
            _empty = pd.DataFrame()
            yield (f"Could not merge files: {exc}", "", None,
                   _empty, _empty, "", _empty, "")
            return
    elif len(uploaded) == 1:
        source, multi = uploaded[0], False
    else:
        source, err = _resolve_source(None, hf_name)
        multi = False
        if not source:
            _empty = pd.DataFrame()
            yield (err or "No dataset provided.", "", None,
                   _empty, _empty, "", _empty, "")
            return

    task_ids     = INTENT_TASK_MAP.get(intent, INTENT_TASK_MAP["visualization"])
    intent_label = INTENT_LABELS.get(intent, intent)

    log_lines:  list[str] = []
    all_scores: list[tuple[str, float]] = []
    download_path: str | None = None
    summary_md     = "_Running…_"
    before_df      = pd.DataFrame()
    after_df       = pd.DataFrame()
    schema_md      = ""
    query_df       = pd.DataFrame()
    final_answer   = ""
    auto_sql_used: str | None = None

    # Track cleaning stats across all tasks (Hard task is what matters for final answer)
    hard_issues_total    = 0
    hard_issues_remain   = 0
    hard_columns_fixed:  list[str] = []
    hard_table           = ""
    hard_columns:        list[str] = []

    def _log(*parts: str) -> None:
        log_lines.append(" ".join(str(p) for p in parts))

    def emit():
        return ("\n".join(log_lines), summary_md, download_path,
                before_df, after_df, schema_md, query_df, final_answer)

    src_label = f"{source} ({len(uploaded)} tables merged)" if multi else source
    _log(f"Intent : {intent_label}")
    _log(f"Dataset: {src_label}")
    _log(f"Format : {output_format}")
    if (user_requirement or "").strip():
        _log(f"Goal   : {(user_requirement or '').strip()[:100]}")
    _log("")
    yield emit()

    for task_id in task_ids:
        difficulty = task_id.split("_")[-1].upper()
        _log(f"{'─'*60}")
        _log(f"[{difficulty}] {task_id}")
        _log(f"{'─'*60}")
        yield emit()

        env     = SQLSherlockEnvironment()
        score   = 0.0
        step    = 0
        done    = False
        obs     = None
        rewards: list[float] = []
        cols_fixed_this_task: list[str] = []

        def _do(action_dict: dict) -> float:
            nonlocal step, done, obs
            if done:
                return 0.0
            step += 1
            filtered = {k: v for k, v in action_dict.items() if v is not None}
            action = SQLSherlockAction(**filtered)
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
            all_tables = list(obs.tables_summary.keys())
            table      = all_tables[0]
            t_info     = obs.tables_summary[table]
            visible    = [c for c in t_info.get("columns", [])
                          if c not in ("id", "_source_format")]

            # Schema for multi-table datasets (built once)
            if len(all_tables) > 1 and not schema_md:
                schema_md = _schema_md(obs.tables_summary)

            # Before state captured from Easy task (first run only)
            if task_id == task_ids[0] and hasattr(env, "_db") and env._db:
                before_df = _rows_to_df(env._db.original_state())

            # Hard task stats
            if task_id.endswith("_hard") and hasattr(env, "_db") and env._db:
                hard_issues_total = env._db.total_issues
                hard_table        = table
                hard_columns      = visible

            tbl_note = f"Tables: {all_tables}" if len(all_tables) > 1 else f"Table: {table}"
            _log(f"  {tbl_note}  ({t_info.get('row_count','?')} rows, {len(visible)} cols)")
            yield emit()

            # Phase 1: Inspect
            r = _do({"action_type": "inspect", "table": table})
            _log(f"  [{step:02d}] inspect                reward={r:+.2f}")
            yield emit()

            # Phase 1b: Multi-table — select + join
            if len(all_tables) > 1:
                r = _do({"action_type": "select_tables", "tables": all_tables})
                _log(f"  [{step:02d}] select_tables({all_tables})  reward={r:+.2f}")
                yield emit()
                if len(all_tables) >= 2 and not done:
                    t2_name = all_tables[1]
                    t2_cols = obs.tables_summary.get(t2_name, {}).get("columns", [])
                    shared  = [c for c in visible if c in t2_cols
                               and c not in ("id", "_source_format")]
                    jkey    = shared[0] if shared else (visible[0] if visible else None)
                    if jkey:
                        r = _do({"action_type": "join_tables",
                                 "table": table, "table2": t2_name, "key": jkey})
                        _log(f"  [{step:02d}] join({table},{t2_name},key={jkey})  reward={r:+.2f}")
                        yield emit()

            # Phase 2: Profile + bulk-fix
            for col in visible[:12]:
                if done:
                    break
                r = _do({"action_type": "profile_column", "table": table, "column": col})
                if obs and obs.query_result:
                    p  = obs.query_result[0]
                    nc = p.get("null_count", 0)
                    dt = p.get("dtype", "?")
                    _log(f"  [{step:02d}] profile({col:<12})  type={dt}  nulls={nc}  reward={r:+.2f}")
                    yield emit()
                    if nc > 0 and not done:
                        fill     = (p.get("median") or p.get("mean") or 0) if dt in ("int","float") \
                                   else (p.get("mode") or "Unknown")
                        strategy = "median" if dt in ("int","float") else "mode"
                        r2 = _do({
                            "action_type": "fix_column", "table": table,
                            "column": col, "value": fill,
                            "reason": f"bulk fix {nc} nulls in {col}, {strategy}={fill}",
                        })
                        _log(f"  [{step:02d}] fix_column({col:<12}, {str(fill)[:8]!r})  reward={r2:+.2f}")
                        if r2 > 0:
                            cols_fixed_this_task.append(col)
                        yield emit()
                else:
                    _log(f"  [{step:02d}] profile({col:<12})  reward={r:+.2f}")
                    yield emit()

            # Phase 3: Validate
            if not done:
                r  = _do({"action_type": "validate"})
                fb = (obs.last_feedback if obs else "")[:100]
                _log(f"  [{step:02d}] validate               reward={r:+.2f}  {fb}")
                yield emit()

            # Phase 4: Business query SQL (run before export so DB is still live)
            if not done and task_id.endswith("_hard"):
                # Determine SQL to run
                _biz = (biz_sql or "").strip()
                run_sql = _biz if _biz else None
                _req   = (user_requirement or "").strip()
                if run_sql is None and intent == "business_query" and _req:
                    run_sql = _nl_to_sql(_req, table, visible)
                    if run_sql:
                        auto_sql_used = run_sql
                        _log(f"  [auto-SQL generated from question]")
                        _log(f"  → {run_sql[:80]}{'…' if len(run_sql) > 80 else ''}")
                        yield emit()

                if run_sql and run_sql.upper().startswith("SELECT"):
                    r = _do({"action_type": "run_sql", "sql": run_sql[:500]})
                    if obs and obs.query_result:
                        query_df = pd.DataFrame(obs.query_result)
                        _log(f"  [{step:02d}] run_sql → {len(obs.query_result)} rows  reward={r:+.2f}")
                    else:
                        _log(f"  [{step:02d}] run_sql → no results  reward={r:+.2f}")
                    yield emit()

            # Phase 5: Export
            if not done:
                # Record issues remaining before export (Hard task only)
                if task_id.endswith("_hard") and hasattr(env, "_db") and env._db:
                    hard_issues_remain  = env._db.issues_remaining()
                    hard_columns_fixed  = cols_fixed_this_task[:]

                r  = _do({"action_type": "export"})
                fb = (obs.last_feedback if obs else "")[:130]
                _log(f"  [{step:02d}] export                 reward={r:+.2f}")
                _log(f"         {fb}")
                m = re.search(r"[Gg]rader\s+score\s*=?\s*(\d+\.\d+)", fb)
                if m:
                    score = float(m.group(1))

                # Capture after-state and download from Hard task
                if task_id.endswith("_hard") and hasattr(env, "_db") and env._db:
                    after_df = _rows_to_df(env._db.current_state())
                    if hasattr(env, "_export_result") and env._export_result:
                        download_path = env._export_result.get("filepath")

                    # Build final answer now that we have all data
                    final_answer = _generate_final_answer(
                        intent         = intent,
                        user_requirement = user_requirement,
                        issues_total   = hard_issues_total,
                        issues_remaining = hard_issues_remain,
                        columns_fixed  = hard_columns_fixed,
                        query_df       = query_df,
                        auto_sql       = auto_sql_used,
                        table          = hard_table or table,
                        columns        = hard_columns or visible,
                    )

                yield emit()

        except Exception as exc:
            _log(f"  [ERROR] {_friendly_error(exc, source)}")
        finally:
            try:
                env.close()
            except Exception:
                pass

        success = score >= 0.50
        _log(f"  Score: {score:.2f}  {'SUCCESS ✓' if success else 'PARTIAL'}  Steps: {step}")
        _log("")
        all_scores.append((task_id, score))
        yield emit()

    # Final summary table
    avg = sum(s for _, s in all_scores) / len(all_scores) if all_scores else 0.0
    rows_md = "\n".join(
        f"| `{tid}` | {s:.2f} | {'Pass ✓' if s >= 0.50 else 'Fail ✗'} |"
        for tid, s in all_scores
    )
    summary_md = f"""## Results — Intent: `{intent}`

| Task | Score | Result |
|------|-------|--------|
{rows_md}

**Average score: {avg:.2f}**

| Detail | Value |
|--------|-------|
| Dataset | `{src_label}` |
| Format | `{output_format}` |
| Tasks run | {len(all_scores)} (easy → medium → hard) |
{"| Download | cleaned file ready ↓ |" if download_path else ""}
{"| Answer | results shown ↓ |" if not query_df.empty else ""}
"""
    yield emit()


# ---------------------------------------------------------------------------
# Build Gradio UI
# ---------------------------------------------------------------------------

def _intent_req_update(intent: str):
    cfg = _REQ_CONFIG.get(intent, _REQ_CONFIG["visualization"])
    return gr.update(label=cfg["label"], placeholder=cfg["placeholder"])


def create_ui() -> gr.Blocks:
    with gr.Blocks(title="SQLSherlock — AI Data Quality Detective") as demo:

        gr.Markdown(
            "# SQLSherlock — AI Data Quality Detective\n"
            "Describe what you want to do with your data — "
            "the agent cleans it across **3 difficulty levels** and answers your question."
        )

        # ==================================================================
        # Tab 1: Data Quality Scanner
        # ==================================================================
        with gr.Tab("Data Quality Scanner"):
            gr.Markdown(
                "Scan any dataset for data quality issues before running the agent.\n\n"
                "> If both a file and a name are provided, the uploaded file takes priority."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### Upload a file")
                    scan_file = gr.File(
                        label="CSV / JSON / JSONL / Parquet / Excel",
                        file_types=[".csv", ".json", ".jsonl", ".parquet", ".xlsx"],
                        type="filepath",
                    )
                    gr.Markdown("#### OR — HuggingFace dataset")
                    scan_hf = gr.Textbox(
                        label="Dataset name", value="phihung/titanic",
                        placeholder="e.g. phihung/titanic",
                    )
                    gr.Examples(examples=_EXAMPLE_HF, inputs=[scan_hf], label="Quick examples")
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
        # Tab 2: Agent Demo
        # ==================================================================
        with gr.Tab("Agent Demo"):
            gr.Markdown(
                "**Describe what you want** — the agent cleans your data "
                "and answers your question or prepares data for your specific use case."
            )

            with gr.Row():
                # --- Dataset ---
                with gr.Column(scale=2):
                    gr.Markdown("#### Dataset (upload up to 3 tables)")
                    with gr.Row():
                        agent_file1 = gr.File(
                            label="Table 1 (primary)",
                            file_types=[".csv", ".json", ".jsonl", ".parquet", ".xlsx"],
                            type="filepath",
                        )
                        agent_file2 = gr.File(
                            label="Table 2 (optional)",
                            file_types=[".csv", ".json", ".jsonl", ".parquet", ".xlsx"],
                            type="filepath",
                        )
                        agent_file3 = gr.File(
                            label="Table 3 (optional)",
                            file_types=[".csv", ".json", ".jsonl", ".parquet", ".xlsx"],
                            type="filepath",
                        )
                    gr.Markdown(
                        "> Upload **2–3 files** for multi-table mode — they are merged into "
                        "one XLSX and the agent will inspect all schemas and attempt joins.\n\n"
                        "> OR enter a **HuggingFace dataset** name below (used only when no file is uploaded)."
                    )
                    agent_hf = gr.Textbox(
                        label="HuggingFace dataset name",
                        value="phihung/titanic",
                        placeholder="phihung/titanic",
                    )

                # --- Intent + Requirement ---
                with gr.Column(scale=2):
                    gr.Markdown("#### Cleaning Intent")
                    agent_intent = gr.Radio(
                        choices=INTENT_CHOICES,
                        label="What will you use this data for?",
                        value="visualization",
                    )
                    # Dynamic text area — label and placeholder change with intent
                    requirement_input = gr.Textbox(
                        label=_REQ_CONFIG["visualization"]["label"],
                        placeholder=_REQ_CONFIG["visualization"]["placeholder"],
                        lines=3,
                        max_lines=6,
                    )

                # --- Format ---
                with gr.Column(scale=1):
                    gr.Markdown("#### Output Format")
                    agent_format = gr.Radio(
                        choices=FORMAT_CHOICES,
                        label="Cleaned file format",
                        value="csv",
                    )

            # Update requirement box when intent changes
            agent_intent.change(
                fn=_intent_req_update,
                inputs=[agent_intent],
                outputs=[requirement_input],
            )

            # Advanced SQL box
            with gr.Accordion("Advanced: Direct SQL Query (optional)", open=False):
                gr.Markdown(
                    "Enter a `SELECT` query to run on the cleaned dataset after the Hard task. "
                    "If left empty and intent is **Business Analytics**, an SQL query is "
                    "auto-generated from your requirement text above.\n\n"
                    "Use the table name shown in the step log (e.g. `dataset`, `titanic`)."
                )
                biz_sql_input = gr.Textbox(
                    label="SQL Query",
                    placeholder=_SQL_PLACEHOLDER,
                    lines=5,
                    max_lines=12,
                )

            gr.Markdown(
                "> **Scoring cascade — Easy matters most:**\n"
                "> Easy `res×0.80 + val×0.20` → Medium `easy×0.70 + advanced×0.30` "
                "→ Hard `medium×0.70 + FK×0.30`"
            )

            gr.Examples(
                examples=_AGENT_EXAMPLES,
                inputs=[agent_hf, agent_intent, requirement_input, agent_format, biz_sql_input],
                label="Quick examples",
            )

            run_btn = gr.Button("Run Agent (3 Tasks)", variant="primary", size="lg")

            # --- Primary outputs ---
            with gr.Row():
                with gr.Column(scale=3):
                    agent_log = gr.Textbox(
                        label="Live Step Log",
                        lines=28, max_lines=120,
                        interactive=False,
                    )
                with gr.Column(scale=2):
                    agent_results = gr.Markdown(
                        value="_Results appear here after the agent finishes._"
                    )

            schema_out = gr.Markdown(value="", label="Schema (multi-table)")

            gr.Markdown("#### Download Cleaned File (Hard task output)")
            download_out = gr.File(
                label="Cleaned dataset",
                interactive=False,
            )

            # --- Before / After ---
            gr.Markdown("#### Before / After Comparison")
            with gr.Row():
                before_tbl = gr.Dataframe(
                    label="Original Data (before agent)",
                    interactive=False, wrap=True,
                )
                after_tbl = gr.Dataframe(
                    label="Cleaned Data (after Hard task)",
                    interactive=False, wrap=True,
                )

            # --- Final Answer ---
            gr.Markdown("#### Final Answer")
            final_answer_out = gr.Markdown(
                value="_Final answer appears here after all 3 tasks complete._"
            )

            # --- Business query result table ---
            gr.Markdown("#### Query Results")
            query_result_tbl = gr.Dataframe(
                label="SQL query output (on cleaned data)",
                interactive=False, wrap=True,
            )

            def _run(f1, f2, f3, hf, intent, fmt, req, sql):
                for log, summary, dl, bdf, adf, schema, qdf, answer in run_agent_streaming(
                    f1, f2, f3, hf, intent, fmt, req, sql
                ):
                    yield log, summary, dl, bdf, adf, schema, qdf, answer

            run_btn.click(
                fn=_run,
                inputs=[agent_file1, agent_file2, agent_file3,
                        agent_hf, agent_intent, agent_format,
                        requirement_input, biz_sql_input],
                outputs=[agent_log, agent_results, download_out,
                         before_tbl, after_tbl, schema_out,
                         query_result_tbl, final_answer_out],
                show_progress="minimal",
            )

        # ==================================================================
        # Tab 3: About
        # ==================================================================
        with gr.Tab("About"):
            gr.Markdown("""
## How It Works

1. **Describe your goal** — dashboard, model, or business question
2. **Agent cleans Easy → Medium → Hard** for the selected intent
3. **Final answer** — cleaning summary + SQL results + narrative

### Scoring (Easy → Medium → Hard cascade)

| Task | Formula | Key weight |
|------|---------|------------|
| **Easy** | `resolution × 0.80 + validation × 0.20` | Nulls/types/whitespace/categories |
| **Medium** | `easy × 0.70 + advanced × 0.30` | Easy = 70% of medium |
| **Hard** | `medium × 0.70 + FK × 0.30` | Medium = 70% of hard |

### Multiple Tables

Upload **2–3 files** — each becomes a table (sheet in a merged XLSX).
The agent will `select_tables`, inspect schemas, and attempt `join_tables`.

### Business Query Answering

- Enter a **plain English question** in the requirement box
- The agent auto-generates SQL from keywords (group-by, average, top-N, etc.)
- Or provide exact SQL in the Advanced SQL field
- Results appear in the **Query Results** table with a narrative

### Intent-Specific Cleaning Focus

| Intent | Primary Issues | Output |
|--------|---------------|--------|
| **Visualization** | Nulls, whitespace, categories | Chart-ready columns list |
| **ML Training** | Nulls, type errors, outliers, constraints | Feature/target column list |
| **Business Analytics** | All types + FK + duplicates | SQL answer + narrative |
""")

    return demo
