# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
DatabaseEngine for SQLSherlock-Env.

Manages one in-memory SQLite database per episode.
Owns: dataset loading, profiling, issue detection, trap planting,
      baseline validation, and all agent-facing read/write operations.
"""

import copy
import math
import random
import re
import sqlite3
from typing import Any, Optional

from server.dataset_loader import load, records_to_sqlite, coerce
from server.schema_profiler import profile_table, find_primary_key
from server.issue_detector import detect_issues, detect_trap, Issue, Trap
from server.validator import Validator, ValidationResult


# ---------------------------------------------------------------------------
# SQL injection block-list
# ---------------------------------------------------------------------------

_BLOCKED = frozenset({
    "DROP", "DELETE", "UPDATE", "INSERT", "ALTER",
    "CREATE", "ATTACH", "DETACH", "LOAD_EXTENSION", "PRAGMA", "VACUUM",
    "REINDEX", "SAVEPOINT", "RELEASE", "BEGIN", "COMMIT", "ROLLBACK",
})
_WORD_RE = re.compile(r"\b(\w+)\b")
_MAX_QUERY_ROWS = 50


# ---------------------------------------------------------------------------
# DatabaseEngine
# ---------------------------------------------------------------------------

class DatabaseEngine:
    """In-memory SQLite environment, isolated per episode.

    Initialisation sequence
    -----------------------
    1. Load dataset from source.
    2. Write records to SQLite.
    3. Deep-copy originals (before any mutation).
    4. Profile all columns.
    5. Capture validator baseline.
    6. Detect real issues (+ synthetic top-up).
    7. Plant trap (task3 only).
    8. Initialise action log.
    """

    def __init__(
        self,
        task_id: str,
        seed: int,
        dataset_source: str,
        max_rows: int = 500,
    ) -> None:
        if not dataset_source or not dataset_source.strip():
            raise ValueError("dataset_source must not be empty.")

        self.task_id = task_id
        self.seed = seed

        # --- 1. Load ---
        table_records = load(dataset_source, max_rows=max_rows)

        # --- 2. SQLite ---
        self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

        self._table_names: list[str] = []
        self._records: dict[str, list[dict]] = {}

        for tname, recs in table_records.items():
            records_to_sqlite(self._conn, tname, recs)
            self._table_names.append(tname)
            self._records[tname] = recs

        # Primary table is always the first one
        self._primary_table: str = self._table_names[0]

        # --- 3. Deep-copy originals (clean snapshot before any mutation) ---
        self._originals: dict[str, list[dict]] = {
            t: copy.deepcopy(recs) for t, recs in self._records.items()
        }

        # --- 4. Profile ---
        self._profiles: dict[str, dict[str, dict]] = {}
        for tname, recs in self._records.items():
            self._profiles[tname] = profile_table(tname, recs, self._conn)

        # Determine PK column for primary table — needed before context table generation
        primary_recs = self._records[self._primary_table]
        self._pk_col: str = (
            find_primary_key(primary_recs) or list(primary_recs[0].keys())[0]
        )

        # --- 4b. Task4: generate context/lookup table for multi-table reasoning ---
        # Context table is generated AFTER initial profile so we can pick a good
        # categorical column; originals are re-snapshotted after to include new column.
        self._context_table: Optional[str] = None
        self._context_join_key: tuple[str, str] = ("", "")  # (primary_col, context_col)
        self._intent: Optional[str] = None      # set externally by environment.py
        self._join_attempts: list[dict] = []    # track join_tables calls for grader

        if task_id == "task4_context_aware_analysis":  # unused in current 3-task design
            ctx = self._generate_context_table(seed)
            if ctx:
                # Re-snapshot originals with new context_ref column included
                for tname in self._table_names:
                    self._records[tname] = self.rows(tname)
                self._originals = {
                    t: copy.deepcopy(recs) for t, recs in self._records.items()
                }
                # Re-profile to include new column
                for tname in self._table_names:
                    self._profiles[tname] = profile_table(
                        tname, self._records[tname], self._conn
                    )
                # Re-determine PK (profile refresh may affect column ordering)
                primary_recs = self._records[self._primary_table]

        # Source format (from injected _source_format key)
        self.source_format: str = (
            primary_recs[0].get("_source_format", "csv") if primary_recs else "csv"
        )
        self.dataset_name: str = dataset_source

        # --- 5. Validator baseline ---
        # Issue registry not yet built — pass empty list for baseline;
        # we rebuild after detection.
        self._validator: Optional[Validator] = None  # initialised after step 6

        # --- 6. Issue detection ---
        primary_profile = self._profiles[self._primary_table]
        self._issues: list[Issue] = detect_issues(
            conn=self._conn,
            profile=primary_profile,
            records=primary_recs,
            task_id=task_id,
            seed=seed,
        )

        # NOW build validator with the real issue registry
        self._validator = Validator(
            conn=self._conn,
            profile=primary_profile,
            issue_registry=self._issues,
        )

        # --- 7. Trap (hard tasks only — any intent) ---
        self._trap: Optional[Trap] = None
        if task_id.endswith("_hard"):
            self._trap = detect_trap(
                conn=self._conn,
                profile=primary_profile,
                records=primary_recs,
                issue_registry=self._issues,
                seed=seed,
            )

        # --- 8. Action log ---
        self._action_log: list[Any] = []

        # Track which columns the agent has touched (for distribution warnings)
        self._touched_columns: set[str] = set()

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def rows(self, table: str) -> list[dict]:
        """Return current rows for *table* as plain dicts."""
        self._require_table(table)
        cur = self._conn.execute(f'SELECT * FROM "{table}"')
        return [dict(row) for row in cur.fetchall()]

    def columns(self, table: str) -> list[str]:
        """Return column names for *table*."""
        self._require_table(table)
        cur = self._conn.execute(f'PRAGMA table_info("{table}")')
        return [row[1] for row in cur.fetchall()]

    def table_names(self) -> list[str]:
        """Return all table names in this episode's database."""
        return list(self._table_names)

    def tables_summary(self) -> dict[str, Any]:
        """Return a compact summary of every table (for observations)."""
        summary = {}
        for tname in self._table_names:
            cols = self.columns(tname)
            profile = self._profiles.get(tname, {})
            dtypes = {col: profile[col]["dtype"] for col in cols if col in profile}
            current_rows = self.rows(tname)
            summary[tname] = {
                "row_count": len(current_rows),
                "columns":   cols,
                "dtypes":    dtypes,
            }
        return summary

    def query(self, sql: str) -> list[dict]:
        """Execute a read-only SELECT query and return up to 50 rows.

        Raises:
            ValueError: If the query is not a SELECT or contains blocked keywords.
        """
        if not sql or not sql.strip():
            raise ValueError("SQL query must not be empty.")

        stripped = sql.strip()
        if not stripped.upper().startswith("SELECT"):
            raise ValueError("Only SELECT queries are permitted.")

        if ";" in stripped:
            raise ValueError("Semicolons are not permitted in queries.")

        # Word-boundary check for blocked keywords
        words = {m.group(1).upper() for m in _WORD_RE.finditer(stripped)}
        blocked_found = words & _BLOCKED
        if blocked_found:
            raise ValueError(
                f"Query contains blocked keyword(s): {sorted(blocked_found)}. "
                "Only SELECT is permitted."
            )

        try:
            cur = self._conn.execute(stripped)
            rows = cur.fetchmany(_MAX_QUERY_ROWS)
            return [dict(row) for row in rows]
        except sqlite3.Error as exc:
            raise ValueError(f"SQL error: {exc}") from exc

    def profile_col(self, table: str, column: str) -> dict:
        """Return statistical profile for one column.

        Returns dict with: mean, std, min, max, null_count,
        z_scores {row_id: z}, must_be_positive.
        """
        self._require_table(table)
        profile = self._profiles.get(table, {})
        if column not in profile:
            # Re-profile on demand (column may have been modified)
            current = self.rows(table)
            updated_profile = profile_table(table, current, self._conn)
            self._profiles[table] = updated_profile
            profile = updated_profile

        if column not in profile:
            raise ValueError(f"Column '{column}' not found in table '{table}'.")

        p = profile[column]

        # Compute median and mode for smarter imputation hints
        current_rows = self.rows(table)
        non_null_vals = [r.get(column) for r in current_rows if not _is_null(r.get(column))]

        median_val = None
        mode_val = None
        if non_null_vals:
            if p.get("dtype") in ("int", "float"):
                nums = sorted(float(v) for v in non_null_vals if _can_cast_float(v))
                if nums:
                    mid = len(nums) // 2
                    median_val = round(nums[mid] if len(nums) % 2 else (nums[mid-1]+nums[mid])/2, 4)
            # Mode: most common value (works for both string and numeric)
            from collections import Counter
            counts = Counter(str(v) for v in non_null_vals)
            if counts:
                mode_val = counts.most_common(1)[0][0]

        return {
            "mean":             p.get("mean"),
            "median":           median_val,
            "mode":             mode_val,
            "std":              p.get("std"),
            "min":              p.get("min"),
            "max":              p.get("max"),
            "null_count":       p.get("null_count", 0),
            "null_rate":        p.get("null_rate", 0.0),
            "z_scores":         p.get("z_scores", {}),
            "must_be_positive": p.get("must_be_positive", False),
            "dtype":            p.get("dtype", "unknown"),
        }

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def fix_cell(self, table: str, row_id: int, column: str, value: Any) -> None:
        """Update one cell in the database.

        Raises:
            ValueError: If table/column not found or row_id does not exist.
        """
        self._require_table(table)
        cols = self.columns(table)
        if column not in cols:
            raise ValueError(f"Column '{column}' not found in table '{table}'.")

        pk = self._pk_col
        existing = self._conn.execute(
            f'SELECT "{pk}" FROM "{table}" WHERE "{pk}" = ?', (row_id,)
        ).fetchone()
        if existing is None:
            raise ValueError(f"Row id={row_id} not found in table '{table}'.")

        # Coerce value to the column's detected dtype so SQLite stores correctly.
        # Without this, an agent sending value="25.5" for a REAL column would
        # store TEXT instead of REAL, causing false type_error flags in validation.
        profile = self._profiles.get(table, {})
        col_dtype = profile.get(column, {}).get("dtype", "str")
        if col_dtype in ("int", "float") and value is not None:
            try:
                fval = float(str(value))
                safe_val = int(fval) if col_dtype == "int" and fval == int(fval) else fval
            except (ValueError, TypeError):
                safe_val = _to_sqlite(value)
        else:
            safe_val = _to_sqlite(value)

        self._conn.execute(
            f'UPDATE "{table}" SET "{column}" = ? WHERE "{pk}" = ?',
            (safe_val, row_id),
        )
        self._conn.commit()
        self._touched_columns.add(column)

        # Invalidate cached profile for this column
        if table in self._profiles and column in self._profiles[table]:
            del self._profiles[table][column]

    def fix_column(self, table: str, column: str, value: Any) -> dict:
        """Fix ALL data quality issues in a column in one bulk operation.

        Fixes: nulls, empty strings, type errors (non-castable values in
        numeric columns), and negative values in must-be-positive columns.

        Returns dict with counts: {nulls_fixed, type_errors_fixed,
        negatives_fixed, total_fixed}.
        """
        self._require_table(table)
        cols = self.columns(table)
        if column not in cols:
            raise ValueError(f"Column '{column}' not found in table '{table}'.")

        profile = self._profiles.get(table, {})
        col_profile = profile.get(column, {})
        col_dtype = col_profile.get("dtype", "str")
        must_be_positive = col_profile.get("must_be_positive", False)

        # Coerce fill value to column dtype
        if col_dtype in ("int", "float") and value is not None:
            try:
                fval = float(str(value))
                safe_val = int(fval) if col_dtype == "int" and fval == int(fval) else fval
            except (ValueError, TypeError):
                safe_val = _to_sqlite(value)
        else:
            safe_val = _to_sqlite(value)

        total = 0

        # 1. Fix NULLs and empty strings
        cur = self._conn.execute(
            f'UPDATE "{table}" SET "{column}" = ? '
            f'WHERE "{column}" IS NULL OR TRIM("{column}") = ?',
            (safe_val, ""),
        )
        nulls_fixed = cur.rowcount
        total += nulls_fixed

        # 2. Fix type errors: non-castable strings in numeric columns
        type_errors_fixed = 0
        if col_dtype in ("int", "float"):
            # Find rows where the value can't be cast to a number
            pk = self._pk_col
            rows = self._conn.execute(
                f'SELECT "{pk}", "{column}" FROM "{table}" '
                f'WHERE "{column}" IS NOT NULL AND TRIM("{column}") != ?',
                ("",),
            ).fetchall()
            for row in rows:
                rid = row[0]
                val = row[1]
                try:
                    float(str(val))
                except (ValueError, TypeError):
                    # This value is not castable to float — it's a type error
                    self._conn.execute(
                        f'UPDATE "{table}" SET "{column}" = ? WHERE "{pk}" = ?',
                        (safe_val, rid),
                    )
                    type_errors_fixed += 1
            total += type_errors_fixed

        # 3. Fix negative values in must-be-positive columns
        negatives_fixed = 0
        if must_be_positive and col_dtype in ("int", "float"):
            cur = self._conn.execute(
                f'UPDATE "{table}" SET "{column}" = ABS(CAST("{column}" AS REAL)) '
                f'WHERE CAST("{column}" AS REAL) < 0',
            )
            negatives_fixed = cur.rowcount
            total += negatives_fixed

        self._conn.commit()
        self._touched_columns.add(column)

        # Invalidate profile cache
        if table in self._profiles and column in self._profiles[table]:
            del self._profiles[table][column]

        return {
            "nulls_fixed": nulls_fixed,
            "type_errors_fixed": type_errors_fixed,
            "negatives_fixed": negatives_fixed,
            "total_fixed": total,
        }

    def delete_row(self, table: str, row_id: int) -> None:
        """Delete a row from the database.

        Raises:
            ValueError: If table not found or row does not exist.
        """
        self._require_table(table)
        pk = self._pk_col
        existing = self._conn.execute(
            f'SELECT "{pk}" FROM "{table}" WHERE "{pk}" = ?', (row_id,)
        ).fetchone()
        if existing is None:
            raise ValueError(f"Row id={row_id} not found in table '{table}'.")

        self._conn.execute(
            f'DELETE FROM "{table}" WHERE "{pk}" = ?', (row_id,)
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> ValidationResult:
        """Run all 6 validator checks against current state."""
        current = self.rows(self._primary_table)
        return self._validator.validate(
            conn=self._conn,
            current_records=current,
            touched_columns=self._touched_columns,
        )

    # ------------------------------------------------------------------
    # State / scoring helpers
    # ------------------------------------------------------------------

    def current_state(self) -> list[dict]:
        """Return current rows of the primary table."""
        return self.rows(self._primary_table)

    def original_state(self) -> list[dict]:
        """Return the deep-copied original rows (before any fixes)."""
        return copy.deepcopy(self._originals[self._primary_table])

    @property
    def primary_table(self) -> str:
        return self._primary_table

    @property
    def pk_col(self) -> str:
        return self._pk_col

    @property
    def trap(self) -> Optional[Trap]:
        return self._trap

    @property
    def issue_registry(self) -> list[Issue]:
        """The ground-truth issue list. NEVER sent to the agent."""
        return self._issues

    @property
    def total_issues(self) -> int:
        return len(self._issues)

    def issues_remaining(self) -> int:
        """Count issues not yet resolved by the current DB state."""
        current = self.rows(self._primary_table)
        pk_col = self._pk_col
        row_map = {row[pk_col]: row for row in current}
        current_ids = set(row_map.keys())

        remaining = 0
        for iss in self._issues:
            if iss.issue_type in ("duplicate", "fk_violation"):
                if iss.row_id in current_ids:
                    remaining += 1
            elif iss.issue_type == "null":
                row = row_map.get(iss.row_id)
                if row is not None and _is_null(row.get(iss.column)):
                    remaining += 1
            elif iss.issue_type == "type_error":
                row = row_map.get(iss.row_id)
                if row is not None:
                    val = row.get(iss.column)
                    # Only count as remaining if non-null AND still non-castable
                    # (prevents null cells being double-counted as type errors)
                    if not _is_null(val) and not _can_cast_float(val):
                        remaining += 1
            elif iss.issue_type == "constraint":
                row = row_map.get(iss.row_id)
                if row is not None:
                    val = row.get(iss.column)
                    if val is not None and _can_cast_float(val) and float(val) < 0:
                        remaining += 1
            elif iss.issue_type == "outlier":
                row = row_map.get(iss.row_id)
                if row is not None:
                    val = row.get(iss.column)
                    if val is not None and _can_cast_float(val):
                        fval = float(val)
                        profile = self._profiles.get(self._primary_table, {})
                        p = profile.get(iss.column, {})
                        std = p.get("std")
                        # Prefer stored correct (median at detection time) over contaminated mean
                        correct = iss.correct
                        if correct is not None and _can_cast_float(correct) and std and std > 0:
                            z_from_median = abs(fval - float(correct)) / std
                            if z_from_median > 2.0:
                                remaining += 1
                        else:
                            # Fallback: contaminated profile mean
                            mean = p.get("mean")
                            if mean is not None and std and std > 0:
                                z = abs(fval - mean) / std
                                if z > 3.0:
                                    remaining += 1
        return remaining

    def log_action(self, action: Any) -> None:
        """Append an action to the episode log."""
        self._action_log.append(action)

    # ------------------------------------------------------------------
    # Multi-table reasoning
    # ------------------------------------------------------------------

    def join_tables(self, table1: str, table2: str, key: str) -> dict:
        """Join two tables on a matching key column and return the result.

        Automatically resolves the matching key in *table2* by looking for:
          - exact column name match
          - "id" column (common for lookup/context tables)
          - "{key_without_ref}_id" pattern

        Returns:
            {
                "rows":       list[dict]  — joined rows (max 50),
                "valid":      bool        — True if not a cartesian product,
                "match_rate": float       — fraction of table1 rows with a match,
                "error":      str | None,
            }
        """
        try:
            self._require_table(table1)
            self._require_table(table2)
        except ValueError as exc:
            return {"rows": [], "valid": False, "match_rate": 0.0, "error": str(exc)}

        t1_cols = self.columns(table1)
        t2_cols = self.columns(table2)

        if key not in t1_cols:
            return {
                "rows": [], "valid": False, "match_rate": 0.0,
                "error": f"Key column '{key}' not found in table '{table1}'.",
            }

        # Resolve the matching column in table2
        t2_key: Optional[str] = None
        if key in t2_cols:
            t2_key = key
        elif key.endswith("_ref"):
            # e.g. "pclass_ref" → look for "pclass_id" or "id"
            base = key[:-4]
            if "id" in t2_cols:
                t2_key = "id"
            elif f"{base}_id" in t2_cols:
                t2_key = f"{base}_id"
        if t2_key is None and "id" in t2_cols:
            t2_key = "id"
        if t2_key is None and t2_cols:
            t2_key = t2_cols[0]   # last resort

        if t2_key is None:
            return {
                "rows": [], "valid": False, "match_rate": 0.0,
                "error": f"No matching key column found in '{table2}' for key '{key}'.",
            }

        try:
            sql = (
                f'SELECT t1.*, t2.* '
                f'FROM "{table1}" t1 '
                f'LEFT JOIN "{table2}" t2 ON t1."{key}" = t2."{t2_key}" '
                f'LIMIT {_MAX_QUERY_ROWS}'
            )
            cur = self._conn.execute(sql)
            rows = [dict(r) for r in cur.fetchall()]
        except sqlite3.Error as exc:
            return {"rows": [], "valid": False, "match_rate": 0.0, "error": str(exc)}

        t1_count = len(self.rows(table1))
        matched = sum(1 for r in rows if r.get(t2_key) is not None)
        match_rate = matched / max(t1_count, 1)

        # Cartesian product heuristic: all rows match all rows (no filtering happened)
        t2_count = len(self.rows(table2))
        is_cartesian = (
            t1_count > 1 and t2_count > 1
            and len(rows) >= min(t1_count * t2_count, _MAX_QUERY_ROWS)
        )
        valid = not is_cartesian and len(rows) > 0

        result = {
            "rows":       rows,
            "valid":      valid,
            "match_rate": round(match_rate, 4),
            "error":      None,
        }
        # Record attempt for grader
        self._join_attempts.append({
            "table1": table1, "table2": table2,
            "key": key, "t2_key": t2_key,
            "valid": valid, "match_rate": round(match_rate, 4),
        })
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _generate_context_table(self, seed: int) -> Optional[str]:
        """Generate a synthetic lookup/context table for task4.

        Picks the best low-cardinality categorical column from the primary
        table, creates a ``{col}_context`` lookup table, adds a ``{col}_ref``
        FK column to the primary table, and plants one FK violation.

        Returns the context table name, or None if no suitable column found.
        """
        rng = random.Random(seed + 100)
        primary_recs = self._records[self._primary_table]
        profile = self._profiles[self._primary_table]

        if not primary_recs:
            return None

        # Pick best categorical column: dtype str/int, 3–15 unique values
        cat_col: Optional[str] = None
        best_cardinality = float("inf")
        for col, p in profile.items():
            if col == self._pk_col or col == "_source_format":
                continue
            if p["dtype"] not in ("str", "unknown", "int"):
                continue
            unique = p.get("unique_count", 0)
            row_count = p.get("row_count", 1)
            if 3 <= unique <= 15 and unique < best_cardinality:
                best_cardinality = unique
                cat_col = col

        # Fallback: any column with 2–20 unique values
        if cat_col is None:
            for col, p in profile.items():
                if col == self._pk_col or col == "_source_format":
                    continue
                unique = p.get("unique_count", 0)
                if 2 <= unique <= 20:
                    cat_col = col
                    break

        if cat_col is None:
            return None

        # Collect unique values (preserve original types)
        seen_lower: set[str] = set()
        unique_vals = []
        for r in primary_recs:
            val = r.get(cat_col)
            if val is None:
                continue
            key = str(val).strip().lower()
            if key not in seen_lower:
                seen_lower.add(key)
                unique_vals.append(val)

        if not unique_vals:
            return None

        # Build context table records
        context_table_name = f"{cat_col}_context"
        descriptions = [
            "Standard category", "Premium tier", "Basic level",
            "Advanced tier", "Economy class", "Elite status",
            "Regular type", "Special category", "Classic group", "Custom segment",
        ]
        context_records = [
            {
                "id": i + 1,
                "label": str(v),
                "description": rng.choice(descriptions),
                "priority": rng.randint(1, 5),
            }
            for i, v in enumerate(unique_vals)
        ]
        # Plant one NULL issue in context table (description of a random row)
        if context_records:
            context_records[rng.randint(0, len(context_records) - 1)]["description"] = None

        # Write context table to SQLite
        records_to_sqlite(self._conn, context_table_name, context_records)
        self._table_names.append(context_table_name)
        self._records[context_table_name] = context_records

        # Add {cat_col}_ref FK column to primary table
        ref_col = f"{cat_col}_ref"
        try:
            self._conn.execute(
                f'ALTER TABLE "{self._primary_table}" ADD COLUMN "{ref_col}" INTEGER'
            )
        except sqlite3.OperationalError:
            return None   # column already exists (shouldn't happen)

        label_to_id = {str(r["label"]).strip().lower(): r["id"] for r in context_records}

        for rec in primary_recs:
            val = rec.get(cat_col)
            if val is None:
                continue
            ctx_id = label_to_id.get(str(val).strip().lower())
            if ctx_id is not None:
                self._conn.execute(
                    f'UPDATE "{self._primary_table}" SET "{ref_col}" = ? '
                    f'WHERE "{self._pk_col}" = ?',
                    (ctx_id, rec[self._pk_col]),
                )

        # Plant ONE FK violation: one row gets an invalid context_ref
        eligible = [r for r in primary_recs if r.get(self._pk_col) is not None]
        if eligible:
            victim = rng.choice(eligible)
            invalid_id = max(r["id"] for r in context_records) + 999
            self._conn.execute(
                f'UPDATE "{self._primary_table}" SET "{ref_col}" = ? '
                f'WHERE "{self._pk_col}" = ?',
                (invalid_id, victim[self._pk_col]),
            )

        self._conn.commit()

        # Profile the new context table
        self._profiles[context_table_name] = profile_table(
            context_table_name, context_records, self._conn
        )

        # Store join metadata for grader/reward use
        self._context_table = context_table_name
        self._context_join_key = (ref_col, "id")   # (primary_col, context_col)

        return context_table_name

    def _require_table(self, table: str) -> None:
        if table not in self._table_names:
            raise ValueError(
                f"Table '{table}' not found. "
                f"Available tables: {self._table_names}"
            )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _to_sqlite(value: Any) -> Any:
    """Convert a Python value to a SQLite-safe scalar."""
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float, str, bytes)):
        return value
    if isinstance(value, float) and math.isnan(value):
        return None
    return str(value)


def _is_null(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def _can_cast_float(value: Any) -> bool:
    try:
        float(str(value))
        return True
    except (ValueError, TypeError):
        return False
