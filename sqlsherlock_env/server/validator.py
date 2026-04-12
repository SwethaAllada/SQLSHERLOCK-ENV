# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Validator for SQLSherlock-Env.

Runs 6 checks comparing the current dataset state against the baseline
captured at reset() time.  Called by:
  - DatabaseEngine.__init__()  → stores baseline_metrics
  - environment.py step()      → on "validate" action
  - graders/universal.py       → final scoring pass
"""

import math
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    name:       str
    passed:     bool
    before:     Any
    after:      Any
    detail:     str = ""
    warnings:   list[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    checks:        dict[str, CheckResult]
    checks_passed: int
    total_checks:  int
    overall:       str          # "PASS" | "PARTIAL" | "FAIL"
    warnings:      list[str]    # distribution drift warnings

    def to_dict(self) -> dict:
        return {
            "checks": {
                name: {
                    "passed":   cr.passed,
                    "before":   cr.before,
                    "after":    cr.after,
                    "detail":   cr.detail,
                    "warnings": cr.warnings,
                }
                for name, cr in self.checks.items()
            },
            "checks_passed": self.checks_passed,
            "total_checks":  self.total_checks,
            "overall":       self.overall,
            "warnings":      self.warnings,
        }


# ---------------------------------------------------------------------------
# Validator class
# ---------------------------------------------------------------------------

class Validator:
    """Stateful validator that stores baseline metrics at construction time.

    Usage::

        v = Validator(conn, profile, issue_registry)
        # ... agent makes fixes ...
        result = v.validate(conn, current_records)
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        profile: dict[str, dict],
        issue_registry: list,           # list[Issue] — typed loosely to avoid circular import
    ) -> None:
        self._profile = profile
        self._issue_registry = issue_registry
        self._baseline = self._scan_baseline(conn, profile, issue_registry)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def validate(
        self,
        conn: sqlite3.Connection,
        current_records: list[dict],
        touched_columns: Optional[set[str]] = None,
    ) -> ValidationResult:
        """Run all 6 checks against the current state.

        Args:
            conn:             Live SQLite connection (current state).
            current_records:  Current rows as list of dicts.
            touched_columns:  Set of column names the agent modified.
                              Used to distinguish false-positive drift warnings.

        Returns:
            ValidationResult with per-check details.
        """
        profile = self._profile
        baseline = self._baseline
        touched = touched_columns or set()

        checks: dict[str, CheckResult] = {}
        warnings: list[str] = []

        # 1. Null check
        checks["null_check"] = self._null_check(current_records, baseline, profile)

        # 2. Type check
        checks["type_check"] = self._type_check(current_records, baseline, profile)

        # 3. Range check
        checks["range_check"] = self._range_check(current_records, baseline, profile)

        # 4. Distribution check
        dist_cr = self._distribution_check(current_records, baseline, profile, touched)
        checks["distribution_check"] = dist_cr
        warnings.extend(dist_cr.warnings)

        # 5. Duplicate check
        checks["duplicate_check"] = self._duplicate_check(current_records, baseline, profile)

        # 6. Outlier check
        checks["outlier_check"] = self._outlier_check(current_records, baseline, profile)

        passed = sum(1 for cr in checks.values() if cr.passed)
        total  = len(checks)

        if passed == total:
            overall = "PASS"
        elif passed == 0:
            overall = "FAIL"
        else:
            overall = "PARTIAL"

        return ValidationResult(
            checks=checks,
            checks_passed=passed,
            total_checks=total,
            overall=overall,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Baseline scan
    # ------------------------------------------------------------------

    def _scan_baseline(
        self,
        conn: sqlite3.Connection,
        profile: dict[str, dict],
        issue_registry: list,
    ) -> dict:
        """Compute baseline metrics from the initial (dirty) state."""
        # We use the profile (computed at load time) as our baseline source
        # plus we do a quick live scan for null/type counts

        baseline: dict = {}

        # Null counts per column (high-confidence issues only)
        high_conf_null_cols: set[str] = set()
        for iss in issue_registry:
            if iss.issue_type == "null" and iss.confidence > 0.50 and iss.column:
                high_conf_null_cols.add(iss.column)

        baseline["null_cols"] = high_conf_null_cols
        baseline["null_counts"] = {
            col: profile[col]["null_count"]
            for col in high_conf_null_cols
            if col in profile
        }

        # Type error columns
        type_error_cols = {
            iss.column
            for iss in issue_registry
            if iss.issue_type == "type_error" and iss.column
        }
        baseline["type_error_cols"] = type_error_cols
        baseline["type_error_counts"] = {col: 0 for col in type_error_cols}
        for iss in issue_registry:
            if iss.issue_type == "type_error" and iss.column:
                baseline["type_error_counts"][iss.column] = (
                    baseline["type_error_counts"].get(iss.column, 0) + 1
                )

        # Must-be-positive columns with negatives
        constraint_cols = {
            iss.column
            for iss in issue_registry
            if iss.issue_type == "constraint" and iss.column
        }
        baseline["constraint_cols"] = constraint_cols
        baseline["constraint_counts"] = {}
        for iss in issue_registry:
            if iss.issue_type == "constraint" and iss.column:
                baseline["constraint_counts"][iss.column] = (
                    baseline["constraint_counts"].get(iss.column, 0) + 1
                )

        # Distribution baseline (mean/std per numeric column)
        baseline["distribution"] = {
            col: {"mean": p["mean"], "std": p["std"]}
            for col, p in profile.items()
            if p["dtype"] in ("int", "float")
            and p["mean"] is not None
        }

        # Duplicate baseline: count of rows with repeated natural-key values
        baseline["duplicate_count"] = sum(
            1 for iss in issue_registry if iss.issue_type == "duplicate"
        )

        # Outlier baseline: set of (row_id, col) pairs with z > 5
        baseline["outlier_cells"] = {
            (iss.row_id, iss.column)
            for iss in issue_registry
            if iss.issue_type == "outlier" and iss.column
        }

        return baseline

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _null_check(
        self,
        records: list[dict],
        baseline: dict,
        profile: dict[str, dict],
    ) -> CheckResult:
        null_cols = baseline.get("null_cols", set())
        before_counts = baseline.get("null_counts", {})

        if not null_cols:
            return CheckResult(
                name="null_check",
                passed=True,
                before=before_counts,
                after={},
                detail="No high-confidence null issues in registry.",
            )

        after_counts: dict[str, int] = {}
        for col in null_cols:
            after_counts[col] = sum(
                1 for row in records if _is_null(row.get(col))
            )

        all_fixed = all(after_counts.get(col, 0) == 0 for col in null_cols)
        return CheckResult(
            name="null_check",
            passed=all_fixed,
            before=before_counts,
            after=after_counts,
            detail=(
                "All high-confidence nulls resolved."
                if all_fixed
                else f"Remaining nulls: { {c:v for c,v in after_counts.items() if v>0} }"
            ),
        )

    def _type_check(
        self,
        records: list[dict],
        baseline: dict,
        profile: dict[str, dict],
    ) -> CheckResult:
        type_cols = baseline.get("type_error_cols", set())
        before_counts = baseline.get("type_error_counts", {})

        if not type_cols:
            return CheckResult(
                name="type_check",
                passed=True,
                before=before_counts,
                after={},
                detail="No type errors in registry.",
            )

        after_counts: dict[str, int] = {}
        for col in type_cols:
            if col not in profile:
                after_counts[col] = 0
                continue
            after_counts[col] = sum(
                1 for row in records
                if not _is_null(row.get(col))
                and not _can_cast_float(row.get(col))
            )

        all_fixed = all(v == 0 for v in after_counts.values())
        return CheckResult(
            name="type_check",
            passed=all_fixed,
            before=before_counts,
            after=after_counts,
            detail=(
                "All type errors resolved."
                if all_fixed
                else f"Remaining type errors: { {c:v for c,v in after_counts.items() if v>0} }"
            ),
        )

    def _range_check(
        self,
        records: list[dict],
        baseline: dict,
        profile: dict[str, dict],
    ) -> CheckResult:
        constraint_cols = baseline.get("constraint_cols", set())
        before_counts = baseline.get("constraint_counts", {})

        if not constraint_cols:
            return CheckResult(
                name="range_check",
                passed=True,
                before=before_counts,
                after={},
                detail="No constraint violations in registry.",
            )

        after_counts: dict[str, int] = {}
        for col in constraint_cols:
            after_counts[col] = sum(
                1 for row in records
                if not _is_null(row.get(col))
                and _can_cast_float(row.get(col))
                and float(row[col]) < 0
            )

        all_fixed = all(v == 0 for v in after_counts.values())
        return CheckResult(
            name="range_check",
            passed=all_fixed,
            before=before_counts,
            after=after_counts,
            detail=(
                "All constraint violations resolved."
                if all_fixed
                else f"Remaining negatives: { {c:v for c,v in after_counts.items() if v>0} }"
            ),
        )

    def _distribution_check(
        self,
        records: list[dict],
        baseline: dict,
        profile: dict[str, dict],
        touched: set[str],
    ) -> CheckResult:
        dist_baseline = baseline.get("distribution", {})
        if not dist_baseline:
            return CheckResult(
                name="distribution_check",
                passed=True,
                before={},
                after={},
                detail="No numeric columns to check.",
            )

        after_dist: dict[str, dict] = {}
        warnings: list[str] = []
        drift_cols: list[str] = []

        for col, bstats in dist_baseline.items():
            b_mean = bstats.get("mean")
            if b_mean is None or b_mean == 0:
                continue
            vals = [
                float(row[col])
                for row in records
                if not _is_null(row.get(col)) and _can_cast_float(row.get(col))
            ]
            if not vals:
                continue
            a_mean = sum(vals) / len(vals)
            drift_pct = abs(a_mean - b_mean) / abs(b_mean) * 100.0
            after_dist[col] = {"mean": round(a_mean, 4), "drift_pct": round(drift_pct, 2)}

            if drift_pct >= 20.0:
                drift_cols.append(col)
            if drift_pct > 5.0 and col not in touched:
                warnings.append(
                    f"Column '{col}' mean drifted {drift_pct:.1f}% but agent did not modify it — "
                    "possible false positive fix in a related column."
                )

        passed = len(drift_cols) == 0
        return CheckResult(
            name="distribution_check",
            passed=passed,
            before={c: {"mean": v["mean"]} for c, v in dist_baseline.items() if "mean" in v},
            after=after_dist,
            detail=(
                "Distribution stable across all numeric columns."
                if passed
                else f"Mean drift ≥20% in: {drift_cols}"
            ),
            warnings=warnings,
        )

    def _duplicate_check(
        self,
        records: list[dict],
        baseline: dict,
        profile: dict[str, dict],
    ) -> CheckResult:
        before_count = baseline.get("duplicate_count", 0)
        if before_count == 0:
            return CheckResult(
                name="duplicate_check",
                passed=True,
                before=0,
                after=0,
                detail="No duplicates in baseline.",
            )

        # Find natural key column from profile
        natural_key = None
        for col, p in profile.items():
            if p.get("all_unique") and p["dtype"] != "float":
                col_lower = col.lower()
                if any(h in col_lower for h in ("name", "email", "code", "ref", "id_", "key", "title")):
                    natural_key = col
                    break

        if natural_key is None:
            return CheckResult(
                name="duplicate_check",
                passed=True,
                before=before_count,
                after=0,
                detail="Natural key column not found; cannot recheck duplicates.",
            )

        seen: set[str] = set()
        after_count = 0
        for row in records:
            val = row.get(natural_key)
            if _is_null(val):
                continue
            key_str = str(val).strip().lower()
            if key_str in seen:
                after_count += 1
            else:
                seen.add(key_str)

        passed = after_count < before_count or after_count == 0
        return CheckResult(
            name="duplicate_check",
            passed=passed,
            before=before_count,
            after=after_count,
            detail=(
                f"Duplicates reduced from {before_count} to {after_count}."
                if passed
                else f"Duplicate count unchanged at {after_count}."
            ),
        )

    def _outlier_check(
        self,
        records: list[dict],
        baseline: dict,
        profile: dict[str, dict],
    ) -> CheckResult:
        outlier_cells = baseline.get("outlier_cells", set())
        if not outlier_cells:
            return CheckResult(
                name="outlier_check",
                passed=True,
                before=set(),
                after=set(),
                detail="No outliers in baseline.",
            )

        pk_col = next(
            (k for k in (records[0].keys() if records else []) if k != "_source_format"),
            "id",
        )
        row_map = {int(r[pk_col]): r for r in records if not _is_null(r.get(pk_col))}

        still_outliers: set[tuple] = set()
        for (rid, col) in outlier_cells:
            if col not in profile:
                continue
            p = profile[col]
            mean = p.get("mean")
            std  = p.get("std")
            if mean is None or std is None or std == 0:
                continue
            row = row_map.get(rid)
            if row is None:
                # Row was deleted — outlier resolved
                continue
            val = row.get(col)
            if _is_null(val) or not _can_cast_float(val):
                continue
            z = abs(float(val) - mean) / std
            if z > 5.0:
                still_outliers.add((rid, col))

        passed = len(still_outliers) == 0
        return CheckResult(
            name="outlier_check",
            passed=passed,
            before=len(outlier_cells),
            after=len(still_outliers),
            detail=(
                "All outliers resolved."
                if passed
                else f"{len(still_outliers)} outlier(s) remain: {list(still_outliers)[:5]}"
            ),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
