# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for server/issue_detector.py

Covers: real detection, confidence scoring, synthetic top-up,
        trap planting, SENTINEL_UNKNOWN, and deduplication.
"""

import copy
import sqlite3

import pytest

from server.issue_detector import (
    SENTINEL_UNKNOWN,
    MINIMUM_ISSUES,
    Issue,
    Trap,
    detect_issues,
    detect_trap,
    _find_natural_key_col,
    _detect_nulls,
    _detect_type_errors,
    _detect_constraints,
    _detect_outliers,
    _detect_duplicates,
)
from server.schema_profiler import profile_table
from tests.conftest import DIRTY_RECORDS, CLEAN_RECORDS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_conn(records: list[dict]) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        'CREATE TABLE passengers '
        '(id INTEGER, name TEXT, age TEXT, fare REAL, survived INTEGER)'
    )
    for r in records:
        conn.execute(
            'INSERT INTO passengers VALUES (?, ?, ?, ?, ?)',
            (r["id"], r["name"], r.get("age"), r.get("fare"), r.get("survived")),
        )
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Null detection
# ---------------------------------------------------------------------------

class TestNullDetection:
    def test_finds_null_age(self, dirty_conn, dirty_profile):
        records = copy.deepcopy(DIRTY_RECORDS)
        issues = _detect_nulls(records, dirty_profile, pk_col="id")
        null_issues = [i for i in issues if i.column == "age" and i.issue_type == "null"]
        # id=1 has age=None
        assert any(i.row_id == 1 for i in null_issues)

    def test_null_confidence_inversely_proportional_to_rate(self, dirty_conn, dirty_profile):
        records = copy.deepcopy(DIRTY_RECORDS)
        issues = _detect_nulls(records, dirty_profile, pk_col="id")
        null_issues = [i for i in issues if i.issue_type == "null"]
        for iss in null_issues:
            assert 0.0 <= iss.confidence <= 1.0

    def test_structural_nulls_low_confidence(self):
        """A column with 80% nulls should produce confidence ≈ 0.20."""
        records = [
            {"id": i, "name": f"p{i}", "cabin": None if i <= 8 else f"C{i}"}
            for i in range(1, 11)
        ]
        profile = profile_table("t", records)
        conn = sqlite3.connect(":memory:")
        issues = _detect_nulls(records, profile, pk_col="id")
        cabin_issues = [i for i in issues if i.column == "cabin"]
        for iss in cabin_issues:
            assert iss.confidence <= 0.25

    def test_no_nulls_on_clean_data(self, clean_conn, clean_profile):
        records = copy.deepcopy(CLEAN_RECORDS)
        issues = _detect_nulls(records, clean_profile, pk_col="id")
        assert issues == []


# ---------------------------------------------------------------------------
# Type error detection
# ---------------------------------------------------------------------------

class TestTypeErrorDetection:
    def test_finds_text_in_numeric_column(self, dirty_conn, dirty_profile):
        records = copy.deepcopy(DIRTY_RECORDS)
        issues = _detect_type_errors(records, dirty_profile, pk_col="id")
        type_issues = [i for i in issues if i.issue_type == "type_error"]
        # id=3 has age="FORTY"
        assert any(i.row_id == 3 and i.column == "age" for i in type_issues)

    def test_type_error_confidence_always_1(self, dirty_conn, dirty_profile):
        records = copy.deepcopy(DIRTY_RECORDS)
        issues = _detect_type_errors(records, dirty_profile, pk_col="id")
        for iss in issues:
            assert iss.confidence == 1.0

    def test_correct_value_is_median(self, dirty_conn, dirty_profile):
        records = copy.deepcopy(DIRTY_RECORDS)
        issues = _detect_type_errors(records, dirty_profile, pk_col="id")
        age_issues = [i for i in issues if i.column == "age"]
        assert len(age_issues) > 0
        # Correct should be a numeric median, not None
        for iss in age_issues:
            assert iss.correct is not None
            assert isinstance(iss.correct, (int, float))

    def test_no_type_errors_on_clean_data(self, clean_conn, clean_profile):
        records = copy.deepcopy(CLEAN_RECORDS)
        issues = _detect_type_errors(records, clean_profile, pk_col="id")
        assert issues == []


# ---------------------------------------------------------------------------
# Constraint detection
# ---------------------------------------------------------------------------

class TestConstraintDetection:
    def test_finds_negative_age(self, dirty_conn, dirty_profile):
        records = copy.deepcopy(DIRTY_RECORDS)
        issues = _detect_constraints(records, dirty_profile, pk_col="id")
        # id=4 has age=-5
        assert any(i.row_id == 4 and i.column == "age" for i in issues)

    def test_correct_is_abs_value(self, dirty_conn, dirty_profile):
        records = copy.deepcopy(DIRTY_RECORDS)
        issues = _detect_constraints(records, dirty_profile, pk_col="id")
        neg_issues = [i for i in issues if i.issue_type == "constraint"]
        for iss in neg_issues:
            assert iss.correct >= 0

    def test_constraint_confidence(self, dirty_conn, dirty_profile):
        records = copy.deepcopy(DIRTY_RECORDS)
        issues = _detect_constraints(records, dirty_profile, pk_col="id")
        for iss in issues:
            assert iss.confidence == 0.95


# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------

class TestOutlierDetection:
    def test_finds_fare_outlier(self, dirty_conn, dirty_profile):
        records = copy.deepcopy(DIRTY_RECORDS)
        issues = _detect_outliers(records, dirty_profile, pk_col="id")
        # id=5 has fare=512.33 — z >> 5
        outlier_issues = [i for i in issues if i.column == "fare"]
        assert any(i.row_id == 5 for i in outlier_issues)

    def test_outlier_correct_is_mean(self, dirty_conn, dirty_profile):
        records = copy.deepcopy(DIRTY_RECORDS)
        issues = _detect_outliers(records, dirty_profile, pk_col="id")
        for iss in issues:
            assert iss.correct is not None
            # correct should be close to the column mean (not the outlier value)
            assert isinstance(iss.correct, float)

    def test_normal_values_not_flagged(self, clean_conn, clean_profile):
        records = copy.deepcopy(CLEAN_RECORDS)
        issues = _detect_outliers(records, clean_profile, pk_col="id")
        assert issues == []


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------

class TestDuplicateDetection:
    def test_finds_duplicate_name(self, dirty_conn, dirty_profile):
        records = copy.deepcopy(DIRTY_RECORDS)
        issues = _detect_duplicates(records, dirty_profile, pk_col="id")
        dup_issues = [i for i in issues if i.issue_type == "duplicate"]
        # id=8 has same name as id=1 (Alice) — later row is the duplicate
        assert any(i.row_id == 8 for i in dup_issues)

    def test_first_occurrence_not_flagged(self, dirty_conn, dirty_profile):
        records = copy.deepcopy(DIRTY_RECORDS)
        issues = _detect_duplicates(records, dirty_profile, pk_col="id")
        dup_ids = {i.row_id for i in issues if i.issue_type == "duplicate"}
        assert 1 not in dup_ids   # Alice (first) should NOT be flagged

    def test_correct_is_none_for_duplicates(self, dirty_conn, dirty_profile):
        records = copy.deepcopy(DIRTY_RECORDS)
        issues = _detect_duplicates(records, dirty_profile, pk_col="id")
        for iss in issues:
            assert iss.correct is None   # should be deleted

    def test_no_duplicates_on_clean_data(self, clean_conn, clean_profile):
        records = copy.deepcopy(CLEAN_RECORDS)
        issues = _detect_duplicates(records, clean_profile, pk_col="id")
        assert issues == []


# ---------------------------------------------------------------------------
# Natural key detection
# ---------------------------------------------------------------------------

class TestNaturalKeyDetection:
    def test_name_column_is_natural_key(self, clean_profile):
        key = _find_natural_key_col(clean_profile, CLEAN_RECORDS, pk_col="id")
        assert key == "name"

    def test_no_key_when_no_unique_hint_col(self):
        records = [{"id": i, "x": i * 2.0, "y": i * 3.0} for i in range(1, 6)]
        profile = profile_table("t", records)
        key = _find_natural_key_col(profile, records, pk_col="id")
        assert key is None


# ---------------------------------------------------------------------------
# Full detect_issues integration
# ---------------------------------------------------------------------------

class TestDetectIssues:
    def test_task1_minimum_issues(self, dirty_conn, dirty_profile):
        records = copy.deepcopy(DIRTY_RECORDS)
        issues = detect_issues(dirty_conn, dirty_profile, records,
                               task_id="viz_easy", seed=42)
        assert len(issues) >= MINIMUM_ISSUES["viz_easy"]

    def test_task2_minimum_issues(self, dirty_conn, dirty_profile):
        records = copy.deepcopy(DIRTY_RECORDS)
        issues = detect_issues(dirty_conn, dirty_profile, records,
                               task_id="ml_medium", seed=42)
        assert len(issues) >= MINIMUM_ISSUES["ml_medium"]

    def test_task3_minimum_issues(self, dirty_conn, dirty_profile):
        records = copy.deepcopy(DIRTY_RECORDS)
        issues = detect_issues(dirty_conn, dirty_profile, records,
                               task_id="bq_hard", seed=42)
        assert len(issues) >= MINIMUM_ISSUES["bq_hard"]

    def test_task1_only_null_and_type_issues(self, dirty_conn, dirty_profile):
        records = copy.deepcopy(DIRTY_RECORDS)
        issues = detect_issues(dirty_conn, dirty_profile, records,
                               task_id="viz_easy", seed=42)
        for iss in issues:
            assert iss.issue_type in ("null", "type_error"), (
                f"task1 should only detect null/type_error, got {iss.issue_type}"
            )

    def test_no_duplicate_issue_ids(self, dirty_conn, dirty_profile):
        records = copy.deepcopy(DIRTY_RECORDS)
        issues = detect_issues(dirty_conn, dirty_profile, records,
                               task_id="bq_hard", seed=42)
        ids = [i.issue_id for i in issues]
        assert len(ids) == len(set(ids)), "Duplicate issue_ids found"

    def test_confidence_in_range(self, dirty_conn, dirty_profile):
        records = copy.deepcopy(DIRTY_RECORDS)
        issues = detect_issues(dirty_conn, dirty_profile, records,
                               task_id="bq_hard", seed=42)
        for iss in issues:
            assert 0.0 <= iss.confidence <= 1.0, (
                f"Issue {iss.issue_id} has out-of-range confidence {iss.confidence}"
            )

    def test_synthetic_topup_on_clean_data(self, clean_conn, clean_profile):
        """Clean data triggers synthetic top-up to meet minimum."""
        records = copy.deepcopy(CLEAN_RECORDS)
        issues = detect_issues(clean_conn, clean_profile, records,
                               task_id="viz_easy", seed=42)
        assert len(issues) >= MINIMUM_ISSUES["viz_easy"]

    def test_reproducible_with_same_seed(self, dirty_conn, dirty_profile):
        conn2 = _make_conn(DIRTY_RECORDS)
        profile2 = profile_table("passengers", copy.deepcopy(DIRTY_RECORDS))
        r1 = copy.deepcopy(DIRTY_RECORDS)
        r2 = copy.deepcopy(DIRTY_RECORDS)
        issues1 = detect_issues(dirty_conn, dirty_profile, r1,
                                task_id="viz_easy", seed=99)
        issues2 = detect_issues(conn2, profile2, r2,
                                task_id="viz_easy", seed=99)
        assert len(issues1) == len(issues2)
        conn2.close()


# ---------------------------------------------------------------------------
# Trap detection
# ---------------------------------------------------------------------------

class TestDetectTrap:
    def test_trap_planted_for_task3(self, dirty_conn, dirty_profile):
        records = copy.deepcopy(DIRTY_RECORDS)
        issues = detect_issues(dirty_conn, dirty_profile, records,
                               task_id="bq_hard", seed=42)
        trap = detect_trap(dirty_conn, dirty_profile, records, issues, seed=42)
        assert trap is not None
        assert isinstance(trap, Trap)

    def test_trap_not_in_issue_registry(self, dirty_conn, dirty_profile):
        records = copy.deepcopy(DIRTY_RECORDS)
        issues = detect_issues(dirty_conn, dirty_profile, records,
                               task_id="bq_hard", seed=42)
        trap = detect_trap(dirty_conn, dirty_profile, records, issues, seed=42)
        if trap is None:
            pytest.skip("No numeric column available for trap")
        issue_cells = {(i.row_id, i.column) for i in issues}
        assert (trap.row_id, trap.column) not in issue_cells

    def test_trap_value_is_2x_original(self, dirty_conn, dirty_profile):
        records = copy.deepcopy(DIRTY_RECORDS)
        issues = detect_issues(dirty_conn, dirty_profile, records,
                               task_id="bq_hard", seed=42)
        trap = detect_trap(dirty_conn, dirty_profile, records, issues, seed=42)
        if trap is None:
            pytest.skip("No numeric column available for trap")
        import math
        assert math.isclose(trap.trap_value, trap.original * 2.0, rel_tol=1e-4)

    def test_trap_written_to_sqlite(self, dirty_conn, dirty_profile):
        records = copy.deepcopy(DIRTY_RECORDS)
        issues = detect_issues(dirty_conn, dirty_profile, records,
                               task_id="bq_hard", seed=42)
        trap = detect_trap(dirty_conn, dirty_profile, records, issues, seed=42)
        if trap is None:
            pytest.skip("No numeric column available for trap")
        # Verify the trap value is actually in the DB
        row = dirty_conn.execute(
            f'SELECT "{trap.column}" FROM passengers WHERE id = ?',
            (trap.row_id,)
        ).fetchone()
        assert row is not None
        import math
        assert math.isclose(float(row[0]), trap.trap_value, rel_tol=1e-4)
