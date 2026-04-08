# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Shared pytest fixtures for SQLSherlock-Env tests.

All fixtures use in-memory SQLite and synthetic data — no network calls,
no HuggingFace token required.
"""

import sqlite3
import sys
import os
import pytest

# Ensure sqlsherlock_env/ is on the path so absolute imports resolve
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sqlsherlock_env"))


# ---------------------------------------------------------------------------
# Minimal synthetic dataset helpers
# ---------------------------------------------------------------------------

CLEAN_RECORDS = [
    {"id": 1, "name": "Alice",   "age": 30,   "fare": 10.50, "survived": 1},
    {"id": 2, "name": "Bob",     "age": 25,   "fare": 7.25,  "survived": 0},
    {"id": 3, "name": "Carol",   "age": 40,   "fare": 15.00, "survived": 1},
    {"id": 4, "name": "Dave",    "age": 35,   "fare": 8.00,  "survived": 0},
    {"id": 5, "name": "Eve",     "age": 28,   "fare": 12.00, "survived": 1},
    {"id": 6, "name": "Frank",   "age": 45,   "fare": 9.75,  "survived": 0},
    {"id": 7, "name": "Grace",   "age": 33,   "fare": 11.50, "survived": 1},
    {"id": 8, "name": "Heidi",   "age": 29,   "fare": 6.50,  "survived": 0},
    {"id": 9, "name": "Ivan",    "age": 38,   "fare": 13.25, "survived": 1},
    {"id": 10, "name": "Judy",   "age": 22,   "fare": 5.00,  "survived": 0},
]

DIRTY_RECORDS = [
    {"id": 1,  "name": "Alice",  "age": None,         "fare": 10.50,  "survived": 1},   # null age
    {"id": 2,  "name": "Bob",    "age": 25,           "fare": 7.25,   "survived": 0},
    {"id": 3,  "name": "Carol",  "age": "FORTY",      "fare": 15.00,  "survived": 1},   # type error
    {"id": 4,  "name": "Dave",   "age": -5,           "fare": 8.00,   "survived": 0},   # constraint
    {"id": 5,  "name": "Eve",    "age": 28,           "fare": 512.33, "survived": 1},   # outlier (z>5)
    {"id": 6,  "name": "Frank",  "age": 45,           "fare": 9.75,   "survived": 0},
    {"id": 7,  "name": "Grace",  "age": 33,           "fare": 11.50,  "survived": 1},
    {"id": 8,  "name": "Alice",  "age": 29,           "fare": 6.50,   "survived": 0},   # duplicate name
    {"id": 9,  "name": "Ivan",   "age": 38,           "fare": 13.25,  "survived": 1},
    {"id": 10, "name": "Judy",   "age": 22,           "fare": 5.00,   "survived": 0},
]

RAW_CSV_TEXT = (
    "id,name,age,fare,survived\n"
    "1,Alice,,10.50,1\n"
    "2,Bob,25,7.25,0\n"
    "3,Carol,FORTY,15.00,1\n"
    "4,Dave,-5,8.00,0\n"
    "5,Eve,28,512.33,1\n"
    "6,Frank,45,9.75,0\n"
    "7,Grace,33,11.50,1\n"
    "8,Alice,29,6.50,0\n"
    "9,Ivan,38,13.25,1\n"
    "10,Judy,22,5.00,0\n"
)


# ---------------------------------------------------------------------------
# SQLite connection fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clean_conn():
    """In-memory SQLite with clean records."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    _create_table(conn, "passengers", CLEAN_RECORDS)
    yield conn
    conn.close()


@pytest.fixture
def dirty_conn():
    """In-memory SQLite with dirty records (nulls, type errors, constraint, outlier, duplicate)."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    _create_table(conn, "passengers", DIRTY_RECORDS)
    yield conn
    conn.close()


def _create_table(conn: sqlite3.Connection, table: str, records: list[dict]) -> None:
    conn.execute(f'DROP TABLE IF EXISTS "{table}"')
    conn.execute(
        f'CREATE TABLE "{table}" '
        f'(id INTEGER, name TEXT, age TEXT, fare REAL, survived INTEGER)'
    )
    for r in records:
        conn.execute(
            f'INSERT INTO "{table}" VALUES (?, ?, ?, ?, ?)',
            (r["id"], r["name"], r.get("age"), r.get("fare"), r.get("survived")),
        )
    conn.commit()


# ---------------------------------------------------------------------------
# Profile fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def dirty_profile():
    """Column profile computed from DIRTY_RECORDS."""
    from server.schema_profiler import profile_table
    return profile_table("passengers", DIRTY_RECORDS)


@pytest.fixture
def clean_profile():
    """Column profile computed from CLEAN_RECORDS."""
    from server.schema_profiler import profile_table
    return profile_table("passengers", CLEAN_RECORDS)


# ---------------------------------------------------------------------------
# DatabaseEngine fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db_task1():
    """DatabaseEngine for task1 loaded from raw CSV text."""
    from server.database import DatabaseEngine
    db = DatabaseEngine(
        task_id="task1_null_and_types",
        seed=42,
        dataset_source=RAW_CSV_TEXT,
        max_rows=50,
    )
    return db


@pytest.fixture
def db_task2():
    """DatabaseEngine for task2 loaded from raw CSV text."""
    from server.database import DatabaseEngine
    db = DatabaseEngine(
        task_id="task2_constraints_and_fk",
        seed=42,
        dataset_source=RAW_CSV_TEXT,
        max_rows=50,
    )
    return db


@pytest.fixture
def db_task3():
    """DatabaseEngine for task3 loaded from raw CSV text."""
    from server.database import DatabaseEngine
    db = DatabaseEngine(
        task_id="task3_full_audit_with_trap",
        seed=42,
        dataset_source=RAW_CSV_TEXT,
        max_rows=50,
    )
    return db


# ---------------------------------------------------------------------------
# Issue registry fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def task1_issues(dirty_conn, dirty_profile):
    """Issues detected for task1 on the dirty dataset."""
    from server.issue_detector import detect_issues
    import copy
    records = copy.deepcopy(DIRTY_RECORDS)
    return detect_issues(
        conn=dirty_conn,
        profile=dirty_profile,
        records=records,
        task_id="task1_null_and_types",
        seed=42,
    )


@pytest.fixture
def task3_issues(dirty_conn, dirty_profile):
    """Issues detected for task3 on the dirty dataset."""
    from server.issue_detector import detect_issues
    import copy
    records = copy.deepcopy(DIRTY_RECORDS)
    return detect_issues(
        conn=dirty_conn,
        profile=dirty_profile,
        records=records,
        task_id="task3_full_audit_with_trap",
        seed=42,
    )
