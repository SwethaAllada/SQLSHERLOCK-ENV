# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for SQLSherlock-Env.

Mounts the OpenEnv core WebSocket/HTTP app and adds extra endpoints:
  GET  /health
  GET  /tasks
  POST /upload_dataset
  GET  /download/{file_id}
  /    Gradio dashboard UI (if gradio is installed)
"""

import os
import tempfile
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from openenv.core.env_server import create_app

from models import SQLSherlockAction, SQLSherlockObservation
from server.environment import SQLSherlockEnvironment, TASKS

# ---------------------------------------------------------------------------
# Core OpenEnv app
# ---------------------------------------------------------------------------

app: FastAPI = create_app(
    SQLSherlockEnvironment,      # class (factory), not instance
    SQLSherlockAction,
    SQLSherlockObservation,
    env_name="sqlsherlock_env",
)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    return {
        "status":            "healthy",
        "version":           "1.0.0",
        "timestamp":         time.time(),
        "tasks":             [t["id"] for t in TASKS],
        "supported_formats": ["csv", "json", "jsonl", "parquet", "hf"],
    }


# ---------------------------------------------------------------------------
# /tasks
# ---------------------------------------------------------------------------

@app.get("/tasks")
async def list_tasks() -> list[dict]:
    return [
        {
            "id":          t["id"],
            "name":        t["name"],
            "difficulty":  t["difficulty"],
            "max_steps":   t["max_steps"],
            "description": t["description"],
        }
        for t in TASKS
    ]


# ---------------------------------------------------------------------------
# /upload_dataset
# ---------------------------------------------------------------------------

@app.post("/upload_dataset")
async def upload_dataset(file: UploadFile = File(...)) -> dict:
    """Accept a dataset file, validate it is loadable, return a preview.

    Supported file types: .csv, .json, .jsonl, .parquet
    """
    from server.dataset_loader import load

    filename = file.filename or "upload"
    suffix   = Path(filename).suffix.lower()

    if suffix not in (".csv", ".json", ".jsonl", ".parquet", ".xlsx"):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{suffix}'. "
                "Upload a .csv, .json, .jsonl, .parquet, or .xlsx file."
            ),
        )

    # Save to temp file
    tmp_path = os.path.join(tempfile.gettempdir(), f"sqlsherlock_upload_{filename}")
    try:
        contents = await file.read()
        with open(tmp_path, "wb") as f:
            f.write(contents)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"File save failed: {exc}")

    # Attempt load
    try:
        table_records = load(tmp_path, max_rows=500)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    table_name = list(table_records.keys())[0]
    records    = table_records[table_name]
    columns    = list(records[0].keys()) if records else []

    issue_preview = _quick_issue_preview(records, columns)

    return {
        "dataset_key":              filename,
        "table_name":               table_name,
        "columns":                  columns,
        "row_count":                len(records),
        "detected_issues_preview":  issue_preview,
        "usage_example": (
            f'{{"dataset": "{filename}", '
            f'"task_id": "viz_easy"}}'
        ),
    }


# ---------------------------------------------------------------------------
# /download/{file_id}
# ---------------------------------------------------------------------------

@app.get("/download/{file_id}")
async def download_file(file_id: str) -> FileResponse:
    """Serve a previously exported cleaned dataset file."""
    tmp_dir = tempfile.gettempdir()
    matches = [
        f for f in os.listdir(tmp_dir)
        if f.startswith(file_id)
    ]
    if not matches:
        raise HTTPException(
            status_code=404,
            detail=f"No exported file found for file_id='{file_id}'.",
        )

    filepath = os.path.join(tmp_dir, matches[0])
    filename = matches[0][len(file_id) + 1:]   # strip "{uuid}_" prefix

    return FileResponse(
        path=filepath,
        filename=filename,
        media_type="application/octet-stream",
    )


# ---------------------------------------------------------------------------
# Gradio dashboard — mounted AFTER all API routes so it acts as fallback
# ---------------------------------------------------------------------------

try:
    import gradio as gr
    from server.ui import create_ui

    _gradio_app = create_ui()
    app = gr.mount_gradio_app(app, _gradio_app, path="/")
except ImportError:
    pass  # gradio not installed — API-only mode


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------

def main(host: str = "0.0.0.0", port: int = 7860):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    main(port=args.port)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _quick_issue_preview(records: list[dict], columns: list[str]) -> int:
    """Count obvious null cells for the upload preview."""
    import math
    count = 0
    for row in records:
        for col in columns:
            val = row.get(col)
            if val is None:
                count += 1
            elif isinstance(val, float) and math.isnan(val):
                count += 1
            elif isinstance(val, str) and val.strip() == "":
                count += 1
    return count
