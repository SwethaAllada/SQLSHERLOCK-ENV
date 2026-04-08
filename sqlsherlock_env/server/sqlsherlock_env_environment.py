# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
MCP-enabled SQLSherlock environment.

Exposes all agent actions as MCP tools that any MCP-compatible LLM
(Claude, GPT, etc.) can discover and invoke dynamically via
ListToolsAction / CallToolAction.

This adds MCP tool discoverability on top of the existing WebSocket/HTTP API.
"""

from typing import Any, Optional

from fastmcp import FastMCP

from openenv.core.env_server.mcp_environment import MCPEnvironment
from openenv.core.env_server.types import Action

from models import SQLSherlockAction, SQLSherlockObservation, SQLSherlockState
from server.environment import SQLSherlockEnvironment


# ---------------------------------------------------------------------------
# FastMCP server — data-quality investigation tools
# ---------------------------------------------------------------------------

mcp = FastMCP("sqlsherlock")


@mcp.tool()
def inspect_table(table: str) -> str:
    """View all rows in a database table.

    Args:
        table: Name of the table to inspect (e.g. 'titanic').
    """
    return f"inspect:{table}"


@mcp.tool()
def profile_column(table: str, column: str) -> str:
    """Get statistical profile: mean, std, min, max, null_count, z-scores.

    IMPORTANT: Always call this BEFORE fixing any numeric value.
    z > 5 = real outlier (fix it). z < 3 = normal (DO NOT touch).

    Args:
        table:  Table name.
        column: Column to profile.
    """
    return f"profile:{table}:{column}"


@mcp.tool()
def run_sql(sql: str) -> str:
    """Execute a read-only SELECT SQL query to investigate data quality.

    Args:
        sql: A SELECT query string. No write operations allowed.
    """
    return f"sql:{sql}"


@mcp.tool()
def fix_cell(table: str, row_id: int, column: str, value: str, reason: str) -> str:
    """Fix a data quality issue in one cell.

    Args:
        table:  Table name.
        row_id: Primary key of the row.
        column: Column to fix.
        value:  Corrected value to write.
        reason: Statistical justification (e.g. 'median=29.0, z-score=N/A').
    """
    return f"fix:{table}:{row_id}:{column}:{value}"


@mcp.tool()
def delete_row(table: str, row_id: int, reason: str) -> str:
    """Delete a duplicate or FK-violation row.

    Args:
        table:  Table name.
        row_id: Primary key to delete.
        reason: Why this row should be removed.
    """
    return f"delete:{table}:{row_id}"


@mcp.tool()
def validate_data() -> str:
    """Run all 6 validation checks comparing current vs raw baseline.

    Returns pass/partial/fail for: null_check, type_check, range_check,
    distribution_check, duplicate_check, outlier_check.
    """
    return "validate"


@mcp.tool()
def submit_investigation() -> str:
    """Submit the investigation for final scoring. Call after all fixes."""
    return "submit"


# ---------------------------------------------------------------------------
# MCP Environment class
# ---------------------------------------------------------------------------

class SQLSherlockMCPEnvironment(MCPEnvironment):
    """SQLSherlock environment with MCP tool discoverability.

    Wraps SQLSherlockEnvironment and exposes all actions as MCP tools.
    MCP agents call ListToolsAction to discover tools, then CallToolAction
    to invoke them.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        super().__init__(mcp_server=mcp)
        self._env = SQLSherlockEnvironment()

    @property
    def state(self) -> SQLSherlockState:
        return self._env.state

    def reset(self, **kwargs) -> SQLSherlockObservation:
        return self._env.reset(**kwargs)

    def _step_impl(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> SQLSherlockObservation:
        """Handle standard SQLSherlock actions (non-MCP)."""
        if isinstance(action, SQLSherlockAction):
            return self._env.step(action, **kwargs)

        # Fallback: construct from dict
        if hasattr(action, "model_dump"):
            d = action.model_dump()
        elif isinstance(action, dict):
            d = action
        else:
            d = {"action_type": "inspect"}

        sa = SQLSherlockAction(**{k: v for k, v in d.items() if v is not None})
        return self._env.step(sa, **kwargs)
