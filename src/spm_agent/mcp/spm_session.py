# src/spm_agent/mcp/spm_session.py
from __future__ import annotations

import json
from typing import Any

from spm_agent.mcp.spm_mcp_client import SPMMCPClient

_client: SPMMCPClient | None = None


def spm() -> SPMMCPClient:
    """Process-wide client: the tool registry is fetched once per session."""
    global _client
    if _client is None:
        _client = SPMMCPClient()
    return _client


class InstrumentError(RuntimeError):
    """The instrument refused or failed a command (ok=False)."""


def unwrap(result: Any, tool: str) -> dict:
    """ToolOutcome envelope -> data dict. Raises loudly on ok=False."""
    if isinstance(result, (list, tuple)):
        result = result[0]
    if isinstance(result, dict) and "text" in result:
        result = json.loads(result["text"])
    elif isinstance(result, str):
        result = json.loads(result)

    if not result.get("ok"):
        err = result.get("error", {}) or {}
        raise InstrumentError(
            f"{tool}: {err.get('code', 'unknown')}: {err.get('message', result)}")
    return result.get("data", {})


async def call(tool: str, args: dict | None = None) -> dict:
    """Call an SPM MCP tool and return its `data` payload."""
    return unwrap(await spm().call_tool(tool, args), tool)