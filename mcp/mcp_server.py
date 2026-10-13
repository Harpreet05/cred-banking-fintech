from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

from fastmcp import FastMCP

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.tool import check_loan_application_status as lookup_loan_application_status


mcp = FastMCP("Cred Banking Support MCP")


@mcp.tool
def check_loan_application_status(record_id: str) -> dict[str, Any]:
    """Look up a loan application's status and escalation assessment by record ID."""
    return lookup_loan_application_status(record_id)


if __name__ == "__main__":
    mcp.run(
        transport="http",
        host="127.0.0.1",
        port=8001,
        path="/mcp",
    )