from __future__ import annotations

import asyncio
from dataclasses import asdict, is_dataclass
import json
import os
from pathlib import Path
import sys
from typing import Any

from fastmcp import Client

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import loan_applications


MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "http://127.0.0.1:8001/mcp")
RECORD_IDS = [application["record_id"] for application in loan_applications[:2]]


def _content_block_payload(block: Any) -> Any:
    if is_dataclass(block):
        return asdict(block)
    if hasattr(block, "model_dump"):
        return block.model_dump(mode="json", by_alias=True)
    return str(block)


def _mcp_response_payload(result: Any) -> dict[str, Any]:
    return {
        "content": [_content_block_payload(block) for block in result.content],
        "structuredContent": result.structured_content,
        "_meta": result.meta,
        "isError": result.is_error,
    }


async def main() -> None:
    if len(set(RECORD_IDS)) < 2:
        raise RuntimeError("The dataset must provide at least two distinct loan record IDs.")

    async with Client(MCP_SERVER_URL) as client:
        for record_id in RECORD_IDS:
            result = await client.call_tool(
                "check_loan_application_status",
                {"record_id": record_id},
            )
            if result.is_error:
                raise RuntimeError(f"MCP tool call failed for {record_id}: {result.content}")
            print(f"MCP response for {record_id}:")
            print(json.dumps(_mcp_response_payload(result), indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    asyncio.run(main())