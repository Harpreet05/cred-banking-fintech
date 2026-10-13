from __future__ import annotations

import asyncio
from pathlib import Path
import sys
from uuid import uuid4

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.graph import CHECKPOINT_DB_PATH, build_agent
from dataset import loan_applications


def _executed_nodes(updates: list[dict[str, object]]) -> list[str]:
    return [node for update in updates for node in update if not node.startswith("__")]


async def run_checkpoint_demo() -> None:
    record_id = loan_applications[0]["record_id"]
    query = f"Check status for {record_id}"
    thread_id = f"checkpoint-demo-{uuid4().hex}"
    config = {"configurable": {"thread_id": thread_id}}

    async with AsyncSqliteSaver.from_conn_string(str(CHECKPOINT_DB_PATH)) as checkpointer:
        graph = build_agent(
            checkpointer=checkpointer,
            interrupt_after=["classify_intent"],
        )
        first_updates = [
            update
            async for update in graph.astream(
                {"query": query},
                config=config,
                stream_mode="updates",
            )
        ]
        first_nodes = _executed_nodes(first_updates)
        paused_state = await graph.aget_state(config)
        if len(first_nodes) < 2 or "classify_intent" not in first_nodes:
            raise RuntimeError(f"Expected two completed nodes before interruption, got {first_nodes}")

        print(f"Paused thread_id: {thread_id}")
        print(f"Nodes executed before pause: {first_nodes}")
        print(
            "Checkpointed state: "
            f"query={paused_state.values.get('query')!r}, "
            f"intent={paused_state.values.get('intent')!r}, "
            f"record_id={paused_state.values.get('record_id')!r}"
        )
        print(f"Next scheduled node(s): {paused_state.next}")

    async with AsyncSqliteSaver.from_conn_string(str(CHECKPOINT_DB_PATH)) as checkpointer:
        resumed_graph = build_agent(
            checkpointer=checkpointer,
            interrupt_after=["classify_intent"],
        )
        restored_state = await resumed_graph.aget_state(config)
        print(
            "Restored from SQLite: "
            f"thread_id={thread_id}, intent={restored_state.values.get('intent')!r}, "
            f"record_id={restored_state.values.get('record_id')!r}"
        )

        resumed_updates = [
            update
            async for update in resumed_graph.astream(
                None,
                config=config,
                stream_mode="updates",
            )
        ]
        resumed_nodes = _executed_nodes(resumed_updates)
        repeated_nodes = sorted(set(first_nodes) & set(resumed_nodes))
        final_state = await resumed_graph.aget_state(config)
        if repeated_nodes:
            raise RuntimeError(f"Completed nodes ran again after resume: {repeated_nodes}")
        if not final_state.values.get("response"):
            raise RuntimeError("Resumed graph did not produce a final response.")

        print(f"Nodes executed after resume: {resumed_nodes}")
        print(f"Previously completed nodes re-executed: {repeated_nodes or 'none'}")
        print(f"Completed response: {final_state.values['response']}")


if __name__ == "__main__":
    asyncio.run(run_checkpoint_demo())