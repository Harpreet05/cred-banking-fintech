from __future__ import annotations

import asyncio
from pathlib import Path
import sys
from time import perf_counter
from typing import TypedDict

from langgraph.errors import NodeTimeoutError
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy, TimeoutPolicy

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.graph import (
    GLOBAL_GRAPH_TIMEOUT_SECONDS,
    LOAN_STATUS_NODE_TIMEOUT_SECONDS,
    LOAN_STATUS_RETRY_POLICY,
    GraphRunTimeoutError,
    invoke_with_global_timeout,
)


class DemoState(TypedDict, total=False):
    retry_succeeded: bool
    fast_step_completed: bool
    slow_step_completed: bool
    downstream_completed: bool


class TransientFailure(Exception):
    pass


def demonstrate_retry_recovery() -> None:
    attempts = 0

    def flaky_node(state: DemoState) -> dict[str, bool]:
        nonlocal attempts
        attempts += 1
        print(f"Transient node attempt {attempts}/{LOAN_STATUS_RETRY_POLICY.max_attempts}")
        if attempts < 3:
            raise TransientFailure("Simulated temporary dependency failure")
        return {"retry_succeeded": True}

    builder = StateGraph(DemoState)
    builder.add_node("transient_node", flaky_node, retry_policy=LOAN_STATUS_RETRY_POLICY)
    builder.add_edge(START, "transient_node")
    builder.add_edge("transient_node", END)
    result = builder.compile().invoke({})
    if attempts != 3 or not result.get("retry_succeeded"):
        raise RuntimeError(f"Retry policy did not recover as expected: attempts={attempts}, result={result}")
    print(
        "Retry recovered: "
        f"attempts={attempts}, initial_interval={LOAN_STATUS_RETRY_POLICY.initial_interval}s, "
        f"max_interval={LOAN_STATUS_RETRY_POLICY.max_interval}s, "
        f"backoff_factor={LOAN_STATUS_RETRY_POLICY.backoff_factor}, "
        f"jitter={LOAN_STATUS_RETRY_POLICY.jitter}"
    )


async def demonstrate_node_timeout() -> None:
    cancellation = {"observed": False}

    async def slow_node(state: DemoState) -> dict[str, bool]:
        try:
            await asyncio.sleep(2.0)
        except asyncio.CancelledError:
            cancellation["observed"] = True
            raise
        return {"slow_step_completed": True}

    builder = StateGraph(DemoState)
    builder.add_node(
        "slow_node",
        slow_node,
        timeout=TimeoutPolicy(run_timeout=0.05),
    )
    builder.add_edge(START, "slow_node")
    builder.add_edge("slow_node", END)
    started_at = perf_counter()
    try:
        await builder.compile().ainvoke({})
    except NodeTimeoutError as error:
        elapsed = perf_counter() - started_at
        if not cancellation["observed"] or elapsed >= 1.0:
            raise RuntimeError("The node timeout did not cancel the simulated slow node cleanly.") from error
        print(f"Per-node timeout fired cleanly: {error}; elapsed={elapsed:.3f}s; node_cancelled=True")
    else:
        raise RuntimeError("The simulated slow node unexpectedly completed without timing out.")


async def demonstrate_global_timeout() -> None:
    execution = {"slow_node_cancelled": False, "downstream_ran": False}

    async def fast_node(state: DemoState) -> dict[str, bool]:
        return {"fast_step_completed": True}

    async def slow_node(state: DemoState) -> dict[str, bool]:
        try:
            await asyncio.sleep(2.0)
        except asyncio.CancelledError:
            execution["slow_node_cancelled"] = True
            raise
        return {"slow_step_completed": True}

    async def downstream_node(state: DemoState) -> dict[str, bool]:
        execution["downstream_ran"] = True
        return {"downstream_completed": True}

    builder = StateGraph(DemoState)
    builder.add_node("fast_node", fast_node)
    builder.add_node("slow_node", slow_node)
    builder.add_node("downstream_node", downstream_node)
    builder.add_edge(START, "fast_node")
    builder.add_edge("fast_node", "slow_node")
    builder.add_edge("slow_node", "downstream_node")
    builder.add_edge("downstream_node", END)

    started_at = perf_counter()
    try:
        await invoke_with_global_timeout(
            builder.compile(),
            {},
            timeout_seconds=0.05,
        )
    except GraphRunTimeoutError as error:
        elapsed = perf_counter() - started_at
        if not execution["slow_node_cancelled"] or execution["downstream_ran"] or elapsed >= 1.0:
            raise RuntimeError("The global timeout did not cancel the complete graph run.") from error
        print(
            f"Global graph timeout fired cleanly: {error}; elapsed={elapsed:.3f}s; "
            "slow_node_cancelled=True; downstream_node_ran=False"
        )
    else:
        raise RuntimeError("The simulated slow graph unexpectedly completed without timing out.")


async def run_resilience_demo() -> None:
    print(
        f"Production policies: loan_status_tool timeout={LOAN_STATUS_NODE_TIMEOUT_SECONDS}s; "
        f"global graph timeout={GLOBAL_GRAPH_TIMEOUT_SECONDS}s"
    )
    demonstrate_retry_recovery()
    await demonstrate_node_timeout()
    await demonstrate_global_timeout()


if __name__ == "__main__":
    asyncio.run(run_resilience_demo())