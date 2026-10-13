from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

from agent.guardrails import mask_pii

DEFAULT_LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "requests.jsonl"


def mask_for_logging(text: str | None) -> str:
    """Mask fixed-format PII before anything is written to disk."""
    if text is None:
        return ""
    return mask_pii(str(text))


def write_request_log(
    path: str | Path = DEFAULT_LOG_PATH,
    *,
    method: str = "",
    route: str = "",
    request_text: str = "",
    trace_id: str | None = None,
    started_at: float | None = None,
    finished_at: float | None = None,
    status_code: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write one JSON-lines request log entry with the request text masked."""
    current_time = perf_counter()
    start = started_at if started_at is not None else current_time
    end = finished_at if finished_at is not None else current_time
    duration_ms = max((end - start) * 1000.0, 0.0)

    entry: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "trace_id": trace_id or str(uuid4()),
        "method": str(method).upper(),
        "route": route,
        "request_text": mask_for_logging(request_text),
        "duration_ms": round(duration_ms, 3),
    }
    if status_code is not None:
        entry["status_code"] = int(status_code)
    if extra:
        entry.update(extra)

    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return entry
