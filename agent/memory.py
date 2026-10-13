import json
from pathlib import Path
from typing import Any


DEFAULT_MEMORY_FILE = Path(__file__).with_name("conversation_history.json")


def _read_memory(memory_file: Path) -> dict[str, Any]:
    if not memory_file.exists():
        return {}
    return json.loads(memory_file.read_text(encoding="utf-8"))


def load_conversation(
    conversation_id: str,
    memory_file: Path = DEFAULT_MEMORY_FILE,
) -> dict[str, Any]:
    """Load one conversation, returning empty history when it is new."""
    return _read_memory(memory_file).get(
        conversation_id,
        {"messages": [], "record_id": ""},
    )


def save_turn(
    conversation_id: str,
    query: str,
    response: str,
    record_id: str = "",
    memory_file: Path = DEFAULT_MEMORY_FILE,
) -> None:
    """Append a user/assistant turn and persist its conversation context."""
    memory = _read_memory(memory_file)
    conversation = memory.setdefault(
        conversation_id,
        {"messages": [], "record_id": ""},
    )
    conversation["messages"].extend([
        {"role": "user", "content": query},
        {"role": "assistant", "content": response},
    ])
    if record_id:
        conversation["record_id"] = record_id

    memory_file.parent.mkdir(parents=True, exist_ok=True)
    memory_file.write_text(json.dumps(memory, indent=2), encoding="utf-8")


def clear_conversation(
    conversation_id: str,
    memory_file: Path = DEFAULT_MEMORY_FILE,
) -> None:
    """Delete one conversation without affecting other saved conversations."""
    memory = _read_memory(memory_file)
    memory.pop(conversation_id, None)
    memory_file.parent.mkdir(parents=True, exist_ok=True)
    memory_file.write_text(json.dumps(memory, indent=2), encoding="utf-8")
