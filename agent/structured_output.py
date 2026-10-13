from typing import Any

from jsonschema import validate


AGENT_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": ["rag", "status"]},
        "answer": {"type": "string", "minLength": 1},
        "fallback": {"type": "boolean"},
        "record_id": {"type": ["string", "null"]},
        "escalation_score": {
            "type": ["number", "null"],
            "minimum": 0,
            "maximum": 1,
        },
        "escalate": {"type": ["boolean", "null"]},
    },
    "required": [
        "intent",
        "answer",
        "fallback",
        "record_id",
        "escalation_score",
        "escalate",
    ],
    "additionalProperties": False,
}


def validate_agent_response(response: dict[str, Any]) -> dict[str, Any]:
    """Validate and return one structured agent response."""
    validate(instance=response, schema=AGENT_RESPONSE_SCHEMA)
    return response
