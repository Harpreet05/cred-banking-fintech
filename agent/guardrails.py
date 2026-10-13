import re
from typing import Any


PII_PATTERNS = [
	(re.compile(r"\b(?:(?:bank\s+)?account(?:\s+(?:number|no\.?))?|a/c(?:\s+(?:number|no\.?))?)\s*[:#-]?\s*\d{9,18}\b", re.IGNORECASE), "[ACCOUNT_REDACTED]"),
	(re.compile(r"(?<!\d)[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}(?!\d)"), "[AADHAAR_REDACTED]"),
	(re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b", re.IGNORECASE), "[PAN_REDACTED]"),
	(re.compile(r"(?<!\d)(?:\+91[ -]?)?[6-9]\d{9}(?!\d)"), "[PHONE_REDACTED]"),
]

INJECTION_PATTERNS = [
	re.compile(r"ignore\s+(all\s+)?(previous|prior)\s+instructions", re.IGNORECASE),
	re.compile(r"disregard\s+(all\s+)?(previous|prior)\s+instructions", re.IGNORECASE),
	re.compile(r"reveal\s+(the\s+)?(system|developer)\s+prompt", re.IGNORECASE),
	re.compile(r"bypass\s+(the\s+)?(guardrails|safety|rules)", re.IGNORECASE),
]

GROUNDING_REFUSAL = "I don't know. The retrieved context does not support an answer to this question."


def mask_pii(text: str) -> str:
	"""Mask Aadhaar, PAN, labeled bank accounts, and Indian mobile numbers."""
	for pattern, replacement in PII_PATTERNS:
		text = pattern.sub(replacement, text)
	return text


def is_prompt_injection(text: str) -> bool:
	"""Detect common attempts to override instructions or reveal hidden prompts."""
	return any(pattern.search(text) for pattern in INJECTION_PATTERNS)


def check_groundedness(result: dict[str, Any], threshold: float = 0.4530) -> dict[str, Any]:
	"""Allow a RAG answer only when retrieval supplied sufficiently similar context."""
	grounded = (
		not result.get("fallback", True)
		and bool(result.get("used_context"))
		and result.get("top_1_similarity", 0.0) >= threshold
	)
	return {
		"grounded": grounded,
		"answer": result["answer"] if grounded else GROUNDING_REFUSAL,
	}
