from __future__ import annotations

from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
DOCS_DIR = BASE_DIR / "docs"


def _read_markdown_document(path: Path) -> dict[str, str]:
    """Read one markdown document and return a doc_id/title/text payload."""
    raw = path.read_text(encoding="utf-8").strip()

    if raw.startswith("---"):
        lines = raw.splitlines()
        if lines and lines[0].strip() == "---":
            end_index = next(
                (idx for idx, line in enumerate(lines[1:], start=1) if line.strip() == "---"),
                None,
            )
            if end_index is not None:
                metadata_lines = lines[1:end_index]
                body = "\n".join(lines[end_index + 1 :]).strip()
                metadata: dict[str, str] = {}
                for line in metadata_lines:
                    if ":" in line:
                        key, value = line.split(":", 1)
                        metadata[key.strip()] = value.strip()
                return {
                    "doc_id": metadata.get("doc_id", path.stem),
                    "title": metadata.get("title", path.stem.replace("-", " ").title()),
                    "text": body,
                }

    return {
        "doc_id": path.stem,
        "title": path.stem.replace("-", " ").title(),
        "text": raw,
    }


def load_knowledge_base_documents(directory: Path | str = DOCS_DIR) -> list[dict[str, str]]:
    """Load all markdown knowledge-base documents from a folder."""
    base_path = Path(directory)
    documents = [
        _read_markdown_document(path)
        for path in sorted(base_path.glob("*.md"))
        if path.is_file()
    ]
    if not documents:
        raise FileNotFoundError(f"No knowledge-base markdown files found in {base_path}")
    return documents


KNOWLEDGE_BASE = load_knowledge_base_documents()
