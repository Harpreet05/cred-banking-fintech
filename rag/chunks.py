from __future__ import annotations
from typing import Dict, List
import re

def chunk_by_fixed_size(text, chunk_size = 75,  overlap = 15) -> List[str]:
    """Split the text into fixed size chunks with a configurable overlap."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []

    chunks: List[str] = []
    start = 0
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(cleaned):
            break
        start = end - overlap
    return chunks

def chunk_by_sentence(text: str, max_chars: int = 220) -> List[str]:
    """Split text into sentence groups whose combined length does not exceed max_chars."""
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text) if sentence.strip()]
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for sentence in sentences:
        sentence_len = len(sentence)
        if current_len and current_len + sentence_len > max_chars:
            chunks.append(" ".join(current))
            current = [sentence]
            current_len = sentence_len
        else:
            current.append(sentence)
            current_len += sentence_len

    if current:
        chunks.append(" ".join(current))
    return chunks


def build_chunk_sets(documents: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    """Build chunk records for both fixed-size and sentence-based retrieval strategies."""
    fixed_chunks: List[Dict[str, str]] = []
    sentence_chunks: List[Dict[str, str]] = []

    for doc in documents:
        for chunk_text in chunk_by_fixed_size(doc["text"]):
            fixed_chunks.append({
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "text": chunk_text,
                "chunk_id": f"{doc['doc_id']}_fixed_{len(fixed_chunks)}",
            })

        for chunk_text in chunk_by_sentence(doc["text"]):
            sentence_chunks.append({
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "text": chunk_text,
                "chunk_id": f"{doc['doc_id']}_sentence_{len(sentence_chunks)}",
            })

    return {"fixed": fixed_chunks, "sentence": sentence_chunks}


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from knowledge_base.documents import KNOWLEDGE_BASE

    chunk_sets = build_chunk_sets(KNOWLEDGE_BASE)

    print("Fixed-size chunks:")
    for chunk in chunk_sets["fixed"]:
        print(f"- {chunk['chunk_id']}: {chunk['text']}")

    print("\nSentence chunks:")
    for chunk in chunk_sets["sentence"]:
        print(f"- {chunk['chunk_id']}: {chunk['text']}")
