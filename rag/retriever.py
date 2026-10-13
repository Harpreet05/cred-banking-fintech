from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.chunks import build_chunk_sets
from rag.embeddings import create_embeddings


DEFAULT_PERSIST_DIRECTORY = "./chroma_db"
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_GROUNDEDNESS_THRESHOLD = 0.4530


@dataclass(frozen=True)
class RetrievedChunk:
    strategy: str
    chunk_id: str
    doc_id: str
    title: str
    text: str
    similarity: float


def load_collections(persist_directory: str = DEFAULT_PERSIST_DIRECTORY) -> Dict[str, Any]:
    """Load both persistent ChromaDB collections."""
    client = PersistentClient(path=persist_directory)
    return {
        strategy: client.get_collection(name=f"knowledge_base_{strategy}")
        for strategy in ("fixed", "sentence")
    }


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the cosine similarity between two normalized vectors."""
    dot_product = sum(a * b for a, b in zip(left, right))
    return float(dot_product)


def _document_text(chunk: RetrievedChunk, persist_directory: str) -> str:
    """Reassemble a parent document from its stored chunks."""
    collection = load_collections(persist_directory)[chunk.strategy]
    stored = collection.get(
        where={"doc_id": chunk.doc_id},
        include=["documents", "metadatas"],
    )
    chunks = sorted(
        zip(stored["metadatas"], stored["documents"]),
        key=lambda item: int(item[0]["chunk_id"].rsplit("_", 1)[-1]),
    )

    text = ""
    for _, part in chunks:
        overlap = min(len(text), len(part))
        while overlap and not text.endswith(part[:overlap]):
            overlap -= 1
        text += part[overlap:] if overlap else f" {part}"
    return text.strip()


def retrieve_top_k(
    query: str,
    top_k: int = 3,
    persist_directory: str = DEFAULT_PERSIST_DIRECTORY,
    model_name: str = DEFAULT_MODEL_NAME,
    strategy: str | None = None,
) -> List[RetrievedChunk]:
    """Retrieve and score the top-k relevant chunks from both collections."""
    if strategy not in (None, "fixed", "sentence"):
        raise ValueError("strategy must be 'fixed', 'sentence', or None")

    model = SentenceTransformer(model_name)
    query_embedding = model.encode(query, normalize_embeddings=True)
    collections = load_collections(persist_directory)
    results: List[RetrievedChunk] = []

    for collection_strategy, collection in collections.items():
        if strategy is not None and strategy != collection_strategy:
            continue
        response = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=["documents", "metadatas", "embeddings"],
        )
        for index, document in enumerate(response["documents"][0]):
            metadata = response["metadatas"][0][index]
            embedding = response["embeddings"][0][index]
            similarity = _cosine_similarity(query_embedding, embedding)
            results.append(
                RetrievedChunk(
                    strategy=collection_strategy,
                    chunk_id=metadata["chunk_id"],
                    doc_id=metadata["doc_id"],
                    title=metadata["title"],
                    text=document,
                    similarity=similarity,
                )
            )

    return sorted(results, key=lambda item: item.similarity, reverse=True)[:top_k]


def grounded_answer(
    query: str,
    top_k: int = 3,
    persist_directory: str = DEFAULT_PERSIST_DIRECTORY,
    model_name: str = DEFAULT_MODEL_NAME,
    threshold: float = DEFAULT_GROUNDEDNESS_THRESHOLD,
    strategy: str | None = None,
) -> Dict[str, Any]:
    """Answer using only retrieved context, or fall back when the top match is below threshold."""
    retrieved = retrieve_top_k(
        query=query,
        top_k=top_k,
        persist_directory=persist_directory,
        model_name=model_name,
        strategy=strategy,
    )

    if not retrieved or retrieved[0].similarity < threshold:
        return {
            "answer": "I don't know. The query is not supported by the available knowledge base.",
            "fallback": True,
            "top_1_similarity": retrieved[0].similarity if retrieved else 0.0,
            "retrieved_chunks": [],
            "used_context": [],
        }

    best_match = retrieved[0]
    document_text = _document_text(best_match, persist_directory)
    answer = f"{best_match.title}: {document_text}"
    relevant_chunks = [chunk for chunk in retrieved if chunk.doc_id == best_match.doc_id]
    return {
        "answer": answer,
        "fallback": False,
        "top_1_similarity": best_match.similarity,
        "retrieved_chunks": [
            {
                "strategy": chunk.strategy,
                "chunk_id": chunk.chunk_id,
                "doc_id": chunk.doc_id,
                "title": chunk.title,
                "text": chunk.text,
                "similarity": chunk.similarity,
            }
            for chunk in relevant_chunks
        ],
        "used_context": [document_text],
    }


def calibrate_threshold(
    queries: Sequence[str],
    in_scope_flags: Sequence[bool],
    persist_directory: str = DEFAULT_PERSIST_DIRECTORY,
    model_name: str = DEFAULT_MODEL_NAME,
) -> Dict[str, Any]:
    """Measure top-1 similarities for labeled queries and return the threshold."""
    if len(queries) != len(in_scope_flags):
        raise ValueError("Each query must have one in-scope flag.")

    model = SentenceTransformer(model_name)
    query_embeddings = model.encode(list(queries), normalize_embeddings=True)
    collections = load_collections(persist_directory)
    measurements: List[Tuple[str, float, bool]] = []

    for query, embedding, in_scope in zip(queries, query_embeddings, in_scope_flags):
        best_similarity = 0.0
        for collection in collections.values():
            response = collection.query(
                query_embeddings=[embedding.tolist()],
                n_results=1,
                include=["embeddings"],
            )
            candidate_embedding = response["embeddings"][0][0]
            best_similarity = max(best_similarity, _cosine_similarity(embedding, candidate_embedding))
        measurements.append((query, best_similarity, in_scope))

    in_scope_values = [value for _, value, in_scope in measurements if in_scope]
    out_of_scope_values = [value for _, value, in_scope in measurements if not in_scope]
    highest_in_scope = max(in_scope_values)
    lowest_out_of_scope = min(out_of_scope_values)
    threshold = (highest_in_scope + lowest_out_of_scope) / 2
    return {
        "measurements": [
            {"query": query, "similarity": value, "in_scope": in_scope}
            for query, value, in_scope in measurements
        ],
        "in_scope_values": sorted(in_scope_values),
        "out_of_scope_values": sorted(out_of_scope_values),
        "threshold": threshold,
    }


def benchmark_queries() -> Dict[str, Any]:
    """Run the documented calibration and grounded-answer demonstration queries."""
    from knowledge_base.documents import KNOWLEDGE_BASE

    if not any(collection.count() for collection in load_collections(DEFAULT_PERSIST_DIRECTORY).values()):
        create_embeddings(KNOWLEDGE_BASE, persist_directory=DEFAULT_PERSIST_DIRECTORY)

    in_scope_queries = [
        "What documents are normally required for KYC?",
        "What are the eligibility requirements for a personal loan?",
        "How are interest rate slabs determined?",
        "How is the account closure process completed?",
        "What are the main factors affecting a credit score?",
    ]
    out_of_scope_queries = [
        "How do I bake a chocolate cake?",
        "How is the weather today?",
    ]

    calibration = calibrate_threshold(
        in_scope_queries + out_of_scope_queries,
        [True] * len(in_scope_queries) + [False] * len(out_of_scope_queries),
        persist_directory=DEFAULT_PERSIST_DIRECTORY,
    )
    results = {
        "threshold": calibration["threshold"],
        "measurements": calibration["measurements"],
        "demonstrations": [],
    }
    for query in in_scope_queries + out_of_scope_queries:
        result = grounded_answer(
            query,
            persist_directory=DEFAULT_PERSIST_DIRECTORY,
            threshold=calibration["threshold"],
        )
        results["demonstrations"].append({
            "query": query,
            "fallback": result["fallback"],
            "top_1_similarity": result["top_1_similarity"],
            "answer": result["answer"],
        })
    return results


if __name__ == "__main__":
    benchmark = benchmark_queries()
    print(f"Selected threshold: {benchmark['threshold']:.4f}")
    for item in benchmark["measurements"]:
        print(f"{item['query']}: {item['similarity']:.4f}")
    print("\nDemonstrations:")
    for item in benchmark["demonstrations"]:
        print(f"- Query: {item['query']}")
        print(f"  Similarity: {item['top_1_similarity']:.4f}")
        print(f"  Fallback: {item['fallback']}")
        print(f"  Answer: {item['answer']}")
