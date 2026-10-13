from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict, List

from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.chunks import build_chunk_sets


def create_embeddings(
    documents: List[Dict[str, str]],
    persist_directory: str = "./chroma_db",
    model_name: str = "all-MiniLM-L6-v2",
) -> Dict[str, Any]:
    """Embed both chunk sets and store each strategy in a separate ChromaDB collection."""
    chunk_sets = build_chunk_sets(documents)
    client = PersistentClient(path=persist_directory)
    model = SentenceTransformer(model_name)

    collections: Dict[str, Any] = {}
    for strategy, chunks in chunk_sets.items():
        collection = client.get_or_create_collection(name=f"knowledge_base_{strategy}")
        ids = [chunk["chunk_id"] for chunk in chunks]
        texts = [chunk["text"] for chunk in chunks]
        embeddings = model.encode(texts, normalize_embeddings=True).tolist()

        metadatas = [{
            "doc_id": chunk["doc_id"],
            "title": chunk["title"],
            "chunk_id": chunk["chunk_id"],
            "strategy": strategy,
        } for chunk in chunks]

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=texts,
        )
        collections[strategy] = collection

    return collections


if __name__ == "__main__":
    from knowledge_base.documents import KNOWLEDGE_BASE

    collections = create_embeddings(KNOWLEDGE_BASE)
    print("Indexed collections:")
    for strategy, collection in collections.items():
        print(f"- {strategy}: {collection.count()} chunks")
