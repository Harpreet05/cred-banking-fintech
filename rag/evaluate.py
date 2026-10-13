from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Sequence, Set

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knowledge_base.documents import KNOWLEDGE_BASE
from rag.embeddings import create_embeddings
from rag.retriever import (
    DEFAULT_GROUNDEDNESS_THRESHOLD,
    DEFAULT_MODEL_NAME,
    DEFAULT_PERSIST_DIRECTORY,
    RetrievedChunk,
    _cosine_similarity,
    load_collections,
    retrieve_top_k,
)


QUERY_DOCUMENTS = {
    "What documents are normally required for KYC?": {"kb_08"},
    "What are the eligibility requirements for a personal loan?": {"kb_01"},
    "How are interest rate slabs determined?": {"kb_11"},
    "How is the account closure process completed?": {"kb_10"},
    "What are the main factors affecting a credit score?": {"kb_14"},
}

REQUIRED_RAG_TOPICS = {
    "loan eligibility criteria by loan type",
    "EMI calculation rules",
    "credit-card fee structure",
    "KYC document requirements",
    "fraud-dispute resolution process",
    "account-closure process",
    "interest-rate slabs",
    "prepayment-penalty rules",
    "minimum-balance requirements",
    "credit-score impact factors",
    "joint-account rules",
    "NRI-account eligibility",
}

RAG_TRIAD_QUERIES = [
    {"query": "What eligibility criteria apply to personal, home, auto, education, and business loans?", "topics": ["loan eligibility criteria by loan type"], "expected_answerable": True},
    {"query": "How is an EMI calculated from the principal, interest rate, and loan tenure?", "topics": ["EMI calculation rules"], "expected_answerable": True},
    {"query": "What fees and charges can apply to a credit card?", "topics": ["credit-card fee structure"], "expected_answerable": True},
    {"query": "Which documents are accepted for completing KYC?", "topics": ["KYC document requirements"], "expected_answerable": True},
    {"query": "How should a customer dispute a suspected fraudulent transaction?", "topics": ["fraud-dispute resolution process"], "expected_answerable": True},
    {"query": "What steps are required to close an account?", "topics": ["account-closure process"], "expected_answerable": True},
    {"query": "How are the interest-rate slabs for a loan determined?", "topics": ["interest-rate slabs"], "expected_answerable": True},
    {"query": "Can a borrower repay a loan early, and when does a prepayment penalty apply?", "topics": ["prepayment-penalty rules"], "expected_answerable": True},
    {"query": "What minimum balance must be maintained, and what happens if it is not?", "topics": ["minimum-balance requirements"], "expected_answerable": True},
    {"query": "Which factors can improve or reduce a borrower's credit score?", "topics": ["credit-score impact factors"], "expected_answerable": True},
    {"query": "What rules apply when opening or operating a joint account?", "topics": ["joint-account rules"], "expected_answerable": True},
    {"query": "Can an NRI open an account, and what eligibility conditions apply?", "topics": ["NRI-account eligibility"], "expected_answerable": True},
    {"query": "If a borrower closes a loan early, how are the closure steps and prepayment charges handled?", "topics": ["account-closure process", "prepayment-penalty rules"], "expected_answerable": True},
    {"query": "How do I bake a chocolate cake without an oven?", "topics": [], "expected_answerable": False},
    {"query": "What will the weather be in Mumbai next week?", "topics": [], "expected_answerable": False},
]

RAG_TRIAD_JUDGE_PROMPT = """You are a strict RAG-triad evaluator. Score each dimension from 1 (poor) to 5 (excellent).
Context relevance: does the supplied context address the query?
Groundedness: are the answer's factual claims supported by the supplied context? A clear refusal is grounded for an explicitly unanswerable query.
Answer relevance: does the answer address the query? A clear refusal is relevant for an explicitly unanswerable query.
Return context_relevance, groundedness, and answer_relevance as numeric scores, with a short rationale for each.

Evaluation input (JSON):
"""

_JUDGE_STOP_WORDS = {
    "a", "an", "and", "apply", "are", "be", "can", "do", "does", "for", "from",
    "how", "i", "if", "in", "is", "it", "of", "on", "or", "the", "to", "what",
    "which", "with", "will", "without",
}
TRIAD_REPORT_PATH = Path(__file__).with_name("rag_triad_report.json")


def _documents_from_chunks(chunks: Sequence[Dict[str, Any]]) -> Set[str]:
    """Return the unique parent document IDs for retrieved chunks."""
    return {chunk["doc_id"] for chunk in chunks}


def _precision_at_k(retrieved: Iterable[str], relevant: Set[str], k: int = 3) -> float:
    """Compute document-level Precision@3 after deduplicating parent documents."""
    retrieved_documents = list(dict.fromkeys(retrieved))[:k]
    relevant_count = len(set(retrieved_documents) & relevant)
    return relevant_count / len(retrieved_documents) if retrieved_documents else 0.0


def _recall_at_k(retrieved: Iterable[str], relevant: Set[str], k: int = 3) -> float:
    """Compute document-level Recall@3 after deduplicating parent documents."""
    retrieved_documents = list(dict.fromkeys(retrieved))[:k]
    if not relevant:
        return 0.0
    relevant_count = len(set(retrieved_documents) & relevant)
    return relevant_count / len(relevant)


def evaluate_collection(
    strategy: str,
    queries: Dict[str, Set[str]],
    persist_directory: str = DEFAULT_PERSIST_DIRECTORY,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """Evaluate one chunking strategy for every query."""
    results: List[Dict[str, Any]] = []

    for query, relevant_documents in queries.items():
        retrieved = retrieve_top_k(
            query=query,
            top_k=top_k,
            persist_directory=persist_directory,
            strategy=strategy,
        )
        retrieved_documents = [chunk.doc_id for chunk in retrieved]
        precision = _precision_at_k(retrieved_documents, relevant_documents, top_k)
        recall = _recall_at_k(retrieved_documents, relevant_documents, top_k)

        results.append({
            "query": query,
            "strategy": strategy,
            "relevant_documents": sorted(relevant_documents),
            "retrieved_documents": retrieved_documents,
            "precision_at_3": precision,
            "recall_at_3": recall,
            "arithmetic": {
                "precision": f"{len(set(retrieved_documents) & relevant_documents)}/{len(set(retrieved_documents)) or 1}",
                "recall": f"{len(set(retrieved_documents) & relevant_documents)}/{len(relevant_documents)}",
            },
        })

    return results


def evaluate_strategies(
    queries: Dict[str, Set[str]] = QUERY_DOCUMENTS,
    persist_directory: str = DEFAULT_PERSIST_DIRECTORY,
    top_k: int = 3,
) -> Dict[str, List[Dict[str, Any]]]:
    """Evaluate both collections and return the per-query metrics."""
    from chromadb import PersistentClient

    client = PersistentClient(path=persist_directory)
    collections_exist = all(
        client.get_collection(name=f"knowledge_base_{strategy}").count() > 0
        for strategy in ("fixed", "sentence")
    )
    if not collections_exist:
        create_embeddings(KNOWLEDGE_BASE, persist_directory=persist_directory)

    return {
        strategy: evaluate_collection(strategy, queries, persist_directory, top_k)
        for strategy in ("fixed", "sentence")
    }


def print_evaluation(results: Dict[str, List[Dict[str, Any]]]) -> None:
    """Print per-query arithmetic and strategy averages."""
    for strategy, scores in results.items():
        print(f"\n{strategy.title()} strategy")
        for item in scores:
            print(
                f"- {item['query']} | "
                f"Precision@3 = {item['precision_at_3']:.2f} "
                f"({item['arithmetic']['precision']}) | "
                f"Recall@3 = {item['recall_at_3']:.2f} "
                f"({item['arithmetic']['recall']})"
            )
        average_precision = sum(item["precision_at_3"] for item in scores) / len(scores)
        average_recall = sum(item["recall_at_3"] for item in scores) / len(scores)
        print(f"Average Precision@3: {average_precision:.2f}")
        print(f"Average Recall@3: {average_recall:.2f}")


def _judge_tokens(text: str) -> Set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 1 and token not in _JUDGE_STOP_WORDS
    }


def _overlap_score(left: Set[str], right: Set[str]) -> tuple[float, str]:
    if not left:
        return 1.0, "No meaningful terms were available to compare."
    matched = len(left & right)
    coverage = matched / len(left)
    return round(1.0 + 4.0 * coverage, 2), f"Matched {matched} of {len(left)} meaningful terms."


def _mock_llm_judge(prompt: str) -> Dict[str, Any]:
    """Run the judge rubric deterministically without a network or API key."""
    marker = "Evaluation input (JSON):\n"
    payload = json.loads(prompt.split(marker, 1)[1])
    query_tokens = _judge_tokens(payload["query"])
    context_tokens = _judge_tokens(" ".join(payload["contexts"]))
    answer_tokens = _judge_tokens(payload["answer"])

    context_score, context_reason = _overlap_score(query_tokens, context_tokens)
    expected_answerable = payload["expected_answerable"]
    is_refusal = payload["answer"].lower().startswith("i don't know")

    if is_refusal and not expected_answerable:
        groundedness_score = 5.0
        groundedness_reason = "The unsupported query received a clear refusal."
        answer_relevance_score = 5.0
        answer_relevance_reason = "The refusal is appropriate for an out-of-scope query."
    elif not context_tokens:
        groundedness_score = 1.0
        groundedness_reason = "The answer has no supporting retrieved context."
        answer_relevance_score = 1.0
        answer_relevance_reason = "The answer does not address an expected in-scope query."
    else:
        groundedness_score, groundedness_reason = _overlap_score(answer_tokens, context_tokens)
        answer_relevance_score, answer_relevance_reason = _overlap_score(query_tokens, answer_tokens)

    return {
        "context_relevance": context_score,
        "groundedness": groundedness_score,
        "answer_relevance": answer_relevance_score,
        "rationales": {
            "context_relevance": context_reason,
            "groundedness": groundedness_reason,
            "answer_relevance": answer_relevance_reason,
        },
    }


def _restore_parent_document(collection: Any, doc_id: str) -> str:
    stored = collection.get(
        where={"doc_id": doc_id},
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


def evaluate_rag_triad(
    queries: Sequence[Dict[str, Any]] = RAG_TRIAD_QUERIES,
    persist_directory: str = DEFAULT_PERSIST_DIRECTORY,
    top_k: int = 3,
    threshold: float = DEFAULT_GROUNDEDNESS_THRESHOLD,
) -> Dict[str, Any]:
    """Evaluate the deployed fixed-size RAG path with the deterministic MOCK_LLM judge."""
    from chromadb import PersistentClient
    from sentence_transformers import SentenceTransformer

    if len(queries) != 15:
        raise ValueError(f"The RAG triad test set must contain exactly 15 queries, got {len(queries)}.")

    covered_topics = {topic for item in queries for topic in item["topics"]}
    missing_topics = REQUIRED_RAG_TOPICS - covered_topics
    if missing_topics:
        raise ValueError(f"RAG triad test set misses required topics: {sorted(missing_topics)}")

    client = PersistentClient(path=persist_directory)
    collection_names = {collection.name for collection in client.list_collections()}
    required_collections = {"knowledge_base_fixed", "knowledge_base_sentence"}
    if not required_collections.issubset(collection_names) or any(
        client.get_collection(name=name).count() == 0 for name in required_collections
    ):
        create_embeddings(KNOWLEDGE_BASE, persist_directory=persist_directory)

    fixed_collection = load_collections(persist_directory)["fixed"]
    model = SentenceTransformer(DEFAULT_MODEL_NAME)
    query_embeddings = model.encode(
        [item["query"] for item in queries],
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    results: List[Dict[str, Any]] = []
    result_count = min(top_k, fixed_collection.count())
    for item, query_embedding in zip(queries, query_embeddings):
        response = fixed_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=result_count,
            include=["documents", "metadatas", "embeddings"],
        )
        matches = []
        for index, text in enumerate(response["documents"][0]):
            metadata = response["metadatas"][0][index]
            matches.append(RetrievedChunk(
                strategy="fixed",
                chunk_id=metadata["chunk_id"],
                doc_id=metadata["doc_id"],
                title=metadata["title"],
                text=text,
                similarity=_cosine_similarity(query_embedding, response["embeddings"][0][index]),
            ))
        matches.sort(key=lambda match: match.similarity, reverse=True)

        best_match = matches[0] if matches else None
        fallback = best_match is None or best_match.similarity < threshold
        if fallback:
            answer = "I don't know. The query is not supported by the available knowledge base."
            contexts: List[str] = []
        else:
            document_text = _restore_parent_document(fixed_collection, best_match.doc_id)
            answer = f"{best_match.title}: {document_text}"
            contexts = [document_text]

        judge_input = {
            "query": item["query"],
            "contexts": contexts,
            "answer": answer,
            "expected_answerable": item["expected_answerable"],
        }
        judge_result = _mock_llm_judge(
            RAG_TRIAD_JUDGE_PROMPT + json.dumps(judge_input, ensure_ascii=False)
        )
        results.append({
            "query": item["query"],
            "topics": list(item["topics"]),
            "expected_answerable": item["expected_answerable"],
            "fallback": fallback,
            "top_1_similarity": round(best_match.similarity, 4) if best_match else 0.0,
            "scores": {
                "context_relevance": judge_result["context_relevance"],
                "groundedness": judge_result["groundedness"],
                "answer_relevance": judge_result["answer_relevance"],
            },
            "rationales": judge_result["rationales"],
        })

    metric_names = ("context_relevance", "groundedness", "answer_relevance")
    averages = {
        metric: round(sum(result["scores"][metric] for result in results) / len(results), 3)
        for metric in metric_names
    }
    return {
        "judge_mode": "MOCK_LLM",
        "score_scale": "1-5",
        "score_method": "Deterministic prompt-based MOCK_LLM using meaningful-token coverage; no external API or network calls.",
        "query_count": len(results),
        "required_topics": sorted(REQUIRED_RAG_TOPICS),
        "covered_required_topics": sorted(covered_topics & REQUIRED_RAG_TOPICS),
        "judge_prompt": RAG_TRIAD_JUDGE_PROMPT,
        "averages": averages,
        "results": results,
    }


def print_rag_triad_report(report: Dict[str, Any]) -> None:
    print(f"RAG triad evaluation ({report['judge_mode']}, 1-5 scale)")
    for index, result in enumerate(report["results"], start=1):
        scores = result["scores"]
        print(
            f"{index:02d}. {result['query']} | "
            f"Context relevance: {scores['context_relevance']:.2f} | "
            f"Groundedness: {scores['groundedness']:.2f} | "
            f"Answer relevance: {scores['answer_relevance']:.2f}"
        )
    print("Averages across all 15 queries:")
    for metric, score in report["averages"].items():
        print(f"- {metric}: {score:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval or the RAG triad.")
    parser.add_argument("--rag-triad", action="store_true", help="Run the 15-query MOCK_LLM RAG triad evaluation.")
    parser.add_argument("--output", type=Path, default=TRIAD_REPORT_PATH, help="Path for the JSON triad report.")
    arguments = parser.parse_args()

    if arguments.rag_triad:
        report = evaluate_rag_triad()
        arguments.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print_rag_triad_report(report)
        print(f"Full report written to {arguments.output}")
    else:
        evaluation = evaluate_strategies()
        print_evaluation(evaluation)
