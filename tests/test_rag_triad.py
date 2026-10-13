import json
from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.evaluate import (
    RAG_TRIAD_JUDGE_PROMPT,
    RAG_TRIAD_QUERIES,
    REQUIRED_RAG_TOPICS,
    _mock_llm_judge,
)


class RagTriadTests(unittest.TestCase):
    def test_test_set_covers_required_topics_and_out_of_scope_queries(self):
        covered_topics = {topic for item in RAG_TRIAD_QUERIES for topic in item["topics"]}
        out_of_scope_count = sum(not item["expected_answerable"] for item in RAG_TRIAD_QUERIES)

        self.assertEqual(len(RAG_TRIAD_QUERIES), 15)
        self.assertTrue(REQUIRED_RAG_TOPICS.issubset(covered_topics))
        self.assertGreaterEqual(out_of_scope_count, 2)

    def test_mock_judge_scores_supported_answers_and_refusals(self):
        supported = _mock_llm_judge(
            RAG_TRIAD_JUDGE_PROMPT
            + json.dumps({
                "query": "KYC identity documents",
                "contexts": ["KYC identity documents are required."],
                "answer": "KYC identity documents are required.",
                "expected_answerable": True,
            })
        )
        refusal = _mock_llm_judge(
            RAG_TRIAD_JUDGE_PROMPT
            + json.dumps({
                "query": "How do I bake a cake?",
                "contexts": [],
                "answer": "I don't know. The query is not supported by the available knowledge base.",
                "expected_answerable": False,
            })
        )

        self.assertEqual(supported["context_relevance"], 5.0)
        self.assertEqual(supported["groundedness"], 5.0)
        self.assertEqual(supported["answer_relevance"], 5.0)
        self.assertEqual(refusal["groundedness"], 5.0)
        self.assertEqual(refusal["answer_relevance"], 5.0)


if __name__ == "__main__":
    unittest.main()
