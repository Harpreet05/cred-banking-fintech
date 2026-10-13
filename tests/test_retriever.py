from __future__ import annotations

import tempfile
import unittest

from knowledge_base.documents import KNOWLEDGE_BASE
from rag.embeddings import create_embeddings
from rag.retriever import grounded_answer, load_collections


class GroundedAnswerTests(unittest.TestCase):
    def test_grounded_answer_returns_context_only_for_in_scope_query(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            create_embeddings(KNOWLEDGE_BASE, persist_directory=temp_dir)
            result = grounded_answer(
                "What documents are normally required for KYC?",
                persist_directory=temp_dir,
                top_k=3,
            )

            self.assertFalse(result["fallback"])
            self.assertGreater(result["top_1_similarity"], 0.5)
            self.assertIn("KYC", result["answer"].upper())
            self.assertIn("identity proof", result["answer"].lower())
            self.assertTrue(result["used_context"])

    def test_grounded_answer_falls_back_for_out_of_scope_query(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            create_embeddings(KNOWLEDGE_BASE, persist_directory=temp_dir)
            result = grounded_answer(
                "Tell me how to bake a chocolate cake.",
                persist_directory=temp_dir,
                top_k=3,
            )

            self.assertTrue(result["fallback"])
            self.assertIn("I don't know", result["answer"])

    def test_local_collections_are_loaded_from_separate_strategies(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
            create_embeddings(KNOWLEDGE_BASE, persist_directory=temp_dir)
            loaded = load_collections(temp_dir)

            self.assertEqual(set(loaded), {"fixed", "sentence"})
            self.assertEqual(loaded["fixed"].count(), 112)
            self.assertEqual(loaded["sentence"].count(), 47)


if __name__ == "__main__":
    unittest.main()
