import unittest
import numpy as np
from minhashlib import DiffChecker


class TestDiffChecker(unittest.TestCase):
    def test_signature_has_expected_length(self):
        checker = DiffChecker(num_perm=64, k=3, seed=42)
        sig = checker.generateSignature("minhash similarity test")

        self.assertEqual(sig.shape, (64,))
        self.assertEqual(sig.dtype, np.int64)

    def test_identical_documents_have_similarity_one(self):
        checker = DiffChecker(num_perm=128, k=2, seed=42)
        doc = "the quick brown fox jumps over the lazy dog"

        sig_a = checker.generateSignature(doc)
        sig_b = checker.generateSignature(doc)
        similarity = checker.checkJaccardSignatureSimilarity(sig_a, sig_b)

        self.assertEqual(similarity, 1.0)

    def test_more_similar_pair_scores_higher(self):
        checker = DiffChecker(num_perm=128, k=3, seed=42)

        base = "minhash is useful for near-duplicate detection in text corpora"
        close = "minhash is useful for near duplicate detection in text corpora"
        far = "completely unrelated sentence about weather and astronomy"

        sig_base = checker.generateSignature(base)
        sig_close = checker.generateSignature(close)
        sig_far = checker.generateSignature(far)

        close_score = checker.checkJaccardSignatureSimilarity(sig_base, sig_close)
        far_score = checker.checkJaccardSignatureSimilarity(sig_base, sig_far)

        self.assertGreater(close_score, far_score)

    def test_append_document_adds_columns(self):
        checker = DiffChecker(num_perm=32, k=2, seed=42)

        checker.appendDocumentAsSignature("first document")
        checker.appendDocumentAsSignature("second document")

        self.assertEqual(checker.signatureMatrix.shape, (32, 2))


if __name__ == "__main__":
    unittest.main()
