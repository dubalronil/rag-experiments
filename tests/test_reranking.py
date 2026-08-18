"""Tests for the second-stage cross-encoder reranker.

The reranker is a pure reordering of a list, which makes its failure modes
quiet ones: a sign flip, a lost chunk, or ranks that no longer start at 1 all
produce a list of the right length and type. The metrics would then be wrong
without anything raising.

None of these tests load the real cross-encoder. The model is replaced with a
scoring function chosen by the test, so what is checked is the reordering, the
truncation, and the renumbering - not whether a 22M-parameter model has good
taste. The last class checks the thing that matters most for cost: that a run
with reranking off never constructs the model at all.

Run with:  python3 -m unittest discover tests
"""

import unittest

import rag.reranking
from rag.reranking import RERANK_DEPTH, RERANK_MODEL, rerank
from rag.types import Chunk, RetrievedChunk


def make_candidates(*texts):
    """A candidate list as the first stage would hand it over: ranks 1..n."""
    return [
        RetrievedChunk(
            chunk=Chunk(chunk_id=f"c{i}", doc_id="doc", text=text,
                        start=0, end=len(text)),
            score=1.0 - i / 100,          # plausible descending cosine scores
            rank=i + 1,
        )
        for i, text in enumerate(texts)
    ]


class FakeCrossEncoder:
    """Scores pairs by a rule the test picks, and records what it was asked."""

    def __init__(self, scorer):
        self.scorer = scorer
        self.seen = []

    def predict(self, pairs):
        self.seen.append(list(pairs))
        return [self.scorer(question, text) for question, text in pairs]


class RerankerTestCase(unittest.TestCase):
    """Swaps in a fake cross-encoder for the duration of each test."""

    def use(self, scorer):
        fake = FakeCrossEncoder(scorer)
        rag.reranking._loaded[RERANK_MODEL] = fake
        return fake

    def setUp(self):
        self._saved = dict(rag.reranking._loaded)

    def tearDown(self):
        rag.reranking._loaded.clear()
        rag.reranking._loaded.update(self._saved)


class TestReordering(RerankerTestCase):
    def test_the_order_follows_the_cross_encoder_not_the_first_stage(self):
        """The whole point: first-stage rank must not survive reranking."""
        self.use(lambda q, text: {"worst": 0.1, "middle": 0.5, "best": 0.9}[text])
        candidates = make_candidates("worst", "middle", "best")

        results = rerank("q", candidates, k=3)

        self.assertEqual([r.chunk.text for r in results], ["best", "middle", "worst"])

    def test_a_low_ranked_candidate_can_be_promoted_to_first(self):
        """The failure this stage exists to fix: gold buried deep in the list."""
        texts = [f"filler{i}" for i in range(30)] + ["gold"]
        self.use(lambda q, text: 10.0 if text == "gold" else 0.0)

        results = rerank("q", make_candidates(*texts), k=5)

        self.assertEqual(results[0].chunk.text, "gold")
        self.assertEqual(results[0].rank, 1)

    def test_scores_are_the_cross_encoder_scores(self):
        self.use(lambda q, text: {"a": -2.5, "b": 3.5}[text])

        results = rerank("q", make_candidates("a", "b"), k=2)

        self.assertAlmostEqual(results[0].score, 3.5, places=5)
        # Cross-encoder logits are unbounded and may be negative - unlike
        # cosine, which the first-stage score would have been.
        self.assertAlmostEqual(results[1].score, -2.5, places=5)

    def test_scores_are_descending(self):
        self.use(lambda q, text: float(len(text)))

        results = rerank("q", make_candidates("a", "bbbb", "cc", "ddddddd"), k=4)
        scores = [r.score for r in results]

        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_ties_break_on_chunk_id(self):
        """Otherwise the order would depend on the first stage's output order."""
        self.use(lambda q, text: 1.0)

        results = rerank("q", make_candidates("a", "b", "c"), k=3)

        self.assertEqual([r.chunk.chunk_id for r in results], ["c0", "c1", "c2"])


class TestTruncationAndRanks(RerankerTestCase):
    def setUp(self):
        super().setUp()
        self.use(lambda q, text: float(text))
        self.candidates = make_candidates(*[str(i) for i in range(20)])

    def test_only_k_results_come_back(self):
        self.assertEqual(len(rerank("q", self.candidates, k=5)), 5)

    def test_ranks_are_renumbered_from_one(self):
        """The returned rank is the post-rerank position, not the candidate's."""
        results = rerank("q", self.candidates, k=5)

        self.assertEqual([r.rank for r in results], [1, 2, 3, 4, 5])
        # Highest text-as-number wins, so these came from deep in the list.
        self.assertEqual([r.chunk.text for r in results],
                         ["19", "18", "17", "16", "15"])

    def test_k_larger_than_the_candidate_set_returns_everything(self):
        self.assertEqual(len(rerank("q", self.candidates, k=99)), 20)

    def test_non_positive_k_is_rejected(self):
        for k in (0, -1):
            with self.assertRaises(ValueError):
                rerank("q", self.candidates, k=k)

    def test_empty_candidates_return_empty(self):
        self.assertEqual(rerank("q", [], k=5), [])


class TestPairsAndInterface(RerankerTestCase):
    def test_every_candidate_is_paired_with_the_question(self):
        fake = self.use(lambda q, text: 0.0)
        candidates = make_candidates("alpha", "beta", "gamma")

        rerank("what happened?", candidates, k=2)

        self.assertEqual(len(fake.seen), 1)
        self.assertEqual(fake.seen[0],
                         [("what happened?", "alpha"),
                          ("what happened?", "beta"),
                          ("what happened?", "gamma")])

    def test_the_chunks_themselves_are_passed_through_unchanged(self):
        self.use(lambda q, text: 1.0 if text == "beta" else 0.0)
        candidates = make_candidates("alpha", "beta")

        result = rerank("q", candidates, k=1)[0]

        self.assertIs(result.chunk, candidates[1].chunk)

    def test_returns_retrieved_chunks(self):
        self.use(lambda q, text: 0.0)

        for result in rerank("q", make_candidates("a", "b"), k=2):
            self.assertIsInstance(result, RetrievedChunk)
            self.assertIsInstance(result.chunk, Chunk)
            self.assertIsInstance(result.score, float)
            self.assertIsInstance(result.rank, int)

    def test_repeated_calls_are_identical(self):
        self.use(lambda q, text: float(len(text)))
        candidates = make_candidates("a", "bb", "ccc", "dd")

        first = [(r.chunk.chunk_id, r.score, r.rank) for r in rerank("q", candidates, k=3)]
        second = [(r.chunk.chunk_id, r.score, r.rank) for r in rerank("q", candidates, k=3)]

        self.assertEqual(first, second)


class TestFixedSettings(unittest.TestCase):
    def test_the_constants_are_the_agreed_values(self):
        self.assertEqual(RERANK_MODEL, "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.assertEqual(RERANK_DEPTH, 50)

    def test_reranking_is_off_by_default(self):
        """Every existing config omits the key, so the default decides."""
        from rag.experiment import DEFAULTS

        self.assertIs(DEFAULTS["retrieval"]["rerank"], False)

    def test_the_model_is_not_loaded_at_import_time(self):
        """Lazy loading is the difference between free and an 80MB download.

        sentence_transformers.CrossEncoder must not even be imported unless
        something actually reranks.
        """
        import inspect

        # The name is not in the module namespace...
        self.assertNotIn("CrossEncoder", dir(rag.reranking))
        # ...because the import sits inside load_reranker's body, indented.
        body = inspect.getsource(rag.reranking.load_reranker)
        self.assertIn("        from sentence_transformers import CrossEncoder", body)
        self.assertNotIn(
            "\nfrom sentence_transformers",
            inspect.getsource(rag.reranking).split("def load_reranker")[0],
        )


if __name__ == "__main__":
    unittest.main()
