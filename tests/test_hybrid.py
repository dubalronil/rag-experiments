"""Tests for hybrid retrieval (Reciprocal Rank Fusion).

Fusion is the easiest part of retrieval to get quietly wrong. A misplaced
constant, a sign flip, or an unstable sort all still return a plausible ranked
list of the right length, and the experiment comparing hybrid against dense
would simply report the wrong winner.

Most of these tests drive HybridIndex with stub halves instead of real ones.
That is deliberate: fusion arithmetic should be checked against rankings chosen
by hand, not against whatever an embedding model happens to produce. The
composition tests at the bottom use real indexes to confirm the halves are
wired together correctly.

Run with:  python3 -m unittest discover tests
"""

import unittest

import rag.retrieval
from rag.retrieval import (
    FUSION_DEPTH,
    RRF_K,
    STRATEGIES,
    Bm25Index,
    HybridIndex,
    VectorIndex,
    build_index,
)
from rag.types import Chunk, RetrievedChunk


def make_chunk(chunk_id, text="text", doc_id="doc"):
    return Chunk(chunk_id=chunk_id, doc_id=doc_id, text=text, start=0, end=len(text))


class StubIndex:
    """A retriever with a ranking chosen by the test rather than computed.

    HybridIndex only ever calls search() and len() on its halves, so a stub is
    substitutable - the same duck typing that lets a config swap strategies.
    """

    def __init__(self, ranking, size=None):
        self.ranking = list(ranking)
        self.size = len(self.ranking) if size is None else size
        self.asked_for = []

    def __len__(self):
        return self.size

    def search(self, query, k=5):
        self.asked_for.append(k)
        return [
            RetrievedChunk(chunk=chunk, score=0.0, rank=position)
            for position, chunk in enumerate(self.ranking[:k], start=1)
        ]


def hybrid(dense_ranking, lexical_ranking, rrf_k=RRF_K, fusion_depth=FUSION_DEPTH):
    # Both halves index the same corpus - the union of what either returns -
    # even when they rank different parts of it.
    corpus = {c.chunk_id for c in [*dense_ranking, *lexical_ranking]}
    return HybridIndex(
        dense=StubIndex(dense_ranking, size=len(corpus)),
        lexical=StubIndex(lexical_ranking, size=len(corpus)),
        rrf_k=rrf_k,
        fusion_depth=fusion_depth,
    )


class TestFusionArithmetic(unittest.TestCase):
    def test_score_is_the_sum_of_reciprocal_ranks(self):
        """Every score computed by hand from 1/(60 + rank)."""
        a, b, c, d = (make_chunk(x) for x in "abcd")
        index = hybrid([a, b, c], [c, a, d])

        scores = {r.chunk.chunk_id: r.score for r in index.search("q", k=4)}

        self.assertAlmostEqual(scores["a"], 1 / 61 + 1 / 62, places=12)
        self.assertAlmostEqual(scores["b"], 1 / 62, places=12)
        self.assertAlmostEqual(scores["c"], 1 / 63 + 1 / 61, places=12)
        self.assertAlmostEqual(scores["d"], 1 / 63, places=12)

    def test_a_chunk_found_by_both_outranks_one_found_by_either(self):
        a, b, c = (make_chunk(x) for x in "abc")
        # b is second-best to both retrievers; a and c each top one list.
        index = hybrid([a, b], [c, b])

        self.assertEqual([r.chunk.chunk_id for r in index.search("q", k=3)],
                         ["b", "a", "c"])

    def test_agreement_outranks_a_single_top_rank_at_this_depth(self):
        """Pins the documented consequence of RRF_K=60 at fusion depth 20.

        A chunk both retrievers rank *last* in a depth-20 list scores
        2/80 = 0.025, beating a chunk only one retriever ranks first, 1/61 =
        0.0164. This is the property the first hybrid experiment is really
        testing, so it should fail loudly if the constant is ever changed
        without that being a deliberate decision.
        """
        # Each list is exactly fusion_depth long, so `shared` sits at rank 20
        # in both - one place deeper and it would be truncated away entirely.
        top = make_chunk("dense-favourite")
        shared = make_chunk("agreed")
        dense = [top] + [make_chunk(f"f{i:02d}") for i in range(18)] + [shared]
        lexical = [make_chunk(f"g{i:02d}") for i in range(19)] + [shared]
        self.assertEqual((len(dense), len(lexical)), (20, 20))

        index = hybrid(dense, lexical)
        results = index.search("q", k=2)

        self.assertEqual(results[0].chunk.chunk_id, "agreed")
        self.assertAlmostEqual(results[0].score, 2 / 80, places=12)
        self.assertAlmostEqual(results[1].score, 1 / 61, places=12)

    def test_only_fusion_depth_candidates_are_considered(self):
        """Each half is asked for exactly fusion_depth results, not more."""
        many = [make_chunk(f"c{i:03d}") for i in range(100)]
        index = hybrid(many, list(reversed(many)), fusion_depth=20)

        index.search("q", k=5)

        self.assertEqual(index.dense.asked_for, [20])
        self.assertEqual(index.lexical.asked_for, [20])

    def test_depth_never_falls_below_k(self):
        """A k larger than fusion_depth must still be fillable."""
        many = [make_chunk(f"c{i:03d}") for i in range(100)]
        index = hybrid(many, many, fusion_depth=20)

        results = index.search("q", k=50)

        self.assertEqual(index.dense.asked_for, [50])
        self.assertEqual(len(results), 50)

    def test_rrf_k_damps_the_value_of_a_top_rank(self):
        """With rrf_k=0 the formula is plain 1/rank, and rank 1 dominates."""
        a, b = make_chunk("a"), make_chunk("b")
        index = hybrid([a, b], [b, a], rrf_k=0)

        scores = {r.chunk.chunk_id: r.score for r in index.search("q", k=2)}

        self.assertAlmostEqual(scores["a"], 1 / 1 + 1 / 2, places=12)
        self.assertAlmostEqual(scores["b"], 1 / 2 + 1 / 1, places=12)


class TestTieBreaking(unittest.TestCase):
    def test_equal_scores_break_on_chunk_id(self):
        """Ties are routine, so the order must not be left to dict insertion."""
        # Each chunk is found by exactly one retriever, at the same rank.
        index = hybrid([make_chunk("zebra"), make_chunk("banana")],
                       [make_chunk("apple"), make_chunk("cherry")])

        results = index.search("q", k=4)
        first_two = sorted(r.score for r in results)[-2:]

        self.assertAlmostEqual(first_two[0], first_two[1], places=12)
        self.assertEqual([r.chunk.chunk_id for r in results],
                         ["apple", "zebra", "banana", "cherry"])

    def test_output_is_independent_of_which_half_is_searched_first(self):
        """Swapping the roles of the two halves must not change the result."""
        a, b, c, d = (make_chunk(x) for x in "abcd")
        forward = hybrid([a, b, c], [c, d, a]).search("q", k=4)
        swapped = hybrid([c, d, a], [a, b, c]).search("q", k=4)

        self.assertEqual([r.chunk.chunk_id for r in forward],
                         [r.chunk.chunk_id for r in swapped])

    def test_repeated_searches_are_identical(self):
        chunks = [make_chunk(f"c{i:02d}") for i in range(30)]
        index = hybrid(chunks, list(reversed(chunks)))

        first = [(r.chunk.chunk_id, r.score, r.rank) for r in index.search("q", k=10)]
        second = [(r.chunk.chunk_id, r.score, r.rank) for r in index.search("q", k=10)]

        self.assertEqual(first, second)


class TestInterface(unittest.TestCase):
    """Hybrid must be substitutable for the other two strategies."""

    def setUp(self):
        self.chunks = [make_chunk(f"c{i}") for i in range(6)]
        self.index = hybrid(self.chunks, list(reversed(self.chunks)))

    def test_returns_retrieved_chunks(self):
        for result in self.index.search("q", k=4):
            self.assertIsInstance(result, RetrievedChunk)
            self.assertIsInstance(result.chunk, Chunk)
            self.assertIsInstance(result.score, float)
            self.assertIsInstance(result.rank, int)

    def test_ranks_are_one_based_and_contiguous(self):
        self.assertEqual([r.rank for r in self.index.search("q", k=6)],
                         [1, 2, 3, 4, 5, 6])

    def test_scores_are_descending(self):
        scores = [r.score for r in self.index.search("q", k=6)]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_k_caps_the_result_count(self):
        self.assertEqual(len(self.index.search("q", k=2)), 2)

    def test_non_positive_k_is_rejected(self):
        for k in (0, -1):
            with self.assertRaises(ValueError):
                self.index.search("q", k=k)

    def test_empty_index_returns_nothing(self):
        empty = HybridIndex(dense=StubIndex([], size=0), lexical=StubIndex([], size=0),
                            rrf_k=RRF_K, fusion_depth=FUSION_DEPTH)

        self.assertEqual(len(empty), 0)
        self.assertEqual(empty.search("q", k=5), [])

    def test_search_signature_matches_the_other_strategies(self):
        import inspect

        expected = list(inspect.signature(VectorIndex.search).parameters)
        self.assertEqual(list(inspect.signature(HybridIndex.search).parameters), expected)
        self.assertEqual(list(inspect.signature(Bm25Index.search).parameters), expected)

    def test_mismatched_halves_are_rejected(self):
        with self.assertRaises(ValueError):
            HybridIndex(dense=StubIndex([make_chunk("a")]),
                        lexical=StubIndex([make_chunk("a"), make_chunk("b")]),
                        rrf_k=RRF_K, fusion_depth=FUSION_DEPTH)

    def test_invalid_fusion_settings_are_rejected(self):
        for rrf_k, depth in ((-1, 20), (60, 0), (60, -5)):
            with self.assertRaises(ValueError):
                HybridIndex(dense=StubIndex([]), lexical=StubIndex([]),
                            rrf_k=rrf_k, fusion_depth=depth)


class TestFixedSettings(unittest.TestCase):
    def test_the_constants_are_the_agreed_values(self):
        """Fixed implementation choices for the first experiment, not knobs."""
        self.assertEqual(RRF_K, 60)
        self.assertEqual(FUSION_DEPTH, 20)

    def test_build_uses_them_by_default(self):
        import inspect

        parameters = inspect.signature(HybridIndex.build).parameters
        self.assertEqual(parameters["rrf_k"].default, RRF_K)
        self.assertEqual(parameters["fusion_depth"].default, FUSION_DEPTH)

    def test_registry_names_all_three_strategies(self):
        self.assertEqual(
            STRATEGIES,
            {"dense": VectorIndex, "bm25": Bm25Index, "hybrid": HybridIndex},
        )


class TestComposition(unittest.TestCase):
    """Real halves, to prove the wiring - not the arithmetic, which is above."""

    def setUp(self):
        import numpy as np

        self.texts = ["alpha beta", "beta gamma", "gamma delta", "delta alpha"]
        self.chunks = [make_chunk(f"c{i}", text=t) for i, t in enumerate(self.texts)]
        self.embed_calls = []

        def fake_embed_documents(texts, model=None, batch_size=32, show_progress=False):
            self.embed_calls.append(("documents", model, len(texts)))
            return np.eye(len(texts), 384, dtype=np.float32)

        def fake_embed_query(text, model=None):
            self.embed_calls.append(("query", model, 1))
            return np.eye(1, 384, dtype=np.float32)[0]

        self._original = (rag.retrieval.embed_documents, rag.retrieval.embed_query)
        rag.retrieval.embed_documents = fake_embed_documents
        rag.retrieval.embed_query = fake_embed_query

    def tearDown(self):
        rag.retrieval.embed_documents, rag.retrieval.embed_query = self._original

    def test_build_index_returns_a_hybrid_with_both_halves(self):
        index = build_index(self.chunks, strategy="hybrid",
                            model="BAAI/bge-small-en-v1.5")

        self.assertIsInstance(index, HybridIndex)
        self.assertIsInstance(index.dense, VectorIndex)
        self.assertIsInstance(index.lexical, Bm25Index)
        self.assertEqual(len(index), 4)

    def test_hybrid_embeds_the_corpus_exactly_once(self):
        """Hybrid pays a dense run's embedding cost - once, not twice."""
        build_index(self.chunks, strategy="hybrid", model="BAAI/bge-small-en-v1.5")

        document_calls = [c for c in self.embed_calls if c[0] == "documents"]
        self.assertEqual(document_calls, [("documents", "BAAI/bge-small-en-v1.5", 4)])

    def test_the_embedding_model_is_reachable_for_recording(self):
        """The runner records this on the summary row, so it must be right."""
        index = build_index(self.chunks, strategy="hybrid",
                            model="BAAI/bge-small-en-v1.5")

        self.assertEqual(index.model, "BAAI/bge-small-en-v1.5")
        self.assertEqual(index.dense.model, "BAAI/bge-small-en-v1.5")

    def test_the_halves_behave_exactly_as_standalone_indexes(self):
        """Hybrid must not perturb either retriever it composes."""
        index = build_index(self.chunks, strategy="hybrid",
                            model="BAAI/bge-small-en-v1.5")

        standalone_dense = build_index(self.chunks, strategy="dense",
                                       model="BAAI/bge-small-en-v1.5")
        standalone_bm25 = build_index(self.chunks, strategy="bm25")

        for query in ("alpha", "beta gamma", "delta"):
            self.assertEqual(
                [(r.chunk.chunk_id, r.score, r.rank)
                 for r in index.dense.search(query, k=4)],
                [(r.chunk.chunk_id, r.score, r.rank)
                 for r in standalone_dense.search(query, k=4)],
            )
            self.assertEqual(
                [(r.chunk.chunk_id, r.score, r.rank)
                 for r in index.lexical.search(query, k=4)],
                [(r.chunk.chunk_id, r.score, r.rank)
                 for r in standalone_bm25.search(query, k=4)],
            )

    def test_fused_results_come_from_the_indexed_chunks(self):
        index = build_index(self.chunks, strategy="hybrid",
                            model="BAAI/bge-small-en-v1.5")

        known = {c.chunk_id for c in self.chunks}
        for result in index.search("alpha beta", k=4):
            self.assertIn(result.chunk.chunk_id, known)


if __name__ == "__main__":
    unittest.main()
