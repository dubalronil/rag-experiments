"""Tests for BM25 retrieval.

BM25 is arithmetic with no ground truth to check against: a wrong exponent or a
flipped IDF still returns a plausible-looking ranked list, and the experiment
comparing it against dense retrieval would just report the wrong winner. So
these tests assert on the properties the formula is supposed to have - rare
terms outrank common ones, long chunks are penalised, repetition saturates -
rather than on particular score values.

They also pin the part that makes the strategies interchangeable: BM25 returns
the same RetrievedChunk shape as dense search, and building it never touches
the embedding layer.

Run with:  python3 -m unittest discover tests
"""

import unittest

import rag.embedding
import rag.retrieval
from rag.retrieval import (
    BM25_B,
    BM25_K1,
    STRATEGIES,
    Bm25Index,
    VectorIndex,
    build_index,
    tokenize,
)
from rag.types import Chunk, RetrievedChunk


def make_chunk(text, chunk_id="c", doc_id="doc"):
    return Chunk(chunk_id=chunk_id, doc_id=doc_id, text=text, start=0, end=len(text))


def make_chunks(*texts):
    return [make_chunk(text, chunk_id=f"c{i}") for i, text in enumerate(texts)]


class TestTokenize(unittest.TestCase):
    """The tokenizer is the whole lexical model - what it splits, BM25 sees."""

    def test_lowercases(self):
        self.assertEqual(tokenize("Chicago BULLS"), ["chicago", "bulls"])

    def test_splits_on_punctuation_and_whitespace(self):
        self.assertEqual(
            tokenize("the Bulls' record: 72-10, best ever."),
            ["the", "bulls", "record", "72", "10", "best", "ever"],
        )

    def test_keeps_digits_as_tokens(self):
        """Numbers are content here - many eval questions turn on a year."""
        self.assertEqual(tokenize("won in 1996"), ["won", "in", "1996"])

    def test_splits_hyphenated_and_decimal_numbers(self):
        """A documented consequence of the simple regex, not an accident.

        "1995-96" is two tokens, so a query for "1995-96" also matches a chunk
        that only says "1995". Pinned so the behaviour cannot drift silently.
        """
        self.assertEqual(tokenize("the 1995-96 season"), ["the", "1995", "96", "season"])
        self.assertEqual(tokenize("$4.8 million"), ["4", "8", "million"])

    def test_drops_symbols_entirely(self):
        self.assertEqual(tokenize("--- *** ###"), [])

    def test_empty_text(self):
        self.assertEqual(tokenize(""), [])


class TestIndexStructure(unittest.TestCase):
    def test_postings_map_terms_to_chunk_positions_and_counts(self):
        index = Bm25Index.build(make_chunks("alpha beta alpha", "beta gamma"))

        self.assertEqual(index.postings["alpha"], {0: 2})
        self.assertEqual(index.postings["beta"], {0: 1, 1: 1})
        self.assertEqual(index.postings["gamma"], {1: 1})
        self.assertNotIn("delta", index.postings)

    def test_lengths_and_average(self):
        index = Bm25Index.build(make_chunks("one two three four", "five two"))

        self.assertEqual(list(index.lengths), [4, 2])
        self.assertEqual(index.average_length, 3.0)

    def test_empty_corpus_builds_without_dividing_by_zero(self):
        index = Bm25Index.build([])

        self.assertEqual(len(index), 0)
        self.assertEqual(index.average_length, 0.0)
        self.assertEqual(index.search("anything", k=5), [])

    def test_length_mismatch_is_rejected(self):
        import numpy as np

        with self.assertRaises(ValueError):
            Bm25Index(
                chunks=tuple(make_chunks("a", "b")),
                postings={},
                lengths=np.asarray([1.0], dtype=np.float32),
                average_length=1.0,
            )


class TestRanking(unittest.TestCase):
    def test_the_chunk_containing_the_query_term_wins(self):
        index = Bm25Index.build(
            make_chunks(
                "the team played well",
                "Jordan scored sixty three points",
                "the season ended",
            )
        )
        results = index.search("Jordan", k=3)

        self.assertEqual(results[0].chunk.chunk_id, "c1")
        self.assertGreater(results[0].score, 0.0)

    def test_rare_terms_outrank_common_ones(self):
        """IDF: a term in every chunk should not decide the ranking.

        Both candidates contain "season". Only one contains "dynasty", which
        appears nowhere else - so a query for both must prefer that one.
        """
        index = Bm25Index.build(
            make_chunks(
                "season season season season",
                "season dynasty",
                "season report",
                "season summary",
            )
        )
        results = index.search("season dynasty", k=4)

        self.assertEqual(results[0].chunk.chunk_id, "c1")

    def test_shorter_chunks_win_when_term_counts_tie(self):
        """Length normalisation: one hit in a short chunk means more."""
        index = Bm25Index.build(
            make_chunks(
                "playoffs",
                "playoffs " + "filler words here and there " * 20,
                "unrelated text",
            )
        )
        results = index.search("playoffs", k=2)

        self.assertEqual(results[0].chunk.chunk_id, "c0")
        self.assertGreater(results[0].score, results[1].score)

    def test_repetition_saturates(self):
        """k1 caps repetition: 10 occurrences are worth well under 10 times 1.

        Without saturation a chunk could win purely by repeating a word, which
        is the failure mode BM25's tf term exists to prevent.
        """
        index = Bm25Index.build(make_chunks("rebound", "rebound " * 10, "other"))
        by_id = {r.chunk.chunk_id: r.score for r in index.search("rebound", k=3)}

        self.assertGreater(by_id["c1"], by_id["c0"])
        self.assertLess(by_id["c1"], by_id["c0"] * 10)

    def test_query_terms_absent_from_the_corpus_score_zero(self):
        index = Bm25Index.build(make_chunks("alpha", "beta"))
        results = index.search("nonexistentword", k=2)

        # Still k results - ranking is meaningless, and the retrieval metrics
        # score it as a miss rather than the retriever pretending it failed.
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r.score == 0.0 for r in results))

    def test_scores_are_descending(self):
        index = Bm25Index.build(
            make_chunks("alpha beta", "alpha", "gamma", "alpha beta gamma")
        )
        scores = [r.score for r in index.search("alpha beta", k=4)]

        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_repeated_query_terms_are_counted_once(self):
        """Documented choice: asking twice is not asking harder."""
        index = Bm25Index.build(make_chunks("alpha beta", "gamma"))
        once = index.search("alpha", k=1)[0].score
        twice = index.search("alpha alpha alpha", k=1)[0].score

        self.assertAlmostEqual(once, twice, places=6)

    def test_matches_the_bm25_formula_by_hand(self):
        """One score computed independently, to catch a rearranged formula."""
        import math

        # Two chunks, three tokens each -> average_length 3, so the length
        # penalty is exactly 1 and drops out.
        index = Bm25Index.build(make_chunks("alpha beta beta", "gamma delta epsilon"))
        score = index.search("beta", k=1)[0].score

        total, document_frequency, term_frequency = 2, 1, 2
        idf = math.log(1 + (total - document_frequency + 0.5) / (document_frequency + 0.5))
        length_penalty = 1 - BM25_B + BM25_B * 1.0
        expected = idf * term_frequency * (BM25_K1 + 1) / (
            term_frequency + BM25_K1 * length_penalty
        )

        self.assertAlmostEqual(score, expected, places=5)


class TestRetrievedChunkInterface(unittest.TestCase):
    """BM25 must be substitutable for dense search everywhere downstream."""

    def setUp(self):
        self.index = Bm25Index.build(
            make_chunks("alpha beta", "beta gamma", "gamma delta", "delta alpha")
        )

    def test_returns_retrieved_chunks(self):
        for result in self.index.search("alpha", k=4):
            self.assertIsInstance(result, RetrievedChunk)
            self.assertIsInstance(result.chunk, Chunk)
            self.assertIsInstance(result.score, float)
            self.assertIsInstance(result.rank, int)

    def test_ranks_are_one_based_and_contiguous(self):
        ranks = [r.rank for r in self.index.search("alpha beta", k=4)]

        self.assertEqual(ranks, [1, 2, 3, 4])

    def test_k_caps_the_result_count(self):
        self.assertEqual(len(self.index.search("alpha", k=2)), 2)

    def test_k_larger_than_the_index_returns_everything(self):
        self.assertEqual(len(self.index.search("alpha", k=99)), 4)

    def test_non_positive_k_is_rejected(self):
        """Same contract as VectorIndex.search."""
        for k in (0, -1):
            with self.assertRaises(ValueError):
                self.index.search("alpha", k=k)

    def test_the_original_chunk_is_reachable(self):
        result = self.index.search("alpha", k=1)[0]

        self.assertEqual(result.chunk.doc_id, "doc")
        self.assertEqual(result.chunk.text, self.index.chunks[0].text)

    def test_search_signature_matches_vector_index(self):
        import inspect

        self.assertEqual(
            list(inspect.signature(Bm25Index.search).parameters),
            list(inspect.signature(VectorIndex.search).parameters),
        )


class TestBuildIndex(unittest.TestCase):
    def test_registry_maps_these_two_strategies(self):
        """Only the two this file covers - completeness is tested in test_hybrid."""
        self.assertIs(STRATEGIES["dense"], VectorIndex)
        self.assertIs(STRATEGIES["bm25"], Bm25Index)

    def test_bm25_never_embeds(self):
        """The point of dispatching in build_index rather than passing a model.

        Both embedding entry points are replaced with something that fails
        loudly, so any accidental call - a model load, an API request, a paid
        OpenAI batch - shows up as a test failure instead of a bill.
        """

        def explode(*args, **kwargs):
            raise AssertionError("bm25 must not embed anything")

        original = rag.embedding.embed_documents, rag.embedding.embed_query
        patched = rag.retrieval.embed_documents, rag.retrieval.embed_query
        rag.embedding.embed_documents = rag.embedding.embed_query = explode
        rag.retrieval.embed_documents = rag.retrieval.embed_query = explode
        try:
            index = build_index(
                make_chunks("alpha beta", "gamma"),
                strategy="bm25",
                # An embedding model is still named, exactly as an inherited
                # config would name one. It must be ignored, not used.
                model="text-embedding-3-large",
            )
            results = index.search("alpha", k=1)
        finally:
            rag.embedding.embed_documents, rag.embedding.embed_query = original
            rag.retrieval.embed_documents, rag.retrieval.embed_query = patched

        self.assertIsInstance(index, Bm25Index)
        self.assertEqual(results[0].chunk.chunk_id, "c0")

    def test_defaults_to_dense(self):
        """A caller that names no strategy gets the pre-existing behaviour."""
        import inspect

        self.assertEqual(
            inspect.signature(build_index).parameters["strategy"].default, "dense"
        )

    def test_dense_builds_a_vector_index(self):
        calls = {}

        def fake_embed_documents(texts, model=None, batch_size=32, show_progress=False):
            import numpy as np

            calls["model"] = model
            calls["texts"] = list(texts)
            return np.eye(len(texts), 384, dtype=np.float32)

        original = rag.retrieval.embed_documents
        rag.retrieval.embed_documents = fake_embed_documents
        try:
            index = build_index(
                make_chunks("alpha", "beta"),
                strategy="dense",
                model="BAAI/bge-small-en-v1.5",
            )
        finally:
            rag.retrieval.embed_documents = original

        self.assertIsInstance(index, VectorIndex)
        # The model reaches the embedding call unchanged, and the index records
        # it so queries are embedded by the same model as the documents.
        self.assertEqual(calls["model"], "BAAI/bge-small-en-v1.5")
        self.assertEqual(calls["texts"], ["alpha", "beta"])
        self.assertEqual(index.model, "BAAI/bge-small-en-v1.5")

    def test_unknown_strategy_is_rejected_by_name(self):
        with self.assertRaises(ValueError) as caught:
            build_index(make_chunks("alpha"), strategy="splade")

        message = str(caught.exception)
        self.assertIn("splade", message)
        self.assertIn("bm25", message)
        self.assertIn("dense", message)


class TestConfigDefault(unittest.TestCase):
    def test_dense_is_the_default_strategy(self):
        """Existing configs name no strategy, so they must still run dense."""
        from rag.experiment import DEFAULTS

        self.assertEqual(DEFAULTS["retrieval"]["strategy"], "dense")


if __name__ == "__main__":
    unittest.main()
