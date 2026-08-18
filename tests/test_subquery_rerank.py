"""Tests for subquery-aware reranking.

The design rests on one claim: cross-encoder logits are comparable *within* a
query and not *across* queries. The traced runs bear that out - one question's
whole candidate pool scored negative while another's fifth place sat above 2.5.
So selection across queries must use ranks only, and the tests below pin that:
a subquery whose scores are wildly higher than everyone else's must still win
exactly its allotted slots and no more.

The cross-encoder is replaced throughout with a scoring function the test
chooses, so what is checked is the selection policy, not the model's taste.

Run with:  python3 -m unittest discover tests
"""

import unittest

import rag.reranking
from rag.reranking import RERANK_MODEL, SELECTION_FLOOR, rerank_subquery_aware
from rag.types import Chunk, RetrievedChunk


def pool(*chunk_ids):
    return [
        RetrievedChunk(
            chunk=Chunk(chunk_id=cid, doc_id="doc", text=cid, start=0, end=len(cid)),
            score=0.0,
            rank=i + 1,
        )
        for i, cid in enumerate(chunk_ids)
    ]


class FakeCrossEncoder:
    """Scores (query, text) pairs from a table the test supplies."""

    def __init__(self, table):
        self.table = table
        self.calls = 0

    def predict(self, pairs):
        self.calls += 1
        return [self.table[q][t] for q, t in pairs]


class SubqueryRerankTestCase(unittest.TestCase):
    def setUp(self):
        self._saved = dict(rag.reranking._loaded)

    def tearDown(self):
        rag.reranking._loaded.clear()
        rag.reranking._loaded.update(self._saved)

    def use(self, table):
        fake = FakeCrossEncoder(table)
        rag.reranking._loaded[RERANK_MODEL] = fake
        return fake


class TestSelectionPolicy(SubqueryRerankTestCase):
    def test_original_keeps_the_floor_and_subqueries_get_the_rest(self):
        chunks = ["a", "b", "c", "d", "e", "f", "g"]
        self.use({
            "Q":  {"a": 9, "b": 8, "c": 7, "d": 6, "e": 5, "f": 4, "g": 3},
            "s1": {"a": 0, "b": 0, "c": 0, "d": 9, "e": 1, "f": 2, "g": 3},
            "s2": {"a": 0, "b": 0, "c": 0, "d": 1, "e": 9, "f": 2, "g": 3},
        })
        results, diag = rerank_subquery_aware(["Q", "s1", "s2"], pool(*chunks), k=5)

        ids = [r.chunk.chunk_id for r in results]
        self.assertEqual(ids, ["a", "b", "c", "d", "e"])
        self.assertEqual([diag[i]["selected_by"] for i in ids], [0, 0, 0, 1, 2])

    def test_a_subquery_slot_promotes_a_chunk_the_original_ranked_low(self):
        """The whole point: 'z' is last for the original, first for a subquery."""
        chunks = ["a", "b", "c", "d", "z"]
        self.use({
            "Q":  {"a": 9, "b": 8, "c": 7, "d": 6, "z": -5},
            "s1": {"a": 0, "b": 0, "c": 0, "d": 0, "z": 9},
            "s2": {"a": 1, "b": 0, "c": 0, "d": 0, "z": 0},
        })
        results, diag = rerank_subquery_aware(["Q", "s1", "s2"], pool(*chunks), k=5)

        ids = [r.chunk.chunk_id for r in results]
        self.assertIn("z", ids)
        self.assertEqual(diag["z"]["selected_by"], 1)
        self.assertEqual(diag["z"]["ranks"][0], 5, "last for the original query")
        self.assertEqual(diag["z"]["ranks"][1], 1, "first for its subquery")

    def test_one_subquery_cannot_dominate_however_high_its_logits(self):
        """Ranks decide across queries, so scale is irrelevant."""
        chunks = ["a", "b", "c", "x", "y", "z"]
        self.use({
            "Q":  {"a": 1.0, "b": 0.9, "c": 0.8, "x": 0.1, "y": 0.05, "z": 0.0},
            # Enormous scores, but still only one slot.
            "s1": {"a": 500, "b": 400, "c": 300, "x": 999, "y": 998, "z": 997},
            "s2": {"a": -9, "b": -9, "c": -9, "x": -9, "y": -1, "z": -9},
        })
        results, diag = rerank_subquery_aware(["Q", "s1", "s2"], pool(*chunks), k=5)

        by_source = [diag[r.chunk.chunk_id]["selected_by"] for r in results]
        self.assertEqual(by_source.count(1), 1, "s1 took more than its slot")
        self.assertEqual(by_source.count(2), 1)
        self.assertEqual(by_source.count(0), 3)

    def test_negative_scores_do_not_disqualify_a_subquery_pick(self):
        """q028's pool scored negative throughout; only ordering matters."""
        self.use({
            "Q":  {"a": -1, "b": -2, "c": -3, "d": -4},
            "s1": {"a": -9, "b": -9, "c": -9, "d": -0.5},
        })
        results, diag = rerank_subquery_aware(["Q", "s1"], pool("a", "b", "c", "d"), k=4)

        self.assertEqual(diag["d"]["selected_by"], 1)
        self.assertEqual([r.chunk.chunk_id for r in results], ["a", "b", "c", "d"])

    def test_duplicates_across_orderings_are_taken_once(self):
        self.use({
            "Q":  {"a": 9, "b": 8, "c": 7, "d": 6},
            "s1": {"a": 9, "b": 0, "c": 0, "d": 1},   # tops with 'a', already taken
            "s2": {"a": 0, "b": 0, "c": 0, "d": 9},
        })
        results, diag = rerank_subquery_aware(["Q", "s1", "s2"], pool("a", "b", "c", "d"), k=4)

        ids = [r.chunk.chunk_id for r in results]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(diag["d"]["selected_by"], 1, "s1 falls through to its next unused pick")

    def test_subqueries_keep_taking_turns_beyond_the_floor(self):
        """Two subqueries, two slots - one each, in query order."""
        self.use({
            "Q":  {"a": 9, "b": 8, "c": 7, "d": 6, "e": 5},
            "s1": {"a": 0, "b": 0, "c": 0, "d": 9, "e": 1},
            "s2": {"a": 0, "b": 0, "c": 0, "d": 1, "e": 9},
        })
        results, diag = rerank_subquery_aware(["Q", "s1", "s2"], pool(*"abcde"), k=5)

        self.assertEqual([diag[r.chunk.chunk_id]["selected_by"] for r in results],
                         [0, 0, 0, 1, 2])

    def test_tops_up_from_the_original_when_there_are_no_subqueries(self):
        """With one query the floor covers 3 slots; the rest are topped up.

        This is the path that keeps single-query behaviour identical, and it is
        the only way the top-up branch is reached - every ordering covers the
        whole pool, so a subquery cannot be exhausted while chunks remain.
        """
        self.use({"Q": {"a": 9, "b": 8, "c": 7, "d": 6, "e": 5}})
        results, diag = rerank_subquery_aware(["Q"], pool(*"abcde"), k=5)

        self.assertEqual([r.chunk.chunk_id for r in results], list("abcde"))
        self.assertEqual({diag[r.chunk.chunk_id]["selected_by"] for r in results}, {0})

    def test_scores_come_from_the_original_question(self):
        """One unit in the score column; per-query values live in diagnostics."""
        self.use({"Q": {"a": 9, "b": 8, "z": -5}, "s1": {"a": 0, "b": 0, "z": 9}})
        results, diag = rerank_subquery_aware(["Q", "s1"], pool("a", "b", "z"), k=3)

        by_id = {r.chunk.chunk_id: r.score for r in results}
        self.assertAlmostEqual(by_id["z"], -5.0, places=5)
        self.assertEqual(diag["z"]["scores"], [-5.0, 9.0])


class TestMechanics(SubqueryRerankTestCase):
    def test_one_pass_per_query(self):
        fake = self.use({"Q": {"a": 1}, "s1": {"a": 1}, "s2": {"a": 1}})
        rerank_subquery_aware(["Q", "s1", "s2"], pool("a"), k=1)

        self.assertEqual(fake.calls, 3, "one predict call per query, no more")

    def test_ranks_are_one_based_and_contiguous(self):
        self.use({"Q": {c: 10 - i for i, c in enumerate("abcde")},
                  "s1": {c: i for i, c in enumerate("abcde")}})
        results, _ = rerank_subquery_aware(["Q", "s1"], pool(*"abcde"), k=5)

        self.assertEqual([r.rank for r in results], [1, 2, 3, 4, 5])

    def test_repeated_calls_are_identical(self):
        table = {"Q": {c: 10 - i for i, c in enumerate("abcdef")},
                 "s1": {c: i for i, c in enumerate("abcdef")}}
        self.use(table)
        first, _ = rerank_subquery_aware(["Q", "s1"], pool(*"abcdef"), k=4)
        second, _ = rerank_subquery_aware(["Q", "s1"], pool(*"abcdef"), k=4)

        self.assertEqual([r.chunk.chunk_id for r in first],
                         [r.chunk.chunk_id for r in second])

    def test_ties_break_on_chunk_id(self):
        self.use({"Q": {"b": 1, "a": 1, "c": 1}})
        results, _ = rerank_subquery_aware(["Q"], pool("b", "a", "c"), k=3)

        self.assertEqual([r.chunk.chunk_id for r in results], ["a", "b", "c"])

    def test_a_single_query_reproduces_plain_reranking(self):
        """With no subqueries the floor plus top-up is ordinary reranking."""
        from rag.reranking import rerank
        self.use({"Q": {"a": 3, "b": 9, "c": 5, "d": 1}})
        plain = rerank("Q", pool("a", "b", "c", "d"), k=3)
        aware, _ = rerank_subquery_aware(["Q"], pool("a", "b", "c", "d"), k=3)

        self.assertEqual([r.chunk.chunk_id for r in plain],
                         [r.chunk.chunk_id for r in aware])

    def test_invalid_inputs(self):
        self.use({"Q": {"a": 1}})
        self.assertEqual(rerank_subquery_aware(["Q"], [], k=5), ([], {}))
        with self.assertRaises(ValueError):
            rerank_subquery_aware(["Q"], pool("a"), k=0)
        with self.assertRaises(ValueError):
            rerank_subquery_aware([], pool("a"), k=1)


class TestDefaults(unittest.TestCase):
    def test_disabled_by_default(self):
        from rag.experiment import DEFAULTS

        self.assertIs(DEFAULTS["retrieval"]["subquery_rerank"], False)

    def test_the_floor_is_the_agreed_value(self):
        self.assertEqual(SELECTION_FLOOR, 3)


if __name__ == "__main__":
    unittest.main()
