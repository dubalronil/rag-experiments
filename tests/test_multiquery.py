"""Tests for deterministic multi-query retrieval.

Two things here fail silently rather than loudly. A splitter that produces a
degenerate fragment still returns a query, and retrieval still returns chunks -
just worse ones. And a merge that drops or reorders candidates still returns a
pool of the right size. Both would show up only as a slightly worse metric,
which is indistinguishable from the experiment simply not working.

The splitter is also the whole leak-prevention argument: every subquery must be
built from the question's own text, never from anything a model knows. That is
structural rather than testable directly, so what is tested is the observable
consequence - the output is always a substring-derived rearrangement of the
input.

Run with:  python3 -m unittest discover tests
"""

import unittest

from rag.multiquery import (
    MAX_SUBQUERIES,
    merge_round_robin,
    queries_for,
    split_question,
)
from rag.types import Chunk, RetrievedChunk


def run(*chunk_ids):
    """A ranked result list identified only by chunk id."""
    return [
        RetrievedChunk(
            chunk=Chunk(chunk_id=cid, doc_id="doc", text=cid, start=0, end=len(cid)),
            score=1.0 - i / 100,
            rank=i + 1,
        )
        for i, cid in enumerate(chunk_ids)
    ]


class TestSplitting(unittest.TestCase):
    def test_splits_on_comma_and(self):
        parts = split_question(
            "Which shuttle disaster contributed to delaying Hubble's launch, "
            "and which orbiter eventually carried the telescope into space?")

        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[0], "Which shuttle disaster contributed to delaying Hubble's launch")
        self.assertEqual(parts[1], "which orbiter eventually carried the telescope into space?")

    def test_splits_on_bare_and(self):
        parts = split_question(
            "Which orbiter both returned the Shuttle fleet to flight after the "
            "Challenger disaster and carried the Hubble Space Telescope into orbit?")

        self.assertEqual(len(parts), 2)
        self.assertIn("Challenger disaster", parts[0])
        self.assertIn("Hubble Space Telescope", parts[1])

    def test_a_leading_sentence_is_prepended_to_every_part(self):
        """Otherwise a trailing clause's pronoun has no referent to embed."""
        parts = split_question(
            "The James Webb Space Telescope is named after a NASA administrator. "
            "Which program did he lead, and which telescope is Webb the successor to?")

        self.assertEqual(len(parts), 2)
        for part in parts:
            self.assertTrue(part.startswith("The James Webb Space Telescope is named after"))
        self.assertIn("Which program did he lead", parts[0])
        self.assertIn("successor", parts[1])

    def test_a_question_with_no_conjunction_does_not_split(self):
        self.assertEqual(split_question("Who is the Hubble Space Telescope named after?"), [])

    def test_a_question_whose_parts_are_implied_does_not_split(self):
        """Space q026: two facts, neither of them a clause in the question."""
        self.assertEqual(
            split_question("Both Voyager probes eventually crossed into interstellar "
                           "space. Roughly how many years separated the two crossings?"),
            [])

    def test_fragments_below_the_token_floor_are_discarded(self):
        """'X and Y' where a side is one word yields nothing worth searching."""
        self.assertEqual(split_question("Which team won and lost?"), [])

    def test_never_returns_more_than_the_cap(self):
        parts = split_question("Name the first thing here, and the second thing there, "
                               "and the third thing everywhere?")

        self.assertLessEqual(len(parts), MAX_SUBQUERIES)

    def test_every_part_is_built_from_the_question_text(self):
        """The leak-prevention property, in its observable form."""
        question = ("Which ISS module made continuous human presence possible, "
                    "and which crew arrived to begin it?")

        for part in split_question(question):
            for word in part.split():
                self.assertIn(word, question, f"{word!r} is not from the question")

    def test_splitting_is_deterministic(self):
        question = "How long was it between the Apollo 1 fire and the launch of Apollo 11?"

        self.assertEqual(split_question(question), split_question(question))


class TestQueriesFor(unittest.TestCase):
    def test_disabled_returns_the_question_alone(self):
        q = "Which team did X beat, and who led them?"

        self.assertEqual(queries_for(q, multi_query=False), [q])

    def test_the_original_question_is_always_first(self):
        q = "Which team did X beat, and who led them?"
        queries = queries_for(q, multi_query=True)

        self.assertEqual(queries[0], q)
        self.assertEqual(len(queries), 3)

    def test_an_unsplittable_question_yields_one_query_even_when_enabled(self):
        q = "Who is the Hubble Space Telescope named after?"

        self.assertEqual(queries_for(q, multi_query=True), [q])


class TestRoundRobinMerge(unittest.TestCase):
    def test_interleaves_one_candidate_per_run_in_turn(self):
        merged, _ = merge_round_robin([run("a1", "a2", "a3"), run("b1", "b2", "b3")], 6)

        self.assertEqual([r.chunk.chunk_id for r in merged],
                         ["a1", "b1", "a2", "b2", "a3", "b3"])

    def test_duplicates_are_taken_once_at_their_earliest_position(self):
        merged, source = merge_round_robin([run("x", "a"), run("x", "b")], 4)

        self.assertEqual([r.chunk.chunk_id for r in merged], ["x", "a", "b"])
        self.assertEqual(source["x"], 0, "first run to offer it owns it")

    def test_the_pool_is_capped_at_the_limit(self):
        merged, _ = merge_round_robin([run(*[f"a{i}" for i in range(50)]),
                                       run(*[f"b{i}" for i in range(50)])], 50)

        self.assertEqual(len(merged), 50)

    def test_a_fixed_budget_displaces_the_first_run_s_deeper_candidates(self):
        """The intended trade, pinned: this is reallocation, not addition.

        Alone, run 0 would fill all six slots with a0-a5. Sharing the same six
        with two other queries, it keeps only its top two.
        """
        merged, source = merge_round_robin(
            [run("a0", "a1", "a2", "a3", "a4", "a5"),
             run("b0", "b1"), run("c0", "c1")], 6)

        kept = [r.chunk.chunk_id for r in merged]
        self.assertEqual(kept, ["a0", "b0", "c0", "a1", "b1", "c1"])
        self.assertNotIn("a2", kept)
        self.assertEqual(sum(1 for cid in kept if source[cid] == 0), 2)

    def test_ranks_are_renumbered_from_one(self):
        merged, _ = merge_round_robin([run("a1", "a2"), run("b1")], 3)

        self.assertEqual([r.rank for r in merged], [1, 2, 3])

    def test_provenance_names_the_contributing_run(self):
        _, source = merge_round_robin([run("a1"), run("b1"), run("c1")], 3)

        self.assertEqual((source["a1"], source["b1"], source["c1"]), (0, 1, 2))

    def test_uneven_run_lengths_are_handled(self):
        merged, _ = merge_round_robin([run("a1"), run("b1", "b2", "b3")], 10)

        self.assertEqual([r.chunk.chunk_id for r in merged], ["a1", "b1", "b2", "b3"])

    def test_merging_is_deterministic(self):
        runs = [run("a1", "a2", "x"), run("x", "b2"), run("c1", "a2")]
        first, _ = merge_round_robin(runs, 5)
        second, _ = merge_round_robin(runs, 5)

        self.assertEqual([r.chunk.chunk_id for r in first],
                         [r.chunk.chunk_id for r in second])

    def test_empty_and_invalid_inputs(self):
        self.assertEqual(merge_round_robin([], 5)[0], [])
        with self.assertRaises(ValueError):
            merge_round_robin([run("a")], 0)


class TestPreservationWhenDisabled(unittest.TestCase):
    def test_disabled_is_the_default(self):
        from rag.experiment import DEFAULTS

        self.assertIs(DEFAULTS["retrieval"]["multi_query"], False)

    def test_a_single_run_merges_to_itself(self):
        """With one query the merge must be an identity on order and ids."""
        only = run("a", "b", "c", "d")
        merged, source = merge_round_robin([only], 4)

        self.assertEqual([r.chunk.chunk_id for r in merged],
                         [r.chunk.chunk_id for r in only])
        self.assertEqual([r.rank for r in merged], [r.rank for r in only])
        self.assertEqual(set(source.values()), {0})


if __name__ == "__main__":
    unittest.main()
