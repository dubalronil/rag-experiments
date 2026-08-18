"""Tests for evidence groups and the two metrics built on them.

An evidence group is one fact a question requires. The quotes inside it are
alternatives - any one proves that fact - and a single quote may appear in
several groups when one sentence proves several facts at once.

Two things make this worth testing carefully. First, Hit@k must keep meaning
exactly what it meant before groups existed, or every recorded result silently
changes definition. Second, Coverage and Complete are averages of averages, and
an off-by-one in either denominator produces numbers that look plausible and
are wrong.

Run with:  python3 -m unittest discover tests
"""

import json
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))

from run_retrieval_eval import (  # noqa: E402
    complete_rank,
    evidence_metrics,
    first_hit_ranks,
    group_ranks,
)
from rag.types import Chunk, RetrievedChunk  # noqa: E402


def results(*texts):
    """A ranked result list whose chunk text is exactly what the test says."""
    return [
        RetrievedChunk(
            chunk=Chunk(chunk_id=f"c{i}", doc_id="doc", text=text, start=0, end=len(text)),
            score=1.0 - i / 100,
            rank=i + 1,
        )
        for i, text in enumerate(texts)
    ]


def question(*groups, doc_id="doc"):
    """Build a question from groups given as tuples of alternative quotes."""
    return {
        "qid": "qtest",
        "type": "multi_hop",
        "supporting": [
            {"quotes": [{"doc_id": doc_id, "quote": q} for q in group]}
            for group in groups
        ],
    }


class TestGroupRanks(unittest.TestCase):
    def test_one_rank_per_group_in_declared_order(self):
        q = question(("alpha",), ("beta",))

        self.assertEqual(group_ranks(q, results("beta here", "alpha here")), [2, 1])

    def test_any_alternative_satisfies_its_group(self):
        """The point of alternatives: either phrasing proves the same fact."""
        q = question(("shepard was first", "first american in space"))

        self.assertEqual(group_ranks(q, results("nothing", "first american in space")), [2])

    def test_the_earliest_alternative_sets_the_rank(self):
        q = question(("alpha", "beta"))

        self.assertEqual(group_ranks(q, results("beta", "alpha")), [1])

    def test_an_unsatisfied_group_is_none(self):
        q = question(("alpha",), ("missing",))

        self.assertEqual(group_ranks(q, results("alpha", "other")), [1, None])

    def test_one_quote_can_satisfy_several_groups(self):
        """A sentence proving two facts counts for both, at the same rank."""
        both = "the Warriors won 73 games during the 2015-16 season"
        q = question((both, "the Warriors"), (both,))

        self.assertEqual(group_ranks(q, results("filler", both)), [2, 2])

    def test_no_results_leaves_every_group_unsatisfied(self):
        self.assertEqual(group_ranks(question(("a",), ("b",)), []), [None, None])


class TestCompleteRank(unittest.TestCase):
    def test_is_the_deepest_group_rank(self):
        self.assertEqual(complete_rank([1, 4, 2]), 4)

    def test_is_none_while_any_group_is_missing(self):
        """One unretrieved fact means the question is not answerable at all."""
        self.assertIsNone(complete_rank([1, None]))

    def test_is_none_for_no_groups(self):
        self.assertIsNone(complete_rank([]))


class TestEvidenceMetrics(unittest.TestCase):
    def test_coverage_is_the_mean_share_of_groups_satisfied(self):
        # One question with 1 of 2, one with 2 of 2 -> (0.5 + 1.0) / 2.
        coverage, _ = evidence_metrics([[1, None], [1, 2]], cutoffs=(5,))

        self.assertAlmostEqual(coverage[5], 0.75, places=12)

    def test_coverage_handles_three_groups(self):
        coverage, _ = evidence_metrics([[1, 2, None]], cutoffs=(5,))

        self.assertAlmostEqual(coverage[5], 2 / 3, places=12)

    def test_complete_requires_every_group(self):
        _, complete = evidence_metrics([[1, None], [1, 2]], cutoffs=(5,))

        self.assertAlmostEqual(complete[5], 0.5, places=12)

    def test_the_cutoff_applies_to_every_group(self):
        """A group satisfied only past k is unsatisfied at k."""
        coverage, complete = evidence_metrics([[1, 7]], cutoffs=(5, 10))

        self.assertAlmostEqual(coverage[5], 0.5, places=12)
        self.assertAlmostEqual(complete[5], 0.0, places=12)
        self.assertAlmostEqual(coverage[10], 1.0, places=12)
        self.assertAlmostEqual(complete[10], 1.0, places=12)

    def test_nothing_retrieved_scores_zero(self):
        coverage, complete = evidence_metrics([[None, None]], cutoffs=(5,))

        self.assertEqual((coverage[5], complete[5]), (0.0, 0.0))

    def test_no_questions_scores_zero_rather_than_dividing_by_zero(self):
        coverage, complete = evidence_metrics([], cutoffs=(5,))

        self.assertEqual((coverage[5], complete[5]), (0.0, 0.0))


class TestInvariants(unittest.TestCase):
    def test_complete_never_exceeds_coverage(self):
        cases = [[1, None], [1, 2], [None, None], [3, 3, 9], [2]]
        coverage, complete = evidence_metrics(cases, cutoffs=(1, 3, 5, 10))

        for k in (1, 3, 5, 10):
            self.assertLessEqual(complete[k], coverage[k] + 1e-12, f"at k={k}")

    def test_coverage_never_exceeds_hit(self):
        """Hit@k is the share of questions with at least one group satisfied."""
        cases = [[1, None], [1, 2], [None, None], [3, 3, 9], [2]]
        coverage, _ = evidence_metrics(cases, cutoffs=(5,))
        hit = sum(any(r is not None and r <= 5 for r in c) for c in cases) / len(cases)

        self.assertLessEqual(coverage[5], hit + 1e-12)

    def test_single_group_questions_collapse_to_hit(self):
        """With one fact per question all three metrics are the same number."""
        cases = [[1], [None], [4], [2]]
        coverage, complete = evidence_metrics(cases, cutoffs=(5,))
        hit = sum(c[0] is not None and c[0] <= 5 for c in cases) / len(cases)

        self.assertAlmostEqual(coverage[5], hit, places=12)
        self.assertAlmostEqual(complete[5], hit, places=12)


class TestHitPreservation(unittest.TestCase):
    """Hit@k must mean exactly what it meant before groups existed."""

    def test_passage_rank_is_the_earliest_satisfied_group(self):
        q = question(("alpha",), ("beta",))

        _, passage = first_hit_ranks(q, results("beta", "alpha"))
        self.assertEqual(passage, 1)

    def test_passage_rank_is_none_when_no_group_is_satisfied(self):
        _, passage = first_hit_ranks(question(("alpha",)), results("other"))

        self.assertIsNone(passage)

    def test_document_rank_sees_every_document_in_every_group(self):
        q = {"qid": "q", "type": "multi_hop", "supporting": [
            {"quotes": [{"doc_id": "a", "quote": "alpha"}]},
            {"quotes": [{"doc_id": "b", "quote": "beta"}]}]}
        found = [
            RetrievedChunk(chunk=Chunk("c0", "b", "nothing useful", 0, 14), score=1.0, rank=1),
            RetrievedChunk(chunk=Chunk("c1", "a", "alpha", 0, 5), score=0.9, rank=2),
        ]

        document, _ = first_hit_ranks(q, found)
        self.assertEqual(document, 1)

    def test_the_old_pooled_definition_is_reproduced(self):
        """Pre-group behaviour: pool every gold quote, first match wins.

        Recomputed here from the flattened quotes, independently of the
        implementation, and compared against first_hit_ranks on the grouped
        question. They must agree however the quotes are grouped.
        """
        shared = "the Warriors won 73 games in the 2015-16 season"
        for groups in ([(shared, "warriors"), (shared,)],
                       [("alpha",), ("beta",)],
                       [("alpha", "beta")],
                       [("missing",), ("beta",)]):
            q = question(*groups)
            found = results("nothing", "beta", shared, "alpha")
            flat = [item["quote"] for g in q["supporting"] for item in g["quotes"]]
            expected = next((r.rank for r in found
                             if any(quote in r.chunk.text for quote in flat)), None)

            _, passage = first_hit_ranks(q, found)
            self.assertEqual(passage, expected, groups)


class TestMigratedEvalSets(unittest.TestCase):
    """The shipped eval files must satisfy the schema the metrics assume."""

    def datasets(self):
        for name in ("nba", "space"):
            path = pathlib.Path(f"data/{name}/eval/questions.jsonl")
            if path.exists():
                yield name, [json.loads(line) for line in
                             path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def test_every_group_is_a_quotes_list(self):
        for name, rows in self.datasets():
            for row in rows:
                for group in row["supporting"]:
                    self.assertIn("quotes", group, f"{name} {row['qid']}")
                    self.assertTrue(group["quotes"], f"{name} {row['qid']} empty group")
                    for item in group["quotes"]:
                        self.assertIn("doc_id", item)
                        self.assertIn("quote", item)

    def test_unanswerable_questions_have_no_groups(self):
        for name, rows in self.datasets():
            for row in rows:
                if row["type"] == "unanswerable":
                    self.assertEqual(row["supporting"], [], f"{name} {row['qid']}")

    def test_multi_hop_questions_require_more_than_one_fact(self):
        for name, rows in self.datasets():
            for row in rows:
                if row["type"] == "multi_hop":
                    self.assertGreater(len(row["supporting"]), 1,
                                       f"{name} {row['qid']} has one group")

    def test_multi_hop_means_no_single_quote_can_answer_it(self):
        """The definition of multi_hop, enforced rather than trusted.

        A question is multi-hop when answering it requires combining at least
        two distinct pieces of evidence - not when its answer happens to assert
        several facts. NBA q028 asks for a team and a year, so it has two
        groups, but one sentence proves both; it is factual. The test is
        therefore whether any single quote appears in *every* group.

        Asserted in both directions, because both errors have happened here:
        questions labelled multi_hop that one sentence answers, and the reverse
        would be just as wrong.
        """
        for name, rows in self.datasets():
            for row in rows:
                if row["type"] == "unanswerable":
                    continue
                groups = [{(q["doc_id"], q["quote"]) for q in g["quotes"]}
                          for g in row["supporting"]]
                answerable_by_one = bool(set.intersection(*groups)) if groups else False

                if row["type"] == "multi_hop":
                    self.assertFalse(
                        answerable_by_one,
                        f"{name} {row['qid']} is labelled multi_hop but one quote "
                        f"satisfies every group")
                else:
                    self.assertTrue(
                        answerable_by_one,
                        f"{name} {row['qid']} needs evidence combined across groups, "
                        f"so it should be multi_hop")

    def test_answerable_questions_have_at_least_one_group(self):
        """Group count tracks facts required, not question type.

        A factual question can require two facts and still not be multi-hop -
        NBA q028 asks for a team and a year, both proven by one sentence. Type
        says whether passages must be combined; group count says how many facts
        the answer asserts. They are independent, so there is no upper bound to
        assert here.
        """
        for name, rows in self.datasets():
            for row in rows:
                if row["type"] != "unanswerable":
                    self.assertGreaterEqual(len(row["supporting"]), 1, f"{name} {row['qid']}")

    def test_every_group_has_a_fact_label_when_a_question_needs_several(self):
        """Multi-group questions must say what each group is for.

        With one group the label is redundant; with two it is the only thing
        distinguishing "alternative phrasings of one fact" from "two separate
        facts", which is exactly what this schema exists to record.
        """
        for name, rows in self.datasets():
            for row in rows:
                if len(row["supporting"]) > 1:
                    for index, group in enumerate(row["supporting"]):
                        self.assertTrue(group.get("fact", "").strip(),
                                        f"{name} {row['qid']} group {index} has no fact label")


if __name__ == "__main__":
    unittest.main()
