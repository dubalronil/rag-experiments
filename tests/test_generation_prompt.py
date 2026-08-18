"""Tests that pin the generation prompt to a corpus-neutral shape.

The prompt used to open with "You answer questions about NBA history". That was
harmless while there was one corpus and quietly wrong once there were two: the
Space runs handed the model astronomy passages under a system prompt telling it
the subject was basketball, and a capable model reads that as a mismatch and
refuses. The Opus run measured that refusal rate, not the generator.

So there are two things to hold still here, and they pull in opposite
directions. The prompt must name no domain, or the same bug returns the next
time a corpus is added. Everything else about it must not move at all, because
the answer metrics of every earlier run were produced under those rules - and
the refusal sentence in particular is compared verbatim by the scorer, so a
changed word there would silently turn every refusal into a wrong answer.

Run with:  python3 -m unittest discover tests
"""

import unittest

from rag.generation import (
    DEFAULT_MODEL, NO_ANSWER, SYSTEM_PROMPT, build_prompt, format_context,
)
from rag.types import Chunk, RetrievedChunk


def retrieved(*texts):
    return [
        RetrievedChunk(
            chunk=Chunk(chunk_id=f"doc_{n}::0001", doc_id=f"doc_{n}",
                        text=text, start=0, end=len(text)),
            score=1.0 - n / 10, rank=n)
        for n, text in enumerate(texts, start=1)
    ]


class TestTheChangeItself(unittest.TestCase):
    """The prompt must describe the task without naming a subject."""

    # Every corpus term that has been in this repository, plus the words a
    # future edit would most plausibly reach for. A term appearing here is not
    # a style objection - it is the exact failure that invalidated a run.
    DOMAIN_WORDS = [
        "NBA", "basketball", "National Basketball Association", "Space",
        "astronomy", "astronomical", "NASA", "spaceflight", "planet",
        "history of", "sports",
    ]

    def test_no_domain_is_named(self):
        lowered = SYSTEM_PROMPT.lower()
        for word in self.DOMAIN_WORDS:
            self.assertNotIn(word.lower(), lowered,
                             f"the prompt names {word!r}, so it is not corpus-neutral")

    def test_the_task_is_still_stated(self):
        """Neutral does not mean vague: it must still say what the job is."""
        opening = SYSTEM_PROMPT.split("Rules:")[0]

        self.assertIn("You answer questions", opening)
        self.assertIn("only the context provided", opening)
        self.assertIn("user's\nmessage", opening)


class TestPreservedInstructions(unittest.TestCase):
    """Everything except the subject line is unchanged, verbatim."""

    def test_the_refusal_string_is_exact(self):
        """Compared character for character by the scorer, so it is pinned here."""
        self.assertEqual(NO_ANSWER,
                         "The provided context does not contain the answer.")

    def test_the_refusal_sentence_is_quoted_in_the_prompt(self):
        self.assertIn(
            "reply with\n  exactly this sentence and nothing else: " + NO_ANSWER,
            SYSTEM_PROMPT)

    def test_the_context_only_rule_survives(self):
        self.assertIn(
            "- Use only the provided context. Do not use anything you know from\n"
            "  outside it, even if you are confident it is correct.",
            SYSTEM_PROMPT)

    def test_the_no_guessing_rule_survives(self):
        self.assertIn(
            "- Do not guess, estimate, or infer a number that is not stated.",
            SYSTEM_PROMPT)

    def test_the_length_rule_survives(self):
        self.assertIn(
            "- Answer in one or two sentences. No preamble, no restating the "
            "question.",
            SYSTEM_PROMPT)

    def test_there_are_still_exactly_four_rules(self):
        """A rule added or dropped is a second variable in every comparison."""
        rules = [line for line in SYSTEM_PROMPT.splitlines()
                 if line.startswith("- ")]

        self.assertEqual(len(rules), 4)

    def test_the_prompt_is_not_addressed_to_one_model(self):
        """Both generators read the identical system prompt."""
        self.assertNotIn("Haiku", SYSTEM_PROMPT)
        self.assertNotIn("Opus", SYSTEM_PROMPT)
        self.assertEqual(DEFAULT_MODEL, "claude-haiku-4-5")


class TestUserMessageUnchanged(unittest.TestCase):
    """The context block is what retrieval hands the model - it did not move.

    Only the system prompt changed. If the rendering of retrieved passages had
    drifted with it, a rerun would differ from the runs it is meant to be
    compared against for a reason nothing recorded.
    """

    def test_passages_render_with_rank_and_source(self):
        self.assertEqual(
            format_context(retrieved("Alpha text.", "Beta text.")),
            "[1] from doc_1:\nAlpha text.\n\n[2] from doc_2:\nBeta text.")

    def test_empty_retrieval_is_stated_not_blank(self):
        self.assertEqual(format_context([]), "(no context was retrieved)")

    def test_the_user_message_is_context_then_question(self):
        self.assertEqual(
            build_prompt("Who?", retrieved("Alpha text.")),
            "Context:\n\n[1] from doc_1:\nAlpha text.\n\nQuestion: Who?")

    def test_the_question_is_passed_through_verbatim(self):
        question = "In what year did the mission land, and who led it?"

        self.assertTrue(
            build_prompt(question, retrieved("Alpha.")).endswith(
                f"Question: {question}"))


class TestRetrievalAndEvalUntouched(unittest.TestCase):
    """The prompt fix must not have reached the parts that are not generation.

    Retrieval defaults decide what the model reads; the eval set decides what
    it is asked. A prompt edit that moved either would make the corrected runs
    incomparable with everything before them, which is the whole reason for
    rerunning.
    """

    def test_retrieval_defaults_are_unchanged(self):
        from rag.experiment import DEFAULTS

        self.assertEqual(DEFAULTS["retrieval"],
                         {"strategy": "dense", "k": 5, "rerank": False,
                          "multi_query": False, "subquery_rerank": False})

    def test_chunking_and_embedding_defaults_are_unchanged(self):
        from rag.experiment import DEFAULTS

        self.assertEqual(DEFAULTS["chunking"],
                         {"strategy": "fixed_size", "size": 512, "overlap": 128})
        self.assertEqual(DEFAULTS["embedding"]["model"],
                         "sentence-transformers/all-MiniLM-L6-v2")

    def test_the_generator_and_judge_defaults_are_unchanged(self):
        from rag.experiment import DEFAULTS

        self.assertEqual(DEFAULTS["generation"],
                         {"enabled": True, "model": "claude-haiku-4-5"})
        self.assertEqual(DEFAULTS["judging"],
                         {"enabled": True, "model": "claude-sonnet-5"})

    def test_the_space_eval_set_is_the_one_the_runs_were_scored_against(self):
        import json

        from rag.datasets import eval_path

        rows = [json.loads(line) for line
                in eval_path("space").read_text(encoding="utf-8").splitlines()
                if line.strip()]

        self.assertEqual(len(rows), 50)
        self.assertEqual(sum(r["type"] == "unanswerable" for r in rows), 8)
        self.assertEqual(rows[18]["qid"], "q019")

    def test_generation_imports_nothing_from_retrieval(self):
        """One direction only: retrieval feeds generation, never the reverse."""
        import rag.generation

        source = open(rag.generation.__file__, encoding="utf-8").read()

        self.assertNotIn("from rag.retrieval", source)
        self.assertNotIn("import rag.retrieval", source)


if __name__ == "__main__":
    unittest.main()
