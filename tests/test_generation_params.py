"""Tests for model-aware generation request parameters.

Two models now generate answers, and they do not accept the same request. Haiku
4.5 still takes sampling parameters; the current-generation models rejected
them, so sending `temperature` to Opus 5 is a 400 rather than a silent no-op.
Opus 5 also has adaptive thinking on by default, which would make a generator
comparison secretly a comparison of "one model, plus reasoning" against "another
model, without it" - so it is disabled explicitly.

Both of those are invisible in an answer string. A run with the wrong parameters
either fails loudly on every question or quietly measures something else, and
the second is the dangerous one. These tests read the exact kwargs handed to the
API instead of trusting the output.

The first class is the one that matters most: Haiku's request must be
byte-identical to what produced every recorded run, or no past result is
comparable with a new one.

Run with:  python3 -m unittest discover tests
"""

import unittest

import rag.generation
from rag.generation import MODELS, SYSTEM_PROMPT, generate_answer
from rag.types import Chunk, RetrievedChunk


def context():
    text = "Sojourner rover, Mars Pathfinder, landed successfully on July 4, 1997."
    return [RetrievedChunk(
        chunk=Chunk(chunk_id="mars_rover::0010", doc_id="mars_rover",
                    text=text, start=0, end=len(text)),
        score=1.0, rank=1)]


class Block:
    def __init__(self, text):
        self.type, self.text = "text", text


class Response:
    def __init__(self, text="an answer"):
        self.content = [Block(text)]
        self.stop_reason = "end_turn"


class FakeMessages:
    def __init__(self):
        self.kwargs = []

    def create(self, **kwargs):
        self.kwargs.append(kwargs)
        return Response()


class FakeClient:
    def __init__(self):
        self.messages = FakeMessages()


class GenerationTestCase(unittest.TestCase):
    def setUp(self):
        self.client = FakeClient()
        self._saved = rag.generation.get_client
        rag.generation.get_client = lambda: self.client

    def tearDown(self):
        rag.generation.get_client = self._saved

    def call(self, model):
        generate_answer("Which rover landed on Mars on July 4, 1997?",
                        context(), model=model)
        return self.client.messages.kwargs[-1]


class TestHaikuPreservation(GenerationTestCase):
    """Haiku's request must not have moved by one field."""

    def test_temperature_is_still_pinned_to_zero(self):
        self.assertEqual(self.call("claude-haiku-4-5")["temperature"], 0)

    def test_thinking_is_not_sent(self):
        """It was never sent before; sending it now would change behaviour."""
        self.assertNotIn("thinking", self.call("claude-haiku-4-5"))

    def test_the_exact_parameter_set_is_unchanged(self):
        sent = self.call("claude-haiku-4-5")

        self.assertEqual(set(sent),
                         {"model", "max_tokens", "temperature", "system", "messages"})
        self.assertEqual(sent["model"], "claude-haiku-4-5")
        self.assertEqual(sent["max_tokens"], MODELS["claude-haiku-4-5"].max_tokens)
        self.assertEqual(sent["system"], SYSTEM_PROMPT)


class TestOpusParameters(GenerationTestCase):
    def test_temperature_is_omitted(self):
        """Sampling parameters are rejected on this model, not ignored."""
        self.assertNotIn("temperature", self.call("claude-opus-5"))

    def test_thinking_is_explicitly_disabled(self):
        """Otherwise the experiment adds adaptive reasoning as a second variable."""
        self.assertEqual(self.call("claude-opus-5")["thinking"], {"type": "disabled"})

    def test_the_exact_parameter_set(self):
        sent = self.call("claude-opus-5")

        self.assertEqual(set(sent),
                         {"model", "max_tokens", "thinking", "system", "messages"})
        self.assertEqual(sent["model"], "claude-opus-5")


class TestSharedAcrossModels(GenerationTestCase):
    def test_the_prompt_and_context_are_identical_for_both(self):
        """Only the model and its parameters may differ - not what it reads."""
        haiku = self.call("claude-haiku-4-5")
        opus = self.call("claude-opus-5")

        self.assertEqual(haiku["system"], opus["system"])
        self.assertEqual(haiku["messages"], opus["messages"])
        self.assertEqual(haiku["max_tokens"], opus["max_tokens"])

    def test_the_question_and_passages_both_reach_the_prompt(self):
        sent = self.call("claude-opus-5")
        content = sent["messages"][0]["content"]

        self.assertIn("Which rover landed on Mars on July 4, 1997?", content)
        self.assertIn("Sojourner", content)

    def test_an_unknown_model_is_rejected_before_any_request(self):
        with self.assertRaises(ValueError) as caught:
            generate_answer("q", context(), model="claude-not-a-model")

        self.assertIn("claude-not-a-model", str(caught.exception))
        self.assertEqual(self.client.messages.kwargs, [], "no request was made")


class TestRegistry(unittest.TestCase):
    def test_both_models_are_registered(self):
        self.assertIn("claude-haiku-4-5", MODELS)
        self.assertIn("claude-opus-5", MODELS)

    def test_the_default_generator_is_unchanged(self):
        from rag.generation import DEFAULT_MODEL

        self.assertEqual(DEFAULT_MODEL, "claude-haiku-4-5")

    def test_the_generator_default_differs_from_the_judge(self):
        """A model grading its own output is the bias the split judge avoids."""
        from rag.experiment import DEFAULTS

        self.assertNotEqual(DEFAULTS["generation"]["model"],
                            DEFAULTS["judging"]["model"])


if __name__ == "__main__":
    unittest.main()
