"""Grade a generated answer on two independent dimensions.

Correctness asks whether the answer says what the reference answer says.
Groundedness asks whether the retrieved passages actually support it. They are
different questions, and an answer can pass one and fail the other: a model
that computes an interval itself and contradicts the source is nearly correct
and completely ungrounded.

The two judgements therefore run as separate calls with deliberately disjoint
inputs. The correctness judge never sees the passages; the groundedness judge
never sees the reference answer. Leak either one into the other prompt and the
metrics collapse into a single fuzzy notion of "good", because a judge that
knows an answer is right will call it supported.

Scores are 0, 1, 2 - fine enough to separate "partly right" from "wrong",
coarse enough that a judge can apply them consistently. A five-point scale
invites precision the judge cannot deliver.
"""

import json
from dataclasses import dataclass

import anthropic

from rag.generation import get_client

# Reasoning is listed first in the schema so the judge commits to an argument
# before committing to a number, and so a disagreement can be read rather than
# guessed at.
VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "string",
            "description": "One sentence justifying the score.",
        },
        "score": {"type": "integer", "enum": [0, 1, 2]},
    },
    "required": ["reasoning", "score"],
    "additionalProperties": False,
}

CORRECTNESS_SYSTEM = """\
You grade one answer against a reference answer.

A question may ask for more than one thing. Work out what it asks for, then
use the reference as the source of truth for what each of those things is:

  2 - supplies every part the question asks for, and each one agrees with
      the reference
  1 - supplies only some of the parts, or gives one that is vague or wrong
  0 - wrong throughout, contradicts the reference, answers a different
      question, or declines to answer

The reference often carries detail the question did not ask for. Leaving that
detail out is not a penalty - only the parts the question actually asks for
are required. If a question asks two things and the answer gets one of them
wrong or omits it, that is a 1.

Wording, length, and formatting do not matter. Extra correct detail is never
a penalty. A wrong or missing number, name, or date in a part the question
asked for is a failure even when the rest of the answer is right. Do not
reward an answer for sounding thorough.\
"""

GROUNDEDNESS_SYSTEM = """\
You check whether an answer is supported by a set of source passages.

Score only support, never correctness:

  2 - every claim in the answer is stated in the passages
  1 - the central claim is stated, but some detail is not
  0 - a central claim is absent from the passages or contradicts them

An answer can be entirely true and still score 0 if the passages do not say
it. Use only the passages; ignore anything you know about the subject.
Arithmetic, date differences, and inferences the passages do not themselves
state are unsupported. If the passages say "three months" and the answer says
"two and a half months", that is a contradiction, not a rounding.\
"""


@dataclass(frozen=True)
class Verdict:
    """One judge's decision on one dimension."""

    score: int
    reasoning: str


@dataclass(frozen=True)
class JudgeModel:
    """Facts about one judge, keyed by its exact API model identifier."""

    max_tokens: int


JUDGE_MODELS = {
    # Deliberately not the model that generates the answers: a model grading
    # its own output rates it generously.
    "claude-sonnet-5": JudgeModel(max_tokens=1024),
}

DEFAULT_JUDGE = "claude-sonnet-5"


def _ask_judge(system, user, judge=DEFAULT_JUDGE):
    """Run one judgement and return a Verdict.

    Thinking is switched off and the answer is constrained to a schema, so the
    response is a parsed object rather than prose to scrape. Sonnet 5 rejects
    temperature, so repeatability comes from the rubric and the schema instead
    of a sampling parameter.
    """
    config = JUDGE_MODELS[judge]

    response = get_client().messages.create(
        model=judge,
        max_tokens=config.max_tokens,
        thinking={"type": "disabled"},
        system=system,
        messages=[{"role": "user", "content": user}],
        output_config={"format": {"type": "json_schema", "schema": VERDICT_SCHEMA}},
    )

    if response.stop_reason == "refusal":
        raise RuntimeError("the judge declined to grade this item")

    text = "".join(
        block.text for block in response.content if block.type == "text"
    ).strip()
    verdict = json.loads(text)
    return Verdict(score=int(verdict["score"]), reasoning=verdict["reasoning"].strip())


def judge_correctness(question, expected, generated, judge=DEFAULT_JUDGE):
    """Does the answer say what the reference answer says?

    The retrieved passages are deliberately not passed in - this judge must
    not be able to excuse a wrong answer because the context explains it.
    """
    user = (
        f"Question: {question}\n\n"
        f"Reference answer: {expected}\n\n"
        f"Answer to grade: {generated}"
    )
    return _ask_judge(CORRECTNESS_SYSTEM, user, judge)


def judge_groundedness(question, retrieved, generated, judge=DEFAULT_JUDGE):
    """Do the retrieved passages support every claim in the answer?

    The reference answer is deliberately not passed in - a judge that knows
    the right answer will call an unsupported claim supported.
    """
    passages = "\n\n".join(
        f"[{item.rank}] {item.chunk.text.strip()}" for item in retrieved
    ) or "(no passages were retrieved)"

    user = (
        f"Passages:\n\n{passages}\n\n"
        f"Question: {question}\n\n"
        f"Answer to check: {generated}"
    )
    return _ask_judge(GROUNDEDNESS_SYSTEM, user, judge)
