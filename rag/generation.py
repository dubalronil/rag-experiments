"""Answer a question from retrieved chunks.

This is the stage that turns retrieval into RAG. It takes the chunks a search
returned, hands them to a model as context, and asks the question.

The prompt does one job beyond answering: it tells the model to answer *only*
from the supplied context, and to say plainly when the context does not
contain the answer. That instruction is what makes the evaluation set's
unanswerable questions meaningful - without it, a model asked for a statistic
the corpus never contained will simply invent a plausible one.

Unlike the rest of the pipeline this calls a paid API and is not
deterministic. Temperature is pinned to 0, which is as close to repeatable as
the API offers.
"""

import textwrap
from dataclasses import dataclass
from pathlib import Path

import anthropic
from dotenv import load_dotenv

# The repository's .env, if there is one. Resolved from this file rather than
# the working directory so it is found no matter where a script is run from.
DOTENV_PATH = Path(__file__).resolve().parent.parent / ".env"

# The exact string the model is told to return when the context is
# insufficient. Kept as a constant so a future scorer can recognise a refusal
# without pattern-matching prose.
NO_ANSWER = "The provided context does not contain the answer."

SYSTEM_PROMPT = textwrap.dedent(
    f"""\
    You answer questions about NBA history using only the context provided in
    the user's message.

    Rules:
    - Use only the provided context. Do not use anything you know from
      outside it, even if you are confident it is correct.
    - If the context does not contain enough information to answer, reply with
      exactly this sentence and nothing else: {NO_ANSWER}
    - Do not guess, estimate, or infer a number that is not stated.
    - Answer in one or two sentences. No preamble, no restating the question.
    """
)


@dataclass(frozen=True)
class GenerationModel:
    """Facts about one model, keyed by its exact API model identifier."""

    max_tokens: int
    """Ceiling on the answer length. Answers here are one or two sentences,
    so this only exists to stop a runaway response."""


MODELS = {
    "claude-haiku-4-5": GenerationModel(max_tokens=1024),
}

DEFAULT_MODEL = "claude-haiku-4-5"

SETUP_HELP = (
    "No Anthropic credentials found. Either:\n"
    "  cp .env.example .env    and put your key in it, or\n"
    "  export ANTHROPIC_API_KEY='sk-ant-...'\n"
    "Get a key at https://console.anthropic.com/settings/keys\n"
    "Generation is the only stage that needs this - corpus loading, chunking,\n"
    "embedding, retrieval, and retrieval scoring all run without it."
)

_client = None


def get_client():
    """Return a shared Anthropic client.

    Credentials are left to the SDK rather than read here, because an unset
    ANTHROPIC_API_KEY does not mean there are none - the SDK also resolves a
    logged-in CLI profile. Checking the environment variable directly would
    reject a perfectly valid setup.
    """
    global _client
    if _client is None:
        # Loaded here rather than at import time so merely importing this
        # module has no side effects. override=False means a key already
        # exported in the shell wins over the file.
        load_dotenv(DOTENV_PATH, override=False)
        _client = anthropic.Anthropic()
    return _client


def format_context(retrieved):
    """Render retrieved chunks into the context block the model sees.

    Each passage is numbered and labelled with its source document, which
    makes an answer traceable back to a chunk when one looks wrong.
    """
    if not retrieved:
        return "(no context was retrieved)"

    return "\n\n".join(
        f"[{item.rank}] from {item.chunk.doc_id}:\n{item.chunk.text.strip()}"
        for item in retrieved
    )


def build_prompt(question, retrieved):
    """Assemble the user message: context first, then the question."""
    return (
        f"Context:\n\n{format_context(retrieved)}\n\n"
        f"Question: {question}"
    )


def generate_answer(question, retrieved, model=DEFAULT_MODEL):
    """Answer a question from retrieved chunks, returning the answer text.

    Raises RuntimeError with setup instructions if no credentials are found.
    """
    if model not in MODELS:
        raise ValueError(
            f"unknown model {model!r} (available: {', '.join(sorted(MODELS))})"
        )

    config = MODELS[model]

    try:
        response = get_client().messages.create(
            model=model,
            max_tokens=config.max_tokens,
            # Sampling parameters were removed on the current-generation
            # models, but Haiku 4.5 still accepts them - so pin temperature to
            # 0 for the most repeatable output the API allows.
            temperature=0,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_prompt(question, retrieved)}],
        )
    except anthropic.AuthenticationError as error:
        # A credential was found and rejected - a different problem from
        # having none, so say so rather than repeating the setup steps.
        raise RuntimeError(
            "Anthropic rejected the credentials that were found.\n"
            f"Checked {DOTENV_PATH} and the ANTHROPIC_API_KEY environment "
            "variable.\nThe key may be mistyped, revoked, or from another "
            "account."
        ) from error
    except TypeError as error:
        # With no credentials at all, the SDK raises TypeError from the
        # request call while resolving auth headers - the client constructs
        # fine, so there is nothing to catch earlier. Only convert the
        # credential case; a TypeError from anything else is a real bug and
        # must not be disguised as a setup problem.
        if "authentication" not in str(error).lower():
            raise
        raise RuntimeError(SETUP_HELP) from error

    if response.stop_reason == "refusal":
        return NO_ANSWER

    return "".join(
        block.text for block in response.content if block.type == "text"
    ).strip()
