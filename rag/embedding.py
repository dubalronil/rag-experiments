"""Turn text into vectors.

Documents and queries get separate functions. For the baseline model they do
the same thing, but that is a property of this model rather than of embedding
in general: some models are asymmetric and expect queries to carry a prefix
that documents do not. Keeping the two paths separate from the start means
swapping in such a model later is a config change, not a refactor.

Vectors are normalized to unit length, which makes cosine similarity the same
thing as a dot product - so retrieval becomes one matrix multiply.

Runs locally on CPU. No API key, no network after the first run, and the same
input gives the same vectors every time, which is what lets a result from
today still be comparable next month.
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# The repository's .env, resolved from this file rather than the working
# directory so it is found no matter where a script is run from.
DOTENV_PATH = Path(__file__).resolve().parent.parent / ".env"

# How many texts to send per API request. Large enough that a corpus is a
# handful of calls, small enough to keep any one request modest.
OPENAI_BATCH_SIZE = 256

OPENAI_SETUP_HELP = (
    "No OpenAI credentials found. Either:\n"
    "  cp .env.example .env    and put your key in it, or\n"
    "  export OPENAI_API_KEY='sk-...'\n"
    "Only the OpenAI embedding models need this - the local models run "
    "without credentials."
)


@dataclass(frozen=True)
class EmbeddingModel:
    """Facts about one model, keyed by its exact Hugging Face identifier."""

    dimensions: int
    """Length of the vectors this model produces. Recorded so a caller can
    check an existing index was built by a model of the same shape."""

    query_prefix: str
    """Text prepended to queries but not to documents.

    Empty for symmetric models like MiniLM, where a query and a document are
    encoded identically. Asymmetric models (the bge family, for example) score
    noticeably worse without their prefix, and get no error to say so - which
    is why this lives in the model's definition rather than in calling code.
    """

    backend: str
    """Which entry of BACKENDS knows how to run this model.

    Stated explicitly rather than inferred from the identifier, so reading the
    dict tells you where each model actually runs.
    """


# Models a config can select. Keys are the exact model identifiers, so a
# config and a saved result name the same string the library is given.
MODELS = {
    "sentence-transformers/all-MiniLM-L6-v2": EmbeddingModel(
        dimensions=384,
        query_prefix="",
        backend="sentence_transformers",
    ),
    # Same 384 dimensions as MiniLM, so the two are directly comparable without
    # touching retrieval. Unlike MiniLM it is asymmetric: the instruction goes
    # on queries only, never on the passages being indexed.
    "BAAI/bge-small-en-v1.5": EmbeddingModel(
        dimensions=384,
        query_prefix="Represent this sentence for searching relevant passages: ",
        backend="sentence_transformers",
    ),
    # Runs over the network instead of locally. Native width, no shortening -
    # requesting fewer dimensions would confound model quality with vector
    # width. Symmetric, so no query prefix.
    "text-embedding-3-small": EmbeddingModel(
        dimensions=1536,
        query_prefix="",
        backend="openai",
    ),
    # The larger sibling, also at its native width. Same backend, so the only
    # thing that differs between the two is the model identifier.
    "text-embedding-3-large": EmbeddingModel(
        dimensions=3072,
        query_prefix="",
        backend="openai",
    ),
}

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Loading a model takes seconds and it is stateless once loaded, so keep each
# one around rather than paying that cost per call.
_loaded = {}


def load_model(model=DEFAULT_MODEL):
    """Return the SentenceTransformer for a named model, loading it once.

    The first call for a given model downloads the weights (~90MB for MiniLM)
    and caches them under ~/.cache/huggingface. Later runs are offline.
    """
    if model not in MODELS:
        raise ValueError(
            f"unknown model {model!r} (available: {', '.join(sorted(MODELS))})"
        )

    if model not in _loaded:
        _loaded[model] = SentenceTransformer(model)
    return _loaded[model]


def _warn_about_truncation(texts, encoder, label):
    """Print a warning for any text longer than the model will actually read.

    This is the quietest failure mode in the whole pipeline. A model with a
    256-token window given a 500-token chunk does not error and does not warn -
    it embeds the first half and silently discards the rest. The resulting
    vector looks perfectly normal. An experiment sweeping chunk sizes upward
    would show quality dropping and read it as a fact about chunking, when it
    is really a fact about the model's window.

    Written to stderr rather than through the warnings module on purpose: the
    default warning filter shows a given warning once per call site, so a
    second run with worse truncation would print nothing at all.
    """
    limit = encoder.max_seq_length
    tokenizer = encoder.tokenizer

    # Tokenizing twice (here and inside encode) is wasted work, but at corpus
    # scale it costs a second or two and buys the only signal there is.
    lengths = [len(tokenizer.encode(text)) for text in texts]
    over_limit = [length for length in lengths if length > limit]

    if not over_limit:
        return

    print(
        f"WARNING: {len(over_limit)} of {len(texts)} {label} exceed the model's "
        f"{limit}-token window and were TRUNCATED.\n"
        f"         Longest was {max(over_limit)} tokens; everything past "
        f"{limit} was silently discarded.\n"
        f"         Shorten the chunks or use a model with a longer window - "
        f"results from these vectors are not trustworthy.",
        file=sys.stderr,
    )


_openai_client = None


def _get_openai_client():
    """Return a shared OpenAI client, loading .env on first use.

    Imported lazily so a machine that only ever runs the local models does not
    need the openai package installed at all.
    """
    global _openai_client
    if _openai_client is None:
        # override=False: a key already exported in the shell wins over the file.
        load_dotenv(DOTENV_PATH, override=False)
        try:
            from openai import OpenAI, OpenAIError
        except ImportError as error:  # pragma: no cover - dependency missing
            raise RuntimeError(
                "the openai package is required for OpenAI embedding models: "
                "pip install -r requirements.txt"
            ) from error
        try:
            _openai_client = OpenAI()
        except OpenAIError as error:
            raise RuntimeError(OPENAI_SETUP_HELP) from error
    return _openai_client


def _normalize(vectors):
    """Scale each row to unit length.

    VectorIndex treats normalization as an invariant - cosine similarity *is*
    the dot product there - so it is enforced here rather than trusted to a
    provider's current behaviour.
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype(np.float32)


def _embed_sentence_transformers(texts, model, batch_size, show_progress, label):
    """Embed locally. Unchanged behaviour for MiniLM and BGE."""
    encoder = load_model(model)
    _warn_about_truncation(texts, encoder, label)

    vectors = encoder.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=show_progress,
    )
    return vectors.astype(np.float32)


def _embed_openai(texts, model, batch_size, show_progress, label):
    """Embed over the API, in batches.

    No truncation check: the request limit is far beyond any chunk this project
    produces, and checking it locally would mean carrying a tokenizer purely to
    warn about a ceiling that cannot be reached. Oversized input is rejected by
    the API instead.
    """
    client = _get_openai_client()

    rows = []
    for start in range(0, len(texts), OPENAI_BATCH_SIZE):
        response = client.embeddings.create(
            model=model,
            input=texts[start:start + OPENAI_BATCH_SIZE],
        )
        # Sort by index rather than trusting response order: the ordering is
        # the only thing tying a vector back to its chunk.
        rows.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))

    return _normalize(np.asarray(rows, dtype=np.float32))


# How to run each model. Reading this dict tells you where every model executes.
BACKENDS = {
    "sentence_transformers": _embed_sentence_transformers,
    "openai": _embed_openai,
}


def _backend_for(model):
    if model not in MODELS:
        raise ValueError(
            f"unknown model {model!r} (available: {', '.join(sorted(MODELS))})"
        )
    return BACKENDS[MODELS[model].backend]


def embed_documents(texts, model=DEFAULT_MODEL, batch_size=32, show_progress=False):
    """Embed a list of document texts into a (len(texts), dimensions) array.

    Returns float32, L2-normalized, one row per input in the same order - the
    ordering is the only thing tying a vector back to the chunk it came from.
    """
    backend = _backend_for(model)
    if not texts:
        return np.zeros((0, MODELS[model].dimensions), dtype=np.float32)

    return backend(texts, model, batch_size, show_progress, "documents")


def embed_query(text, model=DEFAULT_MODEL):
    """Embed one query string into a 1-D (dimensions,) array.

    Separate from embed_documents because a query is not always encoded the
    same way a document is - see EmbeddingModel.query_prefix.
    """
    backend = _backend_for(model)
    prefixed = MODELS[model].query_prefix + text

    return backend([prefixed], model, 1, False, "queries")[0]
