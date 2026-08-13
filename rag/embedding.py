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

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass(frozen=True)
class EmbeddingModel:
    """A model the config can select by name."""

    repo_id: str
    """The Hugging Face identifier the weights are downloaded from."""

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


# Models a config can select by name. One entry today; reading the dict tells
# you every option there is.
MODELS = {
    "minilm": EmbeddingModel(
        repo_id="sentence-transformers/all-MiniLM-L6-v2",
        dimensions=384,
        query_prefix="",
    ),
}

DEFAULT_MODEL = "minilm"

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
        _loaded[model] = SentenceTransformer(MODELS[model].repo_id)
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


def embed_documents(texts, model=DEFAULT_MODEL, batch_size=32, show_progress=False):
    """Embed a list of document texts into a (len(texts), dimensions) array.

    Returns float32, L2-normalized, one row per input in the same order - the
    ordering is the only thing tying a vector back to the chunk it came from.
    """
    if not texts:
        return np.zeros((0, MODELS[model].dimensions), dtype=np.float32)

    encoder = load_model(model)
    _warn_about_truncation(texts, encoder, "documents")

    vectors = encoder.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=show_progress,
    )
    return vectors.astype(np.float32)


def embed_query(text, model=DEFAULT_MODEL):
    """Embed one query string into a 1-D (dimensions,) array.

    Separate from embed_documents because a query is not always encoded the
    same way a document is - see EmbeddingModel.query_prefix.
    """
    encoder = load_model(model)
    prefixed = MODELS[model].query_prefix + text

    _warn_about_truncation([prefixed], encoder, "queries")

    vector = encoder.encode(
        prefixed,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vector.astype(np.float32)
