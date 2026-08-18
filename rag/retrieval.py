"""Find the chunks most relevant to a query.

Two interchangeable strategies, selected by name in a config:

  dense  - embed everything, rank by cosine similarity
  bm25   - rank by lexical overlap, no embeddings at all
  hybrid - run both and fuse the two rankings

Both are exact: every chunk is scored on every query, nothing approximated.
Vector databases exist to avoid that at millions of vectors, buying speed by
returning approximate neighbours; at this corpus size the exact answer costs
about a millisecond, so there is nothing to trade away.

Neither index is written to disk. Rebuilding takes seconds, which also means
there is no stale index to reuse against a different chunking configuration by
mistake.
"""

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from rag.embedding import DEFAULT_MODEL, MODELS, embed_documents, embed_query
from rag.types import RetrievedChunk

# BM25 Okapi's two free parameters, fixed rather than exposed. k1 controls how
# quickly repeated terms stop adding value; b controls how strongly a long
# chunk is penalised. These are the standard defaults, and leaving them fixed
# keeps the retrieval experiment about dense-versus-lexical rather than about
# tuning.
BM25_K1 = 1.5
BM25_B = 0.75

# Runs of letters and digits; everything else is a separator. Deliberately
# simple and visible: "1995-96" becomes two tokens, "$4.8" becomes "4" and
# "8". Tokenisation *is* the experiment when comparing lexical against dense
# retrieval, so it should be readable rather than hidden in a library.
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    """Lowercase, then split into runs of letters and digits."""
    return _TOKEN_RE.findall(text.lower())


@dataclass(frozen=True, eq=False)
class VectorIndex:
    """Chunks plus their embeddings, searchable by query text.

    Build it with VectorIndex.build(chunks); the constructor is for when you
    already hold vectors and want to pair them with chunks yourself.

    eq=False because the dataclass-generated __eq__ would compare NumPy arrays
    with ==, which returns an array rather than a bool.
    """

    chunks: tuple
    """The indexed chunks, in the same order as the rows of `vectors`."""

    vectors: np.ndarray
    """Shape (len(chunks), dimensions), float32, unit-normalized."""

    model: str
    """The exact model identifier that produced the vectors.

    Kept so queries are embedded by the same model as the documents. Scoring a
    query from one model against documents from another produces numbers that
    look fine and mean nothing.
    """

    def __post_init__(self):
        rows = self.vectors.shape[0] if self.vectors.ndim == 2 else -1
        if rows != len(self.chunks):
            raise ValueError(
                f"{len(self.chunks)} chunks but vectors has shape "
                f"{self.vectors.shape} - the two must line up row for row"
            )

        expected = MODELS[self.model].dimensions
        if self.chunks and self.vectors.shape[1] != expected:
            raise ValueError(
                f"model {self.model!r} produces {expected}-dimensional vectors, "
                f"but this array is {self.vectors.shape[1]}-dimensional"
            )

    def __len__(self):
        return len(self.chunks)

    @classmethod
    def build(cls, chunks, model=DEFAULT_MODEL, batch_size=32, show_progress=False):
        """Embed a list of chunks and return an index over them."""
        chunks = tuple(chunks)
        vectors = embed_documents(
            [chunk.text for chunk in chunks],
            model=model,
            batch_size=batch_size,
            show_progress=show_progress,
        )
        return cls(chunks=chunks, vectors=vectors, model=model)

    def search(self, query, k=5):
        """Return the k best matches for a query string, best first.

        Fewer than k results come back if the index is smaller than k.
        """
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if not self.chunks:
            return []

        query_vector = embed_query(query, model=self.model)
        return self.search_vector(query_vector, k=k)

    def search_vector(self, query_vector, k=5):
        """Same as search, but for a query that is already embedded.

        Useful when one query is run against several indexes, so it is only
        embedded once.
        """
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if not self.chunks:
            return []

        # Both sides are unit-normalized, so the dot product is cosine
        # similarity and the whole search is one matrix multiply.
        scores = self.vectors @ query_vector

        # A full sort of a few thousand scores is microseconds. At a scale
        # where it stopped being, np.argpartition would find the top k without
        # ordering the rest.
        order = np.argsort(-scores)[:k]

        return [
            RetrievedChunk(
                chunk=self.chunks[index],
                score=float(scores[index]),
                rank=position,
            )
            for position, index in enumerate(order, start=1)
        ]


@dataclass(frozen=True, eq=False)
class Bm25Index:
    """Chunks indexed by the words they contain, searchable by query text.

    BM25 Okapi: a chunk scores well when it contains the query's terms, more so
    when those terms are rare across the corpus, with diminishing returns for
    repetition and a penalty for length. No embeddings are involved anywhere -
    building this index makes no API calls and loads no model.

    Where dense search matches meaning, this matches words. It cannot connect
    "coach" to "head coach" through anything but the shared token, and it will
    not connect "franchise" to "team" at all. What it does have is exactness:
    a rare token like "1995" or "Kareem" is found reliably rather than
    approximately, which is where dense embeddings tend to be weakest.

    eq=False for consistency with VectorIndex, whose arrays make the generated
    __eq__ unusable.
    """

    chunks: tuple
    """The indexed chunks, in the order their positions refer to."""

    postings: dict
    """term -> {chunk position: how many times the term occurs in that chunk}.

    An inverted index: scoring visits only the chunks that contain a query
    term, rather than every chunk in the corpus. The number of chunks holding a
    term is len(postings[term]), which is the document frequency the IDF needs,
    so it is not stored twice.
    """

    lengths: np.ndarray
    """Token count per chunk, aligned with `chunks`. Used for length normalization."""

    average_length: float
    """Mean token count across the corpus - the yardstick `lengths` is judged against."""

    def __post_init__(self):
        if len(self.lengths) != len(self.chunks):
            raise ValueError(
                f"{len(self.chunks)} chunks but {len(self.lengths)} lengths - "
                f"the two must line up"
            )

    def __len__(self):
        return len(self.chunks)

    @classmethod
    def build(cls, chunks):
        """Tokenize a list of chunks and return an index over them.

        Takes no model argument, because there is nothing to embed. That is the
        whole reason build_index dispatches rather than passing a model through
        to every strategy.
        """
        chunks = tuple(chunks)

        postings = {}
        lengths = []
        for position, chunk in enumerate(chunks):
            tokens = tokenize(chunk.text)
            lengths.append(len(tokens))
            for term, count in Counter(tokens).items():
                postings.setdefault(term, {})[position] = count

        lengths = np.asarray(lengths, dtype=np.float32)
        # Guarded because an empty corpus would otherwise divide by zero in
        # _score. search returns early on an empty index, so the value is never
        # actually used - it just must not be a NaN waiting to happen.
        average_length = float(lengths.mean()) if len(lengths) else 0.0

        return cls(
            chunks=chunks,
            postings=postings,
            lengths=lengths,
            average_length=average_length,
        )

    def search(self, query, k=5):
        """Return the k best matches for a query string, best first.

        Same signature and same return type as VectorIndex.search - that shared
        shape is what lets a config swap one for the other.

        Fewer than k results come back if the index is smaller than k. A query
        whose terms appear nowhere in the corpus still returns k chunks, all
        scoring zero; ranking is meaningless there, and the retrieval metrics
        will correctly count it as a miss.
        """
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if not self.chunks:
            return []

        scores = self._score(tokenize(query))
        order = np.argsort(-scores)[:k]

        return [
            RetrievedChunk(
                chunk=self.chunks[index],
                score=float(scores[index]),
                rank=position,
            )
            for position, index in enumerate(order, start=1)
        ]

    def _score(self, query_tokens):
        """BM25 Okapi score of every chunk against one tokenized query.

        For each query term t and chunk d:

            idf(t) * tf(t,d) * (k1 + 1)
            ---------------------------------------------------
            tf(t,d) + k1 * (1 - b + b * len(d) / average_length)

        summed over the query's terms. Reading it piece by piece:

          idf   rare terms count for more than common ones. A query term in
                nearly every chunk carries almost no information about which
                chunk to pick.
          tf    repetition helps, but with diminishing returns - k1 caps how
                much the fifth occurrence of a word can add over the first.
          len   b scales a penalty for long chunks, which would otherwise win
                on term counts simply by containing more words.

        Query terms are counted once each: repeating a word in the question
        does not double its weight.
        """
        total = len(self.chunks)
        scores = np.zeros(total, dtype=np.float32)

        for term in set(query_tokens):
            posting = self.postings.get(term)
            if not posting:
                # Not in the corpus at all: no document frequency, no signal.
                continue

            document_frequency = len(posting)
            idf = math.log(
                1 + (total - document_frequency + 0.5) / (document_frequency + 0.5)
            )

            for position, term_frequency in posting.items():
                length_penalty = 1 - BM25_B + BM25_B * (
                    self.lengths[position] / self.average_length
                )
                scores[position] += (
                    idf
                    * term_frequency
                    * (BM25_K1 + 1)
                    / (term_frequency + BM25_K1 * length_penalty)
                )

        return scores


# Reciprocal Rank Fusion's two settings. Fixed implementation choices for the
# first hybrid experiment, not knobs: they are deliberately absent from the
# config schema so that a hybrid run differs from a dense run in exactly one
# thing, the strategy name. Promoting either to a config key is a later
# decision, and would have to come with a summary column so runs stay
# distinguishable.
#
# RRF_K is the published constant (Cormack et al., 2009). It damps how much a
# top rank is worth relative to being found by both retrievers at all - and at
# this depth it damps it heavily. A chunk both retrievers rank at depth d
# scores 2/(60+d), which beats a chunk one retriever ranks first, 1/61,
# whenever d < 62. So within any depth used here, agreement outranks position.
# That is a property to measure, not a bug to patch, but it is what the first
# hybrid result will actually be testing.
RRF_K = 60

# How deep each retriever contributes candidates, independent of the final k.
# 20 reaches the gold chunks dense search currently ranks 13th, 14th and 18th,
# which a depth equal to k could never see, without going so deep that every
# additional candidate is noise.
FUSION_DEPTH = 20


@dataclass(frozen=True, eq=False)
class HybridIndex:
    """Dense and BM25 rankings fused into one, by Reciprocal Rank Fusion.

    Composed from the other two indexes rather than reimplementing either, so
    dense and lexical retrieval keep whatever behaviour they had - this class
    only decides how to interleave their outputs.

    RRF scores each chunk by summing 1/(RRF_K + rank) across the retrievers
    that returned it, and ignores the underlying scores entirely. That is the
    point: cosine similarity lives in [-1, 1] and BM25 is unbounded and
    corpus-dependent, so any score-based blend would need the two put on a
    common scale first, and every way of doing that is a tuning decision.
    Ranks need no such calibration.
    """

    dense: VectorIndex
    """The dense half. Holds the vectors and the embedding model identifier."""

    lexical: Bm25Index
    """The lexical half. Holds the inverted index."""

    rrf_k: int
    """The RRF damping constant - see RRF_K."""

    fusion_depth: int
    """How many candidates each retriever contributes - see FUSION_DEPTH."""

    def __post_init__(self):
        if len(self.dense) != len(self.lexical):
            raise ValueError(
                f"the two halves cover different corpora: {len(self.dense)} "
                f"chunks dense, {len(self.lexical)} lexical"
            )
        if self.rrf_k < 0:
            raise ValueError(f"rrf_k must not be negative, got {self.rrf_k}")
        if self.fusion_depth <= 0:
            raise ValueError(
                f"fusion_depth must be positive, got {self.fusion_depth}"
            )

    def __len__(self):
        return len(self.dense)

    @property
    def model(self):
        """The embedding model behind the dense half.

        Hybrid does embed, so callers that record which model a run used - the
        experiment runner's summary row - can ask a HybridIndex the same way
        they ask a VectorIndex.
        """
        return self.dense.model

    @classmethod
    def build(cls, chunks, model=DEFAULT_MODEL, rrf_k=RRF_K,
              fusion_depth=FUSION_DEPTH, **kwargs):
        """Build both halves over the same chunks and pair them.

        The corpus is embedded once, exactly as a dense run would embed it, so
        hybrid costs a dense run plus the fraction of a second BM25 takes.
        """
        chunks = tuple(chunks)
        return cls(
            dense=VectorIndex.build(chunks, model=model, **kwargs),
            lexical=Bm25Index.build(chunks),
            rrf_k=rrf_k,
            fusion_depth=fusion_depth,
        )

    def search(self, query, k=5):
        """Return the k best matches after fusing both rankings, best first.

        Same signature and return type as the other two indexes.
        """
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if not len(self):
            return []

        # Never fuse over fewer candidates than the caller asked to be ranked;
        # otherwise a large k would return fewer results than the index holds.
        depth = max(k, self.fusion_depth)

        scores = defaultdict(float)
        found = {}
        for run in (self.dense.search(query, k=depth),
                    self.lexical.search(query, k=depth)):
            for result in run:
                scores[result.chunk.chunk_id] += 1 / (self.rrf_k + result.rank)
                found[result.chunk.chunk_id] = result.chunk

        # Ties are routine here - any two chunks each found by a single
        # retriever at the same rank score identically - so the tie-break is
        # spelled out. Left implicit, the winner would be decided by dict
        # insertion order, which is repeatable but arbitrary and would shift
        # the moment either retriever's output order changed.
        order = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))

        return [
            RetrievedChunk(
                chunk=found[chunk_id],
                score=scores[chunk_id],
                rank=position,
            )
            for position, chunk_id in enumerate(order[:k], start=1)
        ]


# The retrieval strategies a config can name. Adding one here plus a branch in
# build_index is the whole extension point.
STRATEGIES = {
    "dense": VectorIndex,
    "bm25": Bm25Index,
    "hybrid": HybridIndex,
}

DEFAULT_STRATEGY = "dense"


def build_index(chunks, strategy=DEFAULT_STRATEGY, model=DEFAULT_MODEL, **kwargs):
    """Build the index named by `strategy` over `chunks`.

    Dispatching here rather than giving every strategy a uniform build(model=...)
    signature is deliberate: BM25 has no embedding model, and passing it one it
    would ignore is how a run ends up quietly paying to embed a corpus it never
    reads. The bm25 branch never touches the embedding layer at all; dense and
    hybrid both do.
    """
    if strategy not in STRATEGIES:
        raise ValueError(
            f"unknown retrieval strategy {strategy!r} "
            f"(available: {', '.join(sorted(STRATEGIES))})"
        )

    if strategy == "dense":
        return VectorIndex.build(chunks, model=model, **kwargs)

    if strategy == "hybrid":
        return HybridIndex.build(chunks, model=model, **kwargs)

    return Bm25Index.build(chunks)
