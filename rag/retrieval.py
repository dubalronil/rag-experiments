"""Find the chunks most similar to a query.

The index lives in memory and holds two things that must stay aligned: the
chunks, and one vector per chunk. Row i of the array is the embedding of
chunk i - that pairing is the entire data structure.

Search is exact. Every chunk is scored, nothing is approximated. Vector
databases exist to avoid exactly that at millions of vectors, and they buy
their speed by returning approximate neighbours; at this corpus size the exact
answer costs about a millisecond, so there is nothing to trade away.

Nothing is written to disk. Embedding the corpus takes seconds, so an index is
rebuilt per run rather than cached - which also means there is no stale index
to accidentally reuse across different chunking configurations.
"""

from dataclasses import dataclass

import numpy as np

from rag.embedding import DEFAULT_MODEL, MODELS, embed_documents, embed_query
from rag.types import RetrievedChunk


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
    """Which entry of embedding.MODELS produced the vectors.

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
