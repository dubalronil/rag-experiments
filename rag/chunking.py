"""Split Documents into Chunks.

Chunking is the first real experiment variable: chunk size and overlap change
what a retriever can possibly find, before any embedding or ranking happens.
So the parameters are explicit arguments rather than defaults buried in here,
and every chunk records where it came from.

Sizes are measured in characters. Characters are a rough proxy for tokens, but
they keep this module dependency-free, which matters while the rest of the
pipeline does not exist yet. Worth revisiting once an embedding model is
involved, since the model itself counts tokens, not characters.
"""

from rag.types import Chunk


def chunk_fixed_size(document, size, overlap=0):
    """Cut a Document into fixed-length, optionally overlapping chunks.

    size    - characters per chunk
    overlap - characters each chunk repeats from the previous one

    Chunks advance by (size - overlap), so overlap=0 gives adjacent slices and
    overlap=100 with size=500 means each chunk repeats the last 100 characters
    of the one before it. Overlap exists so that a sentence landing on a
    boundary still appears intact somewhere.

    The text is sliced exactly as-is, never stripped, so the Chunk invariant
    chunk.text == document.text[chunk.start:chunk.end] always holds. Trimming
    whitespace here would make the offsets lie.
    """
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}")
    if overlap < 0:
        raise ValueError(f"overlap must not be negative, got {overlap}")
    if overlap >= size:
        # step would be zero or negative, and the loop below would never end.
        raise ValueError(f"overlap ({overlap}) must be smaller than size ({size})")

    text = document.text
    if not text:
        return []

    step = size - overlap
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(
            Chunk(
                chunk_id=f"{document.doc_id}::{len(chunks):04d}",
                doc_id=document.doc_id,
                text=text[start:end],
                start=start,
                end=end,
            )
        )
        # Once a chunk reaches the end of the document, stop. Continuing would
        # emit trailing chunks wholly contained in the one just added.
        if end == len(text):
            break
        start += step

    return chunks


# Strategies a config can select by name. Kept as a plain dict so that reading
# it tells you every option there is - no decorators, no dynamic registration.
STRATEGIES = {
    "fixed_size": chunk_fixed_size,
}


def chunk_corpus(documents, strategy="fixed_size", **params):
    """Chunk a list of Documents with one strategy, returning a flat list.

    Order follows the input, and Documents arrive from corpus.load_corpus in
    sorted order, so the chunk sequence is reproducible between runs.
    """
    if strategy not in STRATEGIES:
        raise ValueError(
            f"unknown strategy {strategy!r} "
            f"(available: {', '.join(sorted(STRATEGIES))})"
        )

    chunk = STRATEGIES[strategy]
    return [c for document in documents for c in chunk(document, **params)]
