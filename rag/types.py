"""The shared data shapes every other stage of the pipeline passes around.

Only two types exist so far: a Document, which is one file from the corpus, and
a Chunk, which is one piece of a Document. Nothing here has any behaviour - the
point is to fix the vocabulary so that a chunker, a retriever and a scorer can
be written independently and still agree on what they are handing each other.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    """One Markdown file from a dataset's documents/ directory.

    The corpus is meant to stay fixed, so a Document is frozen: loading one
    twice gives you the same thing, and no later stage can quietly rewrite the
    text that a result was produced from.
    """

    doc_id: str
    """Stable identifier, taken from the filename stem (e.g. "chicago_bulls").

    This is what the evaluation set's gold labels refer to, so it must not
    change when article titles change. See corpus.load_document.
    """

    title: str
    """Human-readable article title from the front matter (e.g. "Chicago Bulls")."""

    source_url: str
    """Canonical Wikipedia URL the text was downloaded from."""

    revision_id: str
    """The Wikipedia revision the text came from, kept as a string.

    It is only ever compared or displayed, never used in arithmetic, and the
    fetcher writes whatever the API returned - so parsing it to an int would
    add a way to crash without adding anything useful.
    """

    text: str
    """The document body: the Markdown after the front matter block.

    Chunk offsets are positions into this string, so it is the single thing
    that defines what "position in the document" means.
    """


@dataclass(frozen=True)
class Chunk:
    """A contiguous slice of one Document's text.

    Invariant: chunk.text == document.text[chunk.start:chunk.end]

    Nothing constructs a Chunk yet - chunking is the next component. The type
    exists now so that the chunker and everything downstream of it are written
    against a shape that is already settled.
    """

    chunk_id: str
    """Identifier unique within a run, conventionally "<doc_id>::<index>".

    Chunk IDs are a function of the chunking strategy, so they change whenever
    chunk size or strategy changes. That is exactly why the evaluation set
    stores verbatim quotes instead of chunk IDs.
    """

    doc_id: str
    """The doc_id of the Document this slice came from."""

    text: str
    """The slice itself - what actually gets embedded and retrieved."""

    start: int
    """Character offset where this chunk begins in the parent Document.text."""

    end: int
    """Character offset just past the end of this chunk (Python slice style).

    start and end are not needed to score retrieval, which only asks whether a
    chunk contains a gold quote. They are here for diagnosis: when a question
    scores zero, the offsets show immediately whether the chunker split the
    supporting sentence across a boundary, rather than leaving you to guess.
    """


@dataclass(frozen=True)
class RetrievedChunk:
    """One search result: a Chunk, how well it matched, and where it placed.

    The Chunk is held rather than copied, so everything about the original
    slice - its text, its document, its offsets - stays reachable through
    .chunk without this type having to mirror those fields.
    """

    chunk: Chunk
    """The chunk that was retrieved."""

    score: float
    """How well this chunk matched the query, higher is better.

    The scale depends on the retrieval strategy that produced it. Dense search
    returns cosine similarity between unit-normalized vectors, bounded in
    [-1, 1]; BM25 returns an unbounded, corpus-dependent sum. So a score is
    meaningful for ordering results within one query, and for little else -
    comparing raw scores across strategies compares two different units.

    Nothing downstream relies on the value: every retrieval metric is computed
    from `rank`, which means the same thing whichever strategy ran.
    """

    rank: int
    """Position in the result list, starting at 1 for the best match.

    One-based rather than zero-based because rank is used directly in metrics:
    reciprocal rank is 1/rank, which has no meaning at rank 0.
    """
