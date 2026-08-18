"""Turn one multi-part question into several retrieval queries.

A question like "Which orbiter both returned the fleet to flight after
Challenger and carried Hubble into orbit?" has two halves, and a single
embedding of the whole thing is a blend of both. Retrieval then tends to
surface chunks matching the dominant half and miss the other entirely - which
is exactly the partial-evidence failure the evidence groups measure.

Splitting is deliberately syntactic. Every subquery is a literal substring of
the user's question, plus any leading context sentence from the same question,
so there is no channel through which a language model's own knowledge could
enter the query. That matters twice over: an LLM decomposer would leak world
knowledge into what is meant to be a retrieval measurement, and it would make
retrieval-only runs cost money and stop being reproducible. This costs nothing
and returns the same queries every time.

The tradeoff is coverage. Only questions whose parts appear as clauses can be
split; a question whose two facts are implied rather than stated - "how many
years separated the two crossings?" - falls through unchanged.
"""

import re

# How many subqueries a question may produce, beyond the original. Fixed rather
# than exposed: every multi-part question in both eval sets is two-part, and a
# third split would only fragment a fixed candidate budget further.
MAX_SUBQUERIES = 2

# Minimum tokens for a split part to be worth searching on. Below this a
# fragment carries no content words and would just return noise.
MIN_TOKENS = 3

# Clause boundaries. ", and" first so it wins over the bare " and " inside it.
_SPLIT = re.compile(r",\s+and\s+|\s+and\s+|;\s+")

# Sentence boundary: a terminator followed by a capital. Used to separate a
# leading declarative ("The JWST is named after a NASA administrator.") from
# the question that follows it.
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def split_question(question, max_subqueries=MAX_SUBQUERIES):
    """Split a multi-part question into clause-level queries.

    Returns [] when the question has no splittable structure, which is the
    signal to fall back to searching the question as written.

    Any sentences before the final one are treated as context and prepended to
    every part, because a trailing clause often refers back to them - "Which
    program did he lead" is unsearchable alone and fine once the sentence
    naming the administrator is attached.
    """
    sentences = [s.strip() for s in _SENTENCE.split(question.strip()) if s.strip()]
    if not sentences:
        return []

    context, ask = sentences[:-1], sentences[-1]

    parts = [part.strip() for part in _SPLIT.split(ask)]
    parts = [part for part in parts if len(part.split()) >= MIN_TOKENS]
    if len(parts) < 2:
        return []

    prefix = " ".join(context)
    return [f"{prefix} {part}".strip() if prefix else part
            for part in parts[:max_subqueries]]


def queries_for(question, multi_query=False):
    """The retrieval queries to run for one question, original always first."""
    if not multi_query:
        return [question]
    return [question] + split_question(question)


def merge_round_robin(runs, limit):
    """Interleave several ranked result lists into one pool of `limit` chunks.

    Takes each list's best candidate in turn, then each list's second, and so
    on, skipping chunks already taken. Ranks are renumbered from 1.

    Round-robin rather than score fusion because these lists answer *different*
    questions. Fusing them would reward chunks that rank well for both halves,
    which is redundancy, not corroboration - and would suppress the chunk that
    answers only the second half, which is the one the whole exercise is trying
    to surface. Interleaving instead guarantees every query a share of the pool.

    The pool is a fixed size, so this reallocates a budget rather than adding to
    it: with three queries the original keeps roughly a third of the slots it
    would have had alone, and its deeper candidates are displaced. That is the
    intended trade - deep candidates for one reading of the question, in
    exchange for shallow candidates for every reading.

    Returns (merged results, {chunk_id: index of the run that contributed it}).
    """
    if limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}")

    from rag.types import RetrievedChunk

    merged, source, seen = [], {}, set()
    depth = max((len(run) for run in runs), default=0)

    for position in range(depth):
        for index, run in enumerate(runs):
            if position >= len(run):
                continue
            candidate = run[position]
            chunk_id = candidate.chunk.chunk_id
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            source[chunk_id] = index
            merged.append(
                RetrievedChunk(chunk=candidate.chunk, score=candidate.score,
                               rank=len(merged) + 1)
            )
            if len(merged) == limit:
                return merged, source

    return merged, source
