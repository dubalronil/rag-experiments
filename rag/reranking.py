"""Rescore retrieved chunks by reading the question and the chunk together.

Everything up to this point compares two vectors that were computed apart from
each other: the question became a vector, each chunk became a vector, and
retrieval scored them by angle. That is fast enough to search a whole corpus,
and it is why it cannot tell "career regular season scoring average" from
"career-high in a single game" - the two chunks look nearly identical to an
encoder that never saw the question.

A cross-encoder reads the pair in one forward pass, so the question can decide
what matters about the chunk. The cost is that it cannot be precomputed: every
(question, chunk) pair is its own inference, which is why this runs as a second
stage over a few dozen candidates rather than over the corpus.

So retrieval becomes two stages when this is enabled:

    index.search(question, k=RERANK_DEPTH)   cheap, wide, vector-based
    rerank(question, candidates, k)          expensive, narrow, pair-based

The first stage sets a ceiling the second cannot exceed - a chunk missing from
the candidate set can never be recovered - which is why runs record where the
gold chunk sat in the candidate list as well as where it finished.

Local and free, like the other local models: no API key, no network after the
first run, and the same input gives the same scores every time.
"""

import numpy as np

# The reranker. Small on purpose - 22M parameters, roughly 80MB - so a run
# stays iterable on CPU. BAAI/bge-reranker-base is the stronger model and pairs
# naturally with BGE embeddings, but at 278M parameters it is an order of
# magnitude slower per pair; swapping it in is a clean follow-up experiment
# once this one shows whether reranking helps at all.
#
# Its window is 512 tokens, and a question plus a 512-character chunk is about
# 140, so nothing is truncated here - unlike the embedding layer, where the
# model's window was the binding constraint.
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# How many candidates the first stage hands over. Fixed for the first
# experiment rather than exposed as a config key, so a reranked run differs
# from a dense run in exactly one thing.
#
# 50 is chosen from the failure analysis: of the eight gold passages dense
# retrieval currently misses, seven sit at rank 31 or better, so a candidate
# set of 50 already contains them. Going deeper is the natural second
# experiment - it raises the ceiling and gives the reranker more distractors to
# get wrong, and those two effects need measuring separately.
RERANK_DEPTH = 50

# Loading the model takes seconds and it is stateless once loaded.
_loaded = {}


def load_reranker(model=RERANK_MODEL):
    """Return the CrossEncoder for a named model, loading it once.

    Imported here rather than at module scope so that a run with reranking
    disabled never constructs the model, never downloads the weights, and pays
    nothing for a stage it is not using.
    """
    if model not in _loaded:
        from sentence_transformers import CrossEncoder

        _loaded[model] = CrossEncoder(model)
    return _loaded[model]


def rerank(question, candidates, k=5, model=RERANK_MODEL):
    """Rescore candidate chunks against the question and return the best k.

    Takes and returns RetrievedChunk lists, so this slots between retrieval and
    everything downstream without either end knowing it happened. Ranks are
    renumbered from 1: the returned rank is the position after reranking, and
    the candidate's original position is deliberately not carried along - the
    caller that wants both keeps the input list.

    Scores are the cross-encoder's raw logits. They are unbounded and can be
    negative, which is a third scale after cosine similarity and BM25 - and
    like those, only the ordering it induces is used.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if not candidates:
        return []

    # dataclasses.replace would be tidier, but RetrievedChunk is frozen and
    # both of the fields being set are ones we are replacing anyway.
    from rag.types import RetrievedChunk

    pairs = [(question, candidate.chunk.text) for candidate in candidates]
    scores = np.asarray(load_reranker(model).predict(pairs), dtype=np.float32)

    # Explicit tie-break on chunk_id, as in hybrid fusion: equal scores are
    # rare here but not impossible, and leaving the order to the sort's
    # stability would make it depend on the first stage's output order.
    order = sorted(
        range(len(candidates)),
        key=lambda i: (-float(scores[i]), candidates[i].chunk.chunk_id),
    )

    return [
        RetrievedChunk(
            chunk=candidates[index].chunk,
            score=float(scores[index]),
            rank=position,
        )
        for position, index in enumerate(order[:k], start=1)
    ]


# How many of the final k slots the original question keeps when subquery-aware
# selection is on. Fixed, like the other retrieval constants.
#
# The original query is the only one that represents the whole ask, so it keeps
# a majority; the remaining slots are guaranteed to the subqueries, which is the
# only way a chunk answering just one part of the question can be promoted. At
# k=5 that is 3 for the question as asked and one for each subquery.
SELECTION_FLOOR = 3


def rerank_subquery_aware(queries, candidates, k=5, model=RERANK_MODEL,
                          floor=SELECTION_FLOOR):
    """Rescore the pool against every query, then select for coverage.

    Cross-encoder logits are not comparable across queries - in one traced
    question every candidate scored negative, in another the fifth-place score
    was above 2.5 - so scores are used only to order candidates *within* a
    single query, where they are commensurate. Selection across queries is by
    rank alone. Nothing is summed, averaged, or maxed.

    Domination is prevented structurally rather than numerically: each query
    owns a fixed number of slots, so a subquery whose logits happen to run high
    cannot take more than its share.

    Returns (results, diagnostics), where diagnostics maps chunk_id to the query
    that selected it and its rank and score under every query.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if not candidates:
        return [], {}
    if not queries:
        raise ValueError("need at least one query")

    from rag.types import RetrievedChunk

    encoder = load_reranker(model)
    orderings, scores = [], []
    for query in queries:
        raw = np.asarray(
            encoder.predict([(query, c.chunk.text) for c in candidates]),
            dtype=np.float32,
        )
        by_id = {c.chunk.chunk_id: float(raw[i]) for i, c in enumerate(candidates)}
        order = sorted(range(len(candidates)),
                       key=lambda i: (-float(raw[i]), candidates[i].chunk.chunk_id))
        orderings.append([candidates[i] for i in order])
        scores.append(by_id)

    ranks = [{c.chunk.chunk_id: position
              for position, c in enumerate(ordering, start=1)}
             for ordering in orderings]

    chosen, seen = [], set()

    def take(candidate, source):
        chunk_id = candidate.chunk.chunk_id
        if chunk_id in seen:
            return False
        seen.add(chunk_id)
        chosen.append((candidate, source))
        return True

    # 1. The original question keeps its top `floor`.
    for candidate in orderings[0]:
        if len(chosen) >= min(floor, k):
            break
        take(candidate, 0)

    # 2. Remaining slots go round-robin to the subqueries, each contributing its
    #    highest-ranked chunk not already taken.
    remaining = list(range(1, len(orderings)))
    while len(chosen) < k and remaining:
        progressed = False
        for source in list(remaining):
            if len(chosen) >= k:
                break
            nxt = next((c for c in orderings[source]
                        if c.chunk.chunk_id not in seen), None)
            if nxt is None:
                remaining.remove(source)
                continue
            take(nxt, source)
            progressed = True
        if not progressed:
            break

    # 3. Top up from the original ordering if the subqueries ran dry.
    for candidate in orderings[0]:
        if len(chosen) >= k:
            break
        take(candidate, 0)

    results, diagnostics = [], {}
    for position, (candidate, source) in enumerate(chosen, start=1):
        chunk_id = candidate.chunk.chunk_id
        # Scored against the original question so the column stays one unit;
        # ordering here reflects the selection policy, not this score.
        results.append(RetrievedChunk(chunk=candidate.chunk,
                                      score=scores[0][chunk_id], rank=position))
        diagnostics[chunk_id] = {
            "selected_by": source,
            "ranks": [r[chunk_id] for r in ranks],
            "scores": [round(sc[chunk_id], 4) for sc in scores],
        }
    return results, diagnostics
