#!/usr/bin/env python3
"""Score the retriever against the fixed evaluation set.

This is the first number the project produces. It answers one question: when
we search with each evaluation question, do the chunks that come back actually
contain the answer?

Two levels are reported, because they fail differently.

  Document level - did a retrieved chunk come from a document the answer lives
  in? This is the loose bar. With only ten documents it is easy to score well
  here while still returning the wrong paragraph.

  Passage level - did a retrieved chunk actually contain the gold quote? This
  is the bar that matters for generation, since a model can only answer from
  text it was handed.

The gap between the two is the interesting part: it is the share of questions
where retrieval found the right article and missed the right sentence.

Unanswerable questions are excluded. They exist to test whether generation
declines to answer, and counting them as retrieval failures would make every
score permanently wrong by their share of the set.

Usage:
    python scripts/run_retrieval_eval.py
    python scripts/run_retrieval_eval.py --size 256 --overlap 0
"""

import argparse
import json
import sys
from pathlib import Path

# Run as `python scripts/run_retrieval_eval.py`, which puts scripts/ on
# sys.path rather than the repo root - so point at the root explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.chunking import chunk_corpus  # noqa: E402
from rag.corpus import load_corpus  # noqa: E402
from rag.datasets import DEFAULT_DATASET, resolve  # noqa: E402
from rag.embedding import DEFAULT_MODEL, MODELS  # noqa: E402
from rag.retrieval import VectorIndex  # noqa: E402
from rag.text import contains_quote  # noqa: E402

# Cutoffs reported for both levels.
CUTOFFS = (1, 3, 5)

# The cutoffs an experiment may report, subject to its k. A run can only be
# measured at a cutoff it actually retrieved: reporting @10 for a run that
# fetched five chunks would print recall@5's value under a @10 label, which is
# a fabricated measurement rather than a missing one.
CUTOFF_LADDER = (1, 3, 5, 10)

# MRR is always reported at this cutoff, whatever k is.
#
# Uncapped MRR silently changes meaning with k - a first hit at rank 7 counts
# for 1/7 when k=10 and for 0 when k=5 - which would leave one summary column
# holding two different metrics. Pinning it keeps every row comparable. Every
# run recorded so far used k=5, so capping here reproduces their saved values
# exactly.
MRR_CUTOFF = 5


def cutoffs_for(k):
    """The cutoffs a run with this k can honestly report.

    The ladder, trimmed to what was retrieved, plus k itself so a run is always
    measurable at its own depth. k=5 gives (1, 3, 5) - unchanged from every
    existing run - and k=10 gives (1, 3, 5, 10).
    """
    return tuple(sorted({c for c in CUTOFF_LADDER if c <= k} | {k}))


def load_questions(path):
    """Read the evaluation set, split into answerable and unanswerable."""
    answerable, unanswerable = [], []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        question = json.loads(line)
        if question["type"] == "unanswerable":
            unanswerable.append(question)
        else:
            answerable.append(question)
    return answerable, unanswerable


def group_ranks(question, results):
    """First rank at which each evidence group is satisfied, else None.

    `supporting` is a list of evidence groups, one per fact the question
    actually requires. The quotes inside a group are *alternatives*: any one of
    them proves that fact, so the group is satisfied by whichever appears
    first. One quote may sit in several groups when a single sentence proves
    several facts at once.

    Returns one rank per group, in the order the groups are declared.
    """
    groups = question["supporting"]
    ranks = [None] * len(groups)

    for result in results:
        for index, group in enumerate(groups):
            if ranks[index] is None and any(
                contains_quote(result.chunk.text, item["quote"])
                for item in group["quotes"]
            ):
                ranks[index] = result.rank
        if all(rank is not None for rank in ranks):
            break

    return ranks


def complete_rank(ranks):
    """The rank at which a question first becomes fully answerable.

    The deepest group rank once every group is satisfied, or None while any
    group is still missing - so a question with one unretrieved fact has no
    complete rank at all, however well the rest scored.
    """
    if not ranks or any(rank is None for rank in ranks):
        return None
    return max(ranks)


def first_hit_ranks(question, results):
    """Return (document rank, passage rank) of the first correct result.

    Either is None when nothing correct was retrieved. Ranks are 1-based.

    A question is credited at the document level if *any* one of its gold
    documents appears - the standard reading of recall@k.

    The passage rank is the *earliest* satisfied group, which is exactly what
    this returned before evidence groups existed: back then every gold quote
    was pooled and the first match won. Hit@k therefore means what it always
    meant, and the two new metrics below are what distinguish finding some of
    the required evidence from finding all of it.
    """
    gold_docs = {item["doc_id"]
                 for group in question["supporting"]
                 for item in group["quotes"]}

    document_rank = None
    for result in results:
        if result.chunk.doc_id in gold_docs:
            document_rank = result.rank
            break

    satisfied = [rank for rank in group_ranks(question, results) if rank is not None]
    return document_rank, (min(satisfied) if satisfied else None)


def evidence_metrics(per_question_ranks, cutoffs=CUTOFFS):
    """Evidence Coverage@k and Complete Evidence@k.

    Coverage is the mean *share* of a question's required facts that were
    retrieved; Complete is the share of questions where every required fact
    was. On a question with one group both collapse to Hit@k, so they only
    diverge where a question genuinely needs more than one fact - which is the
    whole point of measuring them.

    Complete@k <= Coverage@k <= Hit@k always holds.
    """
    total = len(per_question_ranks)
    coverage, complete = {}, {}

    for k in cutoffs:
        if not total:
            coverage[k] = complete[k] = 0.0
            continue
        shares, wholes = [], []
        for ranks in per_question_ranks:
            satisfied = [rank is not None and rank <= k for rank in ranks]
            shares.append(sum(satisfied) / len(satisfied) if satisfied else 0.0)
            wholes.append(bool(satisfied) and all(satisfied))
        coverage[k] = sum(shares) / total
        complete[k] = sum(wholes) / total

    return coverage, complete


def summarize(ranks, cutoffs=CUTOFFS, mrr_cutoff=None):
    """Turn a list of first-correct ranks into recall@k plus MRR.

    mrr_cutoff=None leaves MRR uncapped, counting a hit at any rank. Pass a
    cutoff to make MRR mean the same thing across runs that retrieved to
    different depths - see MRR_CUTOFF.
    """
    total = len(ranks)
    recall = {
        k: sum(1 for rank in ranks if rank is not None and rank <= k) / total
        for k in cutoffs
    }
    mrr = sum(
        1 / rank
        for rank in ranks
        if rank is not None and (mrr_cutoff is None or rank <= mrr_cutoff)
    ) / total
    return recall, mrr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=512, help="chunk size in characters")
    parser.add_argument("--overlap", type=int, default=128, help="chunk overlap in characters")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"one of: {', '.join(sorted(MODELS))}")
    parser.add_argument("--dataset", default=DEFAULT_DATASET,
                        help=f"corpus + eval set to use (default: {DEFAULT_DATASET})")
    # Default None so an explicit override is distinguishable from the
    # dataset default - resolve() files an overridden run under "custom".
    parser.add_argument("--eval", default=None,
                        help="override the eval set path")
    parser.add_argument("--corpus", default=None,
                        help="override the corpus directory")
    args = parser.parse_args()
    dataset, corpus_path, eval_file = resolve(
        args.dataset, corpus=args.corpus, eval_file=args.eval)

    documents = load_corpus(corpus_path)
    chunks = chunk_corpus(documents, "fixed_size", size=args.size, overlap=args.overlap)
    index = VectorIndex.build(chunks, model=args.model)

    answerable, unanswerable = load_questions(eval_file)

    # One record per question, so the overall numbers and the per-type
    # breakdown are computed from exactly the same measurements.
    records = []
    for question in answerable:
        results = index.search(question["question"], k=max(CUTOFFS))
        document_rank, passage_rank = first_hit_ranks(question, results)
        records.append((question["type"], document_rank, passage_rank))

    document_ranks = [record[1] for record in records]
    passage_ranks = [record[2] for record in records]

    document_recall, document_mrr = summarize(document_ranks)
    passage_recall, _ = summarize(passage_ranks)

    print("Retrieval evaluation")
    print(f"  corpus      {len(documents)} documents")
    print(
        f"  chunking    fixed_size, size={args.size}, overlap={args.overlap}"
        f" -> {len(chunks):,} chunks"
    )
    print(f"  model       {args.model} ({MODELS[args.model].dimensions}d)")
    print(
        f"  questions   {len(answerable)} answerable"
        f" ({len(unanswerable)} unanswerable excluded)"
    )

    print("\nDocument level - a retrieved chunk came from a gold document")
    for k in CUTOFFS:
        print(f"  Recall@{k}   {document_recall[k]:.3f}")
    print(f"  MRR        {document_mrr:.3f}")

    print("\nPassage level - a retrieved chunk contained the gold quote")
    for k in CUTOFFS:
        print(f"  Hit@{k}      {passage_recall[k]:.3f}")

    # Averaging over the whole set hides the cases that behave differently -
    # a change that helps simple lookups and hurts multi-step questions looks
    # like no change at all in the totals above.
    print("\nBy question type")
    header = (
        f"  {'type':<11} {'n':>3} | "
        + " ".join(f"R@{k}".rjust(5) for k in CUTOFFS)
        + f" {'MRR':>5} | "
        + " ".join(f"H@{k}".rjust(5) for k in CUTOFFS)
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for question_type in sorted({record[0] for record in records}):
        group = [record for record in records if record[0] == question_type]
        type_document_recall, type_mrr = summarize([record[1] for record in group])
        type_passage_recall, _ = summarize([record[2] for record in group])
        print(
            f"  {question_type:<11} {len(group):>3} | "
            + " ".join(f"{type_document_recall[k]:.3f}" for k in CUTOFFS)
            + f" {type_mrr:.3f} | "
            + " ".join(f"{type_passage_recall[k]:.3f}" for k in CUTOFFS)
        )

    gap = document_recall[max(CUTOFFS)] - passage_recall[max(CUTOFFS)]
    print(
        f"\n  Gap at k={max(CUTOFFS)}: {gap:.3f} - the share of questions where the right"
        f"\n  document was retrieved but the right passage was not."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
