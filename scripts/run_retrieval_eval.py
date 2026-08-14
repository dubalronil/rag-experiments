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
from rag.embedding import DEFAULT_MODEL, MODELS  # noqa: E402
from rag.retrieval import VectorIndex  # noqa: E402
from rag.text import contains_quote  # noqa: E402

# Cutoffs reported for both levels.
CUTOFFS = (1, 3, 5)


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


def first_hit_ranks(question, results):
    """Return (document rank, passage rank) of the first correct result.

    Either is None when nothing correct was retrieved. Ranks are 1-based.

    A question is credited at the document level if *any* one of its gold
    documents appears - the standard reading of recall@k. Multi-hop questions
    genuinely need all of their sources to be answerable, so this is the
    generous interpretation and the passage-level number is the honest one.
    """
    gold_docs = {item["doc_id"] for item in question["supporting"]}
    gold_quotes = [item["quote"] for item in question["supporting"]]

    document_rank = None
    passage_rank = None

    for result in results:
        if document_rank is None and result.chunk.doc_id in gold_docs:
            document_rank = result.rank
        if passage_rank is None and any(
            contains_quote(result.chunk.text, quote) for quote in gold_quotes
        ):
            passage_rank = result.rank
        if document_rank is not None and passage_rank is not None:
            break

    return document_rank, passage_rank


def summarize(ranks, cutoffs=CUTOFFS):
    """Turn a list of first-correct ranks into recall@k plus MRR."""
    total = len(ranks)
    recall = {
        k: sum(1 for rank in ranks if rank is not None and rank <= k) / total
        for k in cutoffs
    }
    mrr = sum(1 / rank for rank in ranks if rank is not None) / total
    return recall, mrr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=512, help="chunk size in characters")
    parser.add_argument("--overlap", type=int, default=128, help="chunk overlap in characters")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"one of: {', '.join(sorted(MODELS))}")
    parser.add_argument("--eval", default="data/eval/questions.jsonl")
    parser.add_argument("--corpus", default="data/documents")
    args = parser.parse_args()

    documents = load_corpus(args.corpus)
    chunks = chunk_corpus(documents, "fixed_size", size=args.size, overlap=args.overlap)
    index = VectorIndex.build(chunks, model=args.model)

    answerable, unanswerable = load_questions(args.eval)

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
