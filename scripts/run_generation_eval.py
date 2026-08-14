#!/usr/bin/env python3
"""Generate an answer for each evaluation question and print it for reading.

This does not score anything. It exists so you can look at real answers before
deciding how to grade them - whether a keyword check would do, or whether the
failures are subtle enough to need a judge. Designing the grader first would
mean guessing at what it has to catch.

Every question is shown with whether the gold passage was actually in the
context the model received. Without that, a wrong answer is ambiguous: the
retriever may never have handed over the text, in which case the generator had
no chance. Separating those two is the reason the pipeline has stages.

Unanswerable questions are included, unlike in retrieval scoring - they are
the point here. Each one asks for something the corpus does not contain, and
the interesting question is whether the model says so or invents an answer.

This calls a paid API once per question. Use --limit to try a few first.

Usage:
    python scripts/run_generation_eval.py --limit 5
    python scripts/run_generation_eval.py > answers.txt
"""

import argparse
import json
import sys
import textwrap
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from rag.chunking import chunk_corpus  # noqa: E402
from rag.corpus import load_corpus  # noqa: E402
from rag.generation import NO_ANSWER, generate_answer  # noqa: E402
from rag.retrieval import VectorIndex  # noqa: E402

# Reused rather than reimplemented: two definitions of "was the gold passage
# retrieved" could disagree, and this script and the retrieval scorer would
# quietly describe different things.
from run_retrieval_eval import first_hit_ranks  # noqa: E402

WIDTH = 78


def load_questions(path):
    """Read every question, answerable and unanswerable alike, in file order."""
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def wrap(label, text):
    """Render one labelled field, wrapped and hanging-indented."""
    return textwrap.fill(
        text,
        width=WIDTH,
        initial_indent=f"{label}  ",
        subsequent_indent=" " * (len(label) + 2),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit",
        type=int,
        help="only run the first N questions, in file order "
        "(unanswerable ones are last, so a small limit will miss them)",
    )
    parser.add_argument("--k", type=int, default=5, help="chunks retrieved per question")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--eval", default="data/eval/questions.jsonl")
    parser.add_argument("--corpus", default="data/documents")
    args = parser.parse_args()

    questions = load_questions(args.eval)
    if args.limit:
        questions = questions[: args.limit]

    index = VectorIndex.build(
        chunk_corpus(load_corpus(args.corpus), size=args.size, overlap=args.overlap)
    )

    types = Counter(question["type"] for question in questions)
    print(f"Generating answers for {len(questions)} questions (one API call each)")
    print(f"  retrieval   size={args.size}, overlap={args.overlap}, k={args.k}")
    print(f"  types       {', '.join(f'{n} {t}' for t, n in sorted(types.items()))}")

    refusals = Counter()
    totals = Counter()

    for question in questions:
        retrieved = index.search(question["question"], k=args.k)
        _, passage_rank = first_hit_ranks(question, retrieved)
        answer = generate_answer(question["question"], retrieved)

        unanswerable = question["type"] == "unanswerable"
        refused = answer.strip() == NO_ANSWER

        if unanswerable:
            gold = "n/a (nothing to retrieve)"
        elif passage_rank is not None:
            gold = f"yes (rank {passage_rank})"
        else:
            gold = "NO"

        group = "unanswerable" if unanswerable else "answerable"
        totals[group] += 1
        if refused:
            refusals[group] += 1
            if not unanswerable:
                found = passage_rank is not None
                refusals["with gold passage" if found else "without gold passage"] += 1

        print("\n" + "-" * WIDTH)
        print(f"{question['qid']}  {question['type']:<13} gold passage in context: {gold}")
        print(wrap("Q ", question["question"]))
        print(wrap("E ", question["answer"]))
        print(wrap("A ", answer))

    print("\n" + "=" * WIDTH)
    print("Refusals - answers exactly matching the no-answer sentence")
    print(f'  "{NO_ANSWER}"\n')
    for group in ("unanswerable", "answerable"):
        if totals[group]:
            print(f"  {group:<14} {refusals[group]:>3} / {totals[group]}")
    if totals["answerable"]:
        print(f"     gold passage present  {refusals['with gold passage']:>3}")
        print(f"     gold passage missing  {refusals['without gold passage']:>3}")
    print("\nNo correctness judgement is made here - read the answers above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
