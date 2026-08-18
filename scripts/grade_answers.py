#!/usr/bin/env python3
"""Retrieve, generate, and grade an answer for every evaluation question.

Two scores per answer, judged separately and never mixed:

  correctness   does it say what the reference answer says?
  groundedness  do the retrieved passages actually support it?

An answer can pass one and fail the other, which is the reason for grading
them apart. Retrieval metrics are not computed here - run_retrieval_eval.py
owns those, and mixing them would hide which stage a regression came from.

Unanswerable questions never reach the judge. "Did it refuse?" is an exact
string comparison, which is free and deterministic; spending a judge call
there would add noise to a measurement that is currently exact.

Refusals on answerable questions score correctness 0 and are excluded from the
groundedness average - asking whether a refusal is grounded is meaningless,
and scoring it as supported would inflate the number.

Usage:
    python scripts/grade_answers.py --limit 5
    python scripts/grade_answers.py --out graded.jsonl
"""

import argparse
import json
import sys
import textwrap
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from rag.chunking import chunk_corpus  # noqa: E402
from rag.corpus import load_corpus  # noqa: E402
from rag.datasets import DEFAULT_DATASET, resolve  # noqa: E402
from rag.generation import NO_ANSWER, generate_answer  # noqa: E402
from rag.judging import judge_correctness, judge_groundedness  # noqa: E402
from rag.retrieval import VectorIndex  # noqa: E402
from run_retrieval_eval import first_hit_ranks  # noqa: E402

WIDTH = 78


def load_questions(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def wrap(label, text):
    return textwrap.fill(
        text, width=WIDTH,
        initial_indent=f"{label}  ", subsequent_indent=" " * (len(label) + 2),
    )


def mean(values):
    return sum(values) / len(values) if values else None


def show(value):
    return "  n/a" if value is None else f"{value:.2f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="only the first N questions")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--dataset", default=DEFAULT_DATASET,
                        help=f"corpus + eval set to use (default: {DEFAULT_DATASET})")
    # Default None so an explicit override is distinguishable from the
    # dataset default - resolve() files an overridden run under "custom".
    parser.add_argument("--eval", default=None,
                        help="override the eval set path")
    parser.add_argument("--corpus", default=None,
                        help="override the corpus directory")
    parser.add_argument("--out", help="also write one JSON object per question here")
    args = parser.parse_args()
    dataset, corpus_path, eval_file = resolve(
        args.dataset, corpus=args.corpus, eval_file=args.eval)

    questions = load_questions(eval_file)
    if args.limit:
        questions = questions[: args.limit]

    index = VectorIndex.build(
        chunk_corpus(load_corpus(corpus_path), size=args.size, overlap=args.overlap)
    )

    answerable = [q for q in questions if q["type"] != "unanswerable"]
    unanswerable = [q for q in questions if q["type"] == "unanswerable"]
    print(
        f"Grading {len(questions)} questions: {len(answerable)} answerable "
        f"({len(answerable) * 2} judge calls), {len(unanswerable)} unanswerable "
        f"(exact-match refusal check, no judge)"
    )
    print(f"  retrieval   size={args.size}, overlap={args.overlap}, k={args.k}")

    records = []
    correctness = defaultdict(list)
    groundedness = defaultdict(list)
    refusals = Counter()

    for question in questions:
        retrieved = index.search(question["question"], k=args.k)
        _, passage_rank = first_hit_ranks(question, retrieved)
        answer = generate_answer(question["question"], retrieved)
        refused = answer.strip() == NO_ANSWER
        is_unanswerable = question["type"] == "unanswerable"

        record = {
            "qid": question["qid"],
            "type": question["type"],
            "gold_passage_rank": passage_rank,
            "refused": refused,
            "answer": answer,
        }

        if is_unanswerable:
            # Correct behaviour here is refusing; no judge needed.
            record["correct_refusal"] = refused
            refusals["unanswerable_correct" if refused else "unanswerable_missed"] += 1
        elif refused:
            record["correctness"] = 0
            record["correctness_reasoning"] = "refused to answer"
            record["groundedness"] = None
            correctness[question["type"]].append(0)
            refusals["answerable"] += 1
        else:
            c = judge_correctness(question["question"], question["answer"], answer)
            g = judge_groundedness(question["question"], retrieved, answer)
            record.update(
                correctness=c.score, correctness_reasoning=c.reasoning,
                groundedness=g.score, groundedness_reasoning=g.reasoning,
            )
            correctness[question["type"]].append(c.score)
            groundedness[question["type"]].append(g.score)

        records.append(record)

        print("\n" + "-" * WIDTH)
        gold = "n/a" if is_unanswerable else (
            f"rank {passage_rank}" if passage_rank else "NOT retrieved"
        )
        scores = (
            f"refusal {'correct' if refused else 'MISSED'}" if is_unanswerable
            else f"correctness {record['correctness']}  "
                 f"groundedness {record['groundedness'] if record['groundedness'] is not None else 'n/a'}"
        )
        print(f"{record['qid']}  {question['type']:<13} gold passage: {gold:<13} {scores}")
        print(wrap("Q ", question["question"]))
        print(wrap("E ", question["answer"]))
        print(wrap("A ", answer))
        if record.get("correctness_reasoning") and not is_unanswerable:
            print(wrap("c:", record["correctness_reasoning"]))
        if record.get("groundedness_reasoning"):
            print(wrap("g:", record["groundedness_reasoning"]))

    all_correctness = [s for group in correctness.values() for s in group]
    all_groundedness = [s for group in groundedness.values() for s in group]

    print("\n" + "=" * WIDTH)
    print("Answerable questions — mean score out of 2")
    print(f"  correctness    {show(mean(all_correctness))}   (n={len(all_correctness)})")
    print(f"  groundedness   {show(mean(all_groundedness))}   (n={len(all_groundedness)}, "
          f"{refusals['answerable']} refusals excluded)")

    print("\n  by type" + " " * 15 + "correctness   groundedness")
    for qtype in sorted(correctness):
        print(f"  {qtype:<20} {show(mean(correctness[qtype])):>8}      "
              f"{show(mean(groundedness[qtype])):>8}   (n={len(correctness[qtype])})")

    if unanswerable:
        total = refusals["unanswerable_correct"] + refusals["unanswerable_missed"]
        print(f"\nUnanswerable questions — refused when they should have")
        print(f"  {refusals['unanswerable_correct']} / {total}")

    if args.out:
        Path(args.out).write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
            encoding="utf-8",
        )
        print(f"\nper-question records written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
