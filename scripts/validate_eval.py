#!/usr/bin/env python3
"""Check that the evaluation set is consistent with the document corpus.

Gold labels are stored as verbatim quotes rather than chunk IDs, so that the
same labels keep working no matter how the corpus is chunked. That only holds
if every quote really does appear in the document it claims to come from - a
quote with a typo in it would score as a retrieval failure forever, and nothing
would ever reveal why. This script is what keeps that from happening.

Checks performed:
  1. every line is valid JSON with the required fields
  2. qids are unique
  3. every doc_id names a file in the corpus
  4. every quote appears verbatim in that document
  5. unanswerable questions have no supporting quotes, and others have some

Standard library only - no third-party dependencies.

Usage:
    python scripts/validate_eval.py       # checks data/eval/questions.jsonl
    python scripts/validate_eval.py --eval /tmp/draft-questions.jsonl

Exits non-zero if anything fails, so it can be used as a pre-commit check.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# Run as `python scripts/validate_eval.py`, which puts scripts/ on sys.path
# rather than the repo root - so point at the root explicitly before importing
# from the rag package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.text import normalize  # noqa: E402

REQUIRED_FIELDS = {"qid", "question", "answer", "type", "supporting"}
VALID_TYPES = {"factual", "multi_hop", "temporal", "unanswerable"}


def load_corpus(corpus_dir):
    """Return {doc_id: normalized text} for every Markdown file in the corpus."""
    documents = {}
    for path in sorted(corpus_dir.glob("*.md")):
        documents[path.stem] = normalize(path.read_text(encoding="utf-8"))
    return documents


def load_questions(eval_path):
    """Parse the JSONL file, reporting the line number of any bad row."""
    questions = []
    errors = []
    for number, line in enumerate(eval_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            questions.append((number, json.loads(line)))
        except json.JSONDecodeError as error:
            errors.append(f"line {number}: invalid JSON - {error}")
    return questions, errors


def check(questions, documents):
    """Return a list of problems. An empty list means the eval set is sound."""
    problems = []
    seen_qids = Counter()

    for number, question in questions:
        qid = question.get("qid", f"<line {number}>")

        missing = REQUIRED_FIELDS - question.keys()
        if missing:
            problems.append(f"{qid}: missing field(s) {', '.join(sorted(missing))}")
            continue

        seen_qids[question["qid"]] += 1

        if question["type"] not in VALID_TYPES:
            problems.append(
                f"{qid}: unknown type {question['type']!r} "
                f"(expected one of {', '.join(sorted(VALID_TYPES))})"
            )

        supporting = question["supporting"]

        # An unanswerable question with gold quotes is a contradiction, and an
        # answerable one without them cannot be scored at all.
        if question["type"] == "unanswerable":
            if supporting:
                problems.append(f"{qid}: unanswerable but has supporting quotes")
            continue
        if not supporting:
            problems.append(f"{qid}: no supporting quotes")
            continue

        for index, item in enumerate(supporting):
            where = f"{qid}[{index}]"
            doc_id = item.get("doc_id")
            quote = item.get("quote")

            if not doc_id or not quote:
                problems.append(f"{where}: needs both doc_id and quote")
                continue
            if doc_id not in documents:
                problems.append(f"{where}: no such document {doc_id!r}")
                continue
            if normalize(quote) not in documents[doc_id]:
                problems.append(f"{where}: quote not found verbatim in {doc_id}")

    for qid, count in seen_qids.items():
        if count > 1:
            problems.append(f"{qid}: duplicated {count} times")

    return problems


def summarize(questions, documents):
    """Print the shape of the eval set: type mix and per-document coverage."""
    types = Counter(q.get("type", "?") for _, q in questions)
    docs = Counter(
        item["doc_id"]
        for _, q in questions
        for item in q.get("supporting", [])
        if isinstance(item, dict) and "doc_id" in item
    )

    print(f"{len(questions)} questions\n")
    print("by type:")
    for name, count in sorted(types.items(), key=lambda pair: -pair[1]):
        share = 100 * count / len(questions)
        print(f"  {name:<14} {count:>3}  ({share:.0f}%)")

    print("\ngold quotes per document:")
    for doc_id in sorted(documents):
        marker = "" if docs[doc_id] else "   <- no questions reference this document"
        print(f"  {doc_id:<36} {docs[doc_id]:>3}{marker}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", default="data/eval/questions.jsonl")
    parser.add_argument("--corpus", default="data/documents")
    args = parser.parse_args()

    eval_path = Path(args.eval)
    corpus_dir = Path(args.corpus)

    if not eval_path.exists():
        print(f"no eval file at {eval_path}", file=sys.stderr)
        return 1
    if not corpus_dir.is_dir():
        print(f"no corpus directory at {corpus_dir}", file=sys.stderr)
        return 1

    documents = load_corpus(corpus_dir)
    questions, parse_errors = load_questions(eval_path)

    summarize(questions, documents)

    problems = parse_errors + check(questions, documents)
    print()
    if problems:
        print(f"FAILED - {len(problems)} problem(s):", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1

    print("OK - every quote verified verbatim against the corpus")
    return 0


if __name__ == "__main__":
    sys.exit(main())
