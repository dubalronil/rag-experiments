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
  5. unanswerable questions have no evidence groups, and others have some,
     with every group holding at least one verbatim quote

Standard library only - no third-party dependencies.

Usage:
    python scripts/validate_eval.py                  # checks the nba eval set
    python scripts/validate_eval.py --dataset space
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
from rag.datasets import DEFAULT_DATASET, resolve  # noqa: E402

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


def check_absent(qid, question, documents):
    """Verify the tripwire phrases for an unanswerable question really are absent.

    A question labelled unanswerable is an assertion about the corpus, and
    nothing was checking it - one of these labels turned out to be wrong, with
    the corpus stating the answer in plain text, and the mistake only surfaced
    when a model answered correctly and looked like it was hallucinating.

    So `absent` lists phrases that would appear if the question were in fact
    answerable, and this fails if any of them is present. It is a tripwire,
    not a proof: a badly chosen phrase can still miss, exactly as a badly
    chosen gold quote can point at the wrong sentence.
    """
    problems = []
    for phrase in question.get("absent", []):
        needle = normalize(phrase)
        for doc_id, text in documents.items():
            if needle in text:
                problems.append(
                    f"{qid}: labelled unanswerable, but {phrase!r} appears in "
                    f"{doc_id} - the corpus may answer this after all"
                )
                break
    return problems


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
            if not question.get("notes", "").strip():
                problems.append(
                    f"{qid}: unanswerable questions need a notes field saying "
                    f"why the corpus cannot answer them"
                )
            problems.extend(check_absent(qid, question, documents))
            continue
        if question.get("absent"):
            problems.append(
                f"{qid}: only unanswerable questions may carry 'absent'"
            )
        if not supporting:
            problems.append(f"{qid}: no evidence groups")
            continue

        # `supporting` is a list of evidence groups: one group per fact the
        # question requires, holding alternative quotes that each prove it.
        for index, group in enumerate(supporting):
            where = f"{qid}[{index}]"

            if not isinstance(group, dict) or "quotes" not in group:
                problems.append(
                    f"{where}: evidence groups need a 'quotes' list "
                    f"(the flat doc_id/quote form is no longer accepted)"
                )
                continue
            quotes = group["quotes"]
            if not quotes:
                problems.append(f"{where}: empty evidence group")
                continue

            for position, item in enumerate(quotes):
                spot = f"{where}.quotes[{position}]"
                doc_id = item.get("doc_id")
                quote = item.get("quote")

                if not doc_id or not quote:
                    problems.append(f"{spot}: needs both doc_id and quote")
                    continue
                if doc_id not in documents:
                    problems.append(f"{spot}: no such document {doc_id!r}")
                    continue
                if normalize(quote) not in documents[doc_id]:
                    problems.append(f"{spot}: quote not found verbatim in {doc_id}")

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
        for group in q.get("supporting", [])
        if isinstance(group, dict)
        for item in group.get("quotes", [])
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

    eval_path = Path(eval_file)
    corpus_dir = Path(corpus_path)

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
