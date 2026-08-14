#!/usr/bin/env python3
"""Run one experiment config against the fixed evaluation set and save the result.

An experiment is a config file plus the corpus. This runs the pipeline that
config describes, computes retrieval and answer metrics, and writes everything
needed to interpret the numbers later into results/runs/<run id>/.

The point of the config is that an experiment changes one thing. A config
extends a parent and lists only its overrides, so the other settings are not
in the file to be edited by accident - and if the merged config differs from
its parent in more than one place, this refuses to run without
--multi-variable.

Usage:
    python scripts/run_experiment.py baseline
    python scripts/run_experiment.py chunk-256
    python scripts/run_experiment.py my-config --multi-variable
"""

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from rag.chunking import chunk_corpus  # noqa: E402
from rag.corpus import load_corpus  # noqa: E402
from rag.experiment import (  # noqa: E402
    corpus_hash, diff_configs, dump_toml, load_config,
)
from rag.retrieval import VectorIndex  # noqa: E402
from run_retrieval_eval import CUTOFFS, first_hit_ranks, summarize  # noqa: E402

RESULTS = Path("results")
SUMMARY = RESULTS / "summary.csv"


def load_questions(path):
    return [json.loads(line) for line in
            Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def mean(values):
    return sum(values) / len(values) if values else None


def previous_corpus_hash():
    """The corpus hash the last recorded run measured, if there is one."""
    if not SUMMARY.exists():
        return None
    rows = list(csv.DictReader(SUMMARY.open()))
    return rows[-1]["corpus_hash"] if rows else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="config name in configs/, or a path to a .toml")
    parser.add_argument("--multi-variable", action="store_true",
                        help="allow a config that changes more than one setting")
    parser.add_argument("--any-corpus", action="store_true",
                        help="record the run even if the corpus differs from earlier runs")
    parser.add_argument("--eval", default="data/eval/questions.jsonl")
    parser.add_argument("--corpus", default="data/documents")
    args = parser.parse_args()

    config, parent = load_config(args.config)

    # ---- one variable at a time -------------------------------------------
    changes = diff_configs(parent, config) if parent else []
    variables = diff_configs(parent, config, count_only_variables=True) if parent else []
    print(f"config: {config['name']}" + (f" (extends {config['extends']})" if parent else ""))
    if config.get("note"):
        print(f"  {config['note']}")
    if parent:
        if changes:
            for key, old, new in changes:
                print(f"  {key}: {old} -> {new}")
        else:
            print("  (identical to its parent - this is a repeat run)")
        if len(variables) > 1 and not args.multi_variable:
            print(
                f"\nRefusing to run: this config changes {len(variables)} settings at once, "
                f"so a\nchange in the numbers could not be attributed to any one of them.\n"
                f"Split it into separate configs, or pass --multi-variable if the "
                f"combination\nis the thing you mean to test.",
                file=sys.stderr,
            )
            return 1
    else:
        print("  (no parent - this is a root config)")

    # ---- build ------------------------------------------------------------
    documents = load_corpus(args.corpus)
    digest = corpus_hash(documents)
    earlier = previous_corpus_hash()
    if earlier and earlier != digest and not args.any_corpus:
        print(
            f"\nRefusing to run: the corpus has changed since the last recorded run\n"
            f"({earlier} -> {digest}), so the numbers would not be comparable.\n"
            f"Pass --any-corpus to record it anyway.",
            file=sys.stderr,
        )
        return 1

    chunks = chunk_corpus(
        documents, config["chunking"]["strategy"],
        size=config["chunking"]["size"], overlap=config["chunking"]["overlap"],
    )
    index = VectorIndex.build(chunks, model=config["embedding"]["model"])
    questions = load_questions(args.eval)
    k = config["retrieval"]["k"]

    generating = config["generation"]["enabled"]
    judging = config["judging"]["enabled"] and generating
    if generating:
        from rag.generation import NO_ANSWER, generate_answer
    if judging:
        from rag.judging import judge_correctness, judge_groundedness

    print(f"\ncorpus {digest} | {len(documents)} documents -> {len(chunks):,} chunks | k={k}")
    print(f"generation {'on' if generating else 'off'}, judging {'on' if judging else 'off'}")

    # ---- run --------------------------------------------------------------
    records = []
    doc_ranks, passage_ranks = [], []
    doc_ranks_by_type, passage_ranks_by_type = defaultdict(list), defaultdict(list)
    correctness, groundedness = defaultdict(list), defaultdict(list)
    refusals = Counter()

    for question in questions:
        unanswerable = question["type"] == "unanswerable"
        retrieved = index.search(question["question"], k=k)
        doc_rank, passage_rank = first_hit_ranks(question, retrieved)

        record = {"qid": question["qid"], "type": question["type"],
                  "doc_rank": doc_rank, "passage_rank": passage_rank}

        if not unanswerable:
            doc_ranks.append(doc_rank)
            passage_ranks.append(passage_rank)
            doc_ranks_by_type[question["type"]].append(doc_rank)
            passage_ranks_by_type[question["type"]].append(passage_rank)

        if generating:
            answer = generate_answer(question["question"], retrieved,
                                     model=config["generation"]["model"])
            refused = answer.strip() == NO_ANSWER
            record.update(answer=answer, refused=refused)

            if unanswerable:
                refusals["unanswerable_correct" if refused else "unanswerable_missed"] += 1
            elif refused:
                refusals["answerable"] += 1
                correctness[question["type"]].append(0)
                record.update(correctness=0, groundedness=None)
            elif judging:
                c = judge_correctness(question["question"], question["answer"], answer,
                                      judge=config["judging"]["model"])
                g = judge_groundedness(question["question"], retrieved, answer,
                                       judge=config["judging"]["model"])
                correctness[question["type"]].append(c.score)
                groundedness[question["type"]].append(g.score)
                record.update(correctness=c.score, correctness_reasoning=c.reasoning,
                              groundedness=g.score, groundedness_reasoning=g.reasoning)

        records.append(record)

    # ---- metrics ----------------------------------------------------------
    doc_recall, doc_mrr = summarize(doc_ranks)
    passage_recall, _ = summarize(passage_ranks)
    all_correct = [s for group in correctness.values() for s in group]
    all_grounded = [s for group in groundedness.values() for s in group]
    gold_hit = [r["correctness"] for r in records
                if r.get("correctness") is not None and r["passage_rank"]]
    gold_missed = [r["correctness"] for r in records
                   if r.get("correctness") is not None and not r["passage_rank"]
                   and r["type"] != "unanswerable"]

    metrics = {
        "run_id": None,  # filled below
        "config": config,
        "corpus_hash": digest,
        "documents": len(documents),
        "chunks": len(chunks),
        "questions": {"total": len(questions), "answerable": len(doc_ranks)},
        "retrieval": {
            "document_recall": {str(c): doc_recall[c] for c in CUTOFFS},
            "document_mrr": doc_mrr,
            "passage_hit": {str(c): passage_recall[c] for c in CUTOFFS},
            "by_type": {
                t: {"document_recall": summarize(doc_ranks_by_type[t])[0][max(CUTOFFS)],
                    "document_mrr": summarize(doc_ranks_by_type[t])[1],
                    "passage_hit": summarize(passage_ranks_by_type[t])[0][max(CUTOFFS)]}
                for t in sorted(doc_ranks_by_type)
            },
        },
    }
    if generating:
        metrics["answers"] = {
            "correctness": mean(all_correct),
            "groundedness": mean(all_grounded),
            "correctness_when_gold_retrieved": mean(gold_hit),
            "correctness_when_gold_missed": mean(gold_missed),
            "refusals_answerable": refusals["answerable"],
            "unanswerable_refused": refusals["unanswerable_correct"],
            "unanswerable_total": refusals["unanswerable_correct"] + refusals["unanswerable_missed"],
            "by_type": {t: {"correctness": mean(correctness[t]),
                            "groundedness": mean(groundedness[t])}
                        for t in sorted(correctness)},
        }

    # ---- save -------------------------------------------------------------
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{config['name']}_{stamp}"
    metrics["run_id"] = run_id
    metrics["finished_at"] = stamp

    run_dir = RESULTS / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    # A snapshot, not a reference: editing the original config later must not
    # change what a past run means.
    (run_dir / "config.toml").write_text(dump_toml(config), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (run_dir / "records.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")

    row = {
        "run_id": run_id, "config": config["name"], "extends": config.get("extends", ""),
        "corpus_hash": digest, "chunks": len(chunks),
        "size": config["chunking"]["size"], "overlap": config["chunking"]["overlap"],
        "k": k, "embed_model": config["embedding"]["model"],
        "gen_model": config["generation"]["model"] if generating else "",
        "judge_model": config["judging"]["model"] if judging else "",
        "recall@1": doc_recall[1], "recall@3": doc_recall[3], "recall@5": doc_recall[5],
        "mrr": doc_mrr,
        "hit@1": passage_recall[1], "hit@3": passage_recall[3], "hit@5": passage_recall[5],
        "correctness": mean(all_correct), "groundedness": mean(all_grounded),
        "unanswerable_refused": refusals["unanswerable_correct"] if generating else "",
    }
    RESULTS.mkdir(exist_ok=True)
    write_header = not SUMMARY.exists()
    with SUMMARY.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow({key: ("" if value is None else value) for key, value in row.items()})

    # ---- report -----------------------------------------------------------
    print(f"\nRetrieval ({len(doc_ranks)} answerable questions)")
    for c in CUTOFFS:
        print(f"  Recall@{c}   {doc_recall[c]:.3f}      Hit@{c}   {passage_recall[c]:.3f}")
    print(f"  MRR        {doc_mrr:.3f}")

    if generating:
        print("\nAnswers (mean out of 2)")
        print(f"  correctness    {mean(all_correct):.2f}" if all_correct else "  correctness    n/a")
        if all_grounded:
            print(f"  groundedness   {mean(all_grounded):.2f}   "
                  f"({refusals['answerable']} refusals excluded)")
        if gold_hit and gold_missed:
            print(f"  correctness when gold passage retrieved  {mean(gold_hit):.2f}")
            print(f"  correctness when it was not              {mean(gold_missed):.2f}")
        print(f"  unanswerable refused  {refusals['unanswerable_correct']}"
              f"/{refusals['unanswerable_correct'] + refusals['unanswerable_missed']}")

    print(f"\nsaved to {run_dir}/ and appended to {SUMMARY}")
    print(
        f"\nNote: {len(doc_ranks)} answerable questions, so one question moves a 0-2 mean "
        f"by about\n{2 / len(doc_ranks):.3f}, and generation is not perfectly repeatable. "
        f"Worth keeping in mind\nwhen reading small differences between runs."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
