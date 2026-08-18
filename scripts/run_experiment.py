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
from time import perf_counter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from rag.chunking import chunk_corpus  # noqa: E402
from rag.corpus import load_corpus  # noqa: E402
from rag.datasets import DEFAULT_DATASET, resolve  # noqa: E402
from rag.experiment import (  # noqa: E402
    corpus_hash, diff_configs, dump_toml, load_config,
)
from rag.retrieval import FUSION_DEPTH, RRF_K, build_index  # noqa: E402
from run_retrieval_eval import (  # noqa: E402
    CUTOFF_LADDER, MRR_CUTOFF, complete_rank, cutoffs_for, evidence_metrics,
    first_hit_ranks, group_ranks, summarize,
)

RESULTS = Path("results")
SUMMARY = RESULTS / "summary.csv"


def load_questions(path):
    return [json.loads(line) for line in
            Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def stage(label):
    """Announce a stage on the current progress line, before it blocks.

    Written without a newline and flushed, so the label appears while the call
    is still in flight - the point is seeing where a run is sitting, not
    reading a log afterwards.
    """
    print(f" {label}", end="", flush=True)


def took(started):
    print(f" {perf_counter() - started:.1f}s", end="", flush=True)


def mean(values):
    return sum(values) / len(values) if values else None


def previous_corpus_hash(dataset):
    """The corpus hash the last recorded run measured, if there is one."""
    if not SUMMARY.exists():
        return None
    rows = list(csv.DictReader(SUMMARY.open()))
    # Only rows for the same dataset. Two datasets are *meant* to have
    # different corpora, so comparing across them would fire the guard on every
    # run after the first. Older rows predate the column and are all nba,
    # which the migration backfilled - .get keeps this working either way.
    rows = [r for r in rows if r.get("dataset", DEFAULT_DATASET) == dataset]
    return rows[-1]["corpus_hash"] if rows else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="config name in configs/, or a path to a .toml")
    parser.add_argument("--multi-variable", action="store_true",
                        help="allow a config that changes more than one setting")
    parser.add_argument("--any-corpus", action="store_true",
                        help="record the run even if the corpus differs from earlier runs")
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
    documents = load_corpus(corpus_path)
    digest = corpus_hash(documents)
    earlier = previous_corpus_hash(dataset)
    if earlier and earlier != digest and not args.any_corpus:
        print(
            f"\nRefusing to run: the {dataset} corpus has changed since the last\n"
            f"recorded {dataset} run ({earlier} -> {digest}), so the numbers would\n"
            f"not be comparable.\n"
            f"Pass --any-corpus to record it anyway.",
            file=sys.stderr,
        )
        return 1

    chunks = chunk_corpus(
        documents, config["chunking"]["strategy"],
        size=config["chunking"]["size"], overlap=config["chunking"]["overlap"],
    )
    strategy = config["retrieval"]["strategy"]
    # Which strategies actually embed the corpus. Hybrid does - it builds a
    # dense half - so it must record the model it used, the same as dense.
    embeds = strategy in {"dense", "hybrid"}
    index = build_index(chunks, strategy=strategy, model=config["embedding"]["model"])
    questions = load_questions(eval_file)
    k = config["retrieval"]["k"]
    # What this run is deep enough to measure. Everything below reports these
    # and nothing else, so a shallow run never claims a cutoff it cannot see.
    cutoffs = cutoffs_for(k)

    # Second-stage reranking. Imported lazily, and only when enabled, so a run
    # without it never loads the cross-encoder or downloads its weights.
    reranking = config["retrieval"]["rerank"]
    if reranking:
        from rag.reranking import (RERANK_DEPTH, RERANK_MODEL, SELECTION_FLOOR,
                                   rerank, rerank_subquery_aware)

    # Deterministic clause splitting - no model, no API, same queries every run.
    multi_query = config["retrieval"]["multi_query"]
    # Inert without multi-query: with one query there is nothing to cover.
    subquery_rerank = config["retrieval"]["subquery_rerank"] and reranking
    if multi_query:
        from rag.multiquery import MAX_SUBQUERIES, merge_round_robin, queries_for

    generating = config["generation"]["enabled"]
    judging = config["judging"]["enabled"] and generating
    if generating:
        from rag.generation import NO_ANSWER, generate_answer
    if judging:
        from rag.judging import judge_correctness, judge_groundedness

    print(f"\ndataset {dataset} | corpus {digest} | {len(documents)} documents "
          f"-> {len(chunks):,} chunks")
    print(f"retrieval {strategy} | k={k} | cutoffs {', '.join(f'@{c}' for c in cutoffs)}")
    if reranking:
        print(f"reranking {RERANK_MODEL} over {RERANK_DEPTH} candidates")
    if multi_query:
        print(f"multi-query on: up to {MAX_SUBQUERIES} deterministic subqueries, "
              f"candidate budget shared round-robin")
    if subquery_rerank:
        print(f"subquery-aware selection: original keeps {SELECTION_FLOOR} of {k} slots, "
              f"rest round-robin by per-query rank")
    print(f"generation {'on' if generating else 'off'}, judging {'on' if judging else 'off'}")

    # ---- run --------------------------------------------------------------
    records = []
    doc_ranks, passage_ranks = [], []
    # One list of per-group ranks per answerable question, for the evidence
    # metrics. Unanswerable questions have no required facts and are excluded.
    evidence_ranks = []
    doc_ranks_by_type, passage_ranks_by_type = defaultdict(list), defaultdict(list)
    correctness, groundedness = defaultdict(list), defaultdict(list)
    refusals = Counter()

    for number, question in enumerate(questions, start=1):
        if generating:
            print(f"  [{number}/{len(questions)}] {question['qid']}", end="", flush=True)

        unanswerable = question["type"] == "unanswerable"
        # The candidate budget is fixed. With multi-query it is shared out
        # across the queries rather than spent entirely on the question as
        # written, so the original query's deeper candidates are displaced.
        queries = queries_for(question["question"], multi_query) if multi_query \
            else [question["question"]]
        sources = None
        selection = None
        if reranking:
            if len(queries) > 1:
                runs = [index.search(q, k=RERANK_DEPTH) for q in queries]
                candidates, sources = merge_round_robin(runs, RERANK_DEPTH)
            else:
                candidates = index.search(queries[0], k=RERANK_DEPTH)
            if subquery_rerank and len(queries) > 1:
                # Scored against every query; selected by rank, never by
                # blending logits across queries.
                retrieved, selection = rerank_subquery_aware(queries, candidates, k=k)
            else:
                # Always rescored against the question as asked, never a fragment.
                retrieved = rerank(question["question"], candidates, k=k)
        elif len(queries) > 1:
            runs = [index.search(q, k=k) for q in queries]
            candidates, sources = merge_round_robin(runs, k)
            retrieved = candidates
        else:
            candidates = None
            retrieved = index.search(queries[0], k=k)
        doc_rank, passage_rank = first_hit_ranks(question, retrieved)
        groups = group_ranks(question, retrieved)

        record = {"qid": question["qid"], "type": question["type"],
                  "doc_rank": doc_rank, "passage_rank": passage_rank,
                  "groups": len(groups), "group_ranks": groups,
                  "complete_rank": complete_rank(groups)}

        if reranking:
            # Where the gold sat before reranking. Without this a miss is
            # ambiguous between "the candidate set never held it" and "the
            # reranker held it and failed to promote it" - two failures with
            # completely different fixes.
            candidate_doc_rank, candidate_passage_rank = first_hit_ranks(
                question, candidates)
            record.update(candidate_doc_rank=candidate_doc_rank,
                          candidate_passage_rank=candidate_passage_rank,
                          candidates=len(candidates))

        if multi_query:
            # The queries actually issued, and which of them contributed each
            # chunk that survived into the final list - without these a miss
            # cannot be told apart from a bad split.
            record.update(
                queries=queries,
                query_sources=[sources.get(r.chunk.chunk_id) for r in retrieved]
                if sources else [0] * len(retrieved))
            if selection:
                # Per-query rank and score for each surviving chunk, plus which
                # query's slot it took - the only way a null result here can be
                # told apart from the mechanism not firing.
                record["selection"] = [selection[r.chunk.chunk_id] for r in retrieved]

        if not unanswerable:
            doc_ranks.append(doc_rank)
            passage_ranks.append(passage_rank)
            evidence_ranks.append(groups)
            doc_ranks_by_type[question["type"]].append(doc_rank)
            passage_ranks_by_type[question["type"]].append(passage_rank)

        if generating:
            stage("generating")
            started = perf_counter()
            answer = generate_answer(question["question"], retrieved,
                                     model=config["generation"]["model"])
            took(started)
            # Exact match, deliberately. A model that opens with the refusal
            # sentence and then keeps talking is counted as an answer, not a
            # refusal - so this can under-report refusals (it did once, on
            # Space q030). Loosening it would reclassify answers in runs
            # already recorded, so it is documented in DESIGN.md rather than
            # changed. Correctness and groundedness still judge the text.
            refused = answer.strip() == NO_ANSWER
            record.update(answer=answer, refused=refused)

            if unanswerable:
                refusals["unanswerable_correct" if refused else "unanswerable_missed"] += 1
            elif refused:
                refusals["answerable"] += 1
                correctness[question["type"]].append(0)
                record.update(correctness=0, groundedness=None)
            elif judging:
                stage("correctness")
                started = perf_counter()
                c = judge_correctness(question["question"], question["answer"], answer,
                                      judge=config["judging"]["model"])
                took(started)
                stage("groundedness")
                started = perf_counter()
                g = judge_groundedness(question["question"], retrieved, answer,
                                       judge=config["judging"]["model"])
                took(started)
                correctness[question["type"]].append(c.score)
                groundedness[question["type"]].append(g.score)
                record.update(correctness=c.score, correctness_reasoning=c.reasoning,
                              groundedness=g.score, groundedness_reasoning=g.reasoning)

        if generating:
            print(flush=True)

        records.append(record)

    # ---- metrics ----------------------------------------------------------
    doc_recall, doc_mrr = summarize(doc_ranks, cutoffs, mrr_cutoff=MRR_CUTOFF)
    passage_recall, _ = summarize(passage_ranks, cutoffs, mrr_cutoff=MRR_CUTOFF)
    coverage, complete = evidence_metrics(evidence_ranks, cutoffs)
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
        "dataset": dataset,
        "corpus_hash": digest,
        "documents": len(documents),
        "chunks": len(chunks),
        "questions": {"total": len(questions), "answerable": len(doc_ranks)},
        "retrieval": {
            "cutoffs": list(cutoffs),
            # Fusion settings are module constants rather than config keys, so
            # record them here - otherwise a hybrid run's saved artifacts would
            # not say what fusion produced them.
            **({"rrf_k": RRF_K, "fusion_depth": FUSION_DEPTH}
               if strategy == "hybrid" else {}),
            **({"rerank_model": RERANK_MODEL, "rerank_depth": RERANK_DEPTH}
               if reranking else {}),
            **({"max_subqueries": MAX_SUBQUERIES} if multi_query else {}),
            **({"selection_floor": SELECTION_FLOOR} if subquery_rerank else {}),
            "document_recall": {str(c): doc_recall[c] for c in cutoffs},
            "document_mrr": doc_mrr,
            "mrr_cutoff": MRR_CUTOFF,
            "passage_hit": {str(c): passage_recall[c] for c in cutoffs},
            # Some of the required evidence vs all of it. Equal to passage_hit
            # on every question that needs only one fact.
            "evidence_coverage": {str(c): coverage[c] for c in cutoffs},
            "complete_evidence": {str(c): complete[c] for c in cutoffs},
            "evidence_groups": sum(len(r) for r in evidence_ranks),
            # by_type reports a single cutoff rather than the whole curve, so
            # name it - otherwise the same key means @5 in one run and @10 in
            # the next, with nothing saying which.
            "by_type_cutoff": max(cutoffs),
            "by_type": {
                t: {"document_recall":
                        summarize(doc_ranks_by_type[t], cutoffs)[0][max(cutoffs)],
                    "document_mrr":
                        summarize(doc_ranks_by_type[t], cutoffs, mrr_cutoff=MRR_CUTOFF)[1],
                    "passage_hit":
                        summarize(passage_ranks_by_type[t], cutoffs)[0][max(cutoffs)]}
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
    # New run IDs carry the dataset, since one config now runs against more
    # than one. Existing directories keep the names they were written with.
    run_id = f"{dataset}_{config['name']}_{stamp}"
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
        "run_id": run_id, "dataset": dataset,
        "config": config["name"], "extends": config.get("extends", ""),
        "corpus_hash": digest, "chunks": len(chunks),
        "size": config["chunking"]["size"], "overlap": config["chunking"]["overlap"],
        "k": k, "strategy": strategy,
        # Lowercase to match how the config spells it, and so the column reads
        # the same as the TOML that produced it.
        "rerank": str(reranking).lower(),
        "multi_query": str(multi_query).lower(),
        "subquery_rerank": str(subquery_rerank).lower(),
        # Blank for bm25 only: the config still carries an inherited embedding
        # model, but nothing embedded anything, and recording it would claim
        # otherwise. Dense and hybrid both embed, so both record it.
        "embed_model": config["embedding"]["model"] if embeds else "",
        "gen_model": config["generation"]["model"] if generating else "",
        "judge_model": config["judging"]["model"] if judging else "",
        # .get so a cutoff this run was too shallow to measure stays blank
        # rather than raising, and blank rather than 0 - a k=5 run did not
        # measure @10, which is different from measuring it and finding nothing.
        **{f"recall@{c}": doc_recall.get(c, "") for c in CUTOFF_LADDER},
        "mrr": doc_mrr,
        **{f"hit@{c}": passage_recall.get(c, "") for c in CUTOFF_LADDER},
        **{f"cov@{c}": coverage.get(c, "") for c in CUTOFF_LADDER},
        **{f"comp@{c}": complete.get(c, "") for c in CUTOFF_LADDER},
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
    for c in cutoffs:
        print(f"  Recall@{c}{'':<3}{doc_recall[c]:.3f}      Hit@{c}{'':<3}{passage_recall[c]:.3f}"
              f"      Cov@{c}{'':<3}{coverage[c]:.3f}      Complete@{c}{'':<3}{complete[c]:.3f}")
    print(f"  MRR@{MRR_CUTOFF}     {doc_mrr:.3f}")

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
