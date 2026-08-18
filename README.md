# rag-experiments

A RAG pipeline built from scratch — no frameworks — to measure how each design
choice actually affects answer quality. Every stage is swappable, every run is
scored against a fixed corpus and a fixed 50-question evaluation set, and the
runner refuses to start if a config changes more than one variable at a time.

Twenty controlled experiments later, it runs against **two independent corpora**:
NBA history and spaceflight. The second exists to check that a result is a
property of the pipeline rather than of the corpus it was tuned on.

## Final architecture

512-character chunks with 256 overlap → `BAAI/bge-small-en-v1.5` embeddings →
exact dense search for 50 candidates → `cross-encoder/ms-marco-MiniLM-L-6-v2`
reranks them to the top 5 → Claude Haiku 4.5 answers from those 5 passages →
Claude Sonnet 5 grades correctness and groundedness.

| Config `bge-rerank`, 42 answerable + 8 unanswerable |   NBA | Space |
| --------------------------------------------------- | ----: | ----: |
| Document recall@5                                   | 1.000 | 1.000 |
| MRR@5                                               | 0.944 | 0.976 |
| Passage hit rate@5                                  | 0.857 | 0.976 |
| Evidence coverage@5                                 | 0.845 | 0.917 |
| Complete evidence@5                                 | 0.833 | 0.857 |
| Correctness (0–2)                                   |  1.93 |  1.86 |
| Groundedness (0–2)                                  |  2.00 |  1.90 |
| Refused when unanswerable                           | 8 / 8 | 8 / 8 |
| Refused when answerable                             |     0 |     2 |

### What moved the needle

Each step below changed exactly one setting from the row above it, on NBA:

| Change                            | Passage hit@5 | Correctness |
| --------------------------------- | ------------: | ----------: |
| Baseline — 512/128 chunks, MiniLM |         0.690 |        1.55 |
| Chunk overlap 128 → 256           |         0.738 |        1.60 |
| MiniLM → BGE-small                |         0.810 |        1.79 |
| **+ cross-encoder reranking**     |     **0.881** |    **1.95** |

And what did not move it: 256-character chunks (correctness 1.43, worse), BM25
instead of dense search (hit@5 0.619), hybrid BM25 + dense fusion (0.786),
`text-embedding-3-large` over the API (0.762), k=10 instead of 5, deterministic
multi-query splitting, subquery-aware reranking, and — measured last — swapping
Haiku 4.5 for Opus 5, which scored correctness 1.857 against Haiku's 1.857.

_(The progression above reads runs as they were recorded. The final NBA figures
in the first table are a rerun under the current evidence labels and the
corpus-neutral prompt, which is why the same config reads 0.857 / 1.93 there and
0.881 / 1.95 here. `EXPERIMENTS.md` explains both changes.)_

## What we learned

**The biggest gains came from retrieval — especially cross-encoder reranking —
not from using a larger generator.** Reranking a 50-candidate pool down to 5
produced the largest single improvement and replicated on the second corpus.
With retrieval held fixed, Opus 5 returned the same correctness as Haiku 4.5 to
three decimals at substantially higher cost, so increasing generator size did
not improve the final system. The remaining failures were associated mainly with
missing or incomplete evidence in the final context.

**Simple passage-hit metrics can hide partial multi-hop failures, so evaluation
design mattered almost as much as pipeline design.** Passage hit rate asks only
whether any gold supporting sentence was retrieved. For a question requiring
multiple facts, retrieving one required fact can therefore look like success
even when the context is incomplete. Restructuring gold evidence into one group
per required fact made that visible: on Space, the final system reached hit@5 of
0.976 but complete evidence@5 of 0.857. Five multi-hop questions still had only
partial evidence in the top 5, and both multi-query retrieval and subquery-aware
reranking failed to close that gap.

[`EXPERIMENTS.md`](EXPERIMENTS.md#main-findings) lists the full set of findings,
including the ones that came out negative.

## Quick start

Python 3.11+.

```bash
pip install -r requirements.txt                     # ~1–2 GB, pulls in PyTorch

python3 scripts/run_experiment.py retrieval-only    # free, no credentials
python3 scripts/run_experiment.py bge-rerank        # graded, ~$0.25
python3 scripts/run_experiment.py bge-rerank --dataset space
```

Retrieval runs entirely locally — no key, no network after the first model
download. Generation and judging need `ANTHROPIC_API_KEY`, exported or placed in
`.env` (see `.env.example`); `OPENAI_API_KEY` is needed only for the two OpenAI
embedding configs. `--dataset` selects the corpus and its evaluation set.

## Repository layout

```text
rag/            the pipeline, one module per stage
  chunking, embedding, retrieval, reranking, multiquery,
  generation, judging, corpus, datasets, experiment, text, types
scripts/        run_experiment (the entry point), fetch_corpus,
                validate_eval, and the standalone eval scripts
configs/        one TOML per experiment; each extends a parent
data/<dataset>/ articles.txt, documents/, eval/questions.jsonl
tests/          187 tests, standard library unittest
```

`rag/` has no framework dependency: chunking and corpus loading are standard
library, embedding is `sentence-transformers` plus NumPy, and search is exact
NumPy over a dense matrix.

## Experiments

A config names a parent and lists only what it changes:

```toml
extends = "bge-small"
name = "bge-rerank"

[retrieval]
rerank = true
```

**The runner refuses to start if more than one setting differs from the parent**,
unless `--multi-variable` is passed. It also refuses if the corpus hash has
changed since the last recorded run of that dataset, because a metric computed
against a different corpus is not comparable with the ones already recorded.

[`EXPERIMENTS.md`](EXPERIMENTS.md) is the log: each experiment's hypothesis, the
one variable it moved, and what happened — including the several that changed
nothing. Every number quoted in it comes from a recorded run.

Runs write their merged config, metrics, and per-question records to a local
`results/` directory, which is gitignored. Those artifacts are working files —
they cost API calls to produce and go stale as soon as a corpus or an eval set
moves — so the repository publishes the findings rather than the raw output.
Re-running any config regenerates them. [`DESIGN.md`](DESIGN.md#what-a-run-records-and-where-it-lives)
covers what a run records and why.

## What is measured

Retrieval and answering fail differently, so they are scored separately.

- **Document recall@k** — did a retrieved chunk come from the right document?
- **MRR@5** — how highly the first correct document was ranked.
- **Passage hit rate@k** — did a retrieved chunk contain a gold supporting sentence?
- **Evidence coverage@k / complete evidence@k** — gold evidence is stored as one
  group per required fact. Coverage is the fraction of required facts retrieved;
  complete evidence is the fraction of questions where _all_ of them were. On a
  multi-hop question these separate "found half the evidence" from "found none",
  which passage hit rate scores identically.
- **Correctness / groundedness** — judged 0–2 by Sonnet 5, in separate calls with
  different inputs: the correctness judge sees the reference answer but not the
  passages, the groundedness judge sees the passages but not the reference.

Eight of the 50 questions in each set are deliberately unanswerable. The corpus
does not contain their answers, so the test is whether the model returns its
fixed refusal sentence instead of inventing something. The generation prompt
names no corpus — it says the model answers from the provided context and
nothing about what that context is about — so the same prompt is valid for every
dataset, and tests pin it that way.

## The datasets

| Dataset | Subject                   | Documents |          Questions | Chunks at 512/256 |
| ------- | ------------------------- | --------: | -----------------: | ----------------: |
| `nba`   | NBA history               |        10 | 50 (42 answerable) |             2,255 |
| `space` | Spaceflight and astronomy |        10 | 50 (42 answerable) |             2,358 |

Both are Wikipedia prose fetched at a pinned revision. Gold evidence is stored as
**verbatim quotes rather than chunk IDs**, so an evaluation set stays valid when
chunk size changes.

Adding a dataset is `data/<name>/articles.txt` plus a fetch — no code change:

```bash
python3 scripts/fetch_corpus.py --dataset <name>
python3 scripts/validate_eval.py --dataset <name>
```

## Tests and validators

```bash
python3 -m unittest discover tests        # 187 tests, no credentials, no network
python3 scripts/validate_eval.py --dataset nba
python3 scripts/validate_eval.py --dataset space
```

The validator re-checks every gold quote verbatim against the corpus and exits
non-zero on any mismatch, so an evaluation set cannot silently drift from the
documents it was written against. The tests cover chunking, metrics, evidence
groups, BM25, hybrid fusion, reranking, multi-query, and the generation
request — including tests that pin the prompt as corpus-neutral and Haiku's
request parameters byte-for-byte, so a past run stays comparable with a new one.

## Caveats

Each corpus is fixed on purpose — re-fetching changes what every metric means.
Tables are stripped by the source API, so per-season statistics and championship
lists are not in the NBA corpus; questions are written against the prose.

42 answerable questions means one question moves a 0–2 mean by about 0.05. Small
differences between runs are noise, and generation is not perfectly repeatable
even at temperature 0.

Known limitations — including the refusal detector's exact-match rule, which can
under-report refusals — are in [`DESIGN.md`](DESIGN.md#known-limitations).

[`DESIGN.md`](DESIGN.md) covers the architecture and the reasoning behind it.

## License

Code is MIT ([`LICENSE`](LICENSE)). Everything under `data/` is Wikipedia-derived
and CC BY-SA 4.0 — see [`data/ATTRIBUTION.md`](data/ATTRIBUTION.md).
