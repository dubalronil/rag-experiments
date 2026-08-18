# Design

## What RAG does

A language model can answer from its training knowledge, but retrieval-augmented generation (RAG) supplies relevant external information at query time. The question is turned into a vector, the most similar passages in a corpus are retrieved by comparing vectors, and those passages are handed to the model as the source it should answer from.

The model's job becomes reading rather than recalling — which makes an answer traceable to a source, and makes _"the context doesn't say"_ a correct answer when the corpus is silent.

Two things can go wrong, and they are different failures. **Retrieval** can miss the passage that holds the answer. Or retrieval can succeed and the **answer** can still be wrong — incomplete, incorrect, or asserting something the passage never said. A system that reports one number cannot tell you which happened.

## This pipeline

```text
question ─┐
          ▼
  corpus ─→ chunks ─→ vectors ─→ 50 candidates ─→ rerank ─→ top 5 ─→ answer ─→ two scores
            (512/256)  (BGE-small)  (exact search)  (cross-encoder)   (Haiku)   (Sonnet)
```

Ten Wikipedia articles are cut into 512-character slices overlapping by 256, embedded locally with `BAAI/bge-small-en-v1.5`, and searched exactly. Search returns **50 candidates**, which `cross-encoder/ms-marco-MiniLM-L-6-v2` rescores against the question as asked; the **top 5** become the generation context. Claude Haiku 4.5 answers from those five passages and nothing else, returning one fixed sentence when they are insufficient. Claude Sonnet 5 then grades the answer.

That is the final configuration, `configs/bge-rerank.toml`. The code's defaults are the original baseline — 512/128 chunks, MiniLM, no reranking — which is what a bare `baseline` run produces; every setting above is a change some experiment made to it and kept.

Each stage depends only on the one before it, so chunking, embeddings, retrieval, reranking, generation, or judging can be changed independently.

### Why two stages of retrieval

Embedding search is cheap because it compares vectors that were computed without ever seeing the question. A cross-encoder reads the question and the passage together, which is far more accurate and far too slow to run over a whole corpus. Running the cheap one for recall and the expensive one for precision gets both: the bi-encoder only has to place the right passage somewhere in 50, and the reranker only has to order 50.

This was the single largest improvement measured here (Experiments 16 and 17 in [`EXPERIMENTS.md`](EXPERIMENTS.md), replicated on the second corpus). The candidate pool is deliberately much deeper than the five passages that survive it — a reranker cannot promote what search never returned, and 50 is where recall@50 was effectively saturated on both corpora.

## Two corpora

The pipeline runs against two datasets, `nba` and `space`, each a corpus of ten articles plus a 50-question evaluation set written against it. Each lives under `data/<dataset>/`, and every run records which one it used.

A dataset is deliberately **not** a config setting. Chunk size or embedding model are things an experiment varies to attribute a change; the corpus is the ground the comparison stands on. Running the same config against a second corpus is a replication, not an experiment — which is exactly what the second dataset is for.

## Making experiments controlled

Each corpus and its 50-question evaluation set are **fixed**. Changing either changes what the metrics mean, so every run records a hash of the corpus it measured and the runner refuses to record a run whose corpus differs from the last one for that dataset.

Eight questions per set are deliberately unanswerable. The corpus does not contain their answers, so the test is whether the model refuses instead of inventing something.

Gold supporting evidence is stored as **verbatim quotes rather than chunk IDs**. Chunk IDs change when chunk size changes, while the supporting sentence remains valid across chunking configurations.

An experiment is a TOML config that names a parent and lists only its overrides. **The runner refuses to start if more than one experimental setting differs**, unless that is explicitly allowed. This keeps each result attributable to one change.

### What a run records, and where it lives

A run writes three files to a local `results/runs/<run id>/`, and appends one row to `results/summary.csv`:

- `config.toml` — the **merged** config, not a reference to the config file. Editing `configs/bge-rerank.toml` later must not change what a past run meant.
- `metrics.json` — every metric, plus the corpus hash, dataset, chunk count, and the module constants that no config mentions (rerank depth, fusion parameters), so a number is never left depending on a value the run did not write down.
- `records.jsonl` — one row per question: the ranks, the answer, and the judge's reasoning. This is what makes a bad number diagnosable rather than merely visible.

**`results/` is gitignored.** These are working files: they cost API calls to produce, they go stale the moment a corpus or an eval set moves, and they are regenerated by re-running a config. What a reader needs is the finding and the reasoning behind it, and those live in [`EXPERIMENTS.md`](EXPERIMENTS.md) — every number quoted there came from one of these runs. Publishing the artifacts as well would mean maintaining two records of the same thing, and the raw one is the one that silently rots.

The consequence is deliberate and worth stating: a claim in `EXPERIMENTS.md` cannot be re-derived from this repository alone, only reproduced by re-running the config. Reproduction is the stronger check anyway — it exercises the pipeline rather than re-reading a file it wrote.

## Evaluating retrieval and answers separately

Retrieval is measured at two levels. **Document recall** asks whether a retrieved chunk came from the correct document. **Passage hit rate** asks whether a retrieved chunk actually contained a gold supporting sentence.

### Evidence groups

Passage hit rate is a single bit, and that is not enough for a question needing several facts. Gold evidence is therefore stored as **groups**: one group per required fact, each holding alternative quotes that would each prove it. A single sentence may appear in several groups when it proves several facts.

Two metrics follow from the grouping, both reported at each cutoff:

- **Evidence coverage@k** — the fraction of required facts whose evidence was retrieved.
- **Complete evidence@k** — the fraction of questions where *every* required fact was retrieved.

They are equal to passage hit rate on any question needing one fact, and they diverge exactly where multi-hop questions fail — which is what makes "retrieved half the evidence" visible instead of scoring the same as "retrieved none of it".

Answers are judged on two independent dimensions. **Correctness** asks whether the generated answer matches the reference answer. **Groundedness** asks whether every claim in the generated answer is supported by the retrieved passages. These are separate judge calls with different inputs: the correctness judge sees the reference answer but not the passages, while the groundedness judge sees the passages but not the reference.

## Key decisions

**Search is exact.** Every chunk is scored for every query. Approximate vector search can trade some recall for speed at large scale; with only a few thousand chunks, exact NumPy search is fast enough and easier to reason about.

**Embeddings run locally on a pinned model** to make experiments reproducible and avoid API dependence for retrieval.

**Generation is constrained to retrieved context** so failures in retrieval remain visible instead of being hidden by the model's outside knowledge.

**The generation prompt names no corpus.** It says the model answers using only the provided context and nothing about what that context is about. An earlier version opened "You answer questions about NBA history"; once a second corpus existed, that prompt told the model the subject was basketball while handing it astronomy passages, and a capable model reads the contradiction and refuses. A prompt that names a domain measures compliance with the prompt rather than the pipeline, so the prompt is corpus-neutral by design and pinned by tests.

**The judge is a different model from the generator**, reducing the risk of a model evaluating its own output too generously.

**The generator is Haiku 4.5, and that is a measured choice.** Opus 5 on identical retrieval, context, prompt and judge scored the same correctness to three decimals at roughly twice the cost of the same run. The remaining errors are all evidence that never reached the context — a missing passage, or one of two required facts — and a stronger generator cannot answer from a passage it was not given.

## Known limitations

**The refusal detector matches the refusal sentence exactly.** An answer counts as a refusal only when it equals `NO_ANSWER` after stripping whitespace. A model that opens with that sentence and then keeps talking is scored as an answer, not a refusal — which happened once in the recorded runs, on Space `q030`, where the answer began with the refusal sentence and continued with an explanation. Correctness and groundedness still judge that text on its merits, so no headline number is wrong, but the "answerable refusals" count can be under-reported by such cases. It is documented rather than fixed because loosening the match would silently reclassify answers in runs already recorded, and every comparison in `EXPERIMENTS.md` was made under the exact-match rule.

**Generation is not perfectly repeatable.** Temperature is pinned to 0 where the model accepts it, and disabled thinking is requested where it applies, but identical requests can still produce different text. With 42 answerable questions, one question moves a 0–2 mean by about 0.05; differences smaller than that are not evidence of anything.

**The judge is a model.** Correctness and groundedness are Sonnet 5's readings of an answer, not ground truth. They are stable enough to rank configurations but should not be read as exact.

**Tables are absent from both corpora.** The MediaWiki extracts API strips them, so statistics that live in tables — season records, instrument specifications — cannot be retrieved regardless of the pipeline. Questions are written against the prose, and a handful of headings survive with no body text beneath them.
