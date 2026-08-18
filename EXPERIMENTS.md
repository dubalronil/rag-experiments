# Experiments

This file records the controlled experiments used to build and evaluate this RAG system.

Each numbered experiment changes **one major variable** from a parent configuration — and that is enforced, not just intended: the runner refuses to start when a config differs from its parent in more than one setting, and refuses to record a run whose corpus hash has changed since the last run of that dataset. Unless otherwise noted, experiments 1–17 use the NBA dataset. Later experiments add a second Space dataset to test whether the conclusions generalize.

[`README.md`](README.md) summarises where this ended up; [`DESIGN.md`](DESIGN.md) explains the architecture, the reasoning behind each stage, and the known limitations of the measurements below.

```bash
python3 scripts/run_experiment.py <config>
```

Each run writes its merged config, metrics, and per-question records to a local `results/` directory. That directory is gitignored, so the raw run artifacts are not part of the public repository. The metrics below were copied from those runs.

## Evaluation

Each dataset contains 10 Wikipedia articles and 50 evaluation questions:

- 42 answerable questions
- 8 unanswerable questions

The main metrics are:

- **Recall@K** — whether a retrieved chunk comes from a gold source document.
- **MRR@5** — how highly the first gold document is ranked.
- **Hit@K** — whether any labeled supporting quote appears in the top K chunks.
- **Evidence Coverage@K** — the average fraction of required evidence groups retrieved.
- **Complete Evidence@K** — the fraction of questions for which every required evidence group is retrieved.
- **Correctness** — answer quality, scored 0–2 by a separate Sonnet 5 judge.
- **Groundedness** — whether the answer is supported by the retrieved context, scored 0–2.
- **Unanswerable refused** — whether the generator returned the required refusal when the corpus did not contain the answer.

`Evidence Coverage@K` and `Complete Evidence@K` were added later, after the Space dataset exposed a weakness in the original flat supporting-quote schema. Historical retrieval metrics are preserved; the final cross-dataset comparison uses the current evidence-group evaluation.

---

# Baseline

The initial system used:

- 512-character chunks
- 128-character overlap
- `sentence-transformers/all-MiniLM-L6-v2`
- dense cosine retrieval
- top 5 chunks
- `claude-haiku-4-5` generation
- `claude-sonnet-5` judging

**Config:** `baseline.toml`

| Metric | Result |
| --- | ---: |
| Recall@1 | 0.714 |
| Recall@3 | 0.905 |
| Recall@5 | 0.929 |
| MRR@5 | 0.812 |
| Hit@1 | 0.333 |
| Hit@3 | 0.548 |
| Hit@5 | 0.690 |
| Correctness | 1.55 / 2 |
| Groundedness | 1.82 / 2 |
| Correctness when gold passage retrieved | 1.97 / 2 |
| Correctness when gold passage missed | 0.62 / 2 |
| Unanswerable refused | 8 / 8 |

Approximate API cost: **~$0.25**

---

# Chunking

The first experiment family tested whether chunk size and overlap changed retrieval quality.

## Experiment 1 — Remove overlap

**Hypothesis:** Removing overlap will hurt passage retrieval because evidence near chunk boundaries can be split.

**Change:** overlap `128 → 0`

**Config:** `overlap-0-retrieval.toml`

| Recall@1 | Recall@3 | Recall@5 | MRR@5 | Hit@1 | Hit@3 | Hit@5 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.786 | 0.952 | 0.952 | 0.861 | 0.238 | 0.500 | 0.524 |

Cost: **$0 — local retrieval only**

## Experiment 2 — 64-character overlap

**Hypothesis:** A smaller overlap may preserve boundary evidence while reducing repeated text.

**Change:** overlap `128 → 64`

**Config:** `overlap-64-retrieval.toml`

| Recall@1 | Recall@3 | Recall@5 | MRR@5 | Hit@1 | Hit@3 | Hit@5 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.690 | 0.929 | 0.976 | 0.804 | 0.214 | 0.500 | 0.643 |

Cost: **$0 — local retrieval only**

## Experiment 3 — 256-character chunks

**Hypothesis:** Smaller chunks may rank more precisely because each embedding represents a narrower idea.

**Change:** chunk size `512 → 256`

**Config:** `chunk-256-retrieval.toml`

| Recall@1 | Recall@3 | Recall@5 | MRR@5 | Hit@1 | Hit@3 | Hit@5 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.833 | 0.905 | 0.929 | 0.875 | 0.333 | 0.524 | 0.667 |

Cost: **$0 — local retrieval only**

## Experiment 4 — 768-character chunks

**Hypothesis:** Larger chunks preserve more surrounding context but may make embeddings less precise.

**Change:** chunk size `512 → 768`

**Config:** `chunk-768-retrieval.toml`

| Recall@1 | Recall@3 | Recall@5 | MRR@5 | Hit@1 | Hit@3 | Hit@5 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.738 | 0.881 | 0.929 | 0.812 | 0.238 | 0.476 | 0.571 |

Cost: **$0 — local retrieval only**

## Experiment 5 — 256-character chunks, full pipeline

**Hypothesis:** Better document ranking from smaller chunks may not translate into better answers if each retrieved chunk carries less context.

**Change:** chunk size `512 → 256`

**Config:** `chunk-256.toml`

| Metric | Result |
| --- | ---: |
| Recall@1 / @3 / @5 | 0.833 / 0.905 / 0.929 |
| MRR@5 | 0.875 |
| Hit@1 / @3 / @5 | 0.333 / 0.524 / 0.667 |
| Correctness | 1.43 / 2 |
| Groundedness | 1.90 / 2 |
| Correctness when gold passage retrieved | 1.86 / 2 |
| Correctness when gold passage missed | 0.57 / 2 |
| Unanswerable refused | 7 / 8 |

Approximate API cost: **~$0.18**

The smaller chunks improved some ranking metrics but reduced final-answer quality, so 512-character chunks remained the better tradeoff.

## Experiment 6 — 256-character overlap

**Hypothesis:** More overlap may preserve supporting evidence that crosses chunk boundaries.

**Change:** overlap `128 → 256`

**Config:** `overlap-256-retrieval.toml`

| Recall@1 | Recall@3 | Recall@5 | MRR@5 | Hit@1 | Hit@3 | Hit@5 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.738 | 0.929 | 0.952 | 0.822 | 0.286 | 0.571 | 0.738 |

Cost: **$0 — local retrieval only**

## Experiment 7 — 256-character overlap, full pipeline

**Hypothesis:** Higher passage retrieval from more overlap may improve final answers.

**Change:** overlap `128 → 256`

**Config:** `overlap-256.toml`

| Metric | Result |
| --- | ---: |
| Recall@1 / @3 / @5 | 0.738 / 0.929 / 0.952 |
| MRR@5 | 0.822 |
| Hit@1 / @3 / @5 | 0.286 / 0.571 / 0.738 |
| Correctness | 1.60 / 2 |
| Groundedness | 2.00 / 2 |
| Correctness when gold passage retrieved | 1.94 / 2 |
| Correctness when gold passage missed | 0.64 / 2 |
| Unanswerable refused | 8 / 8 |

Approximate API cost: **~$0.24**

This became the chunking setup used by the later retrieval experiments: **512-character chunks with 256-character overlap**.

---

# Embedding models

All embedding comparisons below use the 512/256 chunking setup and dense top-5 retrieval.

## Experiment 8 — BGE-small

**Hypothesis:** A retrieval-focused embedding model will retrieve supporting passages more reliably than MiniLM.

**Change:** `sentence-transformers/all-MiniLM-L6-v2 → BAAI/bge-small-en-v1.5`

**Config:** `bge-small-retrieval.toml`

| Recall@1 | Recall@3 | Recall@5 | MRR@5 | Hit@1 | Hit@3 | Hit@5 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.786 | 0.905 | 0.952 | 0.853 | 0.476 | 0.714 | 0.810 |

Cost: **$0 — local retrieval only**

## Experiment 9 — BGE-small, full pipeline

**Hypothesis:** BGE-small's stronger passage retrieval will improve final answer quality.

**Change:** embedding model only

**Config:** `bge-small.toml`

| Metric | Result |
| --- | ---: |
| Recall@1 / @3 / @5 | 0.786 / 0.905 / 0.952 |
| MRR@5 | 0.853 |
| Hit@1 / @3 / @5 | 0.476 / 0.714 / 0.810 |
| Correctness | 1.79 / 2 |
| Groundedness | 1.87 / 2 |
| Correctness when gold passage retrieved | 1.94 / 2 |
| Correctness when gold passage missed | 1.12 / 2 |
| Unanswerable refused | 8 / 8 |

Approximate API cost: **~$0.26**

## Experiment 10 — OpenAI `text-embedding-3-small`

**Hypothesis:** OpenAI's small embedding model may improve retrieval over the local models.

**Change:** `BAAI/bge-small-en-v1.5 → text-embedding-3-small`

**Config:** `openai-small-retrieval.toml`

| Recall@1 | Recall@3 | Recall@5 | MRR@5 | Hit@1 | Hit@3 | Hit@5 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.762 | 0.952 | 0.952 | 0.845 | 0.310 | 0.548 | 0.690 |

Cost: **OpenAI embeddings only; exact spend not recorded**

## Experiment 11 — OpenAI `text-embedding-3-large`

**Hypothesis:** A larger embedding model may improve retrieval further.

**Change:** `BAAI/bge-small-en-v1.5 → text-embedding-3-large`

**Config:** `openai-large-retrieval.toml`

| Recall@1 | Recall@3 | Recall@5 | MRR@5 | Hit@1 | Hit@3 | Hit@5 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.833 | 0.929 | 0.952 | 0.883 | 0.452 | 0.667 | 0.762 |

Cost: **OpenAI embeddings only; exact spend not recorded**

BGE-small had the strongest passage-level retrieval of the tested embedding models and remained the default.

---

# Retrieval strategies

## Experiment 12 — BM25

**Hypothesis:** BM25 may outperform dense retrieval when the question and supporting passage share strong lexical cues.

**Change:** retrieval strategy `dense → bm25`

**Config:** `bm25-retrieval.toml`

| Recall@1 | Recall@3 | Recall@5 | MRR@5 | Hit@1 | Hit@3 | Hit@5 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.762 | 0.833 | 0.952 | 0.821 | 0.429 | 0.524 | 0.619 |

Cost: **$0 — local retrieval only**

## Experiment 13 — BM25, full pipeline

**Hypothesis:** If dense BGE retrieves supporting passages more often, it should also produce better final answers than BM25.

**Change:** retrieval strategy `dense → bm25`

**Config:** `bm25.toml`

| Metric | Result |
| --- | ---: |
| Recall@1 / @3 / @5 | 0.762 / 0.833 / 0.952 |
| MRR@5 | 0.821 |
| Hit@1 / @3 / @5 | 0.429 / 0.524 / 0.619 |
| Correctness | 1.67 / 2 |
| Groundedness | 1.92 / 2 |
| Correctness when gold passage retrieved | 1.92 / 2 |
| Correctness when gold passage missed | 1.25 / 2 |
| Unanswerable refused | 8 / 8 |

Approximate API cost: **~$0.25**

BM25 found several lexical matches that BGE missed, but BGE remained stronger overall.

## Experiment 14 — Retrieval depth k=10

**Hypothesis:** Increasing retrieval depth may recover supporting passages just outside the top 5.

**Change:** `k=5 → k=10`

**Config:** `bge-k10-retrieval.toml`

| Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR@5 | Hit@1 | Hit@3 | Hit@5 | Hit@10 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.786 | 0.905 | 0.952 | 1.000 | 0.853 | 0.476 | 0.714 | 0.810 | 0.810 |

Cost: **$0 — local retrieval only**

The correct document always appeared by rank 10, but passage retrieval did not improve beyond rank 5. Simply giving the generator more context was therefore not justified.

## Experiment 15 — Hybrid dense + BM25

**Hypothesis:** Combining dense and lexical retrieval may recover complementary evidence.

**Change:** retrieval strategy `dense → hybrid`

**Config:** `hybrid-retrieval.toml`

| Recall@1 | Recall@3 | Recall@5 | MRR@5 | Hit@1 | Hit@3 | Hit@5 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.786 | 0.952 | 0.976 | 0.871 | 0.405 | 0.643 | 0.786 |

Cost: **$0 — local retrieval only**

Hybrid retrieval improved document-level ranking but slightly reduced passage retrieval compared with dense BGE.

---

# Reranking

## Experiment 16 — Cross-encoder reranking

**Hypothesis:** BGE often retrieves the correct passage somewhere in a larger candidate set but ranks it below the top 5. A cross-encoder should improve the final ordering.

**Change:** reranking `false → true`

**Config:** `bge-rerank-retrieval.toml`

BGE retrieves 50 candidates. `cross-encoder/ms-marco-MiniLM-L-6-v2` then scores each question–chunk pair and returns the final top 5.

| Recall@1 | Recall@3 | Recall@5 | MRR@5 | Hit@1 | Hit@3 | Hit@5 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.929 | 1.000 | 1.000 | 0.960 | 0.667 | 0.810 | 0.881 |

Cost: **$0 — local retrieval and reranking**

## Experiment 17 — Cross-encoder reranking, full pipeline

**Hypothesis:** Better passage ranking should translate into better final answers.

**Change:** reranking `false → true`

**Config:** `bge-rerank.toml`

| Metric | Result |
| --- | ---: |
| Recall@1 / @3 / @5 | 0.929 / 1.000 / 1.000 |
| MRR@5 | 0.960 |
| Hit@1 / @3 / @5 | 0.667 / 0.810 / 0.881 |
| Correctness | 1.95 / 2 |
| Groundedness | 1.98 / 2 |
| Correctness when gold passage retrieved | 2.00 / 2 |
| Correctness when gold passage missed | 1.60 / 2 |
| Unanswerable refused | 8 / 8 |

Reranking produced the strongest NBA end-to-end result of the initial experiment sequence.

---

# Cross-dataset replication

A second dataset was added to test whether the retrieval improvements generalized beyond the NBA benchmark.

The Space dataset contains 10 space-exploration articles and the same 50-question evaluation structure.

The same three configurations were run unchanged:

1. baseline
2. BGE-small
3. BGE + reranker

An initial set of Space generation runs used a system prompt that still said the assistant answered questions about NBA history. Retrieval metrics from those runs remain valid because retrieval never sees the generation prompt, but their answer metrics were superseded after the prompt was corrected to be corpus-neutral.

The corrected Space runs are:

| Space config | Recall@1 | Recall@5 | MRR@5 | Hit@1 | Hit@5 | Coverage@5 | Complete@5 | Correctness | Groundedness | Answerable refusals | Unanswerable refused |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.881 | 1.000 | 0.940 | 0.381 | 0.810 | 0.750 | 0.690 | 1.69 | 1.95 | 4 | 8 / 8 |
| BGE-small | 0.929 | 1.000 | 0.957 | 0.643 | 0.857 | 0.798 | 0.738 | 1.69 | 1.89 | 5 | 8 / 8 |
| BGE + reranker | 0.952 | 1.000 | 0.976 | 0.714 | 0.976 | 0.917 | 0.857 | 1.86 | 1.90 | 2 | 8 / 8 |

(Recall@3 was 1.000 for baseline and BGE + reranker and 0.976 for BGE-small;
Hit@3 was 0.619, 0.786 and 0.976. Coverage and Complete Evidence are computed
under the evidence-group schema described above.)

The retrieval improvements generalized: BGE improved passage retrieval over MiniLM, and cross-encoder reranking improved it again.

The Space dataset also exposed a new failure mode. Several genuinely multi-hop questions retrieved only one of the required pieces of evidence even when `Hit@5` reported success. That led to the evidence-group evaluation described above.

---

# Multi-hop retrieval

With the evidence-group schema, Space BGE + reranker retrieved complete evidence for only **4 of 9** genuine multi-hop questions. The other 5 retrieved partial evidence.

## Experiment 18 — Multi-query retrieval

**Hypothesis:** A single embedding of a two-part question may blend both information needs. Searching clause-level subqueries separately may surface the missing evidence.

**Change:** `retrieval.multi_query: false → true`

**Config:** `multiquery-retrieval.toml --dataset space`

The splitter is deterministic and uses no LLM. The original question and its clause-level subqueries share the same fixed 50-candidate budget through round-robin merging.

| Metric | BGE + reranker | Multi-query |
| --- | ---: | ---: |
| Recall@5 | 1.000 | 1.000 |
| MRR@5 | 0.976 | 0.976 |
| Hit@5 | 0.976 | 0.976 |
| Evidence Coverage@5 | 0.917 | 0.917 |
| Complete Evidence@5 | 0.857 | 0.857 |

Cost: **$0 — local retrieval only**

**Result:** no evidence-level improvement. Multi-hop remained **4 complete / 5 partial**.

The mechanism was not a no-op: subqueries changed the candidate pool and contributed chunks to the final top 5. The evidence outcome simply did not improve.

## Experiment 19 — Subquery-aware reranking

**Hypothesis:** Some chunks answer one part of a multi-hop question well but score poorly against the full question. Giving each subquery a reserved final slot may preserve those specialist chunks.

**Change:** `retrieval.subquery_rerank: false → true`

**Config:** `subquery-rerank-retrieval.toml --dataset space`

The candidate pool is unchanged. The same cross-encoder scores the pool separately against the original question and each subquery. Final selection uses per-query rankings rather than blending raw logits:

- 3 slots from the original-question ranking
- 1 slot from subquery 1
- 1 slot from subquery 2

| Metric | BGE + reranker | Multi-query | Subquery-aware rerank |
| --- | ---: | ---: | ---: |
| Recall@5 | 1.000 | 1.000 | 1.000 |
| MRR@5 | 0.976 | 0.976 | 0.976 |
| Hit@5 | 0.976 | 0.976 | 0.976 |
| Evidence Coverage@5 | 0.917 | 0.917 | 0.917 |
| Complete Evidence@5 | 0.857 | 0.857 | 0.857 |

Cost: **$0 — local retrieval and reranking**

**Result:** again, no question changed from partial to complete.

The allocation worked exactly as designed rather than quietly failing to fire:
`selected_by` was `[0, 0, 0, 1, 2]` on all eight splittable questions, so the
original question kept its three slots and no subquery ever exceeded its one.
Two questions shifted internally without losing completeness — q033's second
evidence group moved from rank 4 to 5, q034's first from 5 to 4. The mechanism
ran; the evidence outcome did not move.

### What Experiments 18 and 19 established

The candidate pool changed substantially under multi-query retrieval, and the final selection policy changed again under subquery-aware reranking, but evidence coverage stayed identical.

Tracing the 5 partial Space multi-hop questions showed:

- **0 / 5** missing facts were completely unreachable by retrieval.
- **1 / 5** was retrieved by a subquery but lost during the fixed 50-candidate merge.
- **4 / 5** entered the candidate pool but were not selected into the final top 5 by the cross-encoder.

So the dominant remaining failure was not initial retrieval. It was the ranking/selection of specialist evidence, with one additional candidate-budget failure.

At this point the multi-hop retrieval branch was stopped rather than tuned further against nine questions.

---

# Generation prompt and model comparison

## Prompt bug discovered

The original generation system prompt began with:

> You answer questions about NBA history using only the context provided in the user's message.

That was appropriate when the project had one NBA corpus, but it became incorrect after adding Space.

Haiku mostly ignored the domain mismatch. Opus 5 followed it much more literally: an initial Space Opus run refused 27 of 42 answerable questions, including questions whose gold evidence was ranked first.

That run (`space_bge-rerank-opus_20260818T002942Z`) was invalidated because it
measured prompt compliance, not generator capability. Its row was removed from
the run summary and its artifacts moved to a local `results/invalid/` directory
with a note recording why — deleting them would have lost the evidence for the
diagnosis.

The prompt was changed to the corpus-neutral:

> You answer questions using only the context provided in the user's message.

Every other generation rule and the exact refusal string remained unchanged.
`tests/test_generation_prompt.py` pins both halves of that: the prompt must name
no domain, and each surviving rule — including the refusal sentence the scorer
matches verbatim — must still be present word for word.

The three Space Haiku configurations were then rerun under the corrected prompt. Their retrieval metrics were unchanged, while their corrected answer metrics are the ones shown in the cross-dataset table above.

## Experiment 20 — Opus 5 vs Haiku 4.5

**Hypothesis:** With retrieval fixed, a stronger generator may use the same retrieved context more effectively.

**Change:** `generation.model: claude-haiku-4-5 → claude-opus-5`

**Config:** `bge-rerank-opus.toml --dataset space`

Everything else remained fixed:

- BGE-small retrieval
- cross-encoder reranking
- 50 candidates
- final k=5
- same neutral system prompt
- same Space eval
- same Sonnet 5 judge

| Metric | Haiku 4.5 | Opus 5 |
| --- | ---: | ---: |
| Correctness | 1.857 | 1.857 |
| Groundedness | 1.900 | 1.923 |
| Answerable refusals | 2 | 3 |
| Unanswerable refused | 8 / 8 | 8 / 8 |
| Correctness when gold passage retrieved | 1.90 | 1.90 |
| Factual correctness | 1.92 | 1.92 |
| Multi-hop correctness | 1.56 | 1.56 |
| Temporal correctness | 2.00 | 2.00 |

**Result:** no correctness improvement.

The remaining failures were not solved by switching generators:

- q010 lacked the required Voyager power-source passage in the final context.
- q030 and q031 were multi-hop questions with incomplete/partial evidence in the final context.
- Opus refused q030 while Haiku produced an incorrect answer, but both received correctness 0.

The 2-vs-3 refusal difference is thinner than it looks. Haiku's q030 answer *opened* with the refusal sentence and then kept going with an explanation, and refusals are detected by exact match, so it was scored as an answer rather than a refusal. Under a looser rule both generators would read as refusing all three. See [`DESIGN.md`](DESIGN.md#known-limitations) for why the detector was left as it is.

Haiku therefore remained the final generator: it matched Opus correctness at substantially lower cost.

Approximate Opus experiment cost: **~$0.60**

---

# Final architecture

The final system is:

```text
documents
→ fixed-size chunking (512 chars, 256 overlap)
→ BAAI/bge-small-en-v1.5
→ dense top-50 candidate retrieval
→ cross-encoder/ms-marco-MiniLM-L-6-v2 reranking
→ final top 5 chunks
→ Claude Haiku 4.5
→ Claude Sonnet 5 evaluation
```

Generation uses the same corpus-neutral system prompt on every dataset.

The final architecture was run unchanged on both corpora:

| Metric | NBA | Space |
| --- | ---: | ---: |
| Chunks | 2,255 | 2,358 |
| Recall@1 | 0.905 | 0.952 |
| Recall@3 | 1.000 | 1.000 |
| Recall@5 | 1.000 | 1.000 |
| MRR@5 | 0.944 | 0.976 |
| Hit@1 | 0.643 | 0.714 |
| Hit@3 | 0.786 | 0.976 |
| Hit@5 | 0.857 | 0.976 |
| Evidence Coverage@5 | 0.845 | 0.917 |
| Complete Evidence@5 | 0.833 | 0.857 |
| Correctness | 1.93 / 2 | 1.86 / 2 |
| Groundedness | 2.00 / 2 | 1.90 / 2 |
| Answerable refusals | 0 | 2 |
| Unanswerable refused | 8 / 8 | 8 / 8 |

The final NBA retrieval numbers differ slightly from the earlier Experiment 17 row (Recall@1 0.929 → 0.905, Hit@1 0.667 → 0.643) because the evidence-group migration corrected the gold quote sets on six NBA questions after that historical run. Exactly one question's rank moved as a result: q029. The final table uses the current evaluation schema; Experiment 17's row is left as it was recorded.

---

# Main findings

1. **Chunking is a tradeoff, not a single retrieval score.** Smaller chunks improved some document-ranking metrics but hurt final answer quality. More overlap improved passage retrieval.

2. **Embedding choice mattered substantially.** BGE-small retrieved supporting passages more reliably than MiniLM, OpenAI `text-embedding-3-small`, and OpenAI `text-embedding-3-large` on this benchmark.

3. **Document recall and passage retrieval are different problems.** Several configurations found the correct document almost every time while still missing the exact evidence needed by the generator.

4. **BM25 and dense retrieval have complementary strengths, but simple hybrid fusion was not automatically better.**

5. **Cross-encoder reranking was the strongest retrieval improvement.** It improved both retrieval metrics and end-to-end correctness, and the improvement replicated on a second corpus.

6. **Evaluation design matters.** A flat supporting-quote metric hid partial multi-hop failures and produced false negatives when equivalent evidence appeared elsewhere. Evidence groups made those cases measurable.

7. **Multi-hop questions need complete evidence, not just a retrieval hit.** On Space, the best retriever often found one required fact while missing another.

8. **More retrieval complexity did not automatically help.** Multi-query retrieval and subquery-aware reranking changed the internals of the pipeline without improving evidence coverage.

9. **A stronger generator did not improve correctness once retrieval was fixed.** Opus 5 matched Haiku 4.5 on the same retrieved context, so Haiku remained the better cost/performance choice for this system.

10. **Prompt correctness is part of the system.** A corpus-specific system prompt silently became wrong when a second dataset was added; a stronger model exposed the bug by following it more literally.

The project stops here deliberately. The goal was not to maximize a small benchmark indefinitely, but to understand which RAG design choices generalized, which failed, and how to measure the difference.
