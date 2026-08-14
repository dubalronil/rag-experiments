# Design

## What RAG does

A language model can answer from its training knowledge, but retrieval-augmented generation (RAG) supplies relevant external information at query time. The question is turned into a vector, the most similar passages in a corpus are retrieved by comparing vectors, and those passages are handed to the model as the source it should answer from.

The model's job becomes reading rather than recalling — which makes an answer traceable to a source, and makes _"the context doesn't say"_ a correct answer when the corpus is silent.

Two things can go wrong, and they are different failures. **Retrieval** can miss the passage that holds the answer. Or retrieval can succeed and the **answer** can still be wrong — incomplete, incorrect, or asserting something the passage never said. A system that reports one number cannot tell you which happened.

## This pipeline

```text
question ─┐
          ▼
  corpus ─→ chunks ─→ vectors ─→ top-k passages ─→ answer ─→ two scores
            (chunking)  (MiniLM)    (exact search)   (Haiku)   (Sonnet)
```

Ten Wikipedia articles on NBA history are cut into fixed-length character slices, embedded locally with `sentence-transformers/all-MiniLM-L6-v2`, and searched. The top five passages go to Claude Haiku 4.5, which is told to answer only from that context and to return one fixed sentence when the context is insufficient. Claude Sonnet 5 then grades the answer.

Each stage depends only on the one before it, so chunking, embeddings, retrieval, generation, or judging can be changed independently.

## Making experiments controlled

The corpus and the 50-question evaluation set are **fixed**. Changing either changes what the metrics mean, so every run records a hash of the corpus it measured.

Eight questions are deliberately unanswerable. The corpus does not contain their answers, so the test is whether the model refuses instead of inventing something.

Gold supporting evidence is stored as **verbatim quotes rather than chunk IDs**. Chunk IDs change when chunk size changes, while the supporting sentence remains valid across chunking configurations.

An experiment is a TOML config that names a parent and lists only its overrides. **The runner refuses to start if more than one experimental setting differs**, unless that is explicitly allowed. This keeps each result attributable to one change.

## Evaluating retrieval and answers separately

Retrieval is measured at two levels. **Document recall** asks whether a retrieved chunk came from the correct document. **Passage hit rate** asks whether a retrieved chunk actually contained the gold supporting sentence.

Answers are judged on two independent dimensions. **Correctness** asks whether the generated answer matches the reference answer. **Groundedness** asks whether every claim in the generated answer is supported by the retrieved passages.

These are separate judge calls with different inputs: the correctness judge sees the reference answer but not the passages, while the groundedness judge sees the passages but not the reference.

## Key decisions

**Search is exact.** Every chunk is scored for every query. Approximate vector search can trade some recall for speed at large scale; with only a few thousand chunks, exact NumPy search is fast enough and easier to reason about.

**Embeddings run locally on a pinned model** to make experiments reproducible and avoid API dependence for retrieval.

**Generation is constrained to retrieved context** so failures in retrieval remain visible instead of being hidden by the model's outside knowledge.

**The judge is a different model from the generator**, reducing the risk of a model evaluating its own output too generously.
