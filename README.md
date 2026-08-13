# rag-experiments

Learning RAG by comparing chunking, embeddings, retrieval, reranking, and generation through controlled experiments against a fixed evaluation set. No frameworks — every piece is written out explicitly.

**Status:** data plus loading and chunking; no retrieval yet. 10 NBA history Wikipedia articles (~97,000 words) in `data/documents/` and 50 evaluation questions in `data/eval/questions.jsonl`; `rag/` turns those files into `Document` and `Chunk` objects. Rebuild the corpus with `python3 scripts/fetch_corpus.py`, check the questions with `python3 scripts/validate_eval.py`, run tests with `python3 -m unittest discover tests`. Standard library only, no dependencies.

**Gold labels are verbatim quotes, not chunk IDs** — so the same labels stay valid no matter how the corpus is chunked. 16% of questions are deliberately unanswerable, to test whether generation admits it rather than confabulating.

**Caveat:** the source API strips tables, so facts that lived in them — per-season stats, championship lists, coaching rosters — are not in the corpus. Questions are written against the prose.

Code is MIT ([`LICENSE`](LICENSE)). Everything under `data/` is Wikipedia-derived and licensed CC BY-SA 4.0, not MIT — see [`data/ATTRIBUTION.md`](data/ATTRIBUTION.md).
