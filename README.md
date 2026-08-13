# rag-experiments

Learning RAG by comparing chunking, embeddings, retrieval, reranking, and generation through controlled experiments against a fixed evaluation set. No frameworks — every piece is written out explicitly.

**Status:** corpus only so far. 10 NBA history Wikipedia articles (~97,000 words) in `data/documents/`, rebuilt with `python3 scripts/fetch_corpus.py` (standard library only, no dependencies).

**Caveat:** the source API strips tables, so facts that lived in them — per-season stats, championship lists, coaching rosters — are not in the corpus. Write evaluation questions against the prose.

Code is MIT ([`LICENSE`](LICENSE)). The corpus is Wikipedia-derived and licensed CC BY-SA 4.0, not MIT — see [`data/ATTRIBUTION.md`](data/ATTRIBUTION.md).
