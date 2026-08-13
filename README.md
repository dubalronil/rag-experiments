# rag-experiments

Learning RAG by comparing chunking, embeddings, retrieval, reranking, and generation through controlled experiments against a fixed evaluation set. No frameworks — every piece is written out explicitly.

**Status:** corpus, chunking, embeddings, search, retrieval scoring, and generation. Answer scoring is not built yet. See [`DESIGN.md`](DESIGN.md) for how the pieces fit together.

## Setup

```bash
pip install -r requirements.txt        # roughly 1–2 GB, pulls in PyTorch
```

Everything except generation runs with no credentials. For generation:

```bash
cp .env.example .env                   # then paste your key into it
```

or `export ANTHROPIC_API_KEY=...`. A shell export takes precedence over `.env`, and `.env` is gitignored.

## Running

```bash
python3 scripts/run_retrieval_eval.py            # score retrieval, prints the table below
python3 scripts/run_retrieval_eval.py --size 256 --overlap 64
python3 scripts/validate_eval.py                 # check gold quotes still match the corpus
python3 scripts/fetch_corpus.py                  # rebuild the corpus (changes it — see below)
python3 -m unittest discover tests
```

Generation is a library call rather than a script so far — `rag.generation.generate_answer(question, retrieved_chunks)`.

## Baseline

10 documents, 1,508 chunks at size 512 / overlap 128, `all-MiniLM-L6-v2`, 42 answerable questions:

| | @1 | @3 | @5 |
| --- | ---: | ---: | ---: |
| **Document recall** — a retrieved chunk came from a gold document | 0.714 | 0.905 | 0.929 |
| **Passage hit rate** — a retrieved chunk contained the gold quote | 0.333 | 0.548 | 0.690 |

Document MRR is 0.812. The gap between the two rows is the interesting part: for roughly a quarter of questions, search finds the right article and misses the right sentence.

## Notes

**Gold labels are verbatim quotes, not chunk IDs** — so the same labels stay valid no matter how the corpus is chunked. 16% of questions are deliberately unanswerable, to test whether generation admits it rather than confabulating; those are excluded from retrieval metrics, where they have nothing correct to retrieve.

**The corpus is fixed on purpose.** Re-running `fetch_corpus.py` pulls newer Wikipedia revisions and changes what every metric means, so results from either side of a re-fetch are not comparable.

**Tables are missing from the corpus.** The source API strips them, so per-season stats, championship lists, and coaching rosters are not present. Questions are written against the prose.

## License

Code is MIT ([`LICENSE`](LICENSE)). Everything under `data/` is Wikipedia-derived and licensed CC BY-SA 4.0, not MIT — see [`data/ATTRIBUTION.md`](data/ATTRIBUTION.md).
