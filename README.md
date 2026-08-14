# rag-experiments

A RAG pipeline built from scratch — no frameworks — to measure how each design choice actually affects answer quality. Every stage is swappable, every run is scored against a fixed corpus and a fixed 50-question evaluation set, and the runner refuses to start if a config changes more than one variable at a time.

## Result

512-character chunks, 128 overlap, `all-MiniLM-L6-v2`, k=5, 42 answerable questions.

| Retrieval | @1 | @3 | @5 |
| --- | ---: | ---: | ---: |
| Document recall | 0.714 | 0.905 | 0.929 |
| Passage hit rate | 0.333 | 0.548 | 0.690 |

| Answers (Haiku 4.5, judged 0–2 by Sonnet 5) | |
| --- | ---: |
| Correctness | 1.52 |
| Groundedness | 1.88 |
| Refused correctly when unanswerable | 8 / 8 |

**Retrieval is the bottleneck, not the model.** Correctness is 1.93 when the gold passage is retrieved and 0.62 when it isn't — and every zero is a refusal, not a wrong answer.

## Run it

```bash
pip install -r requirements.txt                    # ~1–2 GB, pulls in PyTorch
python3 scripts/run_experiment.py retrieval-only   # free, no credentials
```

Generation and judging need `ANTHROPIC_API_KEY` (see `.env.example`); a full graded run is ~$0.25.

## Experiments

A config names a parent and lists only what it changes:

```toml
extends = "baseline"
name = "chunk-256"

[chunking]
size = 256
```

**The runner refuses to start if more than one setting differs from the parent**, unless `--multi-variable` is passed. Each run saves its merged config, metrics, and per-question records to `results/`.

## Caveats

The corpus is fixed on purpose — re-fetching it changes what every metric means. Tables are stripped by the source API, so per-season stats and championship lists aren't in it; questions are written against the prose.

[`DESIGN.md`](DESIGN.md) covers the architecture and the reasoning behind it.

## License

Code is MIT ([`LICENSE`](LICENSE)). Everything under `data/` is Wikipedia-derived and CC BY-SA 4.0 — see [`data/ATTRIBUTION.md`](data/ATTRIBUTION.md).
