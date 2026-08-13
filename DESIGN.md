# Design

How this project fits together today: a fixed corpus and four stages — load,
chunk, embed, search, generate — plus a script that scores retrieval against
the fixed evaluation set. Reranking is not built, and neither is any scoring
of generated answers, so those are not described here.

The guiding constraint is that experiments must be **controlled**: exactly one
thing changes per run, everything else stays identical.

## The pipeline

```
  data/documents/*.md          10 Markdown files, fixed
          │
          │  rag/corpus.py      read the file, split off the front matter
          ▼
      Document                  one article
          │
          │  rag/chunking.py    slice the text into fixed-length pieces
          ▼
        Chunk                   one piece of an article
          │
          │  rag/embedding.py   all-MiniLM-L6-v2, runs locally on CPU
          ▼
   numpy array                  (n_chunks, 384) numbers, one row per chunk
          │
          │  rag/retrieval.py   score a query against every chunk
          ▼
  RetrievedChunk list           the top k matches, best first
          │
          │  rag/generation.py   Claude Haiku 4.5, answering from that context
          ▼
       answer text               one or two sentences, or an explicit refusal
```

Each stage depends only on the stage before it and on the shared types in
`rag/types.py` — never on another stage's internals. That is what lets one
stage be swapped in an experiment without touching the rest.

## The corpus

Ten Wikipedia articles on NBA history — the league, the ABA, the Finals, four
franchises, two players — about 97,000 words, downloaded once by
`scripts/fetch_corpus.py` and committed to the repository.

The corpus is **fixed on purpose**: changing it changes what every metric
means, so results from either side of a change are not comparable. Each file
records the Wikipedia revision it came from, so drift is detectable.

Each file has a front matter block, then the article:

```markdown
---
title: "Chicago Bulls"
source_url: https://en.wikipedia.org/wiki/Chicago_Bulls
revision_id: 1366322186
---

# Chicago Bulls

The Chicago Bulls are an American professional basketball team...
```

**Known limitation:** the source API strips tables along with citations and
infoboxes, so about a quarter of the section headings survive with little or
no body (`Season-by-season record`, `Career statistics`). The prose is intact;
tabular data is not in the corpus at all.

## The two data types

Both live in `rag/types.py` and both are frozen — once loaded, nothing can
modify them, so a result always describes the input it claims to.

**`Document`** — one article: `doc_id` (the filename stem, and the stable
identity everything else refers to), `title`, `source_url`, `revision_id`, and
`text` (the body, without the front matter).

**`Chunk`** — a slice of one document: `chunk_id` (`chicago_bulls::0007`),
`doc_id`, `text` (what gets embedded), and `start`/`end` character offsets
into the parent document.

The offsets let you tell whether a chunker split a sentence across a
boundary, which otherwise looks identical to search ranking it too low. And
chunk IDs shift whenever the chunk size changes, which is why the evaluation
set identifies answers by quoting text rather than by chunk ID.

## Stage 1 — Loading (`rag/corpus.py`)

`load_corpus()` reads every `.md` file in `data/documents/` into `Document`
objects, sorted by `doc_id` so two runs see the same order. Front matter is a
handful of `key: value` lines, parsed directly rather than with a YAML
dependency. Malformed files raise an error naming the file.

## Stage 2 — Chunking (`rag/chunking.py`)

`chunk_fixed_size(document, size, overlap)` cuts a document into fixed-length
character slices. Each chunk starts `size - overlap` characters after the last
one, so `overlap=0` gives adjacent slices and `overlap=128` means each chunk
repeats the previous chunk's last 128 characters.

Overlap exists so a sentence landing on a boundary still appears intact
somewhere. It costs duplication — 512/128 produces 33% more chunks than 512/0
— which is why it is worth measuring rather than assuming.

Sizes are in **characters**, not tokens. That keeps this stage
dependency-free; the mismatch with the model's token limit is handled at the
embedding stage.

`STRATEGIES` is a plain dict mapping a config name to a function, so reading
it tells you every option there is.

## Stage 3 — Embedding (`rag/embedding.py`)

`embed_documents(texts)` turns text into numbers, returning one row per input
in the same order — that ordering is the only thing tying a vector back to its
chunk. The full corpus (1,508 chunks at 512/128) takes about 8.5 seconds and
2.3 MB. Vectors are normalized, which makes comparing two of them a single
multiplication later.

The model is **`all-MiniLM-L6-v2`**, run locally on CPU. Not for cost — an
API would charge under a cent for this corpus — but because a pinned model
returns identical vectors indefinitely, and a public repository is a poor
place for an API key. The price is roughly 1–2 GB of dependencies, since
`sentence-transformers` pulls in PyTorch.

Documents and queries have **separate functions**. For this model they behave
identically, but some models expect queries to carry a prefix that documents
do not, so keeping the paths separate makes swapping models a config change.

**The one thing to watch: MiniLM reads at most 256 tokens.** Longer text is
embedded up to that point and the rest silently dropped — no error, and the
vector looks normal. The ceiling lands between 512 and 1024 characters per
chunk. Without a warning a chunk-size sweep would show quality dropping and
look like a fact about chunking when it is really a fact about the model, so
every call checks and warns.

## Stage 4 — Search (`rag/retrieval.py`)

`VectorIndex.build(chunks)` embeds the chunks and holds them alongside their
vectors. Row *i* is the embedding of chunk *i* — that pairing is the whole
data structure, and the constructor refuses an index where the two do not line
up. `index.search(query, k)` returns the *k* best matches as `RetrievedChunk`
objects, each carrying the chunk, its score, and its rank (starting at 1,
because reciprocal rank is 1/rank).

The search is **exact** — every chunk is scored, nothing approximated. Because
all vectors are normalized, that is one matrix multiply: about **0.06 ms**
across 1,508 chunks, against ~7 ms to embed the query. Vector databases trade
accuracy for speed at millions of vectors; at this size there is nothing to
trade away. Indexes are **not written to disk** — rebuilding takes about eight
seconds, and caching would mostly add a way to reuse vectors built from a
different chunk size by mistake.

## Stage 5 — Generation (`rag/generation.py`)

`generate_answer(question, retrieved)` renders the retrieved chunks into a
numbered context block, sends it with the question, and returns the answer.
The model is **Claude Haiku 4.5** at `temperature=0`, which the current
generation of models no longer accepts but Haiku still does.

Beyond answering, the prompt restricts the model to the supplied context and
tells it to reply with one exact sentence when that context is insufficient.
That sentence is a module constant, so a scorer can recognise a refusal
without pattern-matching prose — and it is what makes the evaluation set's
unanswerable questions measurable.

This is the only stage needing credentials, costing money, or being
non-deterministic. Credentials are left to the SDK, so a `.env` file and a
logged-in CLI profile both work.

## File map

| Path | Purpose |
| --- | --- |
| `data/documents/*.md` | The fixed corpus. |
| `data/ATTRIBUTION.md` | Sources and the CC BY-SA terms the corpus carries. |
| `rag/types.py` | `Document`, `Chunk`, `RetrievedChunk`. Shapes only, no behaviour. |
| `rag/corpus.py` | Markdown files → `Document`. |
| `rag/chunking.py` | `Document` → list of `Chunk`. |
| `rag/embedding.py` | Text → vectors. |
| `rag/generation.py` | Question + retrieved chunks → answer. The only stage needing an API key. |
| `rag/retrieval.py` | `VectorIndex`: holds chunks + vectors, exact top-k search. |
| `rag/text.py` | The one definition of `normalize()`, shared by anything comparing gold quotes to text. |
| `scripts/fetch_corpus.py` | Rebuilds the corpus from Wikipedia. Run rarely — re-fetching changes it. |
| `scripts/run_retrieval_eval.py` | Scores the retriever against the evaluation set. |
| `tests/test_chunking.py` | 12 tests on the chunker, whose bugs are the quietest in the pipeline. |
| `.env.example` | Template for the one credential the project uses. Copy to `.env`, which is gitignored. |
| `requirements.txt` | Needed by the embedding and generation stages only. |

## Not built yet

No reranking, no scoring of generated answers, no experiment configs, and no
result storage. Indexes are built in memory per run and scores are printed
rather than saved, so comparing two configurations means reading two terminal
outputs.
