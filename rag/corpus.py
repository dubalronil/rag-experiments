"""Load a dataset's Markdown corpus into Document objects.

The files were produced by scripts/fetch_corpus.py and all have the same shape:
a front matter block delimited by "---" lines, then the article body.

    ---
    title: "Chicago Bulls"
    source_url: https://en.wikipedia.org/wiki/Chicago_Bulls
    revision_id: 1366322186
    revision_timestamp: 2026-06-24T02:13:03Z
    retrieved: 2026-08-12
    ---

    # Chicago Bulls

    The Chicago Bulls are an American professional basketball team...

The front matter is a handful of "key: value" lines, so it is parsed here
directly rather than by pulling in a YAML dependency.
"""

import re
from pathlib import Path

from rag.datasets import DEFAULT_DATASET, corpus_dir

from rag.types import Document

DEFAULT_CORPUS_DIR = corpus_dir(DEFAULT_DATASET)

# Front matter is the block between the first two "---" lines, which must be at
# the very start of the file. Group 1 is the block's contents.
FRONT_MATTER_RE = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)

# Fields a Document needs. Anything else in the front matter is ignored.
REQUIRED_KEYS = ("title", "source_url", "revision_id")


def parse_front_matter(raw):
    """Split a file's contents into (front matter dict, body text).

    Raises ValueError if the file has no front matter block.
    """
    match = FRONT_MATTER_RE.match(raw)
    if not match:
        raise ValueError("no front matter block found")

    fields = {}
    for line in match.group(1).split("\n"):
        if not line.strip():
            continue
        # Split on the first colon only. Values contain colons of their own -
        # "https://..." and the timestamps both would break a naive split.
        key, separator, value = line.partition(":")
        if not separator:
            raise ValueError(f"front matter line is not 'key: value': {line!r}")
        # The fetcher quotes the title because titles can contain colons.
        fields[key.strip()] = value.strip().strip('"')

    # Everything after the closing "---" is the document body. Leading blank
    # lines are dropped so that offset 0 is the first real character.
    body = raw[match.end():].lstrip("\n")
    return fields, body


def load_document(path):
    """Read one Markdown file into a Document.

    Raises ValueError, naming the file, if it is malformed.
    """
    raw = Path(path).read_text(encoding="utf-8")

    try:
        fields, body = parse_front_matter(raw)
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from error

    missing = [key for key in REQUIRED_KEYS if key not in fields]
    if missing:
        raise ValueError(f"{path}: front matter missing {', '.join(missing)}")

    # doc_id comes from the filename, not from the front matter title. The
    # evaluation set's gold labels reference these filenames, so the id has to
    # stay put even if an article gets renamed upstream.
    return Document(
        doc_id=Path(path).stem,
        title=fields["title"],
        source_url=fields["source_url"],
        revision_id=fields["revision_id"],
        text=body,
    )


def load_corpus(corpus_dir=DEFAULT_CORPUS_DIR):
    """Read every .md file in a directory into a list of Documents.

    Sorted by doc_id, so two runs over the same corpus see the documents in the
    same order - directory listing order is not guaranteed, and an experiment
    that silently reorders its inputs is not a controlled one.
    """
    corpus_dir = Path(corpus_dir)
    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"no corpus directory at {corpus_dir}")

    documents = [load_document(path) for path in sorted(corpus_dir.glob("*.md"))]
    if not documents:
        raise FileNotFoundError(f"no .md files in {corpus_dir}")
    return documents
