"""Where each dataset's corpus and evaluation set live.

A dataset is a corpus plus the evaluation set written against it. The two are
only meaningful together - a gold quote is verbatim text from one particular
corpus - so they are named once, here, rather than as two paths every script
has to be handed consistently.

    data/<name>/articles.txt          the source list the corpus was fetched from
    data/<name>/documents/*.md        the corpus
    data/<name>/eval/questions.jsonl  the evaluation set

A dataset is deliberately *not* a config setting. Chunk size, embedding model
and retrieval strategy are things an experiment varies to attribute a change;
the dataset is the ground the comparison stands on, like the corpus hash. A
config differing only in dataset would not be a controlled experiment, it would
be a replication - so the same config file runs against either dataset, and
which one it ran against is recorded in the results rather than in the config.
"""

from pathlib import Path

DATA_ROOT = Path("data")

DEFAULT_DATASET = "nba"

# The label used when a caller points at a corpus or eval set by hand instead
# of naming a dataset. Such a run must not be filed under the named dataset:
# the corpus guard compares a run against earlier runs of the same dataset, and
# an ad-hoc corpus recorded as "nba" would either trip that guard for real nba
# runs or, worse, let a mismatched corpus pass as one.
CUSTOM_DATASET = "custom"


def dataset_dir(name):
    return DATA_ROOT / name


def corpus_dir(name):
    return dataset_dir(name) / "documents"


def eval_path(name):
    return dataset_dir(name) / "eval" / "questions.jsonl"


def articles_path(name):
    """The list of source article titles the corpus was built from.

    Kept as data rather than as a list inside the fetcher, so adding a dataset
    does not mean editing a script, and so each corpus carries its own
    provenance next to the documents it produced.
    """
    return dataset_dir(name) / "articles.txt"


def available():
    """Dataset names that have a corpus directory, sorted."""
    if not DATA_ROOT.is_dir():
        return []
    return sorted(p.name for p in DATA_ROOT.iterdir() if corpus_dir(p.name).is_dir())


def read_articles(name):
    """Article titles for a dataset, one per line.

    Blank lines and lines starting with '#' are ignored, so the file can carry
    comments about why a particular article is in the corpus.
    """
    path = articles_path(name)
    if not path.exists():
        raise FileNotFoundError(f"no article list at {path}")

    titles = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            titles.append(line)
    if not titles:
        raise ValueError(f"{path} lists no articles")
    return titles


def resolve(name=DEFAULT_DATASET, corpus=None, eval_file=None):
    """Return (label, corpus directory, eval path) for one run.

    `corpus` and `eval_file` are explicit overrides. Passing either means the
    run is no longer the named dataset, whatever --dataset said, so the label
    becomes CUSTOM_DATASET. The label is what gets recorded and what the corpus
    guard groups by; the paths are what actually get read.
    """
    if name not in available() and corpus is None and eval_file is None:
        known = ", ".join(available()) or "none found"
        raise FileNotFoundError(
            f"unknown dataset {name!r} - no corpus at {corpus_dir(name)} "
            f"(available: {known})"
        )

    overridden = corpus is not None or eval_file is not None
    return (
        CUSTOM_DATASET if overridden else name,
        Path(corpus) if corpus is not None else corpus_dir(name),
        Path(eval_file) if eval_file is not None else eval_path(name),
    )
