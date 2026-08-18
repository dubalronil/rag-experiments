#!/usr/bin/env python3
"""Download a dataset's Wikipedia articles and save them as Markdown.

Which articles a dataset contains is read from data/<dataset>/articles.txt, so
this script is not tied to any one corpus.

We ask the MediaWiki API for the "extract" of each article, which is the
article's plain text with section headings included, and with citations,
infoboxes, navigation boxes, and tables already removed. That saves us from
writing an HTML scraper, at the cost of losing tabular data (see README note).

Standard library only - no third-party dependencies.

Usage:
    python scripts/fetch_corpus.py                  # the default dataset
    python scripts/fetch_corpus.py --dataset space
    python scripts/fetch_corpus.py --force          # re-download existing files
    python scripts/fetch_corpus.py --out some/dir
"""

import argparse
import datetime
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.datasets import DEFAULT_DATASET, corpus_dir, read_articles  # noqa: E402

# The corpus is defined by data/<dataset>/articles.txt rather than by a list
# here, so adding a dataset means adding a file, not editing this script -
# and each corpus keeps its provenance next to the documents it produced.

# Sections that carry no article prose. Everything under a top-level heading
# with one of these names is dropped.
SKIP_SECTIONS = {
    "see also",
    "references",
    "notes",
    "citations",
    "sources",
    "footnotes",
    "further reading",
    "bibliography",
    "external links",
}

API_URL = "https://en.wikipedia.org/w/api.php"

# Wikipedia blocks requests that do not identify themselves.
USER_AGENT = (
    "rag-experiments/0.1 (learning project; "
    "https://github.com/dubalronil/rag-experiments)"
)

# Matches a section heading in the API's plain-text output, e.g. "== History =="
# or "=== Early years ===". The number of "=" signs gives the nesting depth.
HEADING_RE = re.compile(r"^(={2,6})\s*(.+?)\s*\1$")


def fetch_article(title, attempts=3):
    """Return the API's data for one article: title, extract, and revision.

    Note the returned title is the *resolved* one. Asking for a redirect such
    as "History of the National Basketball Association" gives back the article
    it points at, under its real name.
    """
    params = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "prop": "extracts|revisions",
        "explaintext": "1",       # plain text instead of HTML
        "exsectionformat": "wiki",  # headings as "== Section ==" markers
        "rvprop": "ids|timestamp",
        "redirects": "1",         # follow "Lakers" -> "Los Angeles Lakers"
        "titles": title,
    }
    url = API_URL + "?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    # Wikipedia answers with 429 if we ask too quickly. Back off and retry
    # rather than losing the article.
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.load(response)
            break
        except urllib.error.HTTPError as error:
            if error.code == 429 and attempt < attempts:
                wait = 5 * attempt
                print(f"        rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            raise

    pages = payload.get("query", {}).get("pages", [])
    if not pages:
        raise RuntimeError(f"no page returned for {title!r}")

    page = pages[0]
    if page.get("missing"):
        raise RuntimeError(f"article does not exist: {title!r}")
    if not page.get("extract"):
        raise RuntimeError(f"article has no text: {title!r}")

    revision = page.get("revisions", [{}])[0]
    return {
        "title": page["title"],
        "extract": page["extract"],
        "revision_id": revision.get("revid"),
        "revision_timestamp": revision.get("timestamp"),
    }


def to_markdown(extract):
    """Convert the API's plain text into Markdown body text.

    Two jobs: turn "== Section ==" into "## Section", and drop the trailing
    reference/link sections listed in SKIP_SECTIONS.
    """
    lines = []
    skipping = False

    for line in extract.split("\n"):
        match = HEADING_RE.match(line.strip())

        if match:
            depth = len(match.group(1))  # 2 for "==", 3 for "===", ...
            heading = match.group(2)

            # Only top-level sections decide what we skip; a subsection stays
            # inside whatever decision its parent section made.
            if depth == 2:
                skipping = heading.strip().lower() in SKIP_SECTIONS
            if skipping:
                continue

            # "==" is the article's top section level, and "#" is reserved for
            # the document title, so "==" becomes "##".
            lines.append("#" * depth + " " + heading)
            continue

        if skipping:
            continue

        lines.append(line.rstrip())

    # Collapse runs of blank lines into one, and trim the ends.
    body = "\n".join(lines)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return body.strip()


def slugify(title):
    """Turn an article title into a stable filename stem.

    The stem becomes the document's ID everywhere else in this project, so it
    needs to stay stable: same article in, same slug out.
    """
    slug = title.lower()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    return slug.strip("_")


def article_url(title):
    """The canonical Wikipedia URL for an article title."""
    return "https://en.wikipedia.org/wiki/" + urllib.parse.quote(
        title.replace(" ", "_")
    )


def build_document(article, retrieved_on):
    """Assemble the final Markdown file contents, front matter included.

    The revision ID pins the exact version of the article we downloaded, so a
    result produced today stays traceable to the text it was produced from.
    """
    header = [
        "---",
        f'title: "{article["title"]}"',
        f"source_url: {article_url(article['title'])}",
        f"revision_id: {article['revision_id']}",
        f"revision_timestamp: {article['revision_timestamp']}",
        f"retrieved: {retrieved_on}",
        "---",
        "",
        f"# {article['title']}",
        "",
        "",
    ]
    return "\n".join(header) + to_markdown(article["extract"]) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"which dataset to fetch (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="override the output directory (default: data/<dataset>/documents)",
    )
    parser.add_argument(
        "--articles",
        default=None,
        help="override the article list (default: data/<dataset>/articles.txt)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-download articles whose file already exists",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="seconds to wait between requests (default: 1.0)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out) if args.out else corpus_dir(args.dataset)
    titles = (
        [line.strip() for line in Path(args.articles).read_text(encoding="utf-8").splitlines()
         if line.strip() and not line.strip().startswith("#")]
        if args.articles else read_articles(args.dataset)
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(titles)} articles -> {out_dir}\n")
    retrieved_on = datetime.date.today().isoformat()

    failures = []
    written = {}  # slug -> resolved title, to catch two entries landing on one file

    for title in titles:
        # Check before fetching so re-runs stay cheap. This uses the requested
        # title; if it turns out to be a redirect we correct the name below.
        if (out_dir / (slugify(title) + ".md")).exists() and not args.force:
            print(f"skip    {title}  (already exists; use --force to replace)")
            continue

        try:
            article = fetch_article(title)
        except (urllib.error.URLError, RuntimeError, ValueError) as error:
            print(f"FAILED  {title}: {error}", file=sys.stderr)
            failures.append(title)
            continue

        # Name the file after the article we actually got, not the one we asked
        # for. Otherwise a redirect silently produces a mislabelled document.
        resolved = article["title"]
        slug = slugify(resolved)
        path = out_dir / (slug + ".md")

        if slug != slugify(title):
            print(f"note    {title!r} redirects to {resolved!r}")

        if slug in written:
            print(
                f"FAILED  {title!r} resolves to {resolved!r}, already saved from "
                f"{written[slug]!r} - remove one from the article list",
                file=sys.stderr,
            )
            failures.append(title)
            continue
        written[slug] = title

        document = build_document(article, retrieved_on)
        path.write_text(document, encoding="utf-8")

        words = len(document.split())
        sections = document.count("\n## ")
        print(f"wrote   {path}  ({words:,} words, {sections} sections)")

        time.sleep(args.delay)

    print(f"\n{len(titles) - len(failures)}/{len(titles)} articles in {out_dir}")
    if failures:
        print("failed: " + ", ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
