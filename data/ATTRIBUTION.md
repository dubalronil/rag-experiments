# Corpus attribution and license

Everything under `data/` is derived from Wikipedia articles and is licensed under
[Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/):

- `data/<dataset>/documents/` — the article text itself, as Markdown.
- `data/<dataset>/eval/questions.jsonl` — the evaluation set. Its `quote` fields
  hold sentences copied verbatim out of the documents, so the file carries the
  same share-alike obligation even though the questions around them are original.
- `data/<dataset>/articles.txt` — the list of article titles each corpus was
  built from.

This license applies to `data/` only. The code in this repository is under the
MIT license — see [`LICENSE`](../LICENSE).

Each article is authored by Wikipedia contributors; the page history linked from
each source URL below lists them.

## Datasets

There are two corpora, each with its own evaluation set. They exist so a result
can be checked against a second, independent body of text rather than only the
one the pipeline was tuned on.

| Dataset | Articles | Documents | Eval questions | Downloaded |
| --- | ---: | ---: | ---: | --- |
| `nba` | NBA history | 10 | 50 (42 answerable, 8 unanswerable) | 2026-08-12 |
| `space` | Spaceflight and astronomy | 10 | 50 (42 answerable, 8 unanswerable) | 2026-08-16 |

Both were downloaded with [`scripts/fetch_corpus.py`](../scripts/fetch_corpus.py).
The revision ID identifies the exact version of each article that was used, and
each document repeats it in its own front matter.

## Source articles — `data/nba/`

| Document | Source article | Revision |
| --- | --- | --- |
| `american_basketball_association` | [American Basketball Association](https://en.wikipedia.org/wiki/American_Basketball_Association) | 1360042842 |
| `boston_celtics` | [Boston Celtics](https://en.wikipedia.org/wiki/Boston_Celtics) | 1369076638 |
| `chicago_bulls` | [Chicago Bulls](https://en.wikipedia.org/wiki/Chicago_Bulls) | 1366322186 |
| `golden_state_warriors` | [Golden State Warriors](https://en.wikipedia.org/wiki/Golden_State_Warriors) | 1368684515 |
| `lebron_james` | [LeBron James](https://en.wikipedia.org/wiki/LeBron_James) | 1369055605 |
| `los_angeles_lakers` | [Los Angeles Lakers](https://en.wikipedia.org/wiki/Los_Angeles_Lakers) | 1369049562 |
| `michael_jordan` | [Michael Jordan](https://en.wikipedia.org/wiki/Michael_Jordan) | 1369032556 |
| `national_basketball_association` | [National Basketball Association](https://en.wikipedia.org/wiki/National_Basketball_Association) | 1363765440 |
| `nba_finals` | [NBA Finals](https://en.wikipedia.org/wiki/NBA_Finals) | 1365379165 |
| `san_antonio_spurs` | [San Antonio Spurs](https://en.wikipedia.org/wiki/San_Antonio_Spurs) | 1364960500 |

## Source articles — `data/space/`

| Document | Source article | Revision |
| --- | --- | --- |
| `apollo_11` | [Apollo 11](https://en.wikipedia.org/wiki/Apollo_11) | 1369197321 |
| `apollo_13` | [Apollo 13](https://en.wikipedia.org/wiki/Apollo_13) | 1369194288 |
| `apollo_program` | [Apollo program](https://en.wikipedia.org/wiki/Apollo_program) | 1368594069 |
| `hubble_space_telescope` | [Hubble Space Telescope](https://en.wikipedia.org/wiki/Hubble_Space_Telescope) | 1369264480 |
| `international_space_station` | [International Space Station](https://en.wikipedia.org/wiki/International_Space_Station) | 1368239995 |
| `james_webb_space_telescope` | [James Webb Space Telescope](https://en.wikipedia.org/wiki/James_Webb_Space_Telescope) | 1369507831 |
| `mars_rover` | [Mars rover](https://en.wikipedia.org/wiki/Mars_rover) | 1368887798 |
| `nasa` | [NASA](https://en.wikipedia.org/wiki/NASA) | 1368866353 |
| `space_shuttle_program` | [Space Shuttle program](https://en.wikipedia.org/wiki/Space_Shuttle_program) | 1364331014 |
| `voyager_program` | [Voyager program](https://en.wikipedia.org/wiki/Voyager_program) | 1360099643 |

## Changes made to the original text

CC BY-SA requires that modifications be stated. The articles were not rewritten,
but they were mechanically reduced — identically for both datasets:

- Retrieved as plain text through the MediaWiki `extracts` API, which removes
  citations, reference markers, infoboxes, navigation boxes, and **tables**.
- Section headings converted from wiki markup (`== History ==`) to Markdown
  (`## History`).
- Trailing sections dropped: See also, References, Notes, Citations, Sources,
  Footnotes, Further reading, Bibliography, External links.
- Front matter added with the title, source URL, revision ID, and retrieval date.

Because tables are stripped, some sections survive as headings with little or no
body text — `Season-by-season record`, `Career statistics`, `Head coaches` in the
NBA corpus, and the mission and instrument tables in the Space corpus. The prose
is intact; tabular data is not present in either corpus.

## Regenerating

If you change an article list and re-download, update the matching table above so
the attribution keeps describing the files actually in the repository.

Re-downloading also changes the corpus hash every run records, and the runner
will refuse to record a run against a corpus that differs from the last one for
that dataset. That is deliberate: a metric computed against a different corpus is
not comparable with the results already here.
