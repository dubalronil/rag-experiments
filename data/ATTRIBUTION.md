# Corpus attribution and license

The Markdown documents in `data/documents/` are derived from Wikipedia articles
and are licensed under
[Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/).

This license applies to the corpus only. The code in this repository is under
the MIT license — see [`LICENSE`](../LICENSE).

Each article is authored by Wikipedia contributors; the page history linked from
each source URL below lists them.

## Source articles

Downloaded 2026-08-12 with [`scripts/fetch_corpus.py`](../scripts/fetch_corpus.py).
The revision ID identifies the exact version of each article that was used, and
each document repeats it in its own front matter.

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

## Changes made to the original text

CC BY-SA requires that modifications be stated. The articles were not rewritten,
but they were mechanically reduced:

- Retrieved as plain text through the MediaWiki `extracts` API, which removes
  citations, reference markers, infoboxes, navigation boxes, and **tables**.
- Section headings converted from wiki markup (`== History ==`) to Markdown
  (`## History`).
- Trailing sections dropped: See also, References, Notes, Citations, Sources,
  Footnotes, Further reading, Bibliography, External links.
- Front matter added with the title, source URL, revision ID, and retrieval date.

Because tables are stripped, some sections survive as headings with little or no
body text — `Season-by-season record`, `Career statistics`, `Head coaches`, and
similar. The prose is intact; tabular data is not present in this corpus.

## Regenerating

If you change the article list and re-download, update the table above so the
attribution keeps matching the files actually in the repository.
