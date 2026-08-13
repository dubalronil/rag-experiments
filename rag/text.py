"""Text normalization shared by everything that compares gold quotes to text.

There is exactly one definition of "does this quote appear in this text", and
it lives here. Two copies would be worse than none: a quote could validate
against the corpus in one place and silently fail to match during scoring in
another, every question would score zero, and nothing would say why.

Standard library only.
"""

import re
import unicodedata

# Characters that differ between the corpus and hand-typed quotes without the
# text actually being different.
_SUBSTITUTIONS = [
    ("‘", "'"),  # left single quote
    ("’", "'"),  # right single quote / apostrophe
    ("“", '"'),  # left double quote
    ("”", '"'),  # right double quote
    ("–", "-"),  # en dash
    ("—", "-"),  # em dash
    (" ", " "),  # non-breaking space
]


def normalize(text):
    """Collapse away differences that are not real mismatches.

    Markdown files wrap and re-wrap, so a quote spanning a line break would
    fail a raw comparison even though the text is identical. Typing a quote by
    hand also tends to produce straight quotes and plain hyphens where the
    source has curly quotes and en dashes.

    What survives normalization is still strict: a changed word, a dropped
    word, a removed comma, or different capitalization all still count as a
    mismatch.
    """
    text = unicodedata.normalize("NFC", text)
    for fancy, plain in _SUBSTITUTIONS:
        text = text.replace(fancy, plain)
    return re.sub(r"\s+", " ", text).strip()


def contains_quote(haystack, quote):
    """True if `quote` appears in `haystack`, comparing normalized forms."""
    return normalize(quote) in normalize(haystack)
