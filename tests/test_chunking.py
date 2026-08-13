"""Tests for rag.chunking.

Chunking bugs are quiet. An off-by-one in the overlap arithmetic does not
crash - it drops or duplicates text, and every result downstream is subtly
wrong with nothing to point at. These tests exist to make that noisy.

Run with:  python3 -m unittest discover tests
"""

import unittest

from rag.chunking import STRATEGIES, chunk_corpus, chunk_fixed_size
from rag.types import Document


def make_document(text, doc_id="doc"):
    return Document(
        doc_id=doc_id,
        title="Test",
        source_url="https://example.invalid",
        revision_id="1",
        text=text,
    )


class TestFixedSize(unittest.TestCase):
    def test_offsets_match_the_text(self):
        """The Chunk invariant: text is exactly the slice the offsets name."""
        document = make_document("abcdefghij" * 12)
        for chunk in chunk_fixed_size(document, size=25, overlap=7):
            self.assertEqual(chunk.text, document.text[chunk.start:chunk.end])

    def test_no_overlap_reassembles_the_document(self):
        """With overlap=0 the chunks must concatenate back to the original."""
        document = make_document("abcdefghij" * 12)
        chunks = chunk_fixed_size(document, size=30, overlap=0)
        self.assertEqual("".join(c.text for c in chunks), document.text)

    def test_every_character_is_covered(self):
        """No character is dropped, whether or not chunks overlap."""
        document = make_document("abcdefghij" * 12)
        for overlap in (0, 1, 13):
            covered = set()
            for chunk in chunk_fixed_size(document, size=25, overlap=overlap):
                covered.update(range(chunk.start, chunk.end))
            self.assertEqual(covered, set(range(len(document.text))), f"overlap={overlap}")

    def test_overlap_repeats_the_previous_tail(self):
        document = make_document("abcdefghij" * 12)
        chunks = chunk_fixed_size(document, size=40, overlap=10)
        for earlier, later in zip(chunks, chunks[1:]):
            self.assertEqual(later.start, earlier.end - 10)
            self.assertEqual(later.text[:10], earlier.text[-10:])

    def test_last_chunk_is_not_redundant(self):
        """No trailing chunk wholly contained in the one before it."""
        document = make_document("x" * 100)
        chunks = chunk_fixed_size(document, size=40, overlap=20)
        self.assertEqual(chunks[-1].end, 100)
        self.assertGreater(chunks[-1].end, chunks[-2].end)

    def test_document_shorter_than_chunk_size(self):
        document = make_document("short")
        chunks = chunk_fixed_size(document, size=500, overlap=50)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, "short")
        self.assertEqual((chunks[0].start, chunks[0].end), (0, 5))

    def test_empty_document_yields_nothing(self):
        self.assertEqual(chunk_fixed_size(make_document(""), size=10), [])

    def test_chunk_ids_are_unique_and_ordered(self):
        document = make_document("abcdefghij" * 12)
        chunks = chunk_fixed_size(document, size=25, overlap=5)
        ids = [c.chunk_id for c in chunks]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(ids[0], "doc::0000")
        self.assertTrue(all(c.doc_id == "doc" for c in chunks))

    def test_rejects_impossible_parameters(self):
        document = make_document("abc")
        for size, overlap in [(0, 0), (-5, 0), (10, -1), (10, 10), (10, 20)]:
            with self.assertRaises(ValueError, msg=f"size={size} overlap={overlap}"):
                chunk_fixed_size(document, size=size, overlap=overlap)


class TestChunkCorpus(unittest.TestCase):
    def test_flattens_in_document_order(self):
        documents = [make_document("a" * 50, "one"), make_document("b" * 50, "two")]
        chunks = chunk_corpus(documents, "fixed_size", size=20, overlap=0)
        self.assertEqual(chunks[0].doc_id, "one")
        self.assertEqual(chunks[-1].doc_id, "two")

    def test_unknown_strategy_is_rejected(self):
        with self.assertRaises(ValueError):
            chunk_corpus([make_document("abc")], "no_such_strategy", size=10)

    def test_strategies_are_callable(self):
        for name, function in STRATEGIES.items():
            self.assertTrue(callable(function), name)


if __name__ == "__main__":
    unittest.main()
