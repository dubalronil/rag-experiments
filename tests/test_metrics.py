"""Tests for the MRR cutoff in run_retrieval_eval.summarize.

MRR is the one retrieval metric with no cutoff in its name, so it is the one
that can silently change meaning when k changes. Uncapped, a first hit at rank
7 contributes 1/7 to a k=10 run and 0 to a k=5 run - and both land in the same
`mrr` column of results/summary.csv, where nothing marks them as different
metrics. Pinning MRR at rank 5 is what keeps that column comparable across
every run.

Nothing raises if the pin is removed. The column just quietly starts drifting
upward with k, which reads as an improvement. Hence this test.

Run with:  python3 -m unittest discover tests
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from run_retrieval_eval import MRR_CUTOFF, summarize  # noqa: E402


class TestMrrCutoff(unittest.TestCase):
    def test_hits_past_rank_5_contribute_nothing_to_mrr_at_5(self):
        """A hit at rank 7 or 10 is a miss as far as MRR@5 is concerned.

        Same three questions either way. The k=5 run found only the rank-2 hit;
        the k=10 run also found hits at 7 and 10. MRR@5 must not notice.
        """
        shallow = [2, None, None]
        deep = [2, 7, 10]

        _, shallow_mrr = summarize(shallow, (1, 3, 5), mrr_cutoff=5)
        _, deep_mrr = summarize(deep, (1, 3, 5, 10), mrr_cutoff=5)

        # One hit at rank 2, over three questions.
        self.assertAlmostEqual(deep_mrr, (1 / 2) / 3, places=12)
        self.assertEqual(deep_mrr, shallow_mrr)

    def test_rank_5_itself_still_counts(self):
        """The cutoff is inclusive - only ranks *past* 5 are dropped."""
        _, mrr = summarize([5, None], (1, 3, 5), mrr_cutoff=5)

        self.assertAlmostEqual(mrr, (1 / 5) / 2, places=12)

    def test_uncapped_mrr_still_includes_deep_ranks(self):
        """The default is unchanged, so the standalone script keeps its numbers."""
        _, mrr = summarize([2, 7, 10], (1, 3, 5, 10))

        self.assertAlmostEqual(mrr, (1 / 2 + 1 / 7 + 1 / 10) / 3, places=12)

    def test_the_pin_is_what_the_runner_uses(self):
        """A pin set to anything but 5 would not match the saved history."""
        self.assertEqual(MRR_CUTOFF, 5)


if __name__ == "__main__":
    unittest.main()
