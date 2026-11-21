import unittest

from vk_mail_search.hybrid import FusionConfig, HybridSearcher, fuse, parse_config


class Fixed:
    """Searcher returning a fixed ranked list; records the depth it was asked for."""

    def __init__(self, hits):
        self.hits = hits
        self.asked = None

    def search(self, query, mailbox, as_of, k):
        self.asked = k
        return self.hits[:k]


class HybridTests(unittest.TestCase):
    def test_rrf_rewards_documents_found_by_both(self):
        lists = {"bm25": [("a", 9.0), ("b", 5.0)], "dense": [("c", 0.9), ("b", 0.8)]}
        self.assertEqual([d for d, _ in fuse(lists, FusionConfig(), 3)], ["b", "a", "c"])

    def test_weights_shift_the_ranking(self):
        lists = {"bm25": [("a", 9.0)], "dense": [("c", 0.9)]}
        config = FusionConfig(weights={"bm25": 0.2, "dense": 0.8})
        self.assertEqual(fuse(lists, config, 2)[0][0], "c")
        only_bm25 = FusionConfig(weights={"bm25": 1.0, "dense": 0.0})
        self.assertEqual([d for d, _ in fuse(lists, only_bm25, 5)], ["a"])

    def test_score_mode_normalises_each_list(self):
        lists = {"bm25": [("a", 30.0), ("b", 10.0)], "dense": [("b", 0.9), ("a", 0.1)]}
        hits = fuse(lists, FusionConfig(mode="score", weights={"bm25": 0.4, "dense": 0.6}), 2)
        self.assertEqual(hits[0], ("b", 0.6))
        self.assertAlmostEqual(hits[1][1], 0.4)

    def test_single_hit_list_and_empty_list(self):
        lists = {"bm25": [], "dense": [("x", 0.3)]}
        self.assertEqual(fuse(lists, FusionConfig(mode="score"), 5), [("x", 1.0)])

    def test_searcher_uses_depth_and_cuts_to_k_without_duplicates(self):
        bm25 = Fixed([(str(i), 10.0 - i) for i in range(10)])
        dense = Fixed([(str(i), 1.0 - i / 10) for i in reversed(range(10))])
        hybrid = HybridSearcher({"bm25": bm25, "dense": dense}, FusionConfig(depth=5))
        hits = hybrid.search("q", "box", None, 3)
        self.assertEqual((bm25.asked, dense.asked), (5, 5))
        self.assertEqual(len(hits), 3)
        self.assertEqual(len({d for d, _ in hits}), 3)
        self.assertEqual(hybrid.search("q", "box", None, 0), [])

    def test_parse_config(self):
        config = parse_config("mode=score, w_dense=0.7, w_bm25=0.3, depth=50")
        self.assertEqual((config.mode, config.depth, dict(config.weights)), ("score", 50, {"dense": 0.7, "bm25": 0.3}))
        self.assertEqual(parse_config(""), FusionConfig())
        for bad in ("mode=max", "depth=0", "w_dense=-1", "alpha=1", "mode"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_config(bad)


if __name__ == "__main__":
    unittest.main()
