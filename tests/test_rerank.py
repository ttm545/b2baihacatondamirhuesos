import threading
import unittest

from vk_mail_search.data import Document
from vk_mail_search.dated import DatedSearcher
from vk_mail_search.rerank import RerankSearcher


class Baseline:
    def __init__(self, docs):
        self.docs = docs

    def search(self, query, mailbox, as_of, k):
        return [(doc.doc_id, float(10 - i)) for i, doc in enumerate(self.docs)
                if doc.available(mailbox, as_of)][:k]


class TestRerank(unittest.TestCase):
    query = "Покажи письма о получении груза, полученные вчера"
    as_of = "2028-09-14"

    def setUp(self):
        records = [
            Document("first", "box", "2028-09-13", "Первое"),
            Document("second", "box", "2028-09-13", "Второе"),
            Document("outside", "box", "2028-09-12", "Вне периода"),
            Document("future", "box", "2028-09-15", "Будущее"),
            Document("another-box", "other", "2028-09-13", "Другой ящик"),
        ]
        self.docs = {doc.doc_id: doc for doc in records}
        self.baseline = DatedSearcher(Baseline(records), records)

    def test_only_period_is_reranked(self):
        class Scorer:
            def score(self, topic, passages, deadline):
                assert "вчера" not in topic and list(passages) == ["Первое", "Второе"]
                return [0.1, 0.9]

        searcher = RerankSearcher(self.baseline, self.docs, Scorer())
        self.addCleanup(searcher.close)
        result = searcher.search(self.query, "box", self.as_of, 10)
        self.assertEqual([doc for doc, _ in result], ["second", "first", "outside"])
        self.assertEqual(searcher.statistics, {"reranked": 1})

    def test_without_period_reranks_the_top(self):
        class Scorer:
            def score(self, topic, passages, deadline):
                assert list(passages) == ["Первое", "Второе", "Вне периода"]
                return [0.1, 0.9, 0.2]

        searcher = RerankSearcher(self.baseline, self.docs, Scorer(), top_n=10)
        self.addCleanup(searcher.close)
        self.assertEqual(searcher.search("груз", "box", self.as_of, 1), [("second", 0.9)])

    def test_timeout_falls_back_and_busy_worker_does_not_queue(self):
        started, release = threading.Event(), threading.Event()

        class SlowScorer:
            def score(self, topic, passages, deadline):
                started.set()
                release.wait(2)
                return [0.1, 0.9]

        searcher = RerankSearcher(self.baseline, self.docs, SlowScorer(), budget_ms=50)
        self.addCleanup(searcher.close)
        self.addCleanup(release.set)
        expected = self.baseline.search(self.query, "box", self.as_of, 1)
        self.assertEqual(searcher.search(self.query, "box", self.as_of, 1), expected)
        self.assertTrue(started.wait(0.5))
        self.assertEqual(searcher.search(self.query, "box", self.as_of, 1), expected)
        self.assertEqual(searcher.statistics, {"timeout": 1, "busy": 1})

    def test_invalid_scores_keep_baseline_order(self):
        class Scorer:
            def score(self, topic, passages, deadline):
                return [float("nan"), 0.5]

        searcher = RerankSearcher(self.baseline, self.docs, Scorer())
        self.addCleanup(searcher.close)
        self.assertEqual(searcher.search(self.query, "box", self.as_of, 10),
                         self.baseline.search(self.query, "box", self.as_of, 10))
        self.assertEqual(searcher.statistics, {"invalid_scores": 1})
