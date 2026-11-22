from unittest import TestCase

from vk_mail_search.data import Document
from vk_mail_search.period_rank import PeriodRanker

from .search import WordVectorSearch


class WordVectorSearchTest(TestCase):
    def test_identical_group_reproduces_prefix_scores(self):
        letters = (
            Document("doc-a", "box", "2028-09-13", "Тема: контейнер\nКонтейнер прибыл."),
            Document("doc-b", "box", "2028-09-13", "Тема: контейнерный\nКонтейнерный рейс."),
            Document("doc-c", "box", "2028-09-12", "Контейнер погружен."),
            Document("doc-future", "box", "2028-09-15", "Контейнер поступил."),
        )
        baseline = PeriodRanker({letter.doc_id: letter for letter in letters})
        semantic = WordVectorSearch(baseline, {"контейнер": (1., 0.), "контейнерный": (1., 0.)})
        args = "контейнер 13.09.2028", "box", "2028-09-14", 10
        expected = baseline.search(*args)
        actual = semantic.search(*args)
        self.assertEqual([doc_id for doc_id, _ in actual], [doc_id for doc_id, _ in expected])
        for (_, old_score), (_, new_score) in zip(expected, actual, strict=True):
            self.assertAlmostEqual(old_score, new_score)

    def test_semantic_counts_in_original_bm25_with_mailbox_and_as_of_limits(self):
        letters = (
            Document("doc-a", "box", "2028-09-13", "сняли"),
            Document("doc-b", "box", "2028-09-13", "сняли сняли"),
            Document("doc-c", "box", "2028-09-13", "прибыл"),
            Document("doc-future", "box", "2028-09-15", "сняли"),
            Document("doc-external", "other", "2028-09-13", "выгрузили"),
        )
        baseline = PeriodRanker({letter.doc_id: letter for letter in letters})
        vectors = {"выгрузить": (1., 0.), "снять": (.8, .6), "прибыть": (0., 1.)}
        searcher = WordVectorSearch(baseline, vectors)
        hits = searcher.period.inner.search("выгрузили", "box", "2028-09-14", 10)
        self.assertEqual([doc_id for doc_id, _ in hits], ["doc-b", "doc-a"])
        self.assertGreater(hits[0][1], hits[1][1])
        self.assertEqual(baseline.inner.search("выгрузили", "box", "2028-09-14", 10), [])
