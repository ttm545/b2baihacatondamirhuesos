from unittest import TestCase

from vk_mail_search.data import Document
from vk_mail_search.period_rank import PeriodRanker

from .search import WordVectorSearch


class WordVectorSearchTest(TestCase):
    def test_soft_frequency_duplicates_and_search_boundaries(self):
        letters = (
            Document("doc-a", "box", "2028-09-13", "сняли"),
            Document("doc-b", "box", "2028-09-13", "сняли сняли"),
            Document("doc-c", "box", "2028-09-13", "прибыл"),
            Document("doc-future", "box", "2028-09-15", "выгрузили"),
            Document("doc-external", "other", "2028-09-13", "выгрузили"),
        )
        baseline = PeriodRanker({letter.doc_id: letter for letter in letters})
        searcher = WordVectorSearch(baseline, {"выгрузить": (1., 0.), "снять": (.8, .6),
                                                "прибыть": (0., 1.)})
        hits = searcher.search("выгрузили 13.09.2028", "box", "2028-09-14", 10)
        self.assertEqual([doc_id for doc_id, _ in hits], ["doc-b", "doc-a", "doc-c"])
        self.assertGreater(hits[0][1], hits[1][1])
        self.assertGreater(hits[1][1], hits[2][1])
        exact = searcher.search("выгрузили 13.09.2028", "box", "2028-09-14", 10, min_similarity=1.)
        self.assertEqual([score for _, score in exact], [0., 0., 0.])

    def test_without_a_period_uses_normal_text_search(self):
        letter = Document("doc-a", "box", "2028-09-13", "Тема: выгрузка")
        period = PeriodRanker({letter.doc_id: letter})
        searcher = WordVectorSearch(period, {})
        args = "выгрузка", "box", "2028-09-14", 5
        self.assertEqual(searcher.search(*args), period.search(*args))
