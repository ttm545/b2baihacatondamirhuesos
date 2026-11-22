from unittest import TestCase

from vk_mail_search.data import Document
from vk_mail_search.period_rank import PeriodRanker

from .searcher import SubjectReranker, fuse


class FakeScorer:
    def __init__(self):
        self.calls = []

    def score(self, query, titles):
        self.calls.append((query, titles))
        return [0.9 if "создан" in title else 0.1 for title in titles]


class SubjectRerankerTest(TestCase):
    def test_fusion_only_moves_positions_eligible_in_the_first_ten(self):
        hits = [(str(i), float(15 - i)) for i in range(12)]
        scored = {"0": 0.05, "2": 0.9, "10": 1.0}
        result = fuse(hits, scored, 0.5)
        self.assertEqual([doc_id for doc_id, _ in result], ["2", "1", "0", *(str(i) for i in range(3, 12))])
        self.assertEqual(fuse(hits, scored, 0), hits)

    def test_sends_only_explicit_subjects_in_period_without_body(self):
        letters = (
            Document("doc-a", "box", "2028-09-13", "Тема: Автовизит создан\nТело письма."),
            Document("doc-b", "box", "2028-09-13", "Тема: Автомашина прибыла\nТело письма."),
            Document("doc-c", "box", "2028-09-13", "Текст без заголовка про автовизит."),
            Document("doc-old", "box", "2028-09-12", "Тема: Автовизит вчера"),
        )
        documents = {doc.doc_id: doc for doc in letters}
        baseline = PeriodRanker(documents)
        model = FakeScorer()
        searcher = SubjectReranker(baseline, documents, model)
        searcher.search("Покажи автовизит 13.09.2028", "box", "2028-09-14", 10)
        self.assertEqual(len(model.calls), 1)
        query, subjects = model.calls[0]
        self.assertNotIn("13.09.2028", query)
        self.assertCountEqual(subjects, ["Автовизит создан", "Автомашина прибыла"])
