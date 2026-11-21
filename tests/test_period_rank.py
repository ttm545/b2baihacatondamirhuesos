import unittest

from vk_mail_search.data import Document
from vk_mail_search.searchers import build_searcher


class PeriodRankTest(unittest.TestCase):
    def setUp(self):
        docs = [
            Document("dispatch", "box", "2028-10-05", "Тема: Отгрузка контейнера\n\nТовар отправили со склада."),
            Document("arrival", "box", "2028-10-05", "Тема: Прибытие контейнера\n\nГруз прибыл на склад."),
            Document("older", "box", "2028-10-04", "Тема: Отгрузка контейнера\n\nТовар отправили."),
            Document("future", "box", "2028-10-07", "Тема: Отгрузка контейнера"),
            Document("other", "other", "2028-10-05", "Тема: Отгрузка контейнера"),
        ]
        self.docs = {doc.doc_id: doc for doc in docs}
        self.engine = build_searcher("period", self.docs)

    def test_candidates_respect_date_priority_and_mailbox(self):
        hits = self.engine.search("Отгрузка контейнера 05.10.2028", "box", "2028-10-06", 10)
        self.assertEqual([doc_id for doc_id, _ in hits], ["dispatch", "arrival", "older"])

    def test_undated_and_date_only_preserve_the_baseline(self):
        baseline = build_searcher("dated:bm25", self.docs)
        for query in ("Отгрузка контейнера", "05.10.2028"):
            with self.subTest(query=query):
                self.assertEqual(self.engine.search(query, "box", "2028-10-06", 10),
                                 baseline.search(query, "box", "2028-10-06", 10))

    def test_top_one_agrees_with_first_result_of_larger_request(self):
        query = "Отгрузка контейнера 5 октября 2028"
        first = self.engine.search(query, "box", "2028-10-06", 1)
        larger = self.engine.search(query, "box", "2028-10-06", 10)
        self.assertEqual(first, larger[:1])


if __name__ == "__main__":
    unittest.main()
