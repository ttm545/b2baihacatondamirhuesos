import json
import tempfile
import unittest
from pathlib import Path

from vk_mail_search.bm25 import BM25Searcher
from vk_mail_search.data import Dataset, Document, Query, load_dataset
from vk_mail_search.evaluate import load_predictions, score_predictions, write_predictions


class BaselineTests(unittest.TestCase):
    def setUp(self):
        documents = [
            Document("older", "own", "2026-11-19", "Назначили курьера"),
            Document("unknown", "own", None, "Курьеры доставят документы"),
            Document("future", "own", "2026-11-22", "Курьер"),
            Document("other", "other", "2026-11-19", "Курьер"),
        ]
        self.dataset = Dataset(
            {doc.doc_id: doc for doc in documents},
            {"q": Query("q", "own", "курьер", "2026-11-21")},
            {"q": {"older": 2, "unknown": 1}},
        )

    def test_bm25_lemmatizes_and_respects_mailbox_and_date(self):
        bm25 = BM25Searcher(self.dataset.documents.values())
        hits = bm25.search("курьер", "own", "2026-11-21", 10)
        self.assertEqual({doc_id for doc_id, _ in hits}, {"older", "unknown"})

    def test_bm25_preserves_sender_addresses_and_removes_html_tags(self):
        docs = [
            Document("plain", "own", None, "От: Иван <ivan.petrov@example.test>\nТема: Доставка"),
            Document("html", "own", None, "От: Анна <anna@example.test>\n<div>Квитанция</div>"),
        ]
        bm25 = BM25Searcher(docs)
        for query, doc_id in (("ivan", "plain"), ("anna", "html")):
            with self.subTest(query=query):
                self.assertEqual([hit for hit, _ in bm25.search(query, "own", None, 10)], [doc_id])
        self.assertEqual(bm25.search("div", "own", None, 10), [])

    def test_metrics_respect_grade_and_rank(self):
        report = score_predictions(self.dataset, {"q": ["unknown", "older"]}, (1, 2))
        self.assertEqual(report["metrics"]["@1"], {"recall": 0.5, "ndcg": 0.333333, "mrr": 1.0})
        self.assertEqual(report["metrics"]["@2"]["recall"], 1.0)
        self.assertLess(report["metrics"]["@2"]["ndcg"], 1.0)

    def test_ineligible_predictions_are_rejected(self):
        for bad in (["future"], ["other"], ["older", "older"]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                score_predictions(self.dataset, {"q": bad}, (10,))

    def test_real_jsonl_schema_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            values = {
                "documents.jsonl": [{"doc_id": "older", "synthetic_user": "own", "doc_dt": None, "doc": "Письмо"}],
                "queries_dev.jsonl": [{"query_id": "q", "synthetic_user": "own", "query_date": "2026-11-21", "query": "письмо"}],
                "qrels_dev.jsonl": [{"query_id": "q", "doc_id": "older", "relevance": 1}],
            }
            for filename, rows in values.items():
                (path / filename).write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
            dataset = load_dataset(path)
            predictions = path / "predictions.jsonl"
            write_predictions(predictions, {"q": ["older"]}, 10)
            self.assertEqual(score_predictions(dataset, load_predictions(predictions), (10,))["metrics"]["@10"]["recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
