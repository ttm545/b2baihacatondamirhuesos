import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from vk_mail_search.bm25 import BM25Searcher
from vk_mail_search.data import Dataset, Document, Query, load_dataset
from vk_mail_search.evaluate import load_paraphrases, load_predictions, main as evaluate, score_predictions, write_predictions


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

    def test_paraphrases_inherit_mailbox_date_and_qrels(self):
        path = Path(self.enterContext(tempfile.TemporaryDirectory())) / "paraphrases.jsonl"
        rows = [
            {"query_id": variant, "source_query_id": "q", "query": text}
            for variant, text in [("p2", "кто привезёт"), ("p1", "доставка")]
        ]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        paraphrases = load_paraphrases(self.dataset, path)
        self.assertEqual(list(paraphrases.queries), ["p2", "p1"])
        self.assertEqual(paraphrases.queries["p2"], Query("p2", "own", "кто привезёт", "2026-11-21"))
        self.assertEqual(paraphrases.qrels, {"p2": self.dataset.qrels["q"], "p1": self.dataset.qrels["q"]})
        report = score_predictions(paraphrases, {"p2": ["older", "unknown"], "p1": []}, (2,))
        self.assertEqual(report["metrics"]["@2"], {"recall": 0.5, "ndcg": 0.5, "mrr": 0.5})

        rows[1]["source_query_id"] = "missing"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "paraphrases.jsonl:2: неизвестный source_query_id"):
            load_paraphrases(self.dataset, path)

    def test_paraphrase_metrics_report_both_splits_and_allow_tune_only(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        dataset = Dataset(self.dataset.documents,
                          {**self.dataset.queries, "second": Query("second", "own", "доставка", "2026-11-21")},
                          {**self.dataset.qrels, "second": {"older": 1}})
        path = root / "paraphrases.jsonl"
        rows = [
            {"query_id": "train", "source_query_id": "q", "query": "доставят", "split": "tune"},
            {"query_id": "held-out", "source_query_id": "second", "query": "привезут", "split": "test"},
        ]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        predictions = root / "predictions.jsonl"
        write_predictions(predictions, {"train": ["older", "unknown"], "held-out": []}, 10)
        args = ["evaluate", "--pred", str(predictions), "--paraphrases", str(path)]
        with patch("sys.argv", args), patch("vk_mail_search.evaluate.load_dataset", return_value=dataset):
            with redirect_stdout(io.StringIO()) as both:
                evaluate()
            write_predictions(predictions, {"train": ["older", "unknown"]}, 10)
            with patch("sys.argv", [*args, "--split", "tune"]), redirect_stdout(io.StringIO()) as tune:
                evaluate()
        report = json.loads(both.getvalue())
        tune_report = json.loads(tune.getvalue())
        self.assertEqual(report["splits"]["tune"]["metrics"], tune_report["metrics"])
        self.assertEqual(tune_report["queries"], 1)
        self.assertEqual(tune_report["split"], "tune")
        self.assertEqual(report["splits"]["test"]["metrics"]["@10"]["ndcg"], 0)

        rows[1]["source_query_id"] = "q"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "source_query_id .* встречается в разных частях"):
            load_paraphrases(dataset, path)

    def test_prefix_matches_related_words_without_crossing_mailboxes(self):
        docs = [
            Document("load", "own", None, "Контейнер погружен"),
            Document("unload", "own", None, "Контейнер выгружен"),
            Document("foreign", "other", None, "Контейнер погружен"),
        ]
        bm25 = BM25Searcher(docs, subject_weight=1, prefix_weight=0.15)
        self.assertEqual([doc_id for doc_id, _ in bm25.search("погрузка", "own", None, 10)], ["load"])

    def test_explicit_subject_has_more_weight_than_body(self):
        docs = [
            Document("a-body", "own", None, "Тема: Почта\nКвитанция"),
            Document("z-topic", "own", None, "Тема: Квитанция\nПочта"),
        ]
        bm25 = BM25Searcher(docs, subject_weight=2, prefix_weight=0)
        self.assertEqual([doc_id for doc_id, _ in bm25.search("квитанция", "own", None, 2)], ["z-topic", "a-body"])

    def test_prepositions_do_not_outweigh_a_topic_match_in_subject(self):
        docs = [
            Document("body", "own", None, "Тема: Склад\nсверка о о о о о о"),
            Document("subject", "own", None, "Тема: Сверка\nсклад"),
        ]
        bm25 = BM25Searcher(docs, b=0, subject_weight=1)
        self.assertEqual(bm25.search("о сверке", "own", None, 1)[0][0], "subject")
        self.assertEqual(bm25.search("о", "own", None, 1)[0][0], "body")

    def test_uses_short_first_line_if_subject_header_is_missing(self):
        docs = [
            Document("a-body", "own", None, "Склад\nКвитанция"),
            Document("z-topic", "own", None, "Квитанция\nСклад"),
        ]
        bm25 = BM25Searcher(docs)
        self.assertEqual(bm25.search("квитанция", "own", None, 1)[0][0], "z-topic")

    def test_single_mrr_uses_only_queries_with_one_positive_judgment(self):
        dataset = Dataset(
            self.dataset.documents,
            {**self.dataset.queries, "single": Query("single", "own", "доставка", "2026-11-21")},
            {**self.dataset.qrels, "single": {"older": 1, "unknown": 0}},
        )
        report = score_predictions(dataset, {"q": ["older", "unknown"], "single": ["unknown", "older"]}, (1, 2))
        self.assertEqual(report["single_relevant_queries"], 1)
        self.assertEqual(report["mrr_single"], {"@1": 0.0, "@2": 0.5})
        self.assertEqual(report["metrics"]["@2"]["mrr"], 0.75)

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
