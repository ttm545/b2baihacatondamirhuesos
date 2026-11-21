import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from vk_mail_search.predict import main


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


class PredictTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.queries = self.root / "private-queries.jsonl"
        self.output = self.root / "predictions.jsonl"
        write_jsonl(self.root / "documents.jsonl", [
            {"doc_id": "past", "synthetic_user": "own", "doc_dt": "2026-11-20", "doc": "Квитанция"},
            {"doc_id": "future", "synthetic_user": "own", "doc_dt": "2026-11-22", "doc": "Квитанция"},
            {"doc_id": "foreign", "synthetic_user": "other", "doc_dt": "2026-11-20", "doc": "Квитанция"},
            {"doc_id": "undated", "synthetic_user": "own", "doc_dt": None, "doc": "Квитанция"},
        ])
        write_jsonl(self.queries, [
            {"query_id": qid, "synthetic_user": mailbox, "query": text, "query_date": "2026-11-21"}
            for qid, mailbox, text in [
                ("z", "own", "квитанция"), ("a", "own", "   "), ("x", "unknown", "квитанция"),
            ]
        ])
        self.args = ["--queries", str(self.queries), "--out", str(self.output)]

    def test_exports_in_input_order_without_relevance_labels(self):
        stderr = io.StringIO()
        with patch.dict(os.environ, VK_DATA_DIR=str(self.root)), contextlib.redirect_stderr(stderr):
            main(self.args)

        rows = [json.loads(line) for line in self.output.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(rows, [
            {"query_id": "z", "doc_ids": ["past", "undated"]},
            {"query_id": "a", "doc_ids": []},
            {"query_id": "x", "doc_ids": []},
        ])
        self.assertIn("пустой запрос", stderr.getvalue())
        self.assertIn("неизвестный ящик", stderr.getvalue())
        summary = json.loads(stderr.getvalue().splitlines()[-1])
        self.assertEqual((summary["queries"], summary["empty"]), (3, 2))
        self.assertEqual(set(summary["search_ms"]), {"p50", "p95", "p99"})

    def test_export_filters_invalid_hits_and_keeps_first_occurrence(self):
        searcher = Mock()
        searcher.search.return_value = [
            (doc, 1.0) for doc in ("foreign", "undated", "undated", "future", "past", "missing")
        ]
        with patch("vk_mail_search.predict.build_searcher", return_value=searcher), contextlib.redirect_stderr(io.StringIO()):
            main([*self.args, "--data-dir", str(self.root), "--k", "2"])
        first = self.output.read_text(encoding="utf-8").splitlines()[0]
        self.assertEqual(json.loads(first)["doc_ids"], ["undated", "past"])
        searcher.search.assert_called_once_with("квитанция", "own", "2026-11-21", 2)


if __name__ == "__main__":
    unittest.main()
