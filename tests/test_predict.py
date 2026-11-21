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

    def test_mixed_file_keeps_valid_rows_and_anchors_relative_dates(self):
        with (self.root / "documents.jsonl").open("a", encoding="utf-8") as documents:
            documents.write(json.dumps({"doc_id": "anchor-yesterday", "synthetic_user": "own",
                                        "doc_dt": "2026-11-21", "doc": "Квитанция"}) + "\n")
        normal = {"query": "квитанция", "synthetic_user": "own", "query_date": "2026-11-21"}
        rows = [
            json.dumps(dict(normal, query_id="valid")),
            "{broken",
            json.dumps(normal),
            json.dumps(dict(normal, query_id="null", query_date=None)),
            json.dumps({"query_id": "relative", "synthetic_user": "own", "query": "Письма за вчера"}),
            json.dumps(dict(normal, query_id="invalid-date", query_date="2026-99-99")),
            json.dumps(dict(normal, query_id="no-mailbox", synthetic_user=None)),
            json.dumps(dict(normal, query_id="unknown", synthetic_user="missing")),
            json.dumps(dict(normal, query_id="empty", query=" ")),
            json.dumps(dict(normal, query_id="valid", synthetic_user="other")),
            json.dumps(dict(normal, query_id="valid-last")),
        ]
        self.queries.write_bytes(("\n".join(rows[:2]) + "\n").encode() + b"\xff\n"
                                 + ("\n".join(rows[2:]) + "\n").encode())
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            main([*self.args, "--data-dir", str(self.root)])
        output = {row["query_id"]: row["doc_ids"]
                  for row in map(json.loads, self.output.read_text(encoding="utf-8").splitlines())}
        self.assertEqual(list(output), ["valid", "null", "relative", "invalid-date", "no-mailbox", "unknown",
                                        "empty", "valid-last"])
        self.assertNotIn("future", output["valid"])
        self.assertEqual(output["valid-last"], output["valid"])
        self.assertIn("future", output["null"])
        self.assertIn("future", output["invalid-date"])
        self.assertIn("undated", output["null"])
        self.assertNotIn("foreign", output["null"])
        self.assertEqual(output["relative"], ["anchor-yesterday"])
        self.assertEqual([output[key] for key in ("no-mailbox", "unknown", "empty")], [[], [], []])
        *warnings, report = stderr.getvalue().splitlines()
        summary = json.loads(report)
        self.assertEqual(summary["warnings"], {"unreadable_row": 2, "missing_query_id": 1, "missing_date": 2,
                                               "invalid_date": 1, "missing_mailbox": 1, "unknown_mailbox": 1,
                                               "empty_query": 1, "duplicate_query_id": 1})
        self.assertEqual(len(warnings), sum(summary["warnings"].values()))
        self.assertEqual((summary["queries"], summary["empty"]), (8, 3))


if __name__ == "__main__":
    unittest.main()
