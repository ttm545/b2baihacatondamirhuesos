import asyncio
from contextlib import asynccontextmanager, redirect_stderr
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from solutions.semantic_ensemble.predict import predict
from vk_mail_search.data import Document


class SemanticPredictTest(TestCase):
    def test_export_uses_the_api_engine_and_preserves_input_order(self):
        docs = {
            "right": Document("right", "box", "2028-09-13", "Выгрузка"),
            "outside": Document("outside", "other", "2028-09-13", "Другое письмо"),
        }

        class Engine:
            def search(self, query, mailbox, as_of, k):
                return [("outside", 2.), ("right", 1.), ("right", 0.)]

        @asynccontextmanager
        async def lifespan(app):
            app.state.docs = docs
            app.state.searcher = Engine()
            yield

        app = SimpleNamespace(router=SimpleNamespace(lifespan_context=lifespan), state=SimpleNamespace())
        base = {"query": "выгрузка", "synthetic_user": "box", "query_date": "2028-09-14"}
        rows = [
            dict(base, query_id="3"),
            dict(base, query_id="2", query_date="bad-date"),
            dict(base, query_id="1", synthetic_user="unknown"),
        ]
        with TemporaryDirectory() as directory, redirect_stderr(StringIO()) as errors:
            corpus = Path(directory)
            queries, output = corpus / "queries.jsonl", corpus / "predictions.jsonl"
            queries.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            with patch("solutions.semantic_ensemble.predict.create_app", return_value=app) as create:
                report = asyncio.run(predict(queries, output, corpus, 100))
            data = [json.loads(line) for line in output.read_text().splitlines()]
        create.assert_called_once_with(corpus)
        self.assertEqual(data, [{"query_id": "3", "doc_ids": ["right"]},
                                {"query_id": "2", "doc_ids": []}, {"query_id": "1", "doc_ids": []}])
        self.assertEqual(report["empty"], 2)
        self.assertIn("некорректная дата", errors.getvalue())
