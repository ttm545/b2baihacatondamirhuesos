import json
import os
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from vk_mail_search.api import create_app


class MailApiTest(unittest.TestCase):
    def setUp(self):
        directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        letters = [
            ("a", "2028-09-25", "one", "От: Касса\nТема: Билет\n\nВаши билеты готовы."),
            ("b", None, "one", "Тема: Заказ\n<p>Билет без даты</p>"),
            ("c", "2028-09-27", "one", "Билеты завтра"),
            ("d", "2028-09-25", "two", "Билеты соседа"),
        ]
        rows = [dict(doc_id=i, doc_dt=dt, synthetic_user=box, doc=body) for i, dt, box, body in letters]
        (directory / "documents.jsonl").write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
        self.enterContext(patch.dict(os.environ, VK_DATA_DIR=str(directory)))
        os.environ.pop("VK_SEARCHER", None)
        self.client = self.enterContext(TestClient(create_app()))

    def search(self, **params):
        return self.client.get("/api/search", params={"q": "билет", "mailbox": "one", **params})

    def test_mailbox_and_reader(self):
        self.assertEqual(self.client.get("/api/mailboxes").json(),
                         [{"mailbox": "one", "count": 3, "latest": "2028-09-27"}, {"mailbox": "two", "count": 1, "latest": "2028-09-25"}])
        rows = self.client.get("/api/mailboxes/one/docs?offset=1&limit=2").json()
        self.assertEqual([r["doc_id"] for r in rows], ["a", "b"])
        self.assertEqual(rows[1]["snippet"], "Билет без даты")
        doc = self.client.get("/api/docs/a").json()
        self.assertEqual(doc["subject"], "Билет")
        self.assertTrue(doc["doc"].startswith("От:"))

    def test_ranked_search(self):
        data = self.search(q="БИЛЕТАМИ вчера", as_of="2028-09-26").json()
        self.assertEqual([r["in_period"] for r in data["results"]], [True, False])
        self.assertEqual(data["topic"], "билетами")
        self.assertEqual(data["period"], "25.09.2028")
        self.assertGreaterEqual(data["took_ms"], 0)
        self.assertEqual([r["doc_id"] for r in data["results"]], ["a", "b"])
        self.assertEqual(data["results"][0]["snippet"], "Ваши билеты готовы.")
        self.assertEqual(len(self.search(k=1).json()["results"]), 1)
        self.assertEqual(self.search(q="уникальнейшее").json()["results"], [])

    def test_period_and_limits(self):
        self.assertIsNone(self.search().json()["period"])
        self.assertIsNone(self.search().json()["period_total"])
        self.assertEqual(self.search(q="сегодня").json()["period"], date.today().strftime("%d.%m.%Y"))
        self.assertEqual(self.search(k=100).status_code, 200)
        for args in ({"k": 101}, {"k": 0}, {"q": " "}, {"as_of": "bad"}):
            self.assertEqual(self.search(**args).status_code, 422)
        self.assertEqual(self.search(mailbox="missing").status_code, 404)
        for path in ("/api/docs/missing", "/api/mailboxes/missing/docs"):
            self.assertEqual(self.client.get(path).status_code, 404)

    def test_similar_letters(self):
        rows = self.client.get("/api/docs/a/similar").json()
        self.assertEqual({row["doc_id"] for row in rows}, {"b", "c"})
        self.assertEqual(self.client.get("/api/docs/a/similar?k=1").json(), rows[:1])
        self.assertEqual(self.client.get("/api/docs/d/similar").json(), [])
        self.assertEqual(self.client.get("/api/docs/missing/similar").status_code, 404)
        self.assertEqual(self.client.get("/api/docs/a/similar?k=101").status_code, 422)

    def test_similar_ignores_subject_date_and_identifiers(self):
        with tempfile.TemporaryDirectory() as root:
            rows = [
                ("a", "2028-09-13", "Погрузка контейнера SYN-123 13.09.2028"),
                ("b", "2028-09-12", "Погрузка контейнера SYN-456 12.09.2028"),
                ("c", "2028-09-13", "Автовизит SYN-123 13.09.2028"),
            ]
            docs = [dict(doc_id=i, doc_dt=dt, synthetic_user="one", doc="Тема: " + s) for i, dt, s in rows]
            (Path(root) / "documents.jsonl").write_text("\n".join(map(json.dumps, docs)), encoding="utf-8")
            with TestClient(create_app(root)) as client:
                self.assertEqual(client.get("/api/docs/a/similar?k=1").json()[0]["doc_id"], "b")

    def test_ui_assets(self):
        for path in ("/", "/static/mail.js", "/static/mail.css"):
            self.assertEqual(self.client.get(path).status_code, 200)

    def test_digest_respects_box_cutoff_and_limit(self):
        params = {"from": "2028-09-25", "to": "2028-09-27", "as_of": "2028-09-26"}
        digest = self.client.get("/api/mailboxes/one/digest", params=params).json()
        self.assertEqual(digest["total"], 1)
        self.assertEqual(digest["groups"], [{"kind": "Другие письма", "count": 1}])
        rows = self.client.get("/api/mailboxes/one/digest", params=dict(params, kind="Другие письма")).json()["results"]
        self.assertEqual([row["doc_id"] for row in rows], ["a"])
        data = self.search(q="за последние 7 дней", as_of="2028-09-27", k=1).json()
        self.assertEqual(data["period_total"], 2)
        self.assertEqual(data["period_bounds"], {"from": "2028-09-21", "to": "2028-09-27"})
        self.assertEqual(len(data["results"]), 1)
        route = "/api/mailboxes/one/digest"
        self.assertEqual(self.client.get(route, params=dict(params, as_of="2028-09-24")).json()["total"], 0)
        self.assertEqual(self.client.get(route, params=dict(params, to="2028-09-24")).status_code, 422)
        self.assertEqual(self.client.get("/api/mailboxes/missing/digest", params=params).status_code, 404)

    def test_digest_groups_structured_notifications_with_different_metadata(self):
        with tempfile.TemporaryDirectory() as root:
            rows = [
                ("a", "one", "Тема: Анна Ивановна Иванова: Погрузка контейнера SYN-123, дата 13.09.2028"),
                ("b", "one", "Тема: Борис Петрович Петров: контейнер SYN-456, дата 13.09.2028\nКонтейнер погружен."),
                ("c", "one", "Тема: Создание автовизита для контейнера SYN-789"),
                ("d", "one", "Тема: Разрешение таможни, контейнер SYN-321"),
                ("e", "two", "Тема: Погрузка контейнера SYN-111"),
            ]
            docs = [dict(doc_id=i, doc_dt="2028-09-13", synthetic_user=box, doc=body) for i, box, body in rows]
            (Path(root) / "documents.jsonl").write_text("\n".join(map(json.dumps, docs)), encoding="utf-8")
            with TestClient(create_app(root)) as client:
                path = "/api/mailboxes/one/digest"
                params = {"from": "2028-09-13", "to": "2028-09-13"}
                digest = client.get(path, params=params).json()
                self.assertEqual(digest["total"], 4)
                self.assertEqual({group["kind"]: group["count"] for group in digest["groups"]},
                                 {"Погрузка контейнера": 2, "Автовизит": 1, "Таможня": 1})
                selected = client.get(path, params=dict(params, kind="Погрузка контейнера", limit=1)).json()
                self.assertEqual([row["doc_id"] for row in selected["results"]], ["a"])
