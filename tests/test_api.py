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

    def test_all_mailboxes_list_and_search(self):
        archive = self.client.get("/api/mailboxes?include_all=true").json()[0]
        self.assertEqual((archive["name"], archive["count"], archive["latest"]),
                         ("Все письма", 4, "2028-09-27"))
        mailbox = archive["mailbox"]
        docs = self.client.get(f"/api/mailboxes/{mailbox}/docs?offset=1&limit=2").json()
        self.assertEqual([(row["doc_id"], row["mailbox"]) for row in docs], [("a", "one"), ("d", "two")])
        hits = self.search(q="билеты вчера", as_of="2028-09-26", mailbox=mailbox).json()["results"]
        self.assertEqual({(row["doc_id"], row["mailbox"]) for row in hits[:2]}, {("a", "one"), ("d", "two")})
        self.assertEqual([row["doc_id"] for row in hits[2:]], ["b"])

    def test_archive_search_prioritizes_text_matches_over_other_mailboxes(self):
        with tempfile.TemporaryDirectory() as root:
            rows = [
                ("a", "first", "О смене системы налогообложения"),
                ("b", "first", "Налоговая декларация"),
                ("c", "second", "Доставка заказа"),
                ("d", "third", "Электронный рецепт"),
            ]
            docs = [dict(doc_id=i, doc_dt="2028-09-25", synthetic_user=box, doc="Тема: " + text)
                    for i, box, text in rows]
            (Path(root) / "documents.jsonl").write_text("\n".join(map(json.dumps, docs)), encoding="utf-8")
            app = create_app(root)
            with TestClient(app) as client:
                with patch.object(app.state.searcher, "search", side_effect=lambda query, box, as_of, k: {
                        "first": [], "second": [("c", 1.0)], "third": [("d", 1.0)],
                }[box]):
                    result = client.get("/api/search", params={"q": "налог", "mailbox": "__all__", "k": 2})
            self.assertEqual(result.status_code, 200)
            self.assertEqual({hit["doc_id"] for hit in result.json()["results"]}, {"a", "b"})

    def test_archive_search_corrects_typos_without_changing_date_or_known_words(self):
        with tempfile.TemporaryDirectory() as root:
            docs = [
                ("a", "2028-09-24", "Тема: Штраф за доставку"),
                ("b", "2028-09-25", "Тема: Придётся оплатить штрафы"),
                ("c", "2028-09-26", "Тема: Новый штраф"),
                ("d", "2028-09-24", "Тема: Шарф для курьера"),
            ]
            rows = [dict(doc_id=doc_id, doc_dt=day, synthetic_user="one", doc=body)
                    for doc_id, day, body in docs]
            (Path(root) / "documents.jsonl").write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
            with TestClient(create_app(root)) as client:
                def search(query):
                    return client.get("/api/search", params={"q": query, "mailbox": "__all__",
                                                             "as_of": "2028-09-25"}).json()

                expected = search("штраф")
                self.assertEqual({row["doc_id"] for row in expected["results"]}, {"a", "b"})
                for typo in ("штроф", "штарф", "штрф", "штрофы"):
                    actual = search(typo)
                    self.assertEqual(actual["topic"], "штраф")
                    self.assertEqual(actual["results"], expected["results"])

                dated = search("штроф за доставку вчера")
                self.assertEqual((dated["topic"], dated["period"]), ("штраф за доставку", "24.09.2028"))
                self.assertEqual(dated["results"][0]["doc_id"], "a")
                self.assertEqual(search("шарф")["results"][0]["doc_id"], "d")

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
