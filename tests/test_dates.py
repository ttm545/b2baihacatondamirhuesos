import unittest
from datetime import date

from vk_mail_search.data import Document
from vk_mail_search.dated import DatedSearcher
from vk_mail_search.query_parser import parse_query

TODAY = "2026-11-21"  # суббота


def period(query, today=TODAY):
    parsed = parse_query(query, today)
    return (parsed.period.start.isoformat(), parsed.period.end.isoformat()) if parsed.period else None


class ParserTests(unittest.TestCase):
    def test_benchmark_forms(self):
        parsed = parse_query("Найди уведомления о сбоях модуля электронных рецептов, полученные 25 сентября 2028 года.",
                             "2028-09-26")
        self.assertEqual(parsed.period.start, date(2028, 9, 25))
        self.assertEqual(parsed.topic, "уведомления о сбоях модуля электронных рецептов")
        parsed = parse_query("Покажи письма о прибытии автомашин, полученные вчера.", "2028-09-21")
        self.assertEqual((parsed.period.start, parsed.topic), (date(2028, 9, 20), "письма о прибытии автомашин"))

    def test_relative_days_and_ranges(self):
        cases = {
            "сегодня": ("2026-11-21", "2026-11-21"),
            "позавчера": ("2026-11-19", "2026-11-19"),
            "3 дня назад": ("2026-11-18", "2026-11-18"),
            "за последние 7 дней": ("2026-11-15", "2026-11-21"),
            "за последнюю неделю": ("2026-11-15", "2026-11-21"),
            "на прошлой неделе": ("2026-11-09", "2026-11-15"),
            "на этой неделе": ("2026-11-16", "2026-11-21"),
            "неделю назад": ("2026-11-09", "2026-11-15"),
            "в прошлом месяце": ("2026-10-01", "2026-10-31"),
            "в этом месяце": ("2026-11-01", "2026-11-21"),
            "в сентябре": ("2026-09-01", "2026-09-30"),
            "в декабре": ("2025-12-01", "2025-12-31"),
            "в мае 2025": ("2025-05-01", "2025-05-31"),
            "в 2025 году": ("2025-01-01", "2025-12-31"),
            "с 1 по 15 ноября": ("2026-11-01", "2026-11-15"),
            "с 28 декабря по 3 января": ("2025-12-28", "2026-01-03"),
            "15.10": ("2026-10-15", "2026-10-15"),
            "5.3.25": ("2025-03-05", "2025-03-05"),
            "2025-03-01": ("2025-03-01", "2025-03-01"),
            "письма от 30 ноября": ("2025-11-30", "2025-11-30"),
            "отчёт 5 марта 2025 г. бухгалтерия": ("2025-03-05", "2025-03-05"),
        }
        for query, expected in cases.items():
            with self.subTest(query=query):
                self.assertEqual(period(query), expected)

    def test_no_false_dates(self):
        for query in ("маршрут доставки", "версия 2.4 релиз", "письма 31 ноября", "март план", "заказ 1773"):
            with self.subTest(query=query):
                self.assertIsNone(period(query))

    def test_topic_cleanup(self):
        self.assertEqual(parse_query("письма от 3 мая", TODAY).topic, "письма")
        self.assertEqual(parse_query("покажи письма за вчера", TODAY).topic, "письма")
        self.assertEqual(parse_query("вчера", TODAY).topic, "")
        self.assertEqual(parse_query("маршрут доставки", TODAY).topic, "маршрут доставки")


class Ranker:
    def __init__(self, hits):
        self.hits, self.calls = hits, []

    def search(self, query, mailbox, as_of, k):
        self.calls.append(query)
        return self.hits[:k]


class DatedSearcherTests(unittest.TestCase):
    def setUp(self):
        self.docs = [
            Document("old_match", "box", "2026-11-10", ""),
            Document("day_match", "box", "2026-11-20", ""),
            Document("day_other", "box", "2026-11-20", ""),
            Document("day_late", "box", "2026-11-20", ""),
            Document("undated", "box", None, ""),
            Document("foreign", "other", "2026-11-20", ""),
        ]

    def test_period_letters_first_then_the_rest(self):
        ranker = Ranker([("old_match", 9.0), ("day_match", 5.0), ("undated", 4.0), ("day_other", 1.0)])
        hits = DatedSearcher(ranker, self.docs).search("счёт вчера", "box", "2026-11-21", 10)
        self.assertEqual([d for d, _ in hits], ["day_match", "day_other", "day_late", "old_match", "undated"])
        self.assertEqual(ranker.calls, ["счет"])

    def test_without_date_the_inner_ranking_is_kept(self):
        ranker = Ranker([("old_match", 9.0), ("day_match", 5.0)])
        hits = DatedSearcher(ranker, self.docs).search("счёт", "box", "2026-11-21", 1)
        self.assertEqual(hits, [("old_match", 9.0)])

    def test_date_only_query_and_as_of(self):
        ranker = Ranker([])
        searcher = DatedSearcher(ranker, self.docs)
        self.assertEqual({d for d, _ in searcher.search("вчера", "box", "2026-11-21", 10)},
                         {"day_match", "day_other", "day_late"})
        self.assertEqual(ranker.calls, [])
        self.assertEqual(searcher.search("в ноябре", "box", "2026-11-15", 10), [("old_match", 0.0)])


if __name__ == "__main__":
    unittest.main()
