"""Extract a received-date period and the topic from a Russian mail search query.

Relative expressions are resolved against the query date (``query_date`` in the
benchmark, today in the UI). A period is an inclusive [start, end] range of days.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta

MONTHS = ("январ[ьяе]", "феврал[ьяе]", "март[ае]?", "апрел[ьяе]", "ма[йяе]", "июн[ьяе]", "июл[ьяе]",
          "август[ае]?", "сентябр[ьяе]", "октябр[ьяе]", "ноябр[ьяе]", "декабр[ьяе]")
MONTH = r"(?P<{name}>" + "|".join(MONTHS) + r")\b"
NUMBERS = {"один": 1, "одну": 1, "два": 2, "две": 2, "три": 3, "четыре": 4, "пять": 5, "шесть": 6,
           "семь": 7, "восемь": 8, "девять": 9, "десять": 10}
COUNT = r"(?P<n>\d{1,3}|" + "|".join(NUMBERS) + ")"
YEAR = r"(?:\s+(?P<{name}>\d{{4}})(?:\s*(?:года|году|год|г\.?)(?!\w))?)?"


@dataclass(frozen=True, slots=True)
class Period:
    start: date
    end: date

    def contains(self, value: str | None) -> bool:
        return value is not None and self.start.isoformat() <= value <= self.end.isoformat()

    def label(self) -> str:
        fmt = "%d.%m.%Y"
        if self.start == self.end:
            return self.start.strftime(fmt)
        return f"{self.start.strftime(fmt)} – {self.end.strftime(fmt)}"


@dataclass(frozen=True, slots=True)
class ParsedQuery:
    text: str
    topic: str
    period: Period | None


def _month(word: str) -> int:
    for index, form in enumerate(MONTHS, 1):
        if re.fullmatch(form, word):
            return index
    raise ValueError(word)


def _count(word: str) -> int:
    return int(word) if word.isdigit() else NUMBERS[word]


def _last_day(year: int, month: int) -> date:
    return (date(year + month // 12, month % 12 + 1, 1)) - timedelta(days=1)


def _past(day: int, month: int, year: str | None, today: date) -> date | None:
    """A date without a year means its most recent occurrence not after today."""
    if year:
        try:
            return date(int(year), month, day)
        except ValueError:
            return None
    for year_back in range(9):  # 29 февраля without a year: the latest leap year
        try:
            value = date(today.year - year_back, month, day)
        except ValueError:
            continue
        if value <= today:
            return value
    return None


def _week_start(value: date) -> date:
    return value - timedelta(days=value.weekday())


WEEKDAYS = ("понедельн", "вторн", "сред", "четверг", "пятниц", "суббот", "воскресен")
WEEKDAY = r"(?P<wd>понедельник\w*|вторник\w*|сред[аыу]|четверг\w*|пятниц[аыу]|суббот[аыу]|воскресень[еяю])"
NUMERIC_DATE = r"\d{4}-\d{2}-\d{2}|\d{1,2}\.\d{1,2}(?:\.\d{2,4})?"


def _weekday(word: str) -> int:
    return next(index for index, stem in enumerate(WEEKDAYS) if word.startswith(stem))


def _last_weekday(today: date, weekday: int) -> date:
    return today - timedelta(days=(today.weekday() - weekday) % 7)


def _numeric(text: str, today: date) -> date | None:
    for pattern, kind in _PATTERNS:
        if kind in ("iso", "dotted"):
            match = pattern.fullmatch(text)
            if match:
                period = _resolve(kind, match, today)
                return period.start if period else None
    return None


_DAY_MONTH = r"(?P<{d}>\d{{1,2}})\s+" + MONTH
_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bс\s+" + _DAY_MONTH.format(d="d1", name="m1") + YEAR.format(name="y1")
                + r"\s+(?:по|до)\s+" + _DAY_MONTH.format(d="d2", name="m2") + YEAR.format(name="y2")), "range"),
    (re.compile(r"\bс\s+(?P<d1>\d{1,2})\s+(?:по|до)\s+" + _DAY_MONTH.format(d="d2", name="m2")
                + YEAR.format(name="y2")), "range_same_month"),
    (re.compile(r"\b(?:с|со|от)\s+(?P<a>" + NUMERIC_DATE + r")\s+(?:по|до)\s+(?P<b>" + NUMERIC_DATE + r")\b"),
     "numeric_range"),
    (re.compile(r"\bс[о]?\s+(?P<word>вчерашн\w*(?:\s+дня)?|вчера|позавчерашн\w*(?:\s+дня)?|позавчера)"), "since_day"),
    (re.compile(r"\bс\s+начала\s+(?P<unit>недели|месяца|года)"), "since_start"),
    (re.compile(r"\b(?:за\s+|на\s+|в\s+)?(?P<prev>прошл\w+\s+|предыдущ\w+\s+)?выходн\w*"
                r"(?:\s+на\s+(?P<which>прошл\w+|предыдущ\w+|эт\w+)\s+неделе)?"), "weekend"),
    (re.compile(r"\bс[о]?\s+" + WEEKDAY), "since_weekday"),
    (re.compile(r"\b(?:в|во)\s+(?P<which>прошл\w+\s+|предыдущ\w+\s+|эт\w+\s+)?" + WEEKDAY), "weekday"),
    (re.compile(r"\b(?P<d>\d{4}-\d{2}-\d{2})\b"), "iso"),
    # "15.10", "15.10.2025", "5.3.25"; a one-digit month needs a year so "версия 2.4" is not a date.
    (re.compile(r"\b(?P<d>\d{1,2})\.(?:(?P<m>\d{2})(?:\.(?P<y>\d{4}|\d{2}))?|(?P<m1>\d)\.(?P<y1>\d{4}|\d{2}))\b"),
     "dotted"),
    (re.compile(r"\b" + _DAY_MONTH.format(d="d", name="m") + YEAR.format(name="y")), "day"),
    (re.compile(r"\bпозавчера\w*"), "day_before_yesterday"),
    (re.compile(r"\bвчера\w*"), "yesterday"),
    (re.compile(r"\bсегодня\w*"), "today"),
    (re.compile(COUNT + r"\s+(?P<unit>дн\w*|день|недел\w*|месяц\w*)\s+назад"), "ago"),
    (re.compile(r"\b(?P<unit>неделю|месяц)\s+назад"), "ago"),
    (re.compile(r"\bза\s+(?:последн\w+\s+)?" + COUNT + r"\s+(?P<unit>дн\w*|день|недел\w*|месяц\w*)"), "last_n"),
    (re.compile(r"\bза\s+последн\w+\s+(?P<unit>недел\w*|месяц\w*|день|год)"), "last_one"),
    (re.compile(r"\b(?:на|за)\s+(?P<which>эт\w+|текущ\w+|прошл\w+|предыдущ\w+)\s+недел\w*"), "week"),
    (re.compile(r"\b(?:в|за)\s+(?P<which>эт\w+|текущ\w+|прошл\w+|предыдущ\w+)\s+месяц\w*"), "month_rel"),
    (re.compile(r"\b(?:в|за)\s+(?P<which>эт\w+|текущ\w+|прошл\w+|предыдущ\w+)\s+год\w*"), "year_rel"),
    (re.compile(r"\b(?:в|за)\s+" + MONTH.format(name="m") + YEAR.format(name="y")), "month"),
    (re.compile(r"\b(?:в|за)\s+(?P<y>\d{4})\s*(?:г\.?|году|год)\b"), "year"),
]

_COMMAND = re.compile(r"^\s*(?:пожалуйста\s+)?(?:найди|найти|покажи|показать|выведи|ищу|поиск|где)\b[\s,:]*",
                      re.I)
_DANGLING = re.compile(r"(?:\s+|^)(?:от|за|в|во|на|с|со|по|до|начиная)$")
_RECEIVED = re.compile(r",?\s*\b(?:полученн|пришедш|присланн|поступивш|отправленн)\w*\s*$", re.I)


_VERSION_BEFORE = re.compile(r"(?:верси\w*|v|ver|релиз\w*|сумм\w*|№|#)\.?\s*$")
_MONEY_AFTER = re.compile(r"^\s*(?:₽|руб|р\.|\$|€|%|usd|eur|rub)")


def _not_a_date(text: str, match: re.Match[str]) -> bool:
    """«версия 15.10», «на сумму 15.10 ₽» are numbers, not dates."""
    return bool(_VERSION_BEFORE.search(text[:match.start()]) or _MONEY_AFTER.match(text[match.end():]))


def _resolve(kind: str, m: re.Match[str], today: date) -> Period | None:
    g = m.groupdict()
    if kind == "range":
        end = _past(int(g["d2"]), _month(g["m2"]), g["y2"], today)
        if end is None:
            return None
        start = _past(int(g["d1"]), _month(g["m1"]), g["y1"] or str(end.year), end)
        if start is None:
            return None
        if start > end:
            start = _past(int(g["d1"]), _month(g["m1"]), str(end.year - 1), end)
        return Period(start, end) if start else None
    if kind == "range_same_month":
        end = _past(int(g["d2"]), _month(g["m2"]), g["y2"], today)
        start = end and _past(int(g["d1"]), end.month, str(end.year), end)
        return Period(start, end) if start and start <= end else None
    if kind == "numeric_range":
        end = _numeric(g["b"], today)
        start = end and _numeric(g["a"], end)
        return Period(start, end) if start and start <= end else None
    if kind == "since_day":
        back = 2 if g["word"].startswith("позавчера") else 1
        return Period(today - timedelta(days=back), today)
    if kind == "since_start":
        unit = g["unit"]
        start = (_week_start(today) if unit.startswith("недел") else
                 today.replace(day=1) if unit.startswith("месяц") else today.replace(month=1, day=1))
        return Period(start, today)
    if kind == "weekend":
        previous = bool(g["prev"]) or bool(g["which"] and g["which"].startswith(("прошл", "предыдущ")))
        saturday = _week_start(today) + timedelta(days=5)
        if previous or saturday > today:
            saturday -= timedelta(days=7)
        return Period(saturday, min(saturday + timedelta(days=1), today))
    if kind == "since_weekday":
        return Period(_last_weekday(today, _weekday(g["wd"])), today)
    if kind == "weekday":
        weekday = _weekday(g["wd"])
        if g["which"] and g["which"].startswith(("прошл", "предыдущ")):
            value = _week_start(today) - timedelta(days=7 - weekday)
        else:
            value = _last_weekday(today, weekday)
        return Period(value, value)
    if kind == "iso":
        try:
            value = date.fromisoformat(g["d"])
        except ValueError:
            return None
        return Period(value, value)
    if kind == "dotted":
        if g["m1"]:
            g["m"], g["y"] = g["m1"], g["y1"]
        year = g["y"]
        if year and len(year) == 2:
            year = "20" + year
        value = _past(int(g["d"]), int(g["m"]), year, today) if 1 <= int(g["m"]) <= 12 else None
        return Period(value, value) if value else None
    if kind == "day":
        value = _past(int(g["d"]), _month(g["m"]), g["y"], today)
        return Period(value, value) if value else None
    if kind in ("today", "yesterday", "day_before_yesterday"):
        value = today - timedelta(days={"today": 0, "yesterday": 1, "day_before_yesterday": 2}[kind])
        return Period(value, value)
    if kind in ("ago", "last_n", "last_one"):
        n = 1 if kind == "last_one" or not g.get("n") else _count(g["n"])
        unit = g["unit"]
        days = n * (7 if unit.startswith("недел") else 30 if unit.startswith("месяц") else
                    365 if unit.startswith("год") else 1)
        if kind == "ago":
            if unit.startswith(("дн", "день")):
                value = today - timedelta(days=n)
                return Period(value, value)
            # "неделю назад": that whole week, "месяц назад": that whole month.
            if unit.startswith("недел"):
                start = _week_start(today - timedelta(days=days))
                return Period(start, start + timedelta(days=6))
            year, month = divmod(today.year * 12 + today.month - 1 - n, 12)
            return Period(date(year, month + 1, 1), _last_day(year, month + 1))
        return Period(today - timedelta(days=days - 1), today)
    if kind == "week":
        start = _week_start(today)
        if g["which"].startswith(("прошл", "предыдущ")):
            start -= timedelta(days=7)
            return Period(start, start + timedelta(days=6))
        return Period(start, today)
    if kind == "month_rel":
        if g["which"].startswith(("прошл", "предыдущ")):
            year, month = divmod(today.year * 12 + today.month - 2, 12)
            return Period(date(year, month + 1, 1), _last_day(year, month + 1))
        return Period(today.replace(day=1), today)
    if kind == "year_rel":
        year = today.year - (1 if g["which"].startswith(("прошл", "предыдущ")) else 0)
        return Period(date(year, 1, 1), min(date(year, 12, 31), today))
    if kind == "month":
        month = _month(g["m"])
        year = int(g["y"]) if g["y"] else today.year if month <= today.month else today.year - 1
        return Period(date(year, month, 1), _last_day(year, month))
    if kind == "year":
        year = int(g["y"])
        return Period(date(year, 1, 1), date(year, 12, 31))
    return None


def parse_query(query: str, today: date | str | None = None) -> ParsedQuery:
    """Find the first date expression; the topic is the query without command, date and 'полученные'."""
    if isinstance(today, str):
        today = date.fromisoformat(today)
    today = today or date.today()
    text = " ".join(query.split())
    lowered = text.casefold().replace("ё", "е")
    best: tuple[int, int, Period] | None = None
    for pattern, kind in _PATTERNS:
        for match in pattern.finditer(lowered):
            if kind == "dotted" and _not_a_date(lowered, match):
                continue
            period = _resolve(kind, match, today)
            if period and (best is None or match.start() < best[0]):
                best = (match.start(), match.end(), period)
            break
    topic = lowered
    if best:
        topic = (lowered[:best[0]] + " " + lowered[best[1]:])
    topic = _COMMAND.sub("", topic)
    topic = re.sub(r"\s+", " ", topic).strip(" ,.;:!?")
    topic = _RECEIVED.sub("", topic).strip(" ,.;:!?")
    topic = _DANGLING.sub("", topic).strip(" ,.;:!?")
    # With a period and nothing else ("письма за вчера") the topic is empty: rank by date only.
    return ParsedQuery(text, topic if best else topic or lowered, best[2] if best else None)
