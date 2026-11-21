"""Mailbox-scoped BM25 with cached Russian lemmatization."""

from __future__ import annotations

import heapq
import html
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache

from .data import Document

TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
RUSSIAN = re.compile(r"[а-я]+")
TAGS = re.compile(r"</?[a-z][^>]*>", re.I)


@dataclass(slots=True)
class MailboxIndex:
    postings: dict[str, list[tuple[Document, float]]]
    idf: dict[str, float]


class BM25Searcher:
    def __init__(self, documents: Iterable[Document], *, k1: float = 1.2, b: float = 0.75):
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("Некорректные параметры BM25.")
        try:
            from pymorphy3 import MorphAnalyzer
        except ImportError as exc:
            raise RuntimeError("Установите лемматизатор: python -m pip install -r requirements-baseline.txt") from exc

        analyzer = MorphAnalyzer(lang="ru")

        @lru_cache(maxsize=100_000)
        def normalize(token: str) -> str:
            if len(token) < 3 or not RUSSIAN.fullmatch(token):
                return token
            forms = analyzer.parse(token)
            return forms[0].normal_form.replace("ё", "е") if forms else token

        def tokenize(text: str) -> list[str]:
            return [normalize(token) for token in TOKEN.findall(text.casefold().replace("ё", "е"))]

        self._tokenize = tokenize
        by_mailbox: dict[str, list[tuple[Document, Counter[str]]]] = defaultdict(list)
        for doc in documents:
            text = html.unescape(TAGS.sub(" ", doc.doc))
            by_mailbox[doc.synthetic_user].append((doc, Counter(tokenize(text))))
        self._mailboxes = {}
        for name, entries in by_mailbox.items():
            postings = defaultdict(list)
            lengths = [sum(counts.values()) for _, counts in entries]
            count = len(entries)
            avg_len = sum(lengths) / count or 1.0
            for (doc, counts), length in zip(entries, lengths, strict=True):
                norm = k1 * (1 - b + b * length / avg_len)
                for term, frequency in counts.items():
                    postings[term].append((doc, frequency * (k1 + 1) / (frequency + norm)))
            idf = {term: math.log1p((count - len(hits) + 0.5) / (len(hits) + 0.5))
                   for term, hits in postings.items()}
            self._mailboxes[name] = MailboxIndex(postings, idf)

    def search(self, query: str, mailbox: str, as_of: str | None, k: int) -> list[tuple[str, float]]:
        index = self._mailboxes.get(mailbox)
        if not index or k <= 0:
            return []
        scores: dict[str, float] = defaultdict(float)
        for term in dict.fromkeys(self._tokenize(query)):
            hits = index.postings.get(term)
            if not hits:
                continue
            idf = index.idf[term]
            for doc, weight in hits:
                if as_of is None or doc.doc_dt is None or doc.doc_dt <= as_of:
                    scores[doc.doc_id] += idf * weight
        return heapq.nsmallest(k, scores.items(), key=lambda item: (-item[1], item[0]))
