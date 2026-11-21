"""Mailbox-scoped BM25 with Russian lemmatization and weighted text signals."""

from __future__ import annotations

import heapq
import html
import math
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache

from .data import HEADER, HTML, Document

TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
RUSSIAN = re.compile(r"[а-я]+")
TAGS = re.compile(r"</?[a-z][\w:-]*(?=[\s/>])[^>]*>", re.I | re.ASCII)


def _prefix_key(lemma: str) -> str | None:
    if len(lemma) >= 6 and RUSSIAN.fullmatch(lemma):
        return "~" + lemma[:5]
    return None


@dataclass(slots=True)
class MailboxIndex:
    postings: dict[str, list[tuple[Document, float]]]
    idf: dict[str, float]


class BM25Searcher:
    def __init__(self, documents: Iterable[Document], *, k1: float = 1.2, b: float = 0.75,
                 subject_weight: float = 2.0, prefix_weight: float = 0.15):
        if (not all(math.isfinite(value) for value in (k1, b, subject_weight, prefix_weight))
                or k1 <= 0 or not 0 <= b <= 1 or subject_weight < 1 or not 0 <= prefix_weight <= 1):
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
        self._prefix_weight = prefix_weight
        by_mailbox: dict[str, list[tuple[Document, dict[str, float], float]]] = defaultdict(list)
        for doc in documents:
            text = html.unescape(TAGS.sub(" ", doc.doc)) if HTML.search(doc.doc) else doc.doc
            counts: dict[str, float] = dict(Counter(tokenize(text)))
            if subject_weight > 1 and (header := HEADER.search(text)):
                for term in tokenize(header[1]):
                    counts[term] = counts.get(term, 0.0) + subject_weight - 1
            # Prefix keys are auxiliary signals and do not count towards document length.
            length = sum(counts.values())
            if prefix_weight:
                for term, frequency in list(counts.items()):
                    if prefix := _prefix_key(term):
                        counts[prefix] = counts.get(prefix, 0.0) + frequency
            by_mailbox[doc.synthetic_user].append((doc, counts, length))
        self._mailboxes = {}
        for name, entries in by_mailbox.items():
            postings = defaultdict(list)
            count = len(entries)
            avg_len = sum(length for _, _, length in entries) / count or 1.0
            for doc, counts, length in entries:
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
        query_weights = dict.fromkeys(self._tokenize(query), 1.0)
        if self._prefix_weight:
            for term in tuple(query_weights):
                if prefix := _prefix_key(term):
                    query_weights[prefix] = self._prefix_weight
        for term, query_weight in query_weights.items():
            hits = index.postings.get(term)
            if not hits:
                continue
            idf = index.idf[term]
            for doc, weight in hits:
                if as_of is None or doc.doc_dt is None or doc.doc_dt <= as_of:
                    scores[doc.doc_id] += query_weight * idf * weight
        return heapq.nsmallest(k, scores.items(), key=lambda item: (-item[1], item[0]))
