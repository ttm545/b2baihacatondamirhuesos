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
RUSSIAN_LETTERS = "абвгдежзийклмнопрстуфхцчшщъыьэюя"
TAGS = re.compile(r"</?[a-z][\w:-]*(?=[\s/>])[^>]*>", re.I | re.ASCII)
FUNCTION_POS = {"PREP", "CONJ", "PRCL", "INTJ"}


def _prefix_key(lemma: str) -> str | None:
    if len(lemma) >= 6 and RUSSIAN.fullmatch(lemma):
        return "~" + lemma[:5]
    return None


def _one_edit_words(word: str):
    for i, original in enumerate(word):
        left, right = word[:i], word[i + 1:]
        yield left + right
        if right and original != right[0]:
            yield left + right[0] + original + right[1:]
        for letter in RUSSIAN_LETTERS:
            if letter != original:
                yield left + letter + right

    for i in range(len(word) + 1):
        left, right = word[:i], word[i:]
        for letter in RUSSIAN_LETTERS:
            yield left + letter + right


@dataclass(slots=True)
class MailboxIndex:
    postings: dict[str, list[tuple[Document, float]]]
    idf: dict[str, float]


class BM25Searcher:
    def __init__(self, documents: Iterable[Document], *, k1: float = 1.2, b: float = 0.75,
                 subject_weight: float = 2.0, prefix_weight: float = 0.15,
                 function_weight: float = 0.1, subject_bonus: float = 0.5):
        parameters = (k1, b, subject_weight, prefix_weight, function_weight, subject_bonus)
        if (not all(math.isfinite(value) for value in parameters)
                or k1 <= 0 or not 0 <= b <= 1 or subject_weight < 1 or not 0 <= prefix_weight <= 1
                or not 0 <= function_weight <= 1 or subject_bonus < 0):
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

        @lru_cache(maxsize=2048)
        def is_function_word(term: str) -> bool:
            forms = analyzer.parse(term) if RUSSIAN.fullmatch(term) else []
            return bool(forms and forms[0].tag.POS in FUNCTION_POS)

        self._tokenize = tokenize
        self._normalize = normalize
        self._is_known_word = analyzer.word_is_known
        self._is_function_word = is_function_word
        self._prefix_weight = prefix_weight
        self._function_weight = function_weight
        self._subject_bonus = subject_bonus
        self._subject_terms: dict[str, set[str]] = {}
        by_mailbox: dict[str, list[tuple[Document, dict[str, float], float]]] = defaultdict(list)
        for doc in documents:
            text = html.unescape(TAGS.sub(" ", doc.doc)) if HTML.search(doc.doc) else doc.doc
            counts: dict[str, float] = dict(Counter(tokenize(text)))
            header = HEADER.search(text) if subject_weight > 1 or subject_bonus else None
            header_terms = tokenize(header[1]) if header else []
            if subject_weight > 1:
                for term in header_terms:
                    counts[term] = counts.get(term, 0.0) + subject_weight - 1
            if subject_bonus:
                # Some exports omit the header but put a short subject on the first line.
                if not header:
                    first_line = text.strip().partition("\n")[0].strip()
                    header_terms = tokenize(first_line) if len(first_line) <= 160 else []
                terms = set(header_terms)
                if prefix_weight:
                    terms.update(prefix for term in tuple(terms) if (prefix := _prefix_key(term)))
                self._subject_terms[doc.doc_id] = terms
            # Prefix keys are auxiliary signals and do not count towards document length.
            length = sum(counts.values())
            if prefix_weight:
                for term, frequency in list(counts.items()):
                    if prefix := _prefix_key(term):
                        counts[prefix] = counts.get(prefix, 0.0) + frequency
            by_mailbox[doc.synthetic_user].append((doc, counts, length))
        self._document_count = sum(len(entries) for entries in by_mailbox.values())
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

        def postings(term):
            if term in index.postings:
                yield index.postings[term], index.idf[term]

        return self._search(query, as_of, k, postings)

    def _document_frequency(self, term: str) -> int:
        return sum(len(index.postings.get(term, ())) for index in self._mailboxes.values())

    def correct_archive_topic(self, topic: str) -> str:
        """Correct unknown Russian words using one edit and the archive vocabulary."""
        def correct(match: re.Match) -> str:
            original = match[0]
            word = original.casefold().replace("ё", "е")
            if not (4 <= len(word) <= 24 and RUSSIAN.fullmatch(word)):
                return original
            if self._is_known_word(word):
                return original
            lemma = self._normalize(word)
            if self._document_frequency(lemma) or (
                self._prefix_weight and len(lemma) >= 5 and self._document_frequency("~" + lemma[:5])
            ):
                return original
            # Use the typed form for edits. The analyzer may guess a new lemma for a typo.
            candidates = {self._normalize(form) for form in set(_one_edit_words(word))
                          if self._is_known_word(form)}
            best = heapq.nlargest(2, ((count, form) for form in candidates
                                      if (count := self._document_frequency(form))))
            if best and (len(best) == 1 or best[0][0] > best[1][0]):
                return best[0][1]
            return original

        return TOKEN.sub(correct, topic)

    def search_all(self, query: str, as_of: str | None, k: int) -> list[tuple[str, float]]:
        """Reuse postings with corpus-wide document frequencies."""
        if k <= 0:
            return []

        def postings(term):
            groups = [index.postings[term] for index in self._mailboxes.values() if term in index.postings]
            matches = sum(len(group) for group in groups)
            idf = math.log1p((self._document_count - matches + 0.5) / (matches + 0.5))
            for group in groups:
                yield group, idf

        return self._search(query, as_of, k, postings, allow_short_prefix=True)

    def _search(self, query, as_of, k, postings, *, allow_short_prefix=False):
        scores: dict[str, float] = defaultdict(float)
        query_weights = dict.fromkeys(self._tokenize(query), 1.0)
        topical = set()
        if self._function_weight < 1 or self._subject_bonus:
            topical = {term for term in query_weights if not self._is_function_word(term)}
        if topical:
            for term in query_weights.keys() - topical:
                query_weights[term] = self._function_weight
        subject_prefixes = set()
        if self._prefix_weight:
            for term in tuple(query_weights):
                prefix = _prefix_key(term)
                if allow_short_prefix and len(term) == 5 and RUSSIAN.fullmatch(term):
                    prefix = "~" + term
                if prefix:
                    query_weights[prefix] = max(query_weights.get(prefix, 0.0),
                                                self._prefix_weight * query_weights[term])
                    if term in topical:
                        subject_prefixes.add(prefix)
        for term, query_weight in query_weights.items():
            if not query_weight:
                continue
            for hits, idf in postings(term):
                for doc, weight in hits:
                    if as_of is None or doc.doc_dt is None or doc.doc_dt <= as_of:
                        scores[doc.doc_id] += query_weight * idf * weight
        if self._subject_bonus and topical:
            for doc_id in scores:
                terms = self._subject_terms[doc_id]
                matches = len(topical & terms) + 0.25 * len(subject_prefixes & terms)
                scores[doc_id] *= 1 + self._subject_bonus * matches
        return heapq.nsmallest(k, scores.items(), key=lambda item: (-item[1], item[0]))
