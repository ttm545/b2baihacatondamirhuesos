"""Rank dated mail by evidence in its subject and opening text."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from html import unescape
from math import log1p, sqrt

from .bm25 import BM25Searcher, TAGS
from .data import Document, HEADER, HTML, SENDER
from .dated import DatedSearcher, Hits
from .query_parser import ParsedQuery, parse_query


_REQUEST_WORDS = frozenset({
    "быть", "весь", "где", "дать", "искать", "какой", "который", "находить", "найти",
    "нужный", "ответ", "письмо", "показать", "получить", "почта", "приходить", "прийти",
    "прислать", "сообщение", "уведомление", "хотеть", "этот",
})


@lru_cache(maxsize=32_768)
def _trigrams(term: str) -> frozenset[str]:
    return frozenset(term[i:i + 3] for i in range(len(term) - 2))


def _fuzzy_match(term: str, words: set[str]) -> float:
    if len(term) < 4:
        return 0.0
    trigrams = _trigrams(term)
    best = 0.0
    for word in words:
        if len(word) < 4 or abs(len(word) - len(term)) > 5:
            continue
        overlap = len(trigrams & _trigrams(word))
        similarity = 2 * overlap / (len(trigrams) + len(_trigrams(word)))
        if similarity > best:
            best = similarity
    return best if best >= 0.40 else 0.0


@dataclass(slots=True)
class _Fields:
    subject: list[str]
    body: list[str]
    subject_terms: set[str]
    lead_terms: set[str]
    counts: Counter[str]


class PeriodRanker(DatedSearcher):
    """A bounded reranker with unchanged date and mailbox boundaries."""

    def __init__(self, documents: Mapping[str, Document], *, mode: str = "learned", depth: int = 60):
        self.mode = mode
        self.depth = depth
        self.docs = documents
        lexical = BM25Searcher(documents.values())
        super().__init__(lexical, documents.values())
        self._tokenize = lexical._tokenize
        self._function_word = lexical._is_function_word
        self._fields: dict[str, _Fields] = {}
        self._frequency: Counter[str] = Counter()
        for doc in documents.values():
            text = unescape(TAGS.sub(" ", doc.doc)) if HTML.search(doc.doc) else doc.doc
            header = HEADER.search(text)
            subject = self._tokenize(header[1] if header else doc.subject)[:50]
            if header:
                text = text[:header.start()] + text[header.end():]
            text = "\n".join(line for line in text.lstrip().splitlines() if not SENDER.match(line))
            body = self._tokenize(text[:5000])[:220]
            subject_set, lead_set = set(subject), set(body[:60])
            counts = Counter(body)
            self._fields[doc.doc_id] = _Fields(subject, body, subject_set, lead_set, counts)
            self._frequency.update(subject_set | counts.keys())

    def topic_terms(self, topic: str) -> list[str]:
        terms = list(dict.fromkeys(self._tokenize(topic[:1000])))[:20]
        content = [word for word in terms if word not in _REQUEST_WORDS and not self._function_word(word)]
        return content or terms

    def _weights(self, terms: list[str]) -> list[float]:
        # A square root tempers uncommon spellings and preserves a contribution from common nouns.
        return [sqrt(log1p(len(self.docs) / (self._frequency[word] + 1))) for word in terms]

    def features(self, terms: list[str], pool: list[tuple[str, float]]) -> list[tuple[float, ...]]:
        weights = self._weights(terms)
        normalizer = sum(weights) or 1.0
        best_score = max((score for _, score in pool), default=0.0) or 1.0
        output = []
        for index, (doc_id, score) in enumerate(pool):
            fields = self._fields[doc_id]
            subject = lead = body = fuzzy = saturation = field_coverage = 0.0
            for term, weight in zip(terms, weights, strict=True):
                in_subject = term in fields.subject_terms
                in_lead = term in fields.lead_terms
                frequency = fields.counts[term]
                subject_match = float(in_subject) or _fuzzy_match(term, fields.subject_terms) * .7
                lead_match = float(in_lead) or _fuzzy_match(term, fields.lead_terms) * .5
                subject += weight * in_subject
                lead += weight * in_lead
                body += weight * bool(frequency)
                fuzzy += weight * max(subject_match, lead_match)
                field_coverage += weight * max(subject_match, .77 * lead_match, .52 * bool(frequency))
                tf = 3 * in_subject + 1.4 * in_lead + frequency / (1 + len(fields.body) / 180)
                saturation += weight * tf / (tf + 1.2)
            proximity = self._proximity(terms, fields.subject, fields.body[:100])
            output.append((score / best_score, 1 / (index + 1), subject / normalizer,
                           lead / normalizer, body / normalizer, fuzzy / normalizer,
                           saturation / normalizer, field_coverage / normalizer, proximity))
        return output

    @staticmethod
    def _proximity(terms: list[str], subject: list[str], lead: list[str]) -> float:
        if len(terms) < 2:
            return 0.0
        hits = 0.0
        for left, right in zip(terms, terms[1:]):
            for tokens, weight in ((subject, 1.0), (lead, .6)):
                if any(right in tokens[i + 1:i + 5] for i, token in enumerate(tokens) if token == left):
                    hits += weight
                    break
        return hits / (len(terms) - 1)

    def candidates(self, query: str, mailbox: str, as_of: str | None, k: int) -> tuple[ParsedQuery, Hits, Hits]:
        parsed = parse_query(query, as_of)
        pool, tail = self._candidates_parsed(parsed, mailbox, as_of, k)
        return parsed, pool, tail

    def _candidates_parsed(self, parsed: ParsedQuery, mailbox: str, as_of: str | None, k: int) -> tuple[Hits, Hits]:
        if parsed.period is None or not parsed.topic:
            return [], super().search_parsed(parsed, mailbox, as_of, k)
        day_count = len(self.in_period(mailbox, parsed, as_of))
        if day_count <= 1:
            return [], super().search_parsed(parsed, mailbox, as_of, k)
        head = min(self.depth, day_count)
        ranked = super().search_parsed(parsed, mailbox, as_of, max(k, head))
        return ranked[:head], ranked[head:]

    def search_parsed(self, parsed: ParsedQuery, mailbox: str, as_of: str | None, k: int) -> Hits:
        if k <= 0:
            return []
        pool, tail = self._candidates_parsed(parsed, mailbox, as_of, k)
        if not pool:
            return tail
        terms = self.topic_terms(parsed.topic)
        features = self.features(terms, pool)
        revised = self.rank(pool, features, self.mode)
        return (revised + tail)[:k]

    @classmethod
    def rank(cls, pool, features, mode):
        weights = PROFILES[mode]
        if mode == "guard" and features[0][2] >= .65:
            weights = PROFILES["conservative"]
        return cls.combine(pool, features, weights)

    @staticmethod
    def combine(pool: list[tuple[str, float]], features: list[tuple[float, ...]], weights: tuple[float, ...]):
        scored = [(doc_id, sum(value * weight for value, weight in zip(row, weights, strict=True)))
                  for (doc_id, _), row in zip(pool, features, strict=True)]
        return sorted(scored, key=lambda pair: -pair[1])


# BM25, original rank, subject, lead, body, fuzz, saturated field frequency, field coverage, proximity.
PROFILES = {
    "cover": (.28, 0, .12, .02, .02, .05, 0, .46, .05),
    "coverage": (0, 0, .13, .03, .03, .12, 0, .63, .06),
    "frequency": (0, 0, .05, 0, 0, 0, .88, 0, .07),
    "subject": (.18, 0, .55, 0, 0, .12, 0, .12, .03),
    "conservative": (.55, 0, .08, 0, .02, .02, 0, .29, .04),
    "guard": (.28, 0, .12, .02, .02, .05, 0, .46, .05),
    # Pairwise fit on the tune families; reproduce with scripts.fit_period, strength .003.
    "learned": (.181024, .095871, .021244, .007053, .000729, .164946, 0, .465536, .063597),
}
