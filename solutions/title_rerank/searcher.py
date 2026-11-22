"""Mix the period rank and a query-to-email-subject relevance score."""

from collections.abc import Mapping, Sequence
from math import isfinite
from typing import Protocol

from vk_mail_search.data import Document, HEADER
from vk_mail_search.dated import Hits
from vk_mail_search.period_rank import PeriodRanker
from vk_mail_search.query_parser import parse_query


DEPTH = 10
DEFAULT_WEIGHT = 0.35


class SubjectScorer(Protocol):
    def score(self, query: str, titles: Sequence[str]) -> Sequence[float]: ...


def fuse(ranked: Hits, relevance: Mapping[str, float], weight: float) -> Hits:
    if not weight or len(relevance) < 2:
        return ranked
    head_size = min(DEPTH, len(ranked))
    eligible = [(position, doc_id) for position, (doc_id, _) in enumerate(ranked[:head_size])
                if doc_id in relevance]
    if len(eligible) < 2:
        return ranked
    scored = [(doc_id, 1 - position / (head_size - 1) + weight * relevance[doc_id])
              for position, doc_id in eligible]
    result = ranked.copy()
    for (position, _), hit in zip(eligible, sorted(scored, key=lambda hit: -hit[1]), strict=True):
        result[position] = hit
    return result


class SubjectReranker:
    def __init__(self, baseline: PeriodRanker, documents: Mapping[str, Document], model: SubjectScorer,
                 weight: float = DEFAULT_WEIGHT):
        if not 0 <= weight <= 1:
            raise ValueError("Subject weight must be between 0 and 1")
        self.baseline = baseline
        self.documents = documents
        self.inner = baseline.inner
        self.model = model
        self.weight = weight
        self.titles = {doc_id: " ".join(match[1].split())[:360] for doc_id, doc in documents.items()
                       if (match := HEADER.search(doc.doc))}

    def prepare(self, query: str, mailbox: str, as_of: str | None, k: int) -> tuple[Hits, dict[str, float]]:
        parsed = parse_query(query, as_of)
        ranked = self.baseline.search_parsed(parsed, mailbox, as_of, max(DEPTH, k))
        if not parsed.topic:
            return ranked, {}
        titles = {doc_id: self.titles[doc_id] for doc_id, _ in ranked[:DEPTH]
                  if self.titles.get(doc_id) and (
                      parsed.period is None or parsed.period.contains(self.documents[doc_id].doc_dt))}
        distinct = list(dict.fromkeys(titles.values()))
        if len(distinct) < 2:
            return ranked, {}
        scores = self.model.score(parsed.topic, distinct)
        if len(scores) != len(distinct) or not all(isfinite(score) and 0 <= score <= 1 for score in scores):
            raise ValueError("Expected one finite relevance score between 0 and 1 per distinct title")
        by_title = dict(zip(distinct, scores, strict=True))
        return ranked, {doc_id: by_title[subject] for doc_id, subject in titles.items()}

    def search(self, query: str, mailbox: str, as_of: str | None, k: int) -> Hits:
        if k <= 0:
            return []
        if not self.weight:
            return self.baseline.search(query, mailbox, as_of, k)
        ranked, relevance = self.prepare(query, mailbox, as_of, k)
        return fuse(ranked, relevance, self.weight)[:k]
