"""Fuse independent rank signals while keeping mailbox and received-date scopes."""

from dataclasses import dataclass, replace
from datetime import date
import json
from pathlib import Path

import numpy as np

from vk_mail_search.query_parser import ParsedQuery, parse_query

from .qwen_context import normalize


@dataclass(frozen=True)
class EnsembleOptions:
    date_mix: tuple[float, float, float]
    date_depth: int
    date_gate: float
    content_mix: tuple[float, float, float, float]

    @classmethod
    def load(cls, path: Path):
        data = json.loads(path.read_text(encoding="utf-8"))
        dated = tuple(float(value) for value in data["date_mix"])
        content = tuple(float(value) for value in data["content_mix"])
        depth, gate = int(data["date_depth"]), float(data["date_gate"])
        if (len(dated) != 3 or len(content) != 4 or not 0 <= depth <= 60 or not 0 <= gate <= 1.01
                or not all(np.isfinite(dated + content)) or any(value < 0 for value in dated + content)
                or sum(dated) <= 0 or sum(content[:3]) <= 0):
            raise ValueError("Invalid ensemble parameters")
        return cls(dated, depth, gate, content)


def inverse_rank(order):
    rank = np.empty(len(order), dtype=np.intp)
    rank[order] = np.arange(len(order))
    return rank


@dataclass
class DatedCandidates:
    ids: list[str]
    original: np.ndarray
    features: np.ndarray
    tail: list[tuple[str, float]]
    confidence: float

    def rank(self, weights, depth, gate, k):
        if not self.ids:
            return self.tail[:k]
        if self.confidence >= gate:
            order, scores = self.original, self.features[:, 0]
        else:
            scores = self.features @ weights
            order = np.lexsort((inverse_rank(self.original), -scores))
            if 0 < depth < len(order):
                admitted = inverse_rank(self.original) < depth
                order = np.concatenate((order[admitted[order]], self.original[depth:]))
        return ([(self.ids[i], float(scores[i])) for i in order[:k]] + self.tail)[:k]


@dataclass
class ContentCandidates:
    ids: list[str]
    features: np.ndarray

    def rank(self, weights, k):
        scores = self.features @ weights
        order = np.argsort(-scores, kind="stable")[:k]
        return [(self.ids[i], float(scores[i])) for i in order if scores[i] > 0]


class EnsembleSearch:
    def __init__(self, context, options, *, semantic_bm25=None):
        self.context = context
        self.options = options
        self.vector = context.vector
        self.documents = self.vector.documents
        self.period = context.period
        self.inner = self.period.inner
        self.semantic_bm25 = semantic_bm25
        self.dates = np.array([date.fromisoformat(doc.doc_dt).toordinal() if doc.doc_dt else 0
                               for doc in self.vector.documents], dtype=np.int32)

    def dated_candidates(self, parsed, mailbox, as_of, k):
        bypass = self.options.date_gate if self.options is not None else None
        profile = self.semantic_bm25.dated if self.semantic_bm25 is not None else None
        if profile is None or not profile.blend or not (profile.weight or profile.dense_weight):
            guarded = self.context.candidates(parsed, mailbox, as_of, k, bypass_at=bypass)
            confidence = guarded.confidence
        else:
            guarded = self.context.candidates(parsed, mailbox, as_of, k, bypass_at=0.)
            confidence = guarded.confidence
            if guarded.ids and (bypass is None or confidence < bypass):
                query = self.vector.encoder.query(parsed.topic)
                first = self.semantic_bm25.candidates(parsed, mailbox, as_of, profile, query_vector=query)
                ranked = first.rank(profile.weight, profile.blend)
                depth = self.period.depth
                pool = ranked[:depth], ranked[depth:] + first.tail
                guarded = self.context.candidates(parsed, mailbox, as_of, k, query_vector=query, pool_override=pool)
        if not guarded.ids:
            return DatedCandidates([], np.empty(0, dtype=np.intp), np.empty((0, 3)), guarded.tail, 1.01)
        original = np.argsort(-guarded.base_scores, kind="stable")
        if bypass is not None and confidence >= bypass:
            return DatedCandidates(guarded.ids, original, guarded.base_scores[:, None], guarded.tail,
                                   confidence)
        cover = np.argsort(-guarded.cover_scores, kind="stable")
        positions = {doc_id: row for row, doc_id in enumerate(guarded.ids)}
        reranked = guarded.ranked(replace(self.context.model, depth=0), len(guarded.ids))
        semantic = np.array([positions[doc_id] for doc_id, _ in reranked], dtype=np.intp)
        features = np.column_stack([1 / (13 + inverse_rank(order)) for order in (original, cover, semantic)])
        return DatedCandidates(guarded.ids, original, features, guarded.tail, confidence)

    def content_candidates(self, topic, mailbox, as_of, *, depth=200):
        rows = self.vector.eligible(mailbox, as_of)
        if not rows.size:
            return ContentCandidates([], np.empty((0, 4)))
        query = self.vector.encoder.query(topic)
        adapted = self.vector.adapter.project(query) if self.vector.adapter is not None else query
        projected = normalize(query @ self.context.model.query_map[:-1] + self.context.model.query_map[-1])
        semantic = np.argsort(-(self.vector.vectors[rows] @ adapted), kind="stable")[:depth]
        contextual = np.argsort(-(self.context.context[rows] @ projected), kind="stable")[:depth]
        profile = self.semantic_bm25.undated if self.semantic_bm25 is not None else None
        if profile is None or not profile.blend or not (profile.weight or profile.dense_weight):
            lexical = self.inner.search(topic, mailbox, as_of, depth)
        else:
            p = ParsedQuery(topic, topic, None)
            lexical = self.semantic_bm25.candidates(p, mailbox, as_of, profile, query_vector=query).rank(
                profile.weight, profile.blend)[:depth]
        positions = {int(row): i for i, row in enumerate(rows)}
        lexical_rows = [positions[self.context.doc_positions[doc_id]] for doc_id, _ in lexical]
        evidence = np.zeros((len(rows), 4), dtype=np.float32)
        for i, order in enumerate((lexical_rows, semantic, contextual)):
            evidence[order, i] = 30 / (30 + np.arange(len(order)))
        selected = np.flatnonzero(evidence[:, :3].any(axis=1))
        if as_of is not None:
            known = self.dates[rows[selected]]
            age = date.fromisoformat(as_of).toordinal() - known
            evidence[selected, 3] = np.where(known > 0, np.exp2(-np.maximum(age, 0) / 30.), 0.)
        return ContentCandidates([self.vector.documents[rows[i]].doc_id for i in selected], evidence[selected])

    def search(self, query, mailbox, as_of, k):
        return self.search_parsed(parse_query(query, as_of), mailbox, as_of, k)

    def search_parsed(self, parsed, mailbox, as_of, k):
        if k <= 0:
            return []
        if parsed.period is not None:
            if self.options.date_mix[1:] == (0., 0.):
                return self.period.search_parsed(parsed, mailbox, as_of, k)
            return self.dated_candidates(parsed, mailbox, as_of, k).rank(
                self.options.date_mix, self.options.date_depth, self.options.date_gate, k)
        if not parsed.topic:
            return []
        return self.content_candidates(parsed.topic, mailbox, as_of).rank(self.options.content_mix, k)
