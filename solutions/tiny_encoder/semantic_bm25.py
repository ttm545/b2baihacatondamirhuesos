"""Expand the sparse BM25 query using terms from contextually similar mail."""

from collections import Counter, defaultdict
from dataclasses import dataclass
import json
from math import log2, sqrt
from pathlib import Path

import numpy as np

from vk_mail_search.period_rank import _REQUEST_WORDS
from vk_mail_search.query_parser import parse_query


@dataclass(frozen=True)
class FeedbackProfile:
    neighbors: int
    max_terms: int
    weight: float
    lexical_share: float = 0.
    context_gate: float = 0.
    dense_weight: float = 0.
    blend: float = 1.

    def __post_init__(self):
        if (not 2 <= self.neighbors <= 32 or not 1 <= self.max_terms <= 32
                or not 0 <= self.weight <= 2 or not 0 <= self.lexical_share <= 1
                or not 0 <= self.context_gate <= 1 or not 0 <= self.dense_weight <= 2
                or not 0 <= self.blend <= 1):
            raise ValueError("Invalid contextual BM25 profile")


def load_profiles(path: Path):
    raw = json.loads(path.read_text(encoding="utf-8"))
    return FeedbackProfile(**raw["dated"]), FeedbackProfile(**raw["undated"])


@dataclass
class CandidatePool:
    scoped_ids: list[str]
    original: dict[str, float]
    extra: dict[str, float]
    tail: list[tuple[str, float]]
    expansion: list[tuple[str, float]]
    dated: bool
    dense_prior: dict[str, float] | None = None

    def rank(self, weight: float, blend: float = 1.) -> list[tuple[str, float]]:
        base_scale = max((self.original.get(doc_id, 0.) for doc_id in self.scoped_ids), default=0.) or 1.
        best_bonus = max(self.extra.values(), default=0.) or 1.
        scale = base_scale * weight / best_bonus
        prior = self.dense_prior or {}
        reweighted = [(doc_id, self.original.get(doc_id, 0.) + scale * self.extra.get(doc_id, 0.)
                       + base_scale * prior.get(doc_id, 0.))
                      for doc_id in self.scoped_ids]
        scored = sorted(((doc_id, score) for doc_id, score in reweighted if score > 0),
                        key=lambda hit: (-hit[1], hit[0]))
        if blend < 1:
            available = set(self.scoped_ids)
            original = [(doc_id, score) for doc_id, score in self.original.items() if doc_id in available]
            joint = defaultdict(float)
            for hits, factor in ((original, 1 - blend), (scored, blend)):
                if factor:
                    for rank, (doc_id, _) in enumerate(hits):
                        joint[doc_id] += factor / (13 + rank)
            scored = sorted(joint.items(), key=lambda hit: (-hit[1], hit[0]))
        if self.dated:
            admitted = {doc_id for doc_id, _ in scored}
            scored.extend((doc_id, 0.) for doc_id in self.scoped_ids if doc_id not in admitted)
        return scored


class SemanticBM25:
    def __init__(self, period, vector, dated: FeedbackProfile, undated: FeedbackProfile, *, rerank=False):
        self.period = period
        self.vector = vector
        self.inner = period.inner
        self.documents = vector.documents
        self.dated = dated
        self.undated = undated
        self.rerank = rerank
        self.positions = {doc.doc_id: i for i, doc in enumerate(self.documents)}
        self.seed_terms = {
            doc_id: {term: 1. if term in fields.subject_terms else .45
                     for term in fields.subject_terms | fields.lead_terms
                     if 3 <= len(term) <= 30 and term.isalpha() and term not in _REQUEST_WORDS
                     and not period._function_word(term)}
            for doc_id, fields in period._fields.items()
        }

    def _feedback(self, query_vector, rows, original, topic, profile):
        adjusted = self.vector.adapter.project(query_vector) if self.vector.adapter is not None else query_vector
        semantic = self.vector.vectors[rows] @ adjusted
        dense_order = np.argsort(-semantic, kind="stable")
        ranks = np.empty(len(rows), dtype=np.intp)
        ranks[dense_order] = np.arange(len(rows))
        terms = set(self.inner._tokenize(topic))
        by_id = [self.documents[row].doc_id for row in rows]
        lexical = np.asarray([original.get(doc_id, 0.) for doc_id in by_id])
        lex_scale = float(lexical.max()) or 1.
        feedback = (1 - profile.lexical_share) / (1 + ranks / 4) + profile.lexical_share * (lexical / lex_scale)
        seed_order = np.argsort(-feedback, kind="stable")[:profile.neighbors]
        concentration, support = defaultdict(float), Counter()
        prefix_keys = {term[:5] for term in terms if len(term) >= 6}
        for rank, seed in enumerate(seed_order):
            for term, field in self.seed_terms[by_id[seed]].items():
                if (term in terms or (len(term) >= 6 and term[:5] in prefix_keys)
                        or (profile.context_gate and "syn" in term)):
                    continue
                support[term] += 1
                concentration[term] += field / log2(rank + 2)
        # Terms need agreement across letters. Corpus-wide weighting discounts common boilerplate.
        weighted = [(term, value * sqrt(support[term]) * self.period._weights([term])[0])
                    for term, value in concentration.items() if support[term] >= 2]
        expansion = sorted(weighted, key=lambda pair: (-pair[1], pair[0]))[:profile.max_terms]
        affinity = {doc_id: 1 / (1 + rank / 12) for doc_id, rank in zip(by_id, ranks, strict=True)}
        prior = {by_id[row]: affinity[by_id[row]] * profile.dense_weight for row in dense_order[:100]}
        return expansion, affinity, prior

    def candidates(self, parsed, mailbox, as_of, profile, *, query_vector=None):
        if not parsed.topic:
            return CandidatePool([], {}, {}, [], [], parsed.period is not None)
        rows = self.vector.eligible(mailbox, as_of)
        original = dict(self.inner.search(parsed.topic, mailbox, as_of, len(rows)))
        inside = self.period.in_period(mailbox, parsed, as_of) if parsed.period is not None else None
        if inside is None:
            scoped = [self.documents[row].doc_id for row in rows]
            tail = []
        else:
            scoped = list(reversed(inside))
            matching = set(inside)
            tail = [(doc_id, score) for doc_id, score in original.items() if doc_id not in matching]
            rows = np.asarray([self.positions[doc_id] for doc_id in inside], dtype=np.intp)
        signals = defaultdict(float)
        expansion = []
        prior = {}
        lexical_index = self.inner._mailboxes.get(mailbox)
        if len(rows) >= 2 and lexical_index is not None and (profile.weight or profile.dense_weight):
            if query_vector is None:
                query_vector = self.vector.encoder.query(parsed.topic)
            expansion, affinity, prior = self._feedback(query_vector, rows, original, parsed.topic, profile)
            for term, weight in expansion:
                for doc, term_bm25 in lexical_index.postings.get(term, ()):
                    if doc.doc_id in affinity:
                        gate = 1 - profile.context_gate * (1 - affinity[doc.doc_id])
                        signals[doc.doc_id] += weight * gate * lexical_index.idf[term] * term_bm25
        return CandidatePool(scoped, original, dict(signals), tail, expansion, inside is not None, prior)

    def order(self, parsed, candidates, weight, k):
        ordered = candidates.rank(weight)
        if self.rerank and parsed.period is not None and len(ordered) > 1:
            pool = ordered[:self.period.depth]
            terms = self.period.topic_terms(parsed.topic)
            ordered = self.period.rank(pool, self.period.features(terms, pool), self.period.mode) + ordered[len(pool):]
        return (ordered + candidates.tail)[:k]

    def search(self, query, mailbox, as_of, k):
        return self.search_parsed(parse_query(query, as_of), mailbox, as_of, k)

    def search_parsed(self, parsed, mailbox, as_of, k):
        if k <= 0:
            return []
        if not parsed.topic:
            return self.period.search_parsed(parsed, mailbox, as_of, k)
        profile = self.dated if parsed.period is not None else self.undated
        return self.order(parsed, self.candidates(parsed, mailbox, as_of, profile), profile.weight, k)
