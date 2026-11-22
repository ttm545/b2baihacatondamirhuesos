"""Mix lexical and word-vector BM25 for the dated candidate pool."""

from copy import copy
from dataclasses import replace

from solutions.tiny_encoder.ensemble import EnsembleSearch
from vk_mail_search.dated import DatedSearcher

from .word_bm25 import DEFAULT_SIMILARITY, SemanticBM25


CANDIDATE_DEPTH = 30
DEFAULT_SEMANTIC_WEIGHT = 0.085


def blend_candidates(lexical, semantic, weight):
    if not lexical or not weight:
        return lexical
    if weight == 1:
        return semantic
    lexical_scale = (1 - weight) / (lexical[0][1] or 1.)
    semantic_scale = weight / (semantic[0][1] if semantic and semantic[0][1] else 1.)
    semantic_scores = dict(semantic)
    combined = [(doc_id, score * lexical_scale + semantic_scores.get(doc_id, 0.) * semantic_scale)
                for doc_id, score in lexical]
    return sorted(combined, key=lambda hit: -hit[1])


class SemanticCandidatePool:
    def __init__(self, period, semantic, weight=DEFAULT_SEMANTIC_WEIGHT):
        if not 0 <= weight <= 1:
            raise ValueError("Semantic candidate weight must be between 0 and 1")
        self.period = period
        self.inner = period.inner
        self.selector = DatedSearcher(semantic, period.docs.values())
        self.weight = weight

    def shortlist(self, parsed, mailbox, as_of, k=CANDIDATE_DEPTH):
        if k <= 0 or parsed.period is None or not parsed.topic:
            return []
        count = len(self.period.in_period(mailbox, parsed, as_of))
        lexical = DatedSearcher.search_parsed(self.period, parsed, mailbox, as_of, count)
        return self._select(parsed, mailbox, as_of, lexical, k)

    def _select(self, parsed, mailbox, as_of, lexical, k):
        if not self.weight or not lexical:
            return lexical[:k]
        semantic = self.selector.search_parsed(parsed, mailbox, as_of, len(lexical))
        return blend_candidates(lexical, semantic, self.weight)[:k]

    def _candidates_parsed(self, parsed, mailbox, as_of, k):
        if parsed.period is None or not parsed.topic:
            return self.period._candidates_parsed(parsed, mailbox, as_of, k)
        count = len(self.period.in_period(mailbox, parsed, as_of))
        if count <= 1:
            return self.period._candidates_parsed(parsed, mailbox, as_of, k)

        # The shortlist changes. The ensemble still receives the original BM25 scores and ordering.
        lexical = DatedSearcher.search_parsed(self.period, parsed, mailbox, as_of, count)
        selected = {doc_id for doc_id, _ in self._select(parsed, mailbox, as_of, lexical, CANDIDATE_DEPTH)}
        pool = [hit for hit in lexical if hit[0] in selected]
        tail = []
        if k > len(pool):
            original = self.period.search_parsed(parsed, mailbox, as_of, k + len(pool))
            tail = [hit for hit in original if hit[0] not in selected]
        return pool, tail

    def topic_terms(self, topic):
        return self.period.topic_terms(topic)

    def features(self, terms, pool):
        return self.period.features(terms, pool)

    def search_parsed(self, parsed, mailbox, as_of, k):
        return self.period.search_parsed(parsed, mailbox, as_of, k)


class SemanticCandidateEnsemble(EnsembleSearch):
    def __init__(self, original, word_vectors, min_similarity=DEFAULT_SIMILARITY,
                 semantic_weight=DEFAULT_SEMANTIC_WEIGHT):
        period = original.period
        self.semantic_bm25 = SemanticBM25(period.inner, period.docs.values(), word_vectors, min_similarity)
        self.pool = SemanticCandidatePool(period, self.semantic_bm25, semantic_weight)
        context = copy(original.context)
        context.period = self.pool
        super().__init__(context, replace(original.options, date_depth=CANDIDATE_DEPTH))
