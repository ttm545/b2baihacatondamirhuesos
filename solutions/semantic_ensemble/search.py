"""Use word-vector BM25 for the dated candidate pool and keep the ensemble scorer."""

from copy import copy
from dataclasses import replace

from solutions.tiny_encoder.ensemble import EnsembleSearch
from vk_mail_search.dated import DatedSearcher

from .word_bm25 import DEFAULT_SIMILARITY, SemanticBM25


CANDIDATE_DEPTH = 30


class SemanticCandidatePool:
    def __init__(self, period, semantic):
        self.period = period
        self.inner = period.inner
        self.selector = DatedSearcher(semantic, period.docs.values())

    def shortlist(self, parsed, mailbox, as_of, k=CANDIDATE_DEPTH):
        if parsed.period is None or not parsed.topic:
            return []
        count = len(self.selector.in_period(mailbox, parsed, as_of))
        return self.selector.search_parsed(parsed, mailbox, as_of, min(k, count))

    def _candidates_parsed(self, parsed, mailbox, as_of, k):
        if parsed.period is None or not parsed.topic:
            return self.period._candidates_parsed(parsed, mailbox, as_of, k)
        count = len(self.period.in_period(mailbox, parsed, as_of))
        if count <= 1:
            return self.period._candidates_parsed(parsed, mailbox, as_of, k)

        selected = {doc_id for doc_id, _ in self.shortlist(parsed, mailbox, as_of)}
        # The shortlist changes. The ensemble still receives the original BM25 scores and ordering.
        lexical = DatedSearcher.search_parsed(self.period, parsed, mailbox, as_of, count)
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
    def __init__(self, original, word_vectors, min_similarity=DEFAULT_SIMILARITY):
        period = original.period
        self.semantic_bm25 = SemanticBM25(period.inner, period.docs.values(), word_vectors, min_similarity)
        self.pool = SemanticCandidatePool(period, self.semantic_bm25)
        context = copy(original.context)
        context.period = self.pool
        super().__init__(context, replace(original.options, date_depth=CANDIDATE_DEPTH))
