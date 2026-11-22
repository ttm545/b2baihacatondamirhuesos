"""Run period with a semantic word expansion in the BM25 prefix channel."""

from collections import Counter, defaultdict
from copy import copy
from functools import lru_cache
import heapq
import math

import numpy as np

from vk_mail_search.bm25 import RUSSIAN


DEFAULT_SIMILARITY = 0.45
# PeriodRanker builds its BM25 index with the standard k1=1.2.
TF_LIMIT = 2.2


class WordVectorSearch:
    def __init__(self, period, vectors, min_similarity=DEFAULT_SIMILARITY):
        self.period = copy(period)
        self.period.inner = SemanticBM25(period.inner, period.docs.values(), vectors, min_similarity)

    def search(self, query, mailbox, as_of, k):
        return self.period.search(query, mailbox, as_of, k)


class SemanticBM25:
    def __init__(self, lexical, documents, vectors, min_similarity):
        self.lexical = lexical
        self.vectors = vectors
        self.min_similarity = min_similarity
        self.mailbox_sizes = Counter(doc.synthetic_user for doc in documents)
        self.words = sorted({
            word for index in lexical._mailboxes.values() for word in index.postings
            if RUSSIAN.fullmatch(word) and word in vectors
        })
        if self.words:
            matrix = np.asarray([vectors[word] for word in self.words], dtype=np.float32)
            self.normalized = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-8)
        else:
            self.normalized = np.empty((0, 0), dtype=np.float32)

    @lru_cache(maxsize=1024)
    def neighbors(self, word):
        if self.min_similarity >= 1 or not self.words or not RUSSIAN.fullmatch(word) or word not in self.vectors:
            return {}
        vector = np.asarray(self.vectors[word], dtype=np.float32)
        similarities = self.normalized @ (vector / max(float(np.linalg.norm(vector)), 1e-8))
        close = np.flatnonzero(similarities >= self.min_similarity)
        top = close[np.argsort(-similarities[close], kind="stable")[:24]]
        return {self.words[i]: min(float(similarities[i]), 1.) for i in top}

    @lru_cache(maxsize=512)
    def expansion(self, mailbox, word):
        index = self.lexical._mailboxes[mailbox]
        frequencies = {}
        subject_matches = {}
        for neighbor, similarity in self.neighbors(word).items():
            for doc, weight in index.postings.get(neighbor, ()):
                # The existing index stores f * (k1 + 1) / (f + norm).
                # Recover f / norm so weighted counts can be added before BM25 saturation.
                ratio = weight / (TF_LIMIT - weight)
                frequency = frequencies.get(doc.doc_id, (doc, 0.))[1]
                frequencies[doc.doc_id] = (doc, frequency + similarity * ratio)
                if neighbor in self.lexical._subject_terms.get(doc.doc_id, ()):
                    subject_matches[doc.doc_id] = max(subject_matches.get(doc.doc_id, 0.), similarity)
        df = len(frequencies)
        idf = math.log1p((self.mailbox_sizes[mailbox] - df + 0.5) / (df + 0.5))
        hits = [(doc, TF_LIMIT * ratio / (1 + ratio)) for doc, ratio in frequencies.values()]
        return hits, idf, subject_matches

    def search(self, query, mailbox, as_of, k):
        index = self.lexical._mailboxes.get(mailbox)
        if not index or k <= 0:
            return []
        scores = defaultdict(float)
        approximate_subject = defaultdict(float)
        query_weights = dict.fromkeys(self.lexical._tokenize(query), 1.)
        topical = set()
        if self.lexical._function_weight < 1 or self.lexical._subject_bonus:
            topical = {term for term in query_weights if not self.lexical._is_function_word(term)}
        if topical:
            for term in query_weights.keys() - topical:
                query_weights[term] = self.lexical._function_weight

        def accumulate(hits, weight):
            if weight:
                for doc, tf in hits:
                    if as_of is None or doc.doc_dt is None or doc.doc_dt <= as_of:
                        scores[doc.doc_id] += weight * tf

        for term, weight in query_weights.items():
            if term in index.postings:
                accumulate(index.postings[term], weight * index.idf[term])
        if self.lexical._prefix_weight and self.min_similarity < 1:
            for term, weight in query_weights.items():
                hits, idf, subject_matches = self.expansion(mailbox, term)
                accumulate(hits, weight * self.lexical._prefix_weight * idf)
                if term in topical:
                    for doc_id, similarity in subject_matches.items():
                        approximate_subject[doc_id] += similarity

        if self.lexical._subject_bonus and topical:
            for doc_id in scores:
                matches = len(topical & self.lexical._subject_terms[doc_id]) + 0.25 * approximate_subject[doc_id]
                scores[doc_id] *= 1 + self.lexical._subject_bonus * matches
        return heapq.nsmallest(k, scores.items(), key=lambda item: (-item[1], item[0]))
