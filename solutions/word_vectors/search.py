"""Rank mail inside a period with soft word frequencies and BM25 saturation."""

from collections import Counter
from functools import lru_cache

import numpy as np
from scipy.sparse import csr_matrix

from vk_mail_search.query_parser import parse_query


DEFAULT_SIMILARITY = 0.35


class WordVectorSearch:
    def __init__(self, period, vectors):
        self.period = period
        self.vectors = vectors
        doc_ids = tuple(period.docs)
        self.rows = {doc_id: row for row, doc_id in enumerate(doc_ids)}
        self.term_ids = {}
        offsets = [0]
        columns = []
        frequencies = []
        for doc_id in doc_ids:
            fields = period._fields[doc_id]
            counts = fields.counts.copy()
            counts.update({word: 3 * count for word, count in Counter(fields.subject).items()})
            for word, count in counts.items():
                columns.append(self.term_ids.setdefault(word, len(self.term_ids)))
                frequencies.append(count)
            offsets.append(len(columns))
        self.counts = csr_matrix(
            (np.asarray(frequencies, dtype=np.float32), columns, offsets),
            shape=(len(doc_ids), len(self.term_ids)),
        )
        self.postings = self.counts.tocsc()
        lengths = np.asarray(self.counts.sum(axis=1)).ravel()
        mean_length = max(float(lengths.mean()), 1.0) if lengths.size else 1.0
        self.length_factor = 1.2 * (0.55 + 0.45 * lengths / mean_length)

        embedded = [(word_id, vectors[word]) for word, word_id in self.term_ids.items() if word in vectors]
        self.vector_term_ids = np.asarray([word_id for word_id, _ in embedded], dtype=np.intp)
        if embedded:
            matrix = np.asarray([vector for _, vector in embedded], dtype=np.float32)
            self.normalized_vectors = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-8)
        else:
            self.normalized_vectors = np.empty((0, 0), dtype=np.float32)

    @lru_cache(maxsize=1024)
    def expansion(self, word, min_similarity):
        """Return vocabulary columns, fractional occurrences, and corpus-wide soft IDF."""
        neighbors = {}
        if min_similarity < 1 and self.vector_term_ids.size and word in self.vectors:
            vector = np.asarray(self.vectors[word], dtype=np.float32)
            similarities = self.normalized_vectors @ (vector / max(float(np.linalg.norm(vector)), 1e-8))
            close = np.flatnonzero(similarities > min_similarity)
            # Keep dominant expansions and bound the number of posting lists per word.
            top = close[np.argsort(-similarities[close], kind="stable")[:24]]
            neighbors.update(
                (int(self.vector_term_ids[i]), (min(float(similarities[i]), 1.) - min_similarity)
                 / (1. - min_similarity))
                for i in top
            )
        exact = self.term_ids.get(word)
        if exact is not None:
            neighbors[exact] = 1.0
        columns = np.fromiter(neighbors, dtype=np.intp)
        weights = np.fromiter(neighbors.values(), dtype=np.float32)
        occupied = np.zeros(self.counts.shape[0], dtype=np.float32)
        for column, weight in zip(columns, weights, strict=True):
            lo, hi = self.postings.indptr[column:column + 2]
            rows = self.postings.indices[lo:hi]
            occupied[rows] = np.maximum(occupied[rows], weight)
        size = len(occupied)
        soft_df = float(occupied.sum())
        idf = float(np.log1p((size - soft_df + 0.5) / (soft_df + 0.5)))
        return columns, weights, idf

    def search(self, query, mailbox, as_of, k, *, min_similarity=DEFAULT_SIMILARITY):
        if k <= 0:
            return []
        parsed = parse_query(query, as_of)
        if parsed.period is None or not parsed.topic:
            return self.period.search_parsed(parsed, mailbox, as_of, k)
        # An independent date pool keeps prefix ranks out of the experimental score.
        candidate_ids = self.period.in_period(mailbox, parsed, as_of)[::-1]
        rows = np.asarray([self.rows[doc_id] for doc_id in candidate_ids], dtype=np.intp)
        candidates = self.counts[rows]
        scores = np.zeros(len(rows), dtype=np.float32)
        for word in self.period.topic_terms(parsed.topic):
            columns, weights, idf = self.expansion(word, min_similarity)
            if columns.size:
                frequencies = candidates[:, columns] @ weights
                scores += 2.2 * idf * frequencies / (frequencies + self.length_factor[rows])
        order = np.argsort(-scores, kind="stable")
        leading = [(candidate_ids[i], float(scores[i])) for i in order[:k]]
        if len(leading) == k:
            return leading
        tail = self.period.search_parsed(parsed, mailbox, as_of, k)
        candidate_set = set(candidate_ids)
        return (leading + [hit for hit in tail if hit[0] not in candidate_set])[:k]
