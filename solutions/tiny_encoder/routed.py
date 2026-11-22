"""Use period for dated requests and a configurable local vector index for other requests."""

from collections import defaultdict

from vk_mail_search.query_parser import parse_query
from vk_mail_search.searchers import Searcher


class RoutedSearch:
    def __init__(self, dated: Searcher, vector: Searcher, *, fuse_undated: bool = False):
        self.dated = dated
        self.vector = vector
        self.fuse_undated = fuse_undated
        self.inner = getattr(dated, "inner", dated)

    def search(self, query: str, mailbox: str, as_of: str | None, k: int):
        if k <= 0:
            return []
        if parse_query(query, as_of).period is not None:
            return self.dated.search(query, mailbox, as_of, k)
        if not self.fuse_undated:
            return self.vector.search(query, mailbox, as_of, k)
        scores = defaultdict(float)
        for engine in (self.dated, self.vector):
            for rank, (doc_id, _) in enumerate(engine.search(query, mailbox, as_of, max(100, k)), 1):
                scores[doc_id] += 1 / (60 + rank)
        return sorted(scores.items(), key=lambda row: (-row[1], row[0]))[:k]
