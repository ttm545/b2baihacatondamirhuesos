"""The search contract shared by the benchmark, API and rankers."""

from collections.abc import Mapping
from typing import Protocol

from .data import Document


class Searcher(Protocol):
    def search(self, query: str, mailbox: str, as_of: str | None, k: int) -> list[tuple[str, float]]:
        """Return at most k unique (doc_id, score) tuples, best first."""
        ...


def build_searcher(name: str, docs: Mapping[str, Document]) -> Searcher:
    if name == "bm25":
        from .bm25 import BM25Searcher

        return BM25Searcher(docs.values())
    raise ValueError(f"Неизвестный поисковик: {name}. Доступен bm25.")
