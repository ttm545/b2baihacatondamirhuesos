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
    if name == "dense":
        try:
            from .dense import DenseSearcher
        except ModuleNotFoundError as exc:
            if exc.name == f"{__package__}.dense":
                raise ValueError("Для dense подключите ветку task/vk-search-dense.") from exc
            raise RuntimeError("Установите зависимости dense из requirements-embeddings.txt.") from exc
        return DenseSearcher(docs)
    raise ValueError(f"Неизвестный поисковик: {name}. Поддерживаются bm25 и dense.")
