"""The search contract shared by the benchmark, API and rankers."""

from collections.abc import Mapping
from typing import Protocol

from .data import Document


class Searcher(Protocol):
    def search(self, query: str, mailbox: str, as_of: str | None, k: int) -> list[tuple[str, float]]:
        """Return at most k unique (doc_id, score) tuples, best first."""
        ...


def build_searcher(name: str, docs: Mapping[str, Document]) -> Searcher:
    """name: bm25 | dense | hybrid | hybrid:<параметры>, например hybrid:mode=score,w_dense=0.7."""
    base, _, spec = name.partition(":")
    if base == "hybrid":
        from .hybrid import HybridSearcher, parse_config

        config = parse_config(spec)
        return HybridSearcher({"bm25": build_searcher("bm25", docs), "dense": build_searcher("dense", docs)}, config)
    if spec:
        raise ValueError(f"Параметры поддерживаются только у hybrid: {name}")
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
    raise ValueError(f"Неизвестный поисковик: {name}. Поддерживаются bm25, dense и hybrid.")
