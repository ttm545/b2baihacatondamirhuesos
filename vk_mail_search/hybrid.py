"""Fuse ranked lists from several searchers (BM25 + dense) into one ranking."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

from .searchers import Searcher

Hits = list[tuple[str, float]]


@dataclass(frozen=True, slots=True)
class FusionConfig:
    """How ranked lists are combined.

    mode="rrf": score = sum(w / (rrf_k + rank)), rank starts at 1. Ignores raw scores,
    robust when BM25 and cosine scales differ.
    mode="score": min-max normalise each list to [0, 1] per query, score = sum(w * norm).
    A document missing from a list contributes 0 for that list.
    """

    mode: str = "rrf"
    weights: Mapping[str, float] | None = None
    depth: int = 100
    rrf_k: float = 60.0

    def __post_init__(self) -> None:
        if self.mode not in ("rrf", "score"):
            raise ValueError(f"Неизвестный режим слияния: {self.mode}")
        if self.depth < 1 or self.rrf_k < 0:
            raise ValueError("depth должен быть положительным, rrf_k неотрицательным")
        if self.weights and any(w < 0 for w in self.weights.values()):
            raise ValueError("Веса не могут быть отрицательными")


def fuse(lists: Mapping[str, Hits], config: FusionConfig, k: int) -> Hits:
    """Combine ranked lists; ties are broken by doc_id for a stable order."""
    scores: dict[str, float] = defaultdict(float)
    for name, hits in lists.items():
        weight = 1.0 if config.weights is None else config.weights.get(name, 0.0)
        if weight == 0 or not hits:
            continue
        hits = hits[:config.depth]
        if config.mode == "rrf":
            for rank, (doc_id, _) in enumerate(hits, 1):
                scores[doc_id] += weight / (config.rrf_k + rank)
        else:
            values = [score for _, score in hits]
            low, high = min(values), max(values)
            span = high - low
            for doc_id, score in hits:
                scores[doc_id] += weight * ((score - low) / span if span > 0 else 1.0)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]


class HybridSearcher:
    def __init__(self, searchers: Mapping[str, Searcher], config: FusionConfig | None = None):
        if not searchers:
            raise ValueError("Нужен хотя бы один поисковик")
        self.searchers = dict(searchers)
        self.config = config or FusionConfig()

    def search(self, query: str, mailbox: str, as_of: str | None, k: int) -> Hits:
        if k <= 0:
            return []
        lists = {name: searcher.search(query, mailbox, as_of, self.config.depth)
                 for name, searcher in self.searchers.items()}
        return fuse(lists, self.config, k)


def parse_config(spec: str) -> FusionConfig:
    """Parse "mode=score,w_dense=0.7,w_bm25=0.3,depth=100,rrf_k=60" (all keys optional)."""
    values: dict[str, str] = {}
    for part in filter(None, (item.strip() for item in spec.split(","))):
        key, sep, value = part.partition("=")
        if not sep:
            raise ValueError(f"Ожидалось ключ=значение: {part}")
        values[key.strip()] = value.strip()
    weights = {key[2:]: float(value) for key, value in values.items() if key.startswith("w_")}
    unknown = values.keys() - {"mode", "depth", "rrf_k"} - {f"w_{name}" for name in weights}
    if unknown:
        raise ValueError(f"Неизвестные параметры гибрида: {', '.join(sorted(unknown))}")
    return FusionConfig(
        mode=values.get("mode", "rrf"),
        weights=weights or None,
        depth=int(values.get("depth", 100)),
        rrf_k=float(values.get("rrf_k", 60)),
    )
