"""Apply a compact Qwen-distilled context score to the period candidate pool."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vk_mail_search.period_rank import PROFILES
from vk_mail_search.query_parser import parse_query

from .routed import RoutedSearch


def normalize(vectors):
    return vectors / np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-8)


def standardize(values, floor):
    return (values - values.mean()) / max(float(values.std()), floor)


@dataclass
class ContextModel:
    doc_map: np.ndarray
    query_map: np.ndarray
    weights: np.ndarray
    strength: float
    threshold: float
    strategy: str = "pairwise"
    depth: int = 0

    @classmethod
    def load(cls, path: Path, signature: str, size: int):
        with np.load(path, allow_pickle=False) as data:
            if str(data["signature"]) != signature:
                raise ValueError("Qwen context adapter does not match the compact encoder")
            doc, query, weights = data["documents"], data["queries"], data["weights"]
            strength, threshold = float(data["strength"]), float(data["threshold"])
            strategy = str(data["strategy"]) if "strategy" in data else "pairwise"
            depth = int(data["depth"]) if "depth" in data else 0
        if (doc.ndim != 2 or doc.shape[0] != size + 1 or query.shape != doc.shape or weights.shape != (4,)
                or not all(np.isfinite(matrix).all() for matrix in (doc, query, weights))
                or not 0 <= strength <= 2 or not 0 <= threshold <= 1.01
                or strategy not in {"pairwise", "rrf"} or (strategy == "rrf" and strength > 1)
                or not 0 <= depth <= 60):
            raise ValueError("Invalid Qwen context adapter coefficients")
        return cls(doc, query, weights, strength, threshold, strategy, depth)

    def save(self, path: Path, signature: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            np.savez(handle, signature=signature, documents=self.doc_map, queries=self.query_map,
                     weights=self.weights, strength=self.strength, threshold=self.threshold, strategy=self.strategy,
                     depth=self.depth)


@dataclass
class Candidates:
    ids: list[str]
    tail: list[tuple[str, float]]
    base_scores: np.ndarray
    prior: np.ndarray
    extra: np.ndarray
    confidence: float
    cover_scores: np.ndarray | None = None

    def ranked(self, model: ContextModel, cutoff: int):
        if not self.ids:
            return self.tail[:cutoff]
        if self.confidence >= model.threshold or model.strength == 0:
            order = np.argsort(-self.base_scores, kind="stable")
            scores = self.base_scores
        elif model.strategy == "pairwise":
            scores = self.prior + model.strength * (self.extra @ model.weights)
            order = np.argsort(-scores, kind="stable")
        else:
            base_rank = np.argsort(np.argsort(-self.base_scores, kind="stable"), kind="stable")
            qwen_weight = np.clip(model.weights[2] / max(abs(float(model.weights[1])), .20), -.50, .50)
            semantic = self.extra[:, 1] + qwen_weight * self.extra[:, 2]
            semantic_rank = np.argsort(np.argsort(-semantic, kind="stable"), kind="stable")
            scores = (1 - model.strength) / (13 + base_rank) + model.strength / (13 + semantic_rank)
            order = np.lexsort((base_rank, -scores))
        if 0 < model.depth < len(order):
            base_order = np.argsort(-self.base_scores, kind="stable")
            within = np.zeros(len(order), dtype=bool)
            within[base_order[:model.depth]] = True
            order = np.concatenate((order[within[order]], base_order[model.depth:]))
        return ([(self.ids[i], float(scores[i])) for i in order] + self.tail)[:cutoff]


class QwenContextSearch:
    def __init__(self, period, vector, model: ContextModel):
        self.period = period
        self.vector = vector
        self.inner = period.inner
        self.model = model
        self.undated = RoutedSearch(period, vector, fuse_undated=True)
        self.doc_positions = {doc.doc_id: i for i, doc in enumerate(vector.documents)}
        self.context = normalize(vector.vectors @ model.doc_map[:-1] + model.doc_map[-1])

    def candidates(self, parsed, mailbox, as_of, k, query_vector=None, contextual=None, *, skip_guarded=False,
                   bypass_at=None):
        pool, tail = self.period._candidates_parsed(parsed, mailbox, as_of, k)
        if not pool:
            return Candidates([], tail, np.empty(0), np.empty(0), np.empty((0, 4)), 1.0)
        ids = [doc_id for doc_id, _ in pool]
        lexical = np.asarray(self.period.features(self.period.topic_terms(parsed.topic), pool), dtype=np.float64)
        scores = np.asarray([sum(value * weight for value, weight in zip(row, PROFILES["learned"], strict=True))
                             for row in lexical])
        cover = lexical @ np.asarray(PROFILES["cover"])
        confidence = float(lexical[:, 2].max())
        if ((skip_guarded and (confidence >= self.model.threshold or self.model.strength == 0))
                or (bypass_at is not None and confidence >= bypass_at)):
            return Candidates(ids, tail, scores, np.empty(0), np.empty((0, 4)), confidence, cover)
        rows = np.asarray([self.doc_positions[doc_id] for doc_id in ids], dtype=np.intp)
        if query_vector is None:
            query_vector = self.vector.encoder.query(parsed.topic)
        raw = self.vector.vectors[rows] @ query_vector
        adapted = self.vector.adapter.project(query_vector) if self.vector.adapter is not None else query_vector
        projected = (normalize(query_vector @ self.model.query_map[:-1] + self.model.query_map[-1])
                     if contextual is None else contextual)
        extra = np.column_stack((
            standardize(raw, .03),
            standardize(self.vector.vectors[rows] @ adapted, .03),
            standardize(self.context[rows] @ projected, .08),
            standardize(lexical @ (np.asarray(PROFILES["cover"]) - np.asarray(PROFILES["learned"])), .10),
        ))
        return Candidates(ids, tail, scores, standardize(scores, .08), extra, confidence, cover)

    def search(self, query: str, mailbox: str, as_of: str | None, k: int):
        if k <= 0:
            return []
        parsed = parse_query(query, as_of)
        if parsed.period is None:
            return self.undated.search(query, mailbox, as_of, k)
        return self.candidates(parsed, mailbox, as_of, k, skip_guarded=True).ranked(self.model, k)
