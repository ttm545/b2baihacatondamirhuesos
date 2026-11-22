"""Build a reusable vector index and search it inside mailbox and date boundaries."""

from collections import defaultdict
from collections.abc import Mapping
from hashlib import sha256
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter

import numpy as np

from vk_mail_search.data import Document
from vk_mail_search.query_parser import ParsedQuery, parse_query

from .adapter import QueryAdapter
from .encoder import Encoder, TEXT_VERSION, letter_text
from .metric_adapter import MetricAdapter
from .topic_adapter import TopicAdapter


def fingerprint(docs: list[Document], encoder: Encoder) -> str:
    digest = sha256(f"{TEXT_VERSION}:{encoder.signature}:{encoder.max_tokens}\n".encode())
    for doc in docs:
        row = (doc.doc_id, doc.synthetic_user, doc.doc_dt, doc.doc)
        digest.update(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def build_index(docs: list[Document], encoder: Encoder, batch_size: int):
    start = perf_counter()
    texts: dict[str, int] = {}
    positions = []
    for doc in docs:
        text = letter_text(doc)
        positions.append(texts.setdefault(text, len(texts)))
    rows = np.empty((len(texts), encoder.dimension), dtype=np.float16)
    for ids, embeddings in encoder.batches(list(texts), batch_size):
        rows[ids] = embeddings
    if not np.isfinite(rows).all():
        raise ValueError("The encoder returned a non-finite document vector")
    return rows[positions], {"documents": len(docs), "unique_texts": len(texts),
                             "vectorization_seconds": round(perf_counter() - start, 3)}


def save_index(path: Path, vectors: np.ndarray, signature: str, info: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(dir=path.parent, prefix=".vectors-", suffix=".npz", delete=False) as handle:
            temporary = Path(handle.name)
            np.savez(handle, vectors=vectors, fingerprint=signature, info=json.dumps(info))
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class TinySearch:
    def __init__(self, documents: Mapping[str, Document], encoder: Encoder, path: Path,
                 batch_size: int = 128, rebuild: bool = False, adapter: QueryAdapter | None = None,
                 metric: MetricAdapter | None = None, topic: TopicAdapter | None = None):
        if sum(item is not None for item in (adapter, metric, topic)) > 1:
            raise ValueError("Select one query adapter")
        self.documents = sorted(documents.values(), key=lambda doc: (doc.synthetic_user, doc.doc_id))
        self.encoder = encoder
        self.adapter = adapter
        self.metric = metric
        self.topic = topic
        key = fingerprint(self.documents, encoder)
        if path.is_file() and not rebuild:
            with np.load(path, allow_pickle=False) as cache:
                if str(cache["fingerprint"]) != key:
                    raise ValueError("Index does not match the archive, model or token limit. Rebuild it explicitly.")
                matrix = cache["vectors"]
                self.index_info = {**json.loads(str(cache["info"])), "cached": True}
        else:
            matrix, info = build_index(self.documents, encoder, batch_size)
            save_index(path, matrix, key, info)
            self.index_info = {**info, "cached": False}
        if matrix.shape != (len(self.documents), encoder.dimension) or not np.isfinite(matrix).all():
            raise ValueError("Invalid shape or values in the vector index")
        self.vectors = matrix.astype(np.float32)
        if metric is not None:
            self.vectors = metric.documents(self.vectors)
        self._topics = topic.document_probabilities(self.vectors) if topic is not None else None
        rows = defaultdict(list)
        for row, doc in enumerate(self.documents):
            rows[doc.synthetic_user].append(row)
        self._mailboxes = {box: np.asarray(indices, dtype=np.intp) for box, indices in rows.items()}
        self._days = np.asarray([doc.doc_dt or "" for doc in self.documents], dtype="U10")

    def eligible(self, mailbox: str, as_of: str | None) -> np.ndarray:
        rows = self._mailboxes.get(mailbox)
        if rows is None:
            return np.empty(0, dtype=np.intp)
        return rows if as_of is None else rows[self._days[rows] <= as_of]

    def in_period(self, rows: np.ndarray, parsed: ParsedQuery) -> np.ndarray:
        if parsed.period is None:
            return np.zeros(len(rows), dtype=bool)
        days = self._days[rows]
        return (days >= parsed.period.start.isoformat()) & (days <= parsed.period.end.isoformat())

    def search(self, query: str, mailbox: str, as_of: str | None, k: int) -> list[tuple[str, float]]:
        return self.search_parsed(parse_query(query, as_of), mailbox, as_of, k)

    def search_parsed(self, parsed: ParsedQuery, mailbox: str, as_of: str | None, k: int) -> list[tuple[str, float]]:
        rows = self.eligible(mailbox, as_of)
        if k <= 0 or not rows.size:
            return []
        inside = self.in_period(rows, parsed)
        if not parsed.topic:
            dates = sorted(rows[inside], key=lambda row: (self._days[row], self.documents[row].doc_id), reverse=True)
            return [(self.documents[row].doc_id, 0.0) for row in dates[:k]]
        if int(inside.sum()) >= k:
            rows = rows[inside]
            inside = np.ones(len(rows), dtype=bool)
        vector = self.encoder.query(parsed.topic)
        if self.adapter is not None:
            vector = self.adapter.project(vector)
        if self.metric is not None:
            vector = self.metric.query(vector)
        scores = self.vectors[rows] @ vector
        if self.topic is not None:
            scores += self.topic.scores(vector, self._topics[rows])
        order = np.lexsort((rows, -scores, ~inside))[:k]
        return [(self.documents[rows[i]].doc_id, float(scores[i])) for i in order]
