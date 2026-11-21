"""Exact cosine search over cached USER2 vectors within one synthetic mailbox."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from itertools import groupby
import json
import logging
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Protocol
from zipfile import BadZipFile

import numpy as np

from .embeddings import EMBEDDING_DIM, MODEL_ID, EmailText, MailEmbedder, prepare_email_text


CHUNK_TOKENS = 512
CHUNK_OVERLAP = 64
logger = logging.getLogger(__name__)


class Document(Protocol):
    doc_id: str
    synthetic_user: str
    doc_dt: str | None
    doc: str


@dataclass(frozen=True)
class _Mailbox:
    docs: tuple[Document, ...]
    vectors: np.ndarray
    starts: np.ndarray


class DenseSearcher:
    """Precompute document/chunk vectors; encode only the query during search."""

    def __init__(
        self,
        docs: Iterable[Document] | Mapping[str, Document],
        *,
        cache_dir: str | Path | None = None,
        device: str | None = None,
        batch_size: int = 16,
        chunk_tokens: int = CHUNK_TOKENS,
    ) -> None:
        records = docs.values() if isinstance(docs, Mapping) else docs
        documents = tuple(sorted(records, key=lambda doc: (doc.synthetic_user, doc.doc_id)))
        if len({doc.doc_id for doc in documents}) != len(documents):
            raise ValueError("Duplicate doc_id in corpus")
        if batch_size < 1 or chunk_tokens <= CHUNK_OVERLAP:
            raise ValueError("Invalid batch size or chunk size")
        self._embedder = MailEmbedder(device=device)
        if chunk_tokens > self._embedder.max_document_tokens:
            raise ValueError("Chunk size exceeds the encoder context length")
        directory = Path(cache_dir or os.environ.get("VK_CACHE_DIR") or ".cache")
        self.cache_path = directory / self._cache_name(documents, chunk_tokens)
        cached = self._read_cache(self.cache_path, len(documents))
        if cached is None:
            cached = self._build_index(documents, batch_size, chunk_tokens)
            self._write_cache(self.cache_path, *cached)
        vectors, counts = cached
        self.document_count = len(documents)
        self.vector_count = len(vectors)
        self.chunked_document_count = int(np.count_nonzero(counts > 1))

        starts = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))
        self._mailboxes = {}
        for mailbox, group in groupby(enumerate(documents), key=lambda item: item[1].synthetic_user):
            indexed = list(group)
            first, stop = indexed[0][0], indexed[-1][0] + 1
            base = starts[first]
            self._mailboxes[mailbox] = _Mailbox(
                tuple(doc for _, doc in indexed), vectors[base:starts[stop]], starts[first:stop] - base
            )

    def _cache_name(self, docs: tuple[Document, ...], chunk_tokens: int) -> str:
        key = sha256()
        key.update(f"dense-v1:{MODEL_ID}:{self._embedder.revision}:{chunk_tokens}:{CHUNK_OVERLAP}\n".encode())
        for doc in docs:
            record = [doc.doc_id, doc.synthetic_user, doc.doc_dt, doc.doc]
            key.update(json.dumps(record, ensure_ascii=False).encode("utf-8"))
            key.update(b"\n")
        return f"dense-{key.hexdigest()}.npz"

    @staticmethod
    def _read_cache(path: Path, size: int) -> tuple[np.ndarray, np.ndarray] | None:
        try:
            with np.load(path, allow_pickle=False) as cache:
                vectors, counts = cache["vectors"], cache["counts"]
            if (counts.shape == (size,) and counts.dtype == np.int32 and (counts > 0).all()
                    and vectors.dtype == np.float32
                    and vectors.shape == (int(counts.sum()), EMBEDDING_DIM)):
                return vectors, counts
        except (OSError, ValueError, KeyError, EOFError, BadZipFile):
            pass
        return None

    def _build_index(
        self, docs: tuple[Document, ...], batch_size: int, max_tokens: int
    ) -> tuple[np.ndarray, np.ndarray]:
        counts: list[int] = []

        def chunks() -> Iterable[str]:
            for index, doc in enumerate(docs, start=1):
                # The corpus text already contains the headers where they are present.
                text = prepare_email_text(EmailText(None, doc.doc))
                parts = self._embedder.document_chunks(text, max_tokens=max_tokens, overlap=CHUNK_OVERLAP)
                counts.append(len(parts))
                yield from parts
                if index % 1000 == 0:
                    logger.info("Prepared %s/%s documents for encoding", index, len(docs))

        vectors = np.fromiter(
            self._embedder.embed_documents(chunks(), batch_size=batch_size),
            dtype=np.dtype((np.float32, EMBEDDING_DIM)),
        )
        return vectors, np.asarray(counts, dtype=np.int32)

    @staticmethod
    def _write_cache(path: Path, vectors: np.ndarray, counts: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as temp:
            temporary_path = Path(temp.name)
            try:
                np.savez(temp, vectors=vectors, counts=counts)
                temp.close()
                temporary_path.replace(path)
            finally:
                temporary_path.unlink(missing_ok=True)

    def search(self, query: str, mailbox: str, as_of: str | None, k: int) -> list[tuple[str, float]]:
        index = self._mailboxes.get(mailbox)
        if index is None or k <= 0 or not query.strip():
            return []
        vector = np.asarray(self._embedder.embed_query(query), dtype=np.float32)
        # Unit vectors make the dot product equal to cosine similarity.
        scores = np.maximum.reduceat(index.vectors @ vector, index.starts)
        matches = [
            (doc.doc_id, float(score))
            for doc, score in zip(index.docs, scores, strict=True)
            if as_of is None or doc.doc_dt is None or doc.doc_dt <= as_of
        ]
        matches.sort(key=lambda hit: (-hit[1], hit[0]))
        return matches[:k]
