"""Build and query the existing mail index with local Qwen3-VL text embeddings."""

from argparse import ArgumentParser
from collections.abc import Sequence
from contextlib import closing
from datetime import date
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import sys
from urllib.error import URLError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

import numpy as np

from vk_mail_search.data import load_dataset, load_documents, load_query_file
from vk_mail_search.evaluate import _load_paraphrase_benchmark, _select_queries, run_search, score_predictions
from vk_mail_search.query_parser import parse_query

from .index import TinySearch, save_index

MODEL_SHA = "feec6d40b22cacd3860863fb2e2fa2c3d20e4842dd6a65f290a3d501ab44ed1e"
MODEL_ALIAS = f"qwen-mail-{MODEL_SHA[:12]}"
DOCUMENT_INSTRUCTION = "Represent the user's input."
QUERY_INSTRUCTION = "Given a search query, retrieve relevant emails that satisfy the request."


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class QwenEncoder:
    dimension = 2048
    max_tokens = 512

    def __init__(self, port: int, text_chars: int, batch_cache: Path):
        self.url = f"http://127.0.0.1:{port}/v1/embeddings"
        self.text_chars = text_chars
        self.batch_cache = batch_cache
        self.signature = f"{MODEL_SHA}:qwen-vl-text-v1:{text_chars}:{DOCUMENT_INSTRUCTION}:{QUERY_INSTRUCTION}"
        self.opener = build_opener(ProxyHandler({}), NoRedirect())

    def encode(self, texts: Sequence[str], instruction: str) -> np.ndarray:
        inputs = [
            f"<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n"
            f"{text[:self.text_chars].replace('<|', '< |')}<|im_end|>\n<|im_start|>assistant\n"
            for text in texts
        ]
        body = json.dumps({"model": MODEL_ALIAS, "input": inputs, "encoding_format": "float"}).encode("utf-8")
        request = Request(self.url, body, headers={"Content-Type": "application/json"})
        try:
            with self.opener.open(request, timeout=300) as result:
                payload = json.load(result)
        except URLError as exc:
            raise RuntimeError(f"Local llama-server embedding request failed: {exc}") from exc
        if payload.get("model") != MODEL_ALIAS:
            raise ValueError("Unexpected llama-server model alias")
        data = sorted(payload["data"], key=lambda row: row["index"])
        if [row["index"] for row in data] != list(range(len(texts))):
            raise ValueError("Missing or duplicate rows in llama-server response")
        matrix = np.asarray([row["embedding"] for row in data], dtype=np.float32)
        if matrix.shape != (len(texts), self.dimension) or not np.isfinite(matrix).all():
            raise ValueError("Invalid Qwen embeddings")
        norm = np.linalg.norm(matrix, axis=1, keepdims=True)
        if (norm == 0).any():
            raise ValueError("Qwen returned a zero embedding")
        return matrix / norm

    def query(self, text: str) -> np.ndarray:
        return self.encode([text], QUERY_INSTRUCTION)[0]

    def batches(self, texts: Sequence[str], batch_size: int):
        self.batch_cache.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.batch_cache, timeout=30)) as db:
            db.execute("CREATE TABLE IF NOT EXISTS embeddings (id BLOB PRIMARY KEY, vector BLOB NOT NULL)")
            for offset in range(0, len(texts), batch_size):
                chunk = texts[offset:offset + batch_size]
                keys = [sha256((self.signature + "\n" + text[:self.text_chars]).encode()).digest() for text in chunk]
                vectors = [db.execute("SELECT vector FROM embeddings WHERE id=?", (key,)).fetchone() for key in keys]
                missing = {key: text for key, text, row in zip(keys, chunk, vectors) if row is None}
                if missing:
                    batch = self.encode(list(missing.values()), DOCUMENT_INSTRUCTION).astype("<f2")
                    new = {key: (row.tobytes(),) for key, row in zip(missing, batch)}
                    with db:
                        db.executemany("INSERT OR IGNORE INTO embeddings VALUES (?, ?)",
                                       ((key, row[0]) for key, row in new.items()))
                    vectors = [new[key] if row is None else row for key, row in zip(keys, vectors)]
                yield list(range(offset, offset + len(chunk))), np.array(
                    [np.frombuffer(row[0], dtype="<f2") for row in vectors])
                done = offset + len(chunk)
                if offset == 0 or done % (batch_size * 8) == 0 or done == len(texts):
                    print(f"Context embeddings: {done}/{len(texts)}", file=sys.stderr, flush=True)


def restrict_documents(documents, mailbox: str | None, day: str | None):
    return {doc_id: doc for doc_id, doc in documents.items()
            if (mailbox is None or doc.synthetic_user == mailbox) and (day is None or doc.doc_dt == day)}


def benchmark_documents(documents, benchmark, paraphrases):
    periods = {}
    for query in (*benchmark.queries.values(), *paraphrases.queries.values()):
        period = parse_query(query.query, query.query_date).period
        if period is not None:
            periods.setdefault(query.synthetic_user, set()).add(period)
    return {doc_id: doc for doc_id, doc in documents.items()
            if any(period.contains(doc.doc_dt) for period in periods.get(doc.synthetic_user, ()))}


def index_query_vectors(source: Path, destination: Path, encoder: QwenEncoder, batch_size: int):
    queries = load_query_file(source)
    if not queries:
        raise ValueError("No queries to index")
    texts = [parse_query(query.query, query.query_date).topic or query.query for query in queries.values()]
    vectors = np.concatenate([
        encoder.encode(texts[start:start + batch_size], QUERY_INSTRUCTION).astype(np.float16)
        for start in range(0, len(texts), batch_size)
    ])
    info = {"queries": len(queries), "query_ids": list(queries), "dimension": encoder.dimension}
    key = sha256(json.dumps([encoder.signature, list(queries), texts], ensure_ascii=False).encode()).hexdigest()
    save_index(destination, vectors, key, info)
    return {"queries": len(queries), "dimension": encoder.dimension, "bytes": destination.stat().st_size}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("index", "index-queries", "search", "evaluate"))
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--cache", type=Path, help="Reuse successful document batches across index runs")
    parser.add_argument("--port", type=int, default=18767)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--text-chars", type=int, default=220)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--benchmark-days", action="store_true")
    parser.add_argument("--scope-mailbox", help="Index one mailbox for a smoke test")
    parser.add_argument("--scope-day", type=date.fromisoformat, help="Index one received day for a smoke test")
    parser.add_argument("--paraphrases", type=Path, default=Path("tests/paraphrases.jsonl"))
    parser.add_argument("--query")
    parser.add_argument("--queries", type=Path, help="Use this benchmark-format JSONL for index-queries")
    parser.add_argument("--mailbox")
    parser.add_argument("--as-of")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535 or not 1 <= args.batch_size <= 32 or not 32 <= args.text_chars <= 384 or args.k < 1:
        parser.error("Invalid port, batch size, context length or K")
    if args.mode == "search" and not (args.query and args.mailbox):
        parser.error("search requires --query and --mailbox")
    if args.mode == "evaluate" and (args.scope_mailbox or args.scope_day):
        parser.error("Use the full archive or --benchmark-days to evaluate")
    cache = args.cache or args.index.with_suffix(".sqlite")
    inputs = {path.resolve() for path in (args.queries, args.paraphrases) if path}
    if len({args.index.resolve(), cache.resolve()}) != 2 or any(
            path.resolve() in inputs or args.data_dir.resolve() in path.resolve().parents
            for path in (args.index, cache)):
        parser.error("Use separate index and cache files outside the data directory")
    encoder = QwenEncoder(args.port, args.text_chars, cache)
    if args.mode == "index-queries":
        info = index_query_vectors(args.queries or args.data_dir / "queries_dev.jsonl", args.index,
                                   encoder, args.batch_size)
        print(json.dumps(info))
        return
    if args.mode in {"search", "evaluate"} and not args.index.is_file():
        parser.error("Build the index before searching or evaluating")

    if args.benchmark_days or args.mode == "evaluate":
        original = load_dataset(args.data_dir)
        documents = original.documents
        benchmark, groups = _load_paraphrase_benchmark(original, args.paraphrases)
        heldout = _select_queries(benchmark, groups["test"])
    else:
        documents = load_documents(args.data_dir)
    if args.benchmark_days:
        documents = benchmark_documents(documents, original, heldout)
    day = args.scope_day.isoformat() if args.scope_day else None
    if args.scope_mailbox or day:
        documents = restrict_documents(documents, args.scope_mailbox, day)
    if not documents:
        parser.error("No documents match the index scope")
    engine = TinySearch(documents, encoder, args.index, args.batch_size)
    result = {"scope": {"benchmark_days": args.benchmark_days, "mailbox": args.scope_mailbox, "day": day},
              "index": engine.index_info}
    if args.mode == "search":
        result["hits"] = [{"id": doc_id, "date": documents[doc_id].doc_dt, "subject": documents[doc_id].subject}
                          for doc_id, _ in engine.search(args.query, args.mailbox, args.as_of, args.k)]
    if args.mode == "evaluate":
        for name, dataset in (("public", original), ("heldout", heldout)):
            predictions, latency = run_search(dataset, engine, max(10, args.k), args.k, warmup=0)
            result[name] = {**score_predictions(dataset, predictions, (5, 10)), "timing": latency}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
