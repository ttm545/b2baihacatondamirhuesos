"""Export ranked VK Mail predictions from documents and queries without relevance labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter_ns

from .data import data_dir, load_documents, load_query_file, percentile
from .searchers import build_searcher


def predict(query_path: Path, out_path: Path, corpus: Path, searcher_name: str, k: int) -> dict:
    if out_path.resolve() in {query_path.resolve(), (corpus / "documents.jsonl").resolve()}:
        raise ValueError("--out должен отличаться от входных файлов.")

    queries = load_query_file(query_path)
    documents = load_documents(corpus)
    start = perf_counter_ns()
    searcher = build_searcher(searcher_name, documents)
    index_ms = (perf_counter_ns() - start) / 1_000_000
    mailboxes = {document.synthetic_user for document in documents.values()}
    elapsed = []
    empty = 0

    # Replace the destination only after all predictions have been written successfully.
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=out_path.parent,
                                prefix=".predictions-", suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            for query in queries.values():
                ids: dict[str, None] = {}
                if not query.query.strip() or query.synthetic_user not in mailboxes:
                    reason = "пустой запрос" if not query.query.strip() else "неизвестный ящик"
                    print(f"Запрос {query.query_id!r}: {reason}", file=sys.stderr)
                else:
                    start = perf_counter_ns()
                    hits = searcher.search(query.query, query.synthetic_user, query.query_date, k)
                    for doc_id, _ in hits:
                        document = documents.get(doc_id)
                        if document and document.available(query.synthetic_user, query.query_date):
                            ids[doc_id] = None
                            if len(ids) == k:
                                break
                    elapsed.append((perf_counter_ns() - start) / 1_000_000)
                empty += not ids
                output.write(json.dumps({"query_id": query.query_id, "doc_ids": list(ids)}, ensure_ascii=False) + "\n")
        temporary.replace(out_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

    latency = {}
    for p in (50, 95, 99):
        value = percentile(elapsed, p / 100)
        latency[f"p{p}"] = round(value, 3) if value is not None else None
    return {"queries": len(queries), "empty": empty, "index_ms": round(index_ms, 3), "search_ms": latency}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True, help="входной JSONL с запросами")
    parser.add_argument("--out", type=Path, required=True, help="выходной JSONL с выдачей")
    parser.add_argument("--data-dir")
    parser.add_argument("--searcher", default="dated:bm25")
    parser.add_argument("--k", type=int, default=100)
    args = parser.parse_args(argv)
    if args.k < 1:
        parser.error("--k должен быть положительным.")
    try:
        summary = predict(args.queries, args.out, data_dir(args.data_dir), args.searcher, args.k)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"{exc}\n")
    print(json.dumps(summary, ensure_ascii=False), file=sys.stderr)


if __name__ == "__main__":
    main()
