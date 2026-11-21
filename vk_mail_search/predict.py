"""Export ranked VK Mail predictions from documents and queries without relevance labels."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter_ns

from .data import _date, data_dir, identifier, load_documents, percentile
from .dated import DatedSearcher
from .query_parser import parse_query
from .searchers import build_searcher


@dataclass(frozen=True, slots=True)
class PredictionQuery:
    line: int
    query_id: str
    mailbox: str | None
    text: str
    as_of: str | None


def warn(path: Path, line: int, problems: Counter[str], kind: str, detail: str) -> None:
    problems[kind] += 1
    print(f"{path.name}:{line}: {detail}", file=sys.stderr)


def read_queries(path: Path, problems: Counter[str]) -> Iterator[PredictionQuery]:
    seen = set()
    with path.open("rb") as source:
        for line, raw in enumerate(source, 1):
            try:
                text = raw.decode("utf-8-sig")
                if not text.strip():
                    continue
                row = json.loads(text)
            except (ValueError, RecursionError):
                warn(path, line, problems, "unreadable_row", "не удалось прочитать строку JSONL; пропуск")
                continue
            try:
                query_id = identifier(row.get("query_id") if isinstance(row, dict) else None)
                query_id.encode("utf-8")
            except ValueError:
                warn(path, line, problems, "missing_query_id", "нет корректного query_id; пропуск")
                continue
            if query_id in seen:
                warn(path, line, problems, "duplicate_query_id", f"повтор query_id {query_id!r}; пропуск")
                continue
            seen.add(query_id)

            try:
                mailbox = identifier(row.get("synthetic_user"))
            except ValueError:
                mailbox = None
                warn(path, line, problems, "missing_mailbox", "нет корректного ящика")
            text = row.get("query")
            if not isinstance(text, str) or not text.strip():
                text = ""
                warn(path, line, problems, "empty_query", "пустой запрос или неверный тип текста")
            as_of = None
            if row.get("query_date") is None:
                warn(path, line, problems, "missing_date", "нет даты запроса; поиск без верхней границы")
            else:
                try:
                    as_of = _date(row["query_date"])
                except ValueError:
                    warn(path, line, problems, "invalid_date", "некорректная дата запроса; поиск без верхней границы")
            yield PredictionQuery(line, query_id, mailbox, text, as_of)


def predict(query_path: Path, out_path: Path, corpus: Path, searcher_name: str, k: int) -> dict:
    if out_path.resolve() in {query_path.resolve(), (corpus / "documents.jsonl").resolve()}:
        raise ValueError("--out должен отличаться от входных файлов.")

    documents = load_documents(corpus)
    start = perf_counter_ns()
    searcher = build_searcher(searcher_name, documents)
    index_ms = (perf_counter_ns() - start) / 1_000_000
    mailboxes: dict[str, str | None] = {}
    for document in documents.values():
        latest = mailboxes.setdefault(document.synthetic_user, None)
        if document.doc_dt is not None and (latest is None or document.doc_dt > latest):
            mailboxes[document.synthetic_user] = document.doc_dt
    problems: Counter[str] = Counter()
    elapsed = []
    empty = 0
    queries = 0

    # Replace the destination only after all predictions have been written successfully.
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=out_path.parent,
                                prefix=".predictions-", suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            for query in read_queries(query_path, problems):
                queries += 1
                ids: dict[str, None] = {}
                mailbox = query.mailbox
                if mailbox is not None and mailbox not in mailboxes:
                    warn(query_path, query.line, problems, "unknown_mailbox", "неизвестный ящик")
                elif mailbox is not None and query.text:
                    start = perf_counter_ns()
                    if isinstance(searcher, DatedSearcher):
                        anchor = query.as_of or mailboxes[mailbox]
                        if anchor is None:
                            hits = searcher.inner.search(query.text, mailbox, None, k)
                        else:
                            try:
                                parsed = parse_query(query.text, anchor)
                            except (ValueError, OverflowError):
                                warn(query_path, query.line, problems, "invalid_period",
                                     "не удалось разобрать период; поиск по тексту")
                                hits = searcher.inner.search(query.text, mailbox, query.as_of, k)
                            else:
                                hits = searcher.search_parsed(parsed, mailbox, query.as_of, k)
                    else:
                        hits = searcher.search(query.text, mailbox, query.as_of, k)
                    for doc_id, _ in hits:
                        document = documents.get(doc_id)
                        if document and document.available(mailbox, query.as_of):
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
    return {"queries": queries, "empty": empty, "warnings": dict(problems),
            "index_ms": round(index_ms, 3), "search_ms": latency}


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
