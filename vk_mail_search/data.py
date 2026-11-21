"""Read the public VK Mail corpus and benchmark from VK_DATA_DIR."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterator

DEFAULT_DATA_DIR = "data"
HTML = re.compile(r"<(?:html|body|div|p|table|br|span|a)(?:\s|/?>)", re.I)
HEADER = re.compile(r"^(?:Subject|Тема)\s*:\s*(.+)$", re.I | re.M)
SENDER = re.compile(r"^(?:From|От|От кого|Отправитель)\s*:", re.I | re.M)
QUOTE = re.compile(r"^\s*>|исходное сообщение|original message|пересланное сообщение|forwarded message", re.I | re.M)


@dataclass(frozen=True, slots=True)
class Document:
    doc_id: str
    synthetic_user: str
    doc_dt: str | None
    doc: str

    @property
    def subject(self) -> str:
        match = HEADER.search(self.doc)
        if match:
            return match[1].strip()
        return self.doc.strip().split("\n", 1)[0][:160]

    def available(self, mailbox: str, as_of: str | None) -> bool:
        return self.synthetic_user == mailbox and (
            self.doc_dt is None or as_of is None or self.doc_dt <= as_of
        )


@dataclass(frozen=True, slots=True)
class Query:
    query_id: str
    synthetic_user: str
    query: str
    query_date: str


@dataclass(frozen=True, slots=True)
class Dataset:
    documents: dict[str, Document]
    queries: dict[str, Query]
    qrels: dict[str, dict[str, float]]


def data_dir(path: str | Path | None = None) -> Path:
    return Path(path or os.environ.get("VK_DATA_DIR") or DEFAULT_DATA_DIR).expanduser()


def jsonl(path: Path) -> Iterator[tuple[int, dict]]:
    try:
        with path.open(encoding="utf-8-sig") as file:
            for line_number, line in enumerate(file, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path.name}:{line_number}: некорректный JSON") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"{path.name}:{line_number}: ожидался JSON-объект")
                yield line_number, row
    except FileNotFoundError as exc:
        raise ValueError(f"Нет {path}. Укажите каталог распакованного ZIP через VK_DATA_DIR или --data-dir.") from exc


def identifier(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value).strip():
        raise ValueError("ожидался непустой идентификатор")
    return str(value)


def _text(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("ожидалась строка")
    return value


def _date(value: object) -> str:
    text = _text(value)
    if date.fromisoformat(text).isoformat() != text:
        raise ValueError("ожидалась дата YYYY-MM-DD")
    return text


def load_documents(path: str | Path | None = None) -> dict[str, Document]:
    documents = {}
    for line, row in jsonl(data_dir(path) / "documents.jsonl"):
        try:
            value = row["doc_dt"]
            item = Document(identifier(row["doc_id"]), identifier(row["synthetic_user"]),
                            None if value is None else _date(value), _text(row["doc"]))
            if item.doc_id in documents:
                raise ValueError("повтор doc_id")
            documents[item.doc_id] = item
        except (KeyError, ValueError) as exc:
            raise ValueError(f"documents.jsonl:{line}: {exc}") from exc
    if not documents:
        raise ValueError("В documents.jsonl нет писем.")
    return documents


def load_queries(path: str | Path | None = None) -> dict[str, Query]:
    queries = {}
    for line, row in jsonl(data_dir(path) / "queries_dev.jsonl"):
        try:
            query = Query(identifier(row["query_id"]), identifier(row["synthetic_user"]),
                          _text(row["query"]), _date(row["query_date"]))
            if query.query_id in queries:
                raise ValueError("повтор query_id")
            queries[query.query_id] = query
        except (KeyError, ValueError) as exc:
            raise ValueError(f"queries_dev.jsonl:{line}: {exc}") from exc
    if not queries:
        raise ValueError("В queries_dev.jsonl нет запросов.")
    return queries


def load_dataset(path: str | Path | None = None) -> Dataset:
    from math import isfinite

    documents = load_documents(path)
    queries = load_queries(path)
    qrels: dict[str, dict[str, float]] = {}
    for line, row in jsonl(data_dir(path) / "qrels_dev.jsonl"):
        try:
            query_id, doc_id = identifier(row["query_id"]), identifier(row["doc_id"])
            grade = row["relevance"]
            if isinstance(grade, bool) or not isinstance(grade, (int, float)) or not isfinite(grade) or grade < 0:
                raise ValueError("оценка релевантности должна быть неотрицательным числом")
            if query_id not in queries or doc_id not in documents:
                raise ValueError("неизвестный query_id или doc_id")
            row_qrels = qrels.setdefault(query_id, {})
            if doc_id in row_qrels:
                raise ValueError("повтор пары query_id/doc_id")
            row_qrels[doc_id] = grade
        except (KeyError, ValueError) as exc:
            raise ValueError(f"qrels_dev.jsonl:{line}: {exc}") from exc
    return Dataset(documents, queries, qrels)


def percentile(values, fraction: float):
    """Nearest rank percentile, including p99 on small query sets."""
    from math import ceil

    ordered = sorted(values)
    return ordered[max(0, ceil(len(ordered) * fraction) - 1)] if ordered else None


def describe(dataset: Dataset) -> dict:
    docs, queries, qrels = dataset.documents, dataset.queries, dataset.qrels
    lengths = [len(entry.doc) for entry in docs.values()]
    grades = [sum(score > 0 for score in qrels.get(query, {}).values()) for query in queries]
    invalid_positive = sum(
        score > 0 and not docs[doc_id].available(queries[query_id].synthetic_user, queries[query_id].query_date)
        for query_id, entries in qrels.items() for doc_id, score in entries.items()
    )
    patterns = {
        "time_words": r"вчер|позавчер|сегодня|недел|месяц|\b\d{4}-\d\d-\d\d\b",
        "sender_words": r"\bот кого\b|отправител|прислал|письмо от",
        "attachment_words": r"вложени|прикреп|\bpdf\b|\bфайл",
    }
    return {
        "documents": len(docs),
        "mailboxes": len({doc.synthetic_user for doc in docs.values()}),
        "document_characters": {f"p{k}": percentile(lengths, k / 100) for k in (50, 90, 95, 99, 100)},
        "explicit_subject_headers": sum(bool(HEADER.search(doc.doc)) for doc in docs.values()),
        "explicit_sender_headers": sum(bool(SENDER.search(doc.doc)) for doc in docs.values()),
        "html_like_documents": sum(bool(HTML.search(doc.doc)) for doc in docs.values()),
        "quote_markers": sum(bool(QUOTE.search(doc.doc)) for doc in docs.values()),
        "unknown_document_dates": sum(doc.doc_dt is None for doc in docs.values()),
        "queries": len(queries),
        "qrels_rows": sum(map(len, qrels.values())),
        "positive_rels_per_query": dict(sorted(Counter(grades).items())),
        "positive_rels_outside_filters": invalid_positive,
        "query_pattern_matches": {
            label: sum(bool(re.search(pattern, query.query, re.I)) for query in queries.values())
            for label, pattern in patterns.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir")
    args = parser.parse_args()
    try:
        print(json.dumps(describe(load_dataset(args.data_dir)), ensure_ascii=False, indent=2))
    except (ValueError, OSError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()
