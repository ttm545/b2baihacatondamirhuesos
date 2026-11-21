"""Score ranked VK Mail results and measure warmed-up search latency."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
from time import perf_counter_ns

from .data import Dataset, identifier, jsonl, load_dataset, percentile
from .searchers import DEFAULT_SEARCHER, Searcher, build_searcher

STANDARD_K = (5, 10, 20)
PARAPHRASE_SPLITS = ("tune", "test")


def load_paraphrases(dataset: Dataset, path: Path) -> Dataset:
    return _load_paraphrase_benchmark(dataset, path)[0]


def _load_paraphrase_benchmark(dataset: Dataset, path: Path) -> tuple[Dataset, dict[str, list[str]]]:
    queries = {}
    qrels = {}
    groups: dict[str, list[str]] = {}
    source_splits: dict[str, str | None] = {}
    for line, row in jsonl(path):
        try:
            query_id = identifier(row["query_id"])
            source_id = identifier(row["source_query_id"])
            text = row["query"]
            if query_id in queries:
                raise ValueError("повтор query_id")
            if source_id not in dataset.queries:
                raise ValueError(f"неизвестный source_query_id {source_id!r}")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("перефразировка должна быть непустой строкой")
            split = row.get("split")
            if "split" in row and split not in PARAPHRASE_SPLITS:
                raise ValueError("split должен быть tune или test")
            if source_splits.setdefault(source_id, split) != split:
                raise ValueError(f"source_query_id {source_id!r} встречается в разных частях")
            if split is not None:
                groups.setdefault(split, []).append(query_id)
            queries[query_id] = replace(dataset.queries[source_id], query_id=query_id, query=text)
            qrels[query_id] = dataset.qrels.get(source_id, {})
        except (KeyError, ValueError) as exc:
            raise ValueError(f"{path.name}:{line}: {exc}") from exc
    if not queries:
        raise ValueError(f"В {path.name} нет перефразировок.")
    if groups and sum(map(len, groups.values())) != len(queries):
        raise ValueError(f"В {path.name} укажите split у всех перефразировок или у всех опустите.")
    return Dataset(dataset.documents, queries, qrels), groups


def _select_queries(dataset: Dataset, ids: list[str]) -> Dataset:
    return Dataset(dataset.documents, {qid: dataset.queries[qid] for qid in ids},
                   {qid: dataset.qrels[qid] for qid in ids})


def load_predictions(path: Path) -> dict[str, list[str]]:
    predictions = {}
    for line, row in jsonl(path):
        try:
            query_id = identifier(row["query_id"])
            values = row["doc_ids"]
            if not isinstance(values, list):
                raise ValueError("doc_ids должен быть массивом")
            if query_id in predictions:
                raise ValueError("повтор query_id")
            predictions[query_id] = [identifier(doc_id) for doc_id in values]
        except (KeyError, ValueError) as exc:
            raise ValueError(f"{path.name}:{line}: {exc}") from exc
    return predictions


def validate_predictions(dataset: Dataset, predictions: dict[str, list[str]]) -> None:
    missing = dataset.queries.keys() - predictions.keys()
    extra = predictions.keys() - dataset.queries.keys()
    if missing or extra:
        raise ValueError(f"Запросы не совпадают с бенчмарком: пропущено {len(missing)}, лишних {len(extra)}.")
    for query_id, ids in predictions.items():
        if len(ids) != len(set(ids)):
            raise ValueError(f"Запрос {query_id}: повтор doc_id.")
        query = dataset.queries[query_id]
        for doc_id in ids:
            doc = dataset.documents.get(doc_id)
            if doc is None:
                raise ValueError(f"Запрос {query_id}: неизвестный doc_id {doc_id}.")
            if not doc.available(query.synthetic_user, query.query_date):
                raise ValueError(f"Запрос {query_id}: письмо {doc_id} из другого ящика или из будущего.")


def _dcg(grades: list[float]) -> float:
    return sum((2 ** grade - 1) / math.log2(i + 2) for i, grade in enumerate(grades))


def score_predictions(dataset: Dataset, predictions: dict[str, list[str]], cutoffs: tuple[int, ...]) -> dict:
    validate_predictions(dataset, predictions)
    sums = {k: {"recall": 0.0, "ndcg": 0.0, "mrr": 0.0} for k in cutoffs}
    single_mrr = dict.fromkeys(cutoffs, 0.0)
    scored = 0
    single_scored = 0
    for query_id, ids in predictions.items():
        qrels = dataset.qrels.get(query_id, {})
        ideal = sorted((grade for grade in qrels.values() if grade > 0), reverse=True)
        if not ideal:
            continue
        scored += 1
        single = len(ideal) == 1
        single_scored += single
        grades = [qrels.get(doc_id, 0) for doc_id in ids]
        for k in cutoffs:
            top = grades[:k]
            first = next((i for i, grade in enumerate(top, 1) if grade > 0), None)
            bucket = sums[k]
            bucket["recall"] += sum(grade > 0 for grade in top) / len(ideal)
            bucket["ndcg"] += _dcg(top) / _dcg(ideal[:k])
            reciprocal_rank = 1 / first if first else 0.0
            bucket["mrr"] += reciprocal_rank
            if single:
                single_mrr[k] += reciprocal_rank
    return {
        "queries": len(dataset.queries),
        "scored_queries": scored,
        "single_relevant_queries": single_scored,
        "max_returned": max(map(len, predictions.values()), default=0),
        "metrics": {f"@{k}": {key: round(value / scored, 6) if scored else None
                              for key, value in values.items()} for k, values in sums.items()},
        "mrr_single": {f"@{k}": round(total / single_scored, 6) if single_scored else None
                       for k, total in single_mrr.items()},
    }


def run_search(dataset: Dataset, searcher: Searcher, quality_k: int, timing_k: int,
               warmup: int) -> tuple[dict[str, list[str]], dict]:
    queries = tuple(dataset.queries.values())

    def search_at(query, k: int) -> list[tuple[str, float]]:
        return searcher.search(query.query, query.synthetic_user, query.query_date, k)

    for i in range(warmup):
        search_at(queries[i % len(queries)], timing_k)

    elapsed = []
    predictions = {}
    for query in queries:
        start = perf_counter_ns()
        timed_hits = search_at(query, timing_k)
        elapsed.append((perf_counter_ns() - start) / 1_000_000)
        hits = timed_hits if timing_k == quality_k else search_at(query, quality_k)
        if len(hits) > quality_k or len(timed_hits) > timing_k:
            raise ValueError(f"Запрос {query.query_id}: поисковик вернул больше K результатов.")
        predictions[query.query_id] = [identifier(doc_id) for doc_id, _ in hits]
    latency = {f"p{k}": round(percentile(elapsed, k / 100), 3) for k in (50, 95, 99)}
    return predictions, {"k": timing_k, "warmup_queries": warmup, "latency_ms": latency}


def write_predictions(path: Path, predictions: dict[str, list[str]], k: int) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for query_id, ids in predictions.items():
            handle.write(json.dumps({"query_id": query_id, "doc_ids": ids[:k]}, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--searcher", default=None, help=f"поисковик (по умолчанию {DEFAULT_SEARCHER})")
    mode.add_argument("--pred", type=Path, help="оценить готовый JSONL")
    parser.add_argument("--paraphrases", type=Path, help="JSONL с query_id, source_query_id и query")
    parser.add_argument("--split", choices=PARAPHRASE_SPLITS, help="оценить только выбранную часть --paraphrases")
    parser.add_argument("--k", type=int, default=10, help="K для тайминга и экспорта; качество также считается на 5, 10, 20")
    parser.add_argument("--warmup", type=int, default=10, help="количество прогревочных вызовов")
    parser.add_argument("--data-dir")
    parser.add_argument("--out", type=Path, help="экспорт выдачи глубиной --k")
    args = parser.parse_args()
    if args.k < 1 or args.warmup < 0:
        parser.error("--k должен быть положительным, --warmup неотрицательным")
    if args.pred and args.out and args.pred.resolve() == args.out.resolve():
        parser.error("--out должен отличаться от входного --pred")
    if args.paraphrases and args.out and args.paraphrases.resolve() == args.out.resolve():
        parser.error("--out должен отличаться от входного --paraphrases")
    if args.split and not args.paraphrases:
        parser.error("--split используется вместе с --paraphrases")
    try:
        dataset = load_dataset(args.data_dir)
        groups = {}
        if args.paraphrases:
            dataset, groups = _load_paraphrase_benchmark(dataset, args.paraphrases)
            if args.split:
                if args.split not in groups:
                    raise ValueError(f"В {args.paraphrases.name} нет части {args.split}.")
                dataset = _select_queries(dataset, groups[args.split])
        cutoffs = tuple(sorted({*STANDARD_K, args.k}))
        if args.pred:
            predictions = load_predictions(args.pred)
            timing = None
        else:
            name = args.searcher or DEFAULT_SEARCHER
            predictions, timing = run_search(dataset, build_searcher(name, dataset.documents),
                                             max(cutoffs), args.k, args.warmup)
        report = score_predictions(dataset, predictions, cutoffs)
        report["search_timing"] = timing
        if args.split:
            report["split"] = args.split
        elif groups:
            report["splits"] = {
                split: score_predictions(_select_queries(dataset, groups[split]),
                                         {qid: predictions[qid] for qid in groups[split]}, cutoffs)
                for split in PARAPHRASE_SPLITS if split in groups
            }
        if args.out:
            write_predictions(args.out, predictions, args.k)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()
