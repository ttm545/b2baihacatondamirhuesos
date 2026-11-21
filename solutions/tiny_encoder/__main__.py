"""Index the archive, export document IDs, or evaluate binary relevance."""

from argparse import ArgumentParser
from collections import Counter
from pathlib import Path
from time import perf_counter
import json

from vk_mail_search.data import Dataset, load_dataset, load_documents
from vk_mail_search.evaluate import (_load_paraphrase_benchmark, _select_queries, run_search,
                                     score_predictions, write_predictions)
from vk_mail_search.predict import read_queries, warn
from vk_mail_search.query_parser import ParsedQuery, parse_query

from .adapter import QueryAdapter
from .encoder import Encoder
from .index import TinySearch
from .student import FastMailEncoder


def binary_dataset(dataset: Dataset) -> Dataset:
    return Dataset(dataset.documents, dataset.queries,
                   {qid: {doc_id: int(grade > 0) for doc_id, grade in labels.items()}
                    for qid, labels in dataset.qrels.items()})


def export(engine, queries: Path, destination: Path, cutoff: int):
    latest = {}
    for doc in engine.documents:
        previous = latest.get(doc.synthetic_user)
        if previous is None or (doc.doc_dt is not None and doc.doc_dt > previous):
            latest[doc.synthetic_user] = doc.doc_dt
    output, warnings = {}, Counter()
    for query in read_queries(queries, warnings):
        ids = []
        if query.mailbox is not None and query.mailbox not in latest:
            warn(queries, query.line, warnings, "unknown_mailbox", "неизвестный ящик")
        elif query.mailbox is not None and query.text:
            anchor = query.as_of or latest[query.mailbox]
            parsed = ParsedQuery(query.text, query.text, None)
            if anchor is not None:
                try:
                    parsed = parse_query(query.text, anchor)
                except (ValueError, OverflowError):
                    warn(queries, query.line, warnings, "invalid_period", "не удалось разобрать период; поиск по теме")
            ids = [doc_id for doc_id, _ in engine.search_parsed(parsed, query.mailbox, query.as_of, cutoff)]
        output[query.query_id] = ids
    write_predictions(destination, output, cutoff)
    return {"queries": len(output), "empty": sum(not doc_ids for doc_ids in output.values()), "warnings": dict(warnings)}


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("index", "evaluate", "predict"))
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--projection", type=Path, help="Use the distilled fast document tower")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--queries", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--paraphrases", type=Path)
    parser.add_argument("--split", choices=("tune", "test"))
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    if args.k < 1 or args.batch_size < 1:
        parser.error("K and batch size must be positive")
    if args.split and not args.paraphrases:
        parser.error("--split requires --paraphrases")
    if args.mode == "predict" and (args.queries is None or args.out is None):
        parser.error("predict requires --queries and --out")
    protected = {candidate.resolve() for candidate in (args.queries, args.paraphrases, args.adapter, args.projection,
                 *(args.data_dir / name for name in ("documents.jsonl", "qrels_dev.jsonl", "queries_dev.jsonl"))) if candidate}
    if args.index.resolve() in protected or args.model.resolve() in args.index.resolve().parents:
        parser.error("The index path must be outside the inputs")
    if args.out and (args.out.resolve() in protected | {args.index.resolve()} or args.model.resolve() in args.out.resolve().parents):
        parser.error("The output path must be outside the inputs")

    start = perf_counter()
    data = load_dataset(args.data_dir) if args.mode == "evaluate" else None
    documents = data.documents if data else load_documents(args.data_dir)
    encoder = Encoder(args.model, max_tokens=args.max_tokens, threads=args.threads, device=args.device)
    adapter = QueryAdapter.load(args.adapter, encoder.signature, args.max_tokens, encoder.dimension) if args.adapter else None
    tower = FastMailEncoder(encoder, args.projection) if args.projection else encoder
    engine = TinySearch(documents, tower, args.index, args.batch_size, args.rebuild, adapter)
    report = {"index": engine.index_info, "dimension": encoder.dimension, "parameters": encoder.parameters,
              "max_tokens": args.max_tokens, "device": args.device, "threads": args.threads,
              "document_tower": "distilled" if args.projection else "transformer",
              "startup_seconds": round(perf_counter() - start, 3), "index_bytes": args.index.stat().st_size}
    if adapter is not None:
        report["adapter_parameters"] = sum(parameter.numel() for parameter in adapter.parameters())
    if args.mode == "predict":
        report["export"] = export(engine, args.queries, args.out, args.k)
    if data:
        if args.paraphrases:
            data, groups = _load_paraphrase_benchmark(data, args.paraphrases)
            if args.split:
                data = _select_queries(data, groups[args.split])
        data = binary_dataset(data)
        predictions, latency = run_search(data, engine, max(100, args.k), args.k, warmup=10)
        cutoffs = tuple(sorted({1, 5, 10, 20, 100, args.k}))
        ideal = {qid: [doc_id for doc_id, grade in data.qrels.get(qid, {}).items() if grade > 0]
                 for qid in data.queries}
        report["metrics"] = score_predictions(data, predictions, cutoffs)
        report["oracle"] = score_predictions(data, ideal, cutoffs)["metrics"]
        full_gain_candidates = [qid for qid, target in ideal.items() if len(target) >= args.k]
        report["binary_gain"] = {"k": args.k, "can_fill_top_k": len(full_gain_candidates),
                                 "all_top_k_relevant": sum(len(predictions[qid]) >= args.k and all(
                                     data.qrels[qid].get(doc_id, 0) == 1 for doc_id in predictions[qid][:args.k]
                                 ) for qid in full_gain_candidates)}
        report["search_timing"] = latency
        if args.out:
            write_predictions(args.out, predictions, args.k)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
