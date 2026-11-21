"""Score one reranker with the shared evaluator; run as python -m scripts.bench_rerank."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
from time import monotonic, perf_counter_ns

from vk_mail_search.data import jsonl, load_dataset, percentile
from vk_mail_search.evaluate import score_predictions
from vk_mail_search.rerank import MODELS, RerankSearcher, TransformersScorer
from vk_mail_search.searchers import build_searcher


def remap_queries(source, path, split):
    queries, qrels = {}, {}
    for lineno, row in jsonl(path):
        if row.get("split") not in {"tune", "test"}:
            raise ValueError(f"{path}:{lineno}: expected a tune or test split")
        if row["split"] != split:
            continue
        original = source.queries[row["source_query_id"]]
        qid = str(row.get("paraphrase_id", row.get("query_id", f"paraphrase-{lineno}")))
        if qid in queries:
            raise ValueError(f"Duplicate paraphrase ID: {qid}")
        queries[qid] = replace(original, query_id=qid, query=row["query"])
        qrels[qid] = source.qrels.get(original.query_id, {})
    return replace(source, queries=queries, qrels=qrels)


def evaluate(dataset, searcher, warmup):
    queries = tuple(dataset.queries.values())
    if not queries:
        raise ValueError("No queries to evaluate")

    def search(query):
        return searcher.search(query.query, query.synthetic_user, query.query_date, 10)

    def wait_idle():
        if isinstance(searcher, RerankSearcher):
            searcher.wait_idle()

    for index in range(warmup):
        wait_idle()
        search(queries[index % len(queries)])
    before = getattr(searcher, "statistics", {})
    times, predictions = [], {}
    for query in queries:
        wait_idle()
        start = perf_counter_ns()
        predictions[query.query_id] = [doc_id for doc_id, _ in search(query)]
        times.append((perf_counter_ns() - start) / 1_000_000)
    result = score_predictions(dataset, predictions, (10,))
    result["latency_ms"] = {f"p{p}": round(percentile(times, p / 100), 3) for p in (50, 95, 99)}
    result["outcomes"] = {key: value - before.get(key, 0)
                          for key, value in getattr(searcher, "statistics", {}).items()}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["bm25", *MODELS], required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--threads", type=int)
    parser.add_argument("--top-n", type=int, choices=(10, 20, 30), nargs="+", default=[10, 20, 30])
    parser.add_argument("--budget-ms", type=float, nargs="+", default=[400])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--data-dir")
    parser.add_argument("--paraphrases", type=Path)
    parser.add_argument("--split", choices=("tune", "test"))
    args = parser.parse_args()
    if args.warmup < 0 or (args.threads is not None and args.threads < 1):
        parser.error("warmup must be nonnegative and threads positive")
    if bool(args.paraphrases) != bool(args.split):
        parser.error("pass --paraphrases and --split together")
    if args.split == "test" and args.model != "bm25" and (len(args.top_n) != 1 or len(args.budget_ms) != 1):
        parser.error("the held-out test accepts one model/N/budget configuration chosen on tune")
    if args.threads is not None:
        import torch
        torch.set_num_threads(args.threads)
    dataset = load_dataset(args.data_dir)
    if args.paraphrases:
        dataset = remap_queries(dataset, args.paraphrases, args.split)
    baseline = build_searcher("dated:bm25", dataset.documents)
    if args.model == "bm25":
        print(json.dumps(dict(model="bm25", **evaluate(dataset, baseline, args.warmup))), flush=True)
        return
    model = TransformersScorer(args.model, device=args.device, dtype=args.dtype)
    print(json.dumps(dict(model_id=MODELS[args.model][0], revision=MODELS[args.model][1],
                          parameters=model.parameter_count, device=args.device, dtype=args.dtype)), flush=True)
    if args.warmup:
        query = next(iter(dataset.queries.values()))
        hits = baseline.search(query.query, query.synthetic_user, query.query_date, 4)
        if hits:
            model.score(query.query, [dataset.documents[doc_id].doc for doc_id, _ in hits], monotonic() + 120)
    for budget_ms in args.budget_ms:
        for top_n in args.top_n:
            searcher = RerankSearcher(baseline, dataset.documents, model, top_n=top_n, budget_ms=budget_ms)
            try:
                result = evaluate(dataset, searcher, args.warmup)
            finally:
                searcher.close()
            print(json.dumps(dict(top_n=top_n, budget_ms=budget_ms, **result)), flush=True)


if __name__ == "__main__":
    main()
