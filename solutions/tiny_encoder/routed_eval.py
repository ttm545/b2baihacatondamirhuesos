"""Compare full-mail search after masking query dates while keeping the original judgments."""

from argparse import ArgumentParser
from dataclasses import replace
import json
from pathlib import Path

from vk_mail_search.data import Dataset, load_dataset
from vk_mail_search.evaluate import _load_paraphrase_benchmark, _select_queries, run_search, score_predictions
from vk_mail_search.query_parser import parse_query
from vk_mail_search.searchers import build_searcher

from .adapter import QueryAdapter
from .encoder import Encoder
from .index import TinySearch
from .routed import RoutedSearch
from .student import FastMailEncoder


def main():
    parser = ArgumentParser(description=__doc__)
    for name in ("data-dir", "model", "projection", "index"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--paraphrases", type=Path, default=Path("tests/paraphrases.jsonl"))
    args = parser.parse_args()
    if not args.index.is_file():
        parser.error("Build the full vector index before comparing search modes")
    original = load_dataset(args.data_dir)
    paraphrases, groups = _load_paraphrase_benchmark(original, args.paraphrases)
    heldout = _select_queries(paraphrases, groups["test"])
    encoder = Encoder(args.model, max_tokens=80, threads=1, device="cpu")
    adapter = QueryAdapter.load(args.adapter, encoder.signature, 80, encoder.dimension) if args.adapter else None
    vector = TinySearch(original.documents, FastMailEncoder(encoder, args.projection), args.index, adapter=adapter)
    period = build_searcher("period", original.documents)
    auto = RoutedSearch(period, vector)
    hybrid = RoutedSearch(period, vector, fuse_undated=True)
    for name, dataset in (("public", original), ("heldout", heldout)):
        topics = {qid: replace(query, query=parse_query(query.query, query.query_date).topic)
                  for qid, query in dataset.queries.items()}
        undated = Dataset(dataset.documents, topics, dataset.qrels)
        for key, engine in (("period", period), ("vector", vector), ("hybrid", hybrid)):
            rankings, timing = run_search(undated, engine, 100, 10, warmup=5)
            print(json.dumps({"slice": name, "mode": key, "date_masked": True,
                              "metrics": score_predictions(undated, rankings, (5, 10, 100))["metrics"],
                              "timing": timing}), flush=True)
        mismatches = 0
        for query in dataset.queries.values():
            params = (query.query, query.synthetic_user, query.query_date, 10)
            expected = period.search(*params)
            mismatches += any(engine.search(*params) != expected for engine in (auto, hybrid))
        print(json.dumps({"slice": name, "dated_queries": len(dataset.queries),
                          "period_mismatches": mismatches}), flush=True)
        if mismatches:
            raise RuntimeError("Automatic routing changed a dated query")


if __name__ == "__main__":
    main()
