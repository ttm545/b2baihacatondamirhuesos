"""Compare the original and semantic-top-30 ensemble on the public VK corpus."""

from argparse import ArgumentParser
from dataclasses import replace
import json
from pathlib import Path

from navec import Navec

from solutions.tiny_encoder.adapter import QueryAdapter
from solutions.tiny_encoder.encoder import Encoder
from solutions.tiny_encoder.ensemble import EnsembleOptions, EnsembleSearch
from solutions.tiny_encoder.index import TinySearch
from solutions.tiny_encoder.qwen_context import ContextModel, QwenContextSearch
from solutions.tiny_encoder.student import FastMailEncoder
from vk_mail_search.data import jsonl, load_dataset
from vk_mail_search.evaluate import load_paraphrases, run_search, score_predictions, write_predictions
from vk_mail_search.period_rank import PeriodRanker
from vk_mail_search.query_parser import parse_query

from .search import SemanticCandidateEnsemble


CUTOFFS = (1, 2, 3, 5, 10, 20, 30, 100)


def candidate_recall(dataset, baseline, semantic):
    totals = {key: 0. for key in ("period20", "period30", "semantic20", "semantic30")}
    cases = []
    for key, query in dataset.queries.items():
        truth = {doc_id for doc_id, rating in dataset.qrels[key].items() if rating > 0}
        args = query.query, query.synthetic_user, query.query_date
        parsed = parse_query(query.query, query.query_date)
        if not truth or parsed.period is None or not parsed.topic:
            continue
        eligible = set(baseline.period.in_period(query.synthetic_user, parsed, query.query_date))
        base = [doc_id for doc_id, _ in baseline.period.search(*args, 30) if doc_id in eligible]
        changed = [doc_id for doc_id, _ in semantic.pool.shortlist(parsed, query.synthetic_user, query.query_date)]
        recalls = {"period20": base[:20], "period30": base,
                   "semantic20": changed[:20], "semantic30": changed}
        for variant, ids in recalls.items():
            totals[variant] += len(truth.intersection(ids)) / len(truth)
        cases.append({"query_id": key, "query": query.query,
                      "added_from_outside_period20": sorted((truth & set(changed)) - set(base[:20])),
                      "added_from_outside_period30": sorted((truth & set(changed)) - set(base)),
                      "lost_from_period30": sorted((truth & set(base)) - set(changed))})
    return {key: round(value / len(cases), 6) if cases else None for key, value in totals.items()}, cases


def main():
    parser = ArgumentParser(description=__doc__)
    for name in ("data-dir", "model", "projection", "index", "context", "config", "word-vectors", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--paraphrases", type=Path, default=Path("tests/paraphrases.jsonl"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data = load_dataset(args.data_dir)
    paraphrases = load_paraphrases(data, args.paraphrases)
    ids = [row["query_id"] for _, row in jsonl(args.paraphrases) if row.get("split") == "test"]
    held_out = replace(paraphrases, queries={key: paraphrases.queries[key] for key in ids},
                      qrels={key: paraphrases.qrels[key] for key in ids})
    encoder = Encoder(args.model, max_tokens=80, threads=1, device="cpu")
    tower = FastMailEncoder(encoder, args.projection)
    adapter = QueryAdapter.load(args.adapter, encoder.signature, 80, encoder.dimension) if args.adapter else None
    vectors = TinySearch(data.documents, tower, args.index, adapter=adapter)
    context = ContextModel.load(args.context, tower.signature, tower.dimension)
    original = EnsembleSearch(QwenContextSearch(PeriodRanker(data.documents), vectors, context),
                              EnsembleOptions.load(args.config))
    semantic = SemanticCandidateEnsemble(original, Navec.load(args.word_vectors))
    results = {}
    for name, dataset in (("public", data), ("held_out", held_out)):
        item = {}
        for label, search in (("ensemble", original), ("semantic_ensemble", semantic)):
            prediction, timing = run_search(dataset, search, quality_k=100, timing_k=10, warmup=8)
            item[label] = {**score_predictions(dataset, prediction, CUTOFFS), "search_ms": timing["latency_ms"]}
            write_predictions(args.out / f"{name}-{label}.jsonl", prediction, 100)
        item["candidate_recall"], item["candidate_cases"] = candidate_recall(dataset, original, semantic)
        results[name] = item
        print(name, item["candidate_recall"], flush=True)
    (args.out / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
