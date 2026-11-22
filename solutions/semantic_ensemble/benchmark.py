"""Tune the candidate mix, then compare final ranking and latency on VK."""

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
MIX_WEIGHTS = (0.10, 0.25, 0.50, 0.75, 0.90)


def candidate_recall(dataset, baseline, selectors):
    totals = dict.fromkeys(("period20", *selectors), 0.)
    cases = []
    for key, query in dataset.queries.items():
        truth = {doc_id for doc_id, rating in dataset.qrels[key].items() if rating > 0}
        parsed = parse_query(query.query, query.query_date)
        if not truth or parsed.period is None or not parsed.topic:
            continue
        eligible = set(baseline.period.in_period(query.synthetic_user, parsed, query.query_date))
        base = baseline.period.search_parsed(parsed, query.synthetic_user, query.query_date, 20)
        groups = {"period20": {doc_id for doc_id, _ in base if doc_id in eligible}}
        for name, search in selectors.items():
            hits = search.pool.shortlist(parsed, query.synthetic_user, query.query_date)
            groups[name] = {doc_id for doc_id, _ in hits}
        for name, group in groups.items():
            totals[name] += len(truth & group) / len(truth)
        cases.append({"query_id": key, "query": query.query,
                      "added_over_lexical30": sorted((truth & groups["hybrid30"]) - groups["lexical30"]),
                      "lost_from_lexical30": sorted((truth & groups["lexical30"]) - groups["hybrid30"])})
    return {key: round(value / len(cases), 6) if cases else None for key, value in totals.items()}, cases


def paraphrase_split(dataset, rows, split):
    ids = [row["query_id"] for row in rows if row.get("split") == split]
    return replace(dataset, queries={key: dataset.queries[key] for key in ids},
                   qrels={key: dataset.qrels[key] for key in ids})


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
    rows = [row for _, row in jsonl(args.paraphrases)]
    tune = paraphrase_split(paraphrases, rows, "tune")
    held_out = paraphrase_split(paraphrases, rows, "test")
    encoder = Encoder(args.model, max_tokens=80, threads=1, device="cpu")
    tower = FastMailEncoder(encoder, args.projection)
    adapter = QueryAdapter.load(args.adapter, encoder.signature, 80, encoder.dimension) if args.adapter else None
    vectors = TinySearch(data.documents, tower, args.index, adapter=adapter)
    context = ContextModel.load(args.context, tower.signature, tower.dimension)
    original = EnsembleSearch(QwenContextSearch(PeriodRanker(data.documents), vectors, context),
                              EnsembleOptions.load(args.config))
    word_vectors = Navec.load(args.word_vectors)
    by_weight = {weight: SemanticCandidateEnsemble(original, word_vectors, semantic_weight=weight)
                 for weight in (0., *MIX_WEIGHTS, 1.)}
    tuning = {}
    for weight, search in by_weight.items():
        hits, _ = run_search(tune, search, quality_k=10, timing_k=10, warmup=4)
        tuning[weight] = score_predictions(tune, hits, (5, 10))["metrics"]
    chosen = max(MIX_WEIGHTS, key=lambda w: (tuning[w]["@5"]["ndcg"], tuning[w]["@5"]["recall"],
                                              tuning[w]["@10"]["ndcg"], -w))
    print("Selected semantic weight:", chosen, flush=True)
    selectors = {"lexical30": by_weight[0.], "semantic30": by_weight[1.], "hybrid30": by_weight[chosen]}
    results = {"semantic_weight": chosen, "tuning": tuning}
    for name, dataset in (("public", data), ("held_out", held_out)):
        item = {}
        for label, search in {"ensemble": original, **selectors}.items():
            if isinstance(search, SemanticCandidateEnsemble):
                search.semantic_bm25.expansion.cache_clear()
                search.semantic_bm25.neighbors.cache_clear()
            prediction, timing = run_search(dataset, search, quality_k=100, timing_k=10, warmup=8)
            item[label] = {**score_predictions(dataset, prediction, CUTOFFS), "search_ms": timing["latency_ms"]}
            write_predictions(args.out / f"{name}-{label}.jsonl", prediction, 100)
        item["candidate_recall"], item["candidate_cases"] = candidate_recall(dataset, original, selectors)
        results[name] = item
        print(name, item["candidate_recall"], flush=True)
    (args.out / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
