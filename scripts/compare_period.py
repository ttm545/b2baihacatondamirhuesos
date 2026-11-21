"""Compare date-first ranking signals without fitting on the test split."""

import argparse
import json
from pathlib import Path
from time import perf_counter

from vk_mail_search.data import load_dataset, percentile
from vk_mail_search.evaluate import _load_paraphrase_benchmark, _select_queries, score_predictions
from vk_mail_search.period_rank import PROFILES, PeriodRanker


def normalized(values):
    low, high = min(values), max(values)
    if low == high:
        return [0.0] * len(values)
    return [(value - low) / (high - low) for value in values]


def model_passage(doc):
    return doc.subject[:230] + "\n" + doc.doc[:1300]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("tune", "test", "public"), default="tune")
    parser.add_argument("--minilm-dir", type=Path)
    parser.add_argument("--potion-dir", type=Path)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--model-depth", type=int, default=16)
    parser.add_argument("--lex-depth", type=int, default=60)
    parser.add_argument("--max-length", type=int, default=112)
    parser.add_argument("--group-templates", action="store_true")
    parser.add_argument("--profiles", nargs="+")
    args = parser.parse_args()

    dataset = load_dataset(args.data_dir)
    if args.split != "public":
        paraphrases, groups = _load_paraphrase_benchmark(dataset, Path("tests/paraphrases.jsonl"))
        dataset = _select_queries(paraphrases, groups[args.split])
    probes = ("baseline", *PROFILES)
    potion = None
    model = None
    if args.minilm_dir:
        from vk_mail_search.minilm import MiniLMScorer, grouped_passages

        model = MiniLMScorer(args.minilm_dir, threads=args.threads, max_length=args.max_length)
        probes = (*probes, "semantic", "semantic50", "semantic25", "semantic75", "semantic-gate")
    if args.potion_dir:
        from vk_mail_search.potion import PotionScorer

        potion = PotionScorer(args.potion_dir)
        probes = (*probes, "potion", "potion50", "potion25", "potion75", "potion-cover", "potion-cover25")
    if args.profiles:
        unknown = set(args.profiles) - set(probes)
        if unknown:
            parser.error(f"Unknown profiles: {sorted(unknown)}")
        probes = args.profiles

    exp = PeriodRanker(dataset.documents, depth=args.lex_depth)
    results = {mode: {} for mode in probes}
    times = {"candidates": [], "features": [], "model": [], "full_model_path": [], "potion": []}
    for qid, query in dataset.queries.items():
        start = perf_counter()
        parsed, pool, rest = exp.candidates(query.query, query.synthetic_user, query.query_date, 100)
        times["candidates"].append((perf_counter() - start) * 1000)
        if not pool:
            for records in results.values():
                records[qid] = [doc_id for doc_id, _ in rest]
            continue
        start = perf_counter()
        terms = exp.topic_terms(parsed.topic)
        features = exp.features(terms, pool)
        times["features"].append((perf_counter() - start) * 1000)
        candidates = pool[:args.model_depth] if not args.group_templates else pool
        if model:
            start = perf_counter()
            if args.group_templates:
                docs = [dataset.documents[doc_id] for doc_id, _ in candidates]
                passages, assigned = grouped_passages(docs, parsed.topic, args.model_depth)
                group_scores = model.score(parsed.topic, passages)
                floor = min(group_scores)
                raw_scores = [group_scores[idx] if idx is not None else floor for idx in assigned]
            else:
                raw_scores = model.score(parsed.topic, [model_passage(dataset.documents[doc_id]) for doc_id, _ in candidates])
            times["model"].append((perf_counter() - start) * 1000)
            times["full_model_path"].append(sum(times[name][-1] for name in ("candidates", "features", "model")))
            semantic = normalized(raw_scores)
            ranks = normalized([score for _, score in candidates])
        if potion:
            start = perf_counter()
            potion_scores = normalized(potion.score(terms, [dataset.documents[doc_id] for doc_id, _ in pool]))
            times["potion"].append((perf_counter() - start) * 1000)
            cover_scores = normalized([sum(value * weight for value, weight in zip(row, PROFILES["cover"], strict=True))
                                       for row in features])
            full_ranks = normalized([score for _, score in pool])
        for mode, records in results.items():
            if mode == "baseline":
                ranked = pool
            elif mode in PROFILES:
                ranked = exp.rank(pool, features, mode)
            elif mode.startswith("potion"):
                weight = {"potion": 1.0, "potion50": .50, "potion25": .25, "potion75": .75,
                          "potion-cover": .50, "potion-cover25": .25}[mode]
                reference = cover_scores if "cover" in mode else full_ranks
                mixed = [weight * s + (1 - weight) * r for s, r in zip(potion_scores, reference, strict=True)]
                ranked = sorted(zip((doc_id for doc_id, _ in pool), mixed, strict=True), key=lambda item: -item[1])
            else:
                weight = {"semantic": 1.0, "semantic50": .50, "semantic25": .25, "semantic75": .75,
                          "semantic-gate": .2 if features[0][2] >= .65 else .75}[mode]
                mixed = [weight * s + (1 - weight) * r for s, r in zip(semantic, ranks, strict=True)]
                ranked = sorted(zip((doc_id for doc_id, _ in candidates), mixed, strict=True), key=lambda item: -item[1])
                ranked += pool[len(candidates):]
            records[qid] = [doc_id for doc_id, _ in (ranked + rest)[:100]]
    for mode, predictions in results.items():
        report = score_predictions(dataset, predictions, (5, 10, 20, 100))["metrics"]
        print(json.dumps({"mode": mode, "split": args.split, **report}), flush=True)
    print(json.dumps({"p99_ms": {key: percentile(values, .99) for key, values in times.items() if values}}))


if __name__ == "__main__":
    main()
