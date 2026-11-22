"""Select a subject score weight, then evaluate it on the VK search data."""

from argparse import ArgumentParser
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter

from vk_mail_search.data import jsonl, load_dataset, percentile
from vk_mail_search.evaluate import load_paraphrases, score_predictions, write_predictions
from vk_mail_search.period_rank import PeriodRanker

from .models import MODELS
from .searcher import DEPTH, SubjectReranker, fuse


CUTOFFS = (1, 2, 3, 5, 10)
WEIGHTS = (0., 0.10, 0.20, 0.35, 0.50)


def collect(searcher, dataset, label):
    prepared = {}
    elapsed = []
    for index, (key, query) in enumerate(dataset.queries.items(), 1):
        start = perf_counter()
        hits, relevance = searcher.prepare(query.query, query.synthetic_user, query.query_date, DEPTH)
        elapsed.append((perf_counter() - start) * 1000)
        prepared[key] = (hits, relevance)
        if index % 12 == 0 or index == len(dataset.queries):
            print(f"{label}: {index}/{len(dataset.queries)}", flush=True)
    timings = {f"p{p}": round(percentile(elapsed, p / 100), 3) for p in (50, 95, 99)}
    return prepared, timings


def predictions(prepared, weight):
    return {key: [doc_id for doc_id, _ in fuse(hits, scores, weight)]
            for key, (hits, scores) in prepared.items()}


def paraphrase_split(dataset, path, split):
    selected = [row["query_id"] for _, row in jsonl(path) if row.get("split") == split]
    return replace(dataset, queries={key: dataset.queries[key] for key in selected},
                   qrels={key: dataset.qrels[key] for key in selected})


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--paraphrases", type=Path, default=Path("tests/paraphrases.jsonl"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--model", choices=MODELS, default="jina")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = load_dataset(args.data_dir)
    paraphrases = load_paraphrases(data, args.paraphrases)
    tune = paraphrase_split(paraphrases, args.paraphrases, "tune")
    held_out = paraphrase_split(paraphrases, args.paraphrases, "test")
    args.out.mkdir(parents=True, exist_ok=True)
    baseline = PeriodRanker(data.documents)
    scorer, weight = MODELS[args.model]
    searcher = SubjectReranker(baseline, data.documents, scorer(args.device), weight)
    for query in list(tune.queries.values())[:3]:
        searcher.prepare(query.query, query.synthetic_user, query.query_date, DEPTH)
    prepared, timings = collect(searcher, tune, "tune")
    comparison = {weight: score_predictions(tune, predictions(prepared, weight), CUTOFFS)
                  for weight in WEIGHTS}
    selected = max(WEIGHTS[1:], key=lambda weight: (comparison[weight]["metrics"]["@5"]["ndcg"], -weight))
    report = {"model": args.model, "device": args.device, "selected_weight": selected,
              "tune_metrics": comparison, "tune_search_ms": timings, "sets": {}}
    for name, dataset in (("held_out", held_out), ("public", data)):
        prepared, timings = collect(searcher, dataset, name)
        base, updated = (predictions(prepared, weight) for weight in (0., selected))
        item = {
            "period": score_predictions(dataset, base, CUTOFFS),
            "subject": score_predictions(dataset, updated, CUTOFFS),
            "search_ms": timings,
            "scored_subject_queries": sum(bool(scores) for _, scores in prepared.values()),
            "changed_queries": sum(base[key] != updated[key] for key in base),
        }
        report["sets"][name] = item
        write_predictions(args.out / f"{name}-period.jsonl", base, DEPTH)
        write_predictions(args.out / f"{name}-subject.jsonl", updated, DEPTH)
        audit = {key: {"query": dataset.queries[key].query, "period": base[key], "subject": updated[key],
                       "subject_scores": scores}
                 for key, (_, scores) in prepared.items()}
        (args.out / f"{name}-positions.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
                                                        encoding="utf-8")
    (args.out / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Selected weight: {selected}", flush=True)


if __name__ == "__main__":
    main()
