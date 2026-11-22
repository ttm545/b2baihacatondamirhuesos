"""Compare soft word counts with exact counts and period on the same requests."""

from argparse import ArgumentParser
from dataclasses import replace
from functools import partial
import json
from pathlib import Path
from time import perf_counter

from navec import Navec

from vk_mail_search.data import jsonl, load_dataset, percentile
from vk_mail_search.evaluate import load_paraphrases, score_predictions, write_predictions
from vk_mail_search.period_rank import PeriodRanker
from vk_mail_search.query_parser import parse_query

from .search import WordVectorSearch


CUTOFFS = (1, 2, 3, 5, 10, 100)
THRESHOLDS = (0.25, 0.35, 0.45, 0.55, 0.65)


def evaluate(search, dataset):
    elapsed = []
    predictions = {}
    for key, query in dataset.queries.items():
        start = perf_counter()
        hits = search(query.query, query.synthetic_user, query.query_date, 100)
        elapsed.append((perf_counter() - start) * 1000)
        predictions[key] = [doc_id for doc_id, _ in hits]
    metrics = score_predictions(dataset, predictions, CUTOFFS)
    metrics["search_ms"] = {f"p{p}": round(percentile(elapsed, p / 100), 3) for p in (50, 95, 99)}
    return metrics, predictions


def split_paraphrases(dataset, path):
    dataset = load_paraphrases(dataset, path)
    splits = {}
    for name in ("tune", "test"):
        ids = [row["query_id"] for _, row in jsonl(path) if row.get("split") == name]
        splits[name] = replace(dataset, queries={key: dataset.queries[key] for key in ids},
                               qrels={key: dataset.qrels[key] for key in ids})
    return splits


def audit_pairs(period, searchers, dataset):
    results = []
    for _, row in jsonl(Path(__file__).with_name("pairs.jsonl")):
        wanted, other = (dataset.documents[row[key]] for key in ("wanted", "other"))
        assert wanted.doc_dt == other.doc_dt and wanted.synthetic_user == other.synthetic_user
        as_of, mailbox = wanted.doc_dt, wanted.synthetic_user
        parsed = parse_query(row["query"], as_of)
        assert parsed.period.start.isoformat() == parsed.period.end.isoformat() == as_of
        query_terms = period.topic_terms(parsed.topic)
        document_terms = set(period._fields[wanted.doc_id].subject) | period._fields[wanted.doc_id].counts.keys()
        shared_prefixes = {term for term in query_terms if term[:5] in {word[:5] for word in document_terms}}
        positions = {}
        for name, search in searchers.items():
            ids = [doc_id for doc_id, _ in search(row["query"], mailbox, as_of, 100)]
            positions[name] = [ids.index(doc_id) + 1 if doc_id in ids else 101
                               for doc_id in (wanted.doc_id, other.doc_id)]
        results.append({**row, "shared_prefixes": sorted(shared_prefixes), "positions": positions})
    no_overlap = [row for row in results if not row["shared_prefixes"]]
    return {
        "examples": results,
        "zero_prefix_overlap_queries": len(no_overlap),
        "zero_prefix_overlap_correct_preference": {
            name: sum(row["positions"][name][0] < row["positions"][name][1] for row in no_overlap)
            for name in searchers
        },
    }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--paraphrases", type=Path, default=Path("tests/paraphrases.jsonl"))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(args.data_dir)
    start = perf_counter()
    period = PeriodRanker(dataset.documents)
    period_index_s = perf_counter() - start
    start = perf_counter()
    vectors = WordVectorSearch(period, Navec.load(args.model))
    vector_index_s = perf_counter() - start
    data = split_paraphrases(dataset, args.paraphrases)
    # Pick one parameter with tune only, before looking at either evaluation set.
    tune = {threshold: evaluate(partial(vectors.search, min_similarity=threshold), data["tune"])[0]
            for threshold in THRESHOLDS}
    chosen = max(tune, key=lambda threshold: (tune[threshold]["metrics"]["@10"]["ndcg"], threshold))
    engines = {
        "period": period.search,
        "exact": partial(vectors.search, min_similarity=1.0),
        "vectors": partial(vectors.search, min_similarity=chosen),
    }
    report = {
        "period_index_s": round(period_index_s, 3),
        "additional_vector_index_s_including_model_load": round(vector_index_s, 3),
        "vocabulary_size": len(vectors.term_ids),
        "embedded_vocabulary_size": len(vectors.vector_term_ids),
        "tune": tune,
        "selected_min_similarity": chosen,
        "sets": {},
    }
    for label, sample in (("public", dataset), ("held_out", data["test"])):
        report["sets"][label] = {}
        for name, search in engines.items():
            # Compare cold word-expansion caches at both slices.
            vectors.expansion.cache_clear()
            metrics, predictions = evaluate(search, sample)
            report["sets"][label][name] = metrics
            write_predictions(args.out / f"{label}-{name}.jsonl", predictions, 100)
    report["pairs"] = audit_pairs(period, engines, dataset)
    (args.out / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
    print(json.dumps({"selected_min_similarity": chosen, "sets": report["sets"],
                      "pairs": report["pairs"]["zero_prefix_overlap_correct_preference"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
