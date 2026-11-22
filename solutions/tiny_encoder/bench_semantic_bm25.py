"""Tune contextual BM25 on development families and compare full-archive search modes."""

from argparse import ArgumentParser
from dataclasses import asdict, replace
from datetime import date, timedelta
from itertools import product
import json
from pathlib import Path
import sys

from vk_mail_search.data import Dataset, load_dataset
from vk_mail_search.evaluate import _load_paraphrase_benchmark, _select_queries, run_search, score_predictions
from vk_mail_search.query_parser import ParsedQuery, parse_query
from vk_mail_search.searchers import build_searcher

from .adapter import QueryAdapter
from .corpora import vk_corpus
from .encoder import Encoder
from .ensemble import EnsembleOptions, EnsembleSearch
from .index import TinySearch
from .qwen_context import ContextModel, QwenContextSearch
from .routed import RoutedSearch
from .semantic_bm25 import FeedbackProfile, SemanticBM25
from .student import FastMailEncoder


def evaluate(dataset, ids, answers):
    return score_predictions(_select_queries(dataset, list(ids)), {qid: answers[qid] for qid in ids},
                             (5, 10, 100))["metrics"]


def mask_dates(dataset, days=0):
    queries = {}
    for qid, q in dataset.queries.items():
        text = parse_query(q.query, q.query_date).topic
        query_date = (date.fromisoformat(q.query_date) + timedelta(days=days)).isoformat()
        queries[qid] = replace(q, query=text, query_date=query_date)
    return Dataset(dataset.documents, queries, dataset.qrels)


def main():
    parser = ArgumentParser(description=__doc__)
    for name in ("data-dir", "model", "projection", "index", "context", "ensemble", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--paraphrases", type=Path, default=Path("tests/paraphrases.jsonl"))
    parser.add_argument("--feedback", choices=("plain", "gated"), default="gated")
    args = parser.parse_args()
    if not args.index.is_file():
        parser.error("Build the full archive index first")
    original = load_dataset(args.data_dir)
    paraphrases, splits = _load_paraphrase_benchmark(original, args.paraphrases)
    corpus = vk_corpus(args.data_dir, args.paraphrases, expanded=True)
    canonical_dev = [qid for qid in corpus.train + corpus.validation if qid in original.queries]
    tune, heldout = splits["tune"], splits["test"]
    all_data = Dataset(original.documents, {**original.queries, **paraphrases.queries},
                       {**original.qrels, **paraphrases.qrels})
    masked = mask_dates(all_data)
    shifted = mask_dates(all_data, 30)
    encoder = Encoder(args.model, max_tokens=80, threads=1, device="cpu")
    tower = FastMailEncoder(encoder, args.projection)
    adapter = QueryAdapter.load(args.adapter, encoder.signature, 80, encoder.dimension) if args.adapter else None
    vector = TinySearch(original.documents, tower, args.index, adapter=adapter)
    period = build_searcher("period", original.documents)
    dated_bm25 = build_searcher("dated:bm25", original.documents)
    context_model = ContextModel.load(args.context, tower.signature, tower.dimension)
    context = QwenContextSearch(period, vector, context_model)
    ensemble = EnsembleSearch(context, EnsembleOptions.load(args.ensemble))
    initial = FeedbackProfile(8, 6, 0.)
    semantic = SemanticBM25(period, vector, initial, initial, rerank=True)
    parsed = {qid: parse_query(q.query, q.query_date) for qid, q in all_data.queries.items()}
    raw = {}
    ids = list(all_data.queries)
    for rows, vectors in encoder.batches([parsed[qid].topic for qid in ids], 32):
        raw.update({ids[row]: value for row, value in zip(rows, vectors, strict=True)})

    def candidates(dataset, ids, profile, *, with_date):
        result = {}
        for qid in ids:
            q = dataset.queries[qid]
            form = parsed[qid] if with_date else ParsedQuery(q.query, q.query, None)
            pool = semantic.candidates(form, q.synthetic_user, q.query_date, profile, query_vector=raw[qid])
            result[qid] = form, pool
        return result

    def predictions(prepared, ids, weight, *, rerank):
        if rerank:
            return {qid: [doc_id for doc_id, _ in semantic.order(*prepared[qid], weight, 100)] for qid in ids}
        return {qid: [doc_id for doc_id, _ in (prepared[qid][1].rank(weight) + prepared[qid][1].tail)[:100]]
                for qid in ids}

    date_base = {qid: [doc_id for doc_id, _ in period.search(q.query, q.synthetic_user, q.query_date, 100)]
                 for qid, q in ((qid, all_data.queries[qid]) for qid in canonical_dev)}
    floor = evaluate(all_data, canonical_dev, date_base)["@5"]["ndcg"] - .005
    best_date, best_undated = (-1., initial, {}), (-1., initial, {})
    dated_dev = tune + canonical_dev
    grid = (list(product((4, 8, 16), (4, 8), (0., .25), (0.,), (0.,))) if args.feedback == "plain"
            else list(product((4, 8), (8,), (0., .25), (.70, 1.), (0., .4))))
    for neighbors, max_terms, share, gate, dense in grid:
        draft = FeedbackProfile(neighbors, max_terms, 1., share, gate, dense)
        with_date = candidates(all_data, dated_dev, draft, with_date=True)
        no_date = candidates(masked, tune, draft, with_date=False)
        later = candidates(shifted, tune, draft, with_date=False)
        for weight in (0., .15, .35, .60, .90):
            draft_setting = replace(draft, weight=weight)
            reranked = predictions(with_date, dated_dev, weight, rerank=True)
            if evaluate(all_data, canonical_dev, reranked)["@5"]["ndcg"] >= floor:
                m = evaluate(all_data, tune, reranked)
                first = evaluate(all_data, tune, predictions(with_date, tune, weight, rerank=False))
                value = (.45 * m["@5"]["ndcg"] + .25 * m["@5"]["recall"] + .15 * m["@10"]["ndcg"]
                         + .15 * first["@5"]["ndcg"])
                if value > best_date[0]:
                    best_date = value, draft_setting, {"rerank": m, "first": first}
            normal = evaluate(masked, tune, predictions(no_date, tune, weight, rerank=False))
            delayed = evaluate(shifted, tune, predictions(later, tune, weight, rerank=False))
            value = (.35 * normal["@10"]["ndcg"] + .35 * delayed["@10"]["ndcg"]
                     + .15 * normal["@100"]["recall"] + .15 * delayed["@100"]["recall"])
            if value > best_undated[0]:
                best_undated = value, draft_setting, {"masked": normal, "plus_30_days": delayed}
    profiles = {"dated": asdict(best_date[1]), "undated": asdict(best_undated[1])}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(profiles, indent=2) + "\n", encoding="utf-8")
    report = {"profiles": profiles, "tune": {"dated": best_date[2], "undated": best_undated[2]}, "slices": {}}
    hybrid = RoutedSearch(period, vector, fuse_undated=True)
    routes = {
        "bm25": period.inner, "dated:bm25": dated_bm25, "period": period,
        "period:cover": build_searcher("period:cover", original.documents), "rubert": vector,
        "auto": RoutedSearch(period, vector), "hybrid": hybrid, "context": context,
        "ensemble": ensemble,
        "semantic_bm25": SemanticBM25(period, vector, best_date[1], best_undated[1]),
        "semantic_bm25_period": SemanticBM25(period, vector, best_date[1], best_undated[1], rerank=True),
    }
    for label, selected in (("public", list(original.queries)), ("heldout", heldout)):
        for suffix, datasets in (("dated", all_data), ("masked", masked), ("masked+30d", shifted)):
            scope = _select_queries(datasets, selected)
            key = f"{label}_{suffix}"
            report["slices"][key] = {}
            for name, engine in routes.items():
                predictions_by_id, timing = run_search(scope, engine, 100, 5, warmup=3)
                report["slices"][key][name] = {**score_predictions(scope, predictions_by_id, (5, 10, 100))["metrics"],
                                                "timing": timing}
            print(f"completed {key}", file=sys.stderr, flush=True)
    examples = {}
    for qid in heldout:
        p = candidates(all_data, [qid], best_date[1], with_date=True)[qid][1]
        retrieved = [doc_id for doc_id, _ in p.rank(best_date[1].weight)[:5]]
        new = sum(p.original.get(doc_id, 0.) == 0. and all_data.qrels[qid].get(doc_id, 0) > 0
                  for doc_id in retrieved)
        if new:
            examples[qid] = {"expansion": [term for term, _ in p.expansion], "new_positives_in_top5": new}
    report["heldout_lexically_unmatched_examples"] = examples
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
