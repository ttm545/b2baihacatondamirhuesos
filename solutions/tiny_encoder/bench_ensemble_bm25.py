"""Tune contextual BM25 inside the existing ensemble and compare both ensembles."""

from argparse import ArgumentParser
from collections import defaultdict
from copy import copy
from dataclasses import asdict, replace
import json
from pathlib import Path

from vk_mail_search.data import Dataset, jsonl, load_dataset
from vk_mail_search.evaluate import _load_paraphrase_benchmark
from vk_mail_search.query_parser import parse_query
from vk_mail_search.searchers import build_searcher

from .adapter import QueryAdapter
from .bench_semantic_bm25 import evaluate, mask_dates
from .corpora import vk_corpus
from .encoder import Encoder
from .ensemble import EnsembleOptions, EnsembleSearch
from .index import TinySearch
from .qwen_context import ContextModel, QwenContextSearch
from .semantic_bm25 import SemanticBM25, load_profiles
from .student import FastMailEncoder
from .tune import fit as fit_adapter, prepare as prepare_adapter


def main():
    parser = ArgumentParser(description=__doc__)
    for name in ("data-dir", "model", "projection", "index", "context", "ensemble", "feedback", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--rule", choices=("scale", "rank-mix"), default="rank-mix")
    parser.add_argument("--paraphrases", type=Path, default=Path("tests/paraphrases.jsonl"))
    args = parser.parse_args()
    published = load_dataset(args.data_dir)
    paraphrases, splits = _load_paraphrase_benchmark(published, args.paraphrases)
    data = Dataset(published.documents, {**published.queries, **paraphrases.queries},
                   {**published.qrels, **paraphrases.qrels})
    corpus = vk_corpus(args.data_dir, args.paraphrases, expanded=True)
    canonical_dev = [qid for qid in corpus.train + corpus.validation if qid in published.queries]
    tune, heldout = splits["tune"], splits["test"]
    masked, shifted = (mask_dates(data, offset) for offset in (0, 30))
    encoder = Encoder(args.model, max_tokens=80, threads=1, device="cpu")
    tower = FastMailEncoder(encoder, args.projection)
    adapter = QueryAdapter.load(args.adapter, encoder.signature, 80, encoder.dimension) if args.adapter else None
    vector = TinySearch(published.documents, tower, args.index, adapter=adapter)
    period = build_searcher("period", published.documents)
    model = ContextModel.load(args.context, tower.signature, tower.dimension)
    context = QwenContextSearch(period, vector, model)
    options = EnsembleOptions.load(args.ensemble)
    profiles = load_profiles(args.feedback)
    base = EnsembleSearch(context, options)

    def ranks(engine, dataset, ids):
        pred = {}
        for qid in ids:
            q = dataset.queries[qid]
            pred[qid] = [doc_id for doc_id, _ in engine.search(q.query, q.synthetic_user, q.query_date, 100)]
        return pred

    def score(engine, dataset, ids):
        return evaluate(dataset, ids, ranks(engine, dataset, ids))

    def scaled(profile, value):
        if args.rule == "rank-mix":
            return replace(profile, blend=value)
        return replace(profile, weight=profile.weight * value, dense_weight=profile.dense_weight * value)

    def create(date_scale, content_scale, runtime=context, template=None):
        feedback = SemanticBM25(period, runtime.vector, *profiles) if template is None else copy(template)
        feedback.dated = scaled(profiles[0], date_scale)
        feedback.undated = scaled(profiles[1], content_scale)
        return EnsembleSearch(runtime, options, semantic_bm25=feedback)

    trials = []
    cases = {"canonical_dev": (data, canonical_dev), "dated": (data, tune),
             "masked": (masked, tune), "masked+30d": (shifted, tune)}
    if args.rule == "scale":
        for scale in (0., .25, .50, 1.):
            branch = create(scale, scale) if scale else base
            trials.append({"scale": scale, **{key: score(branch, dataset, ids)
                                             for key, (dataset, ids) in cases.items()}})
    else:
        sources = {row["query_id"]: row["source_query_id"] for _, row in jsonl(args.paraphrases)}
        family = {}
        for qid in set(tune + canonical_dev):
            source = data.queries[sources.get(qid, qid)]
            family[qid] = parse_query(source.query, source.query_date).topic
        examples = defaultdict(list)
        if adapter is not None:
            for qid in sorted(set(tune) | {sources[qid] for qid in tune}):
                q = data.queries[qid]
                topic = parse_query(q.query, q.query_date).topic
                examples[family[qid]].append(prepare_adapter(vector, q, data.qrels[qid], encoder.query(topic)))
        scales = (0., .15, .35, .60, 1.)
        cv = {scale: {key: {} for key in cases} for scale in scales}
        for key in sorted(set(family.values())):
            fold_vector = copy(vector)
            if examples:
                fit = {group: items for group, items in examples.items() if key != group}
                fold_vector.adapter = fit_adapter(fit, tower.dimension, penalty=24., epochs=45)
            runtime = QwenContextSearch(period, fold_vector, model)
            template = SemanticBM25(period, fold_vector, *profiles)
            for scale in scales:
                branch = create(scale, scale, runtime, template)
                for label, (dataset, ids) in cases.items():
                    held = [qid for qid in ids if family[qid] == key]
                    cv[scale][label].update(ranks(branch, dataset, held))
        for scale in scales:
            trials.append({"scale": scale, **{key: evaluate(dataset, ids, cv[scale][key])
                                             for key, (dataset, ids) in cases.items()}})
    floor = trials[0]["canonical_dev"]["@5"]["ndcg"] - .002
    dated = max((item for item in trials if item["canonical_dev"]["@5"]["ndcg"] >= floor),
                key=lambda item: .55 * item["dated"]["@5"]["ndcg"] + .25 * item["dated"]["@5"]["recall"]
                + .20 * item["dated"]["@10"]["ndcg"])
    undated = max(trials, key=lambda item:
                  .35 * item["masked"]["@10"]["ndcg"] + .35 * item["masked+30d"]["@10"]["ndcg"]
                  + .15 * item["masked"]["@100"]["recall"] + .15 * item["masked+30d"]["@100"]["recall"])
    winner = create(dated["scale"], undated["scale"])
    final = {"dated": asdict(winner.semantic_bm25.dated), "undated": asdict(winner.semantic_bm25.undated)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")
    report = {"profiles": final, "rule": args.rule, "prepared": len(published.documents), "chosen_dated_scale": dated["scale"],
              "chosen_undated_scale": undated["scale"], "trials": trials, "slices": {}}
    for name, ids in (("public", list(published.queries)), ("heldout", heldout)):
        for label, dataset in (("dated", data), ("masked", masked), ("masked+30d", shifted)):
            report["slices"][f"{name}_{label}"] = {
                "ensemble": score(base, dataset, ids),
                "ensemble-semantic-bm25": score(winner, dataset, ids),
            }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
