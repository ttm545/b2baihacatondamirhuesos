"""Train a small query adapter on public tune families, with grouped validation."""

from argparse import ArgumentParser
from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from vk_mail_search.data import jsonl, load_dataset
from vk_mail_search.evaluate import _load_paraphrase_benchmark, _select_queries, score_predictions
from vk_mail_search.query_parser import parse_query

from .__main__ import binary_dataset
from .adapter import QueryAdapter
from .encoder import Encoder
from .index import TinySearch


@dataclass(slots=True)
class Example:
    query: torch.Tensor
    candidates: torch.Tensor
    ids: list[str]
    positive: torch.Tensor


def prepare(engine, query, qrels, vector):
    parsed = parse_query(query.query, query.query_date)
    elig = engine.eligible(query.synthetic_user, query.query_date)
    rows = elig[engine.in_period(elig, parsed)] if parsed.period else elig
    ids = [engine.documents[row].doc_id for row in rows]
    return Example(torch.from_numpy(vector), torch.from_numpy(engine.vectors[rows]), ids,
                   torch.tensor([qrels.get(doc_id, 0) > 0 for doc_id in ids], dtype=torch.bool))


def fit(examples, dimension, penalty, epochs):
    torch.manual_seed(202621)
    model = QueryAdapter(dimension)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.015)
    train = []
    for family in examples.values():
        for example in family:
            matches, misses = example.candidates[example.positive], example.candidates[~example.positive]
            if len(matches) and len(misses):
                scores = misses @ example.query
                negatives = misses[scores.topk(min(32, len(misses))).indices]
                train.append((example.query, matches, negatives, 1 / len(family)))
    if not train:
        raise ValueError("Tune has no positive–negative pairs inside query scopes")

    for _ in range(epochs):
        optimizer.zero_grad()
        loss = 0.0
        for vector, positives, negatives, weight in train:
            revised = model(vector)
            pair_margins = positives @ revised - (negatives @ revised)[:, None]
            loss = loss + weight * (F.softplus(-20 * pair_margins).mean() + penalty * (1 - revised @ vector))
        loss.backward()
        optimizer.step()
    return model.eval()


@torch.inference_mode()
def order(example, model=None):
    vector = model(example.query) if model is not None else example.query
    ranking = torch.argsort(example.candidates @ vector, descending=True, stable=True).tolist()
    return [example.ids[index] for index in ranking[:100]]


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=80)
    parser.add_argument("--epochs", type=int, default=45)
    args = parser.parse_args()
    if not args.index.is_file() or args.index.resolve() == args.out.resolve() or args.epochs < 1:
        parser.error("Build the index first and choose a different adapter output path and positive epoch count")
    if args.model.resolve() in args.out.resolve().parents or args.data_dir.resolve() in args.out.resolve().parents:
        parser.error("Use an adapter output outside the model and dataset directories")

    dataset = binary_dataset(load_dataset(args.data_dir))
    source = Path("tests/paraphrases.jsonl")
    paraphrases, splits = _load_paraphrase_benchmark(dataset, source)
    tune = _select_queries(paraphrases, splits["tune"])
    records = {row["query_id"]: row["source_query_id"] for _, row in jsonl(source) if row["split"] == "tune"}
    encoder = Encoder(args.model, max_tokens=args.max_tokens, threads=1)
    engine = TinySearch(dataset.documents, encoder, args.index)
    train, validate = defaultdict(list), {}
    queries = {**tune.queries, **{qid: dataset.queries[qid] for qid in set(records.values())}}
    query_ids = list(queries)
    texts = [parse_query(query.query, query.query_date).topic for query in queries.values()]
    vectors = {}
    for indices, encoded in encoder.batches(texts, batch_size=32):
        vectors.update({query_ids[index]: row for index, row in zip(indices, encoded, strict=True)})
    originals_seen = set()
    for qid, query in tune.queries.items():
        family = qid.rsplit("-", 1)[0]
        validate[qid] = prepare(engine, query, tune.qrels[qid], vectors[qid])
        train[family].append(validate[qid])
        source_id = records[qid]
        if source_id not in originals_seen:
            originals_seen.add(source_id)
            train[family].append(prepare(engine, dataset.queries[source_id], dataset.qrels[source_id], vectors[source_id]))

    base = score_predictions(tune, {qid: order(example) for qid, example in validate.items()}, (5, 10))["metrics"]
    trials = []
    for penalty in (2., 8., 24.):
        predictions = {}
        for group in sorted(train):
            model = fit({family: examples for family, examples in train.items() if family != group},
                        encoder.dimension, penalty, args.epochs)
            for qid, example in validate.items():
                if qid.rsplit("-", 1)[0] == group:
                    predictions[qid] = order(example, model)
        report = score_predictions(tune, predictions, (5, 10))["metrics"]
        trials.append({"penalty": penalty, "metrics": report})
        print(json.dumps(trials[-1]), flush=True)
    selected = max(trials, key=lambda trial: (trial["metrics"]["@5"]["recall"], trial["metrics"]["@5"]["ndcg"]))
    adapter = fit(train, encoder.dimension, selected["penalty"], args.epochs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    adapter.save(args.out, encoder.signature, encoder.max_tokens)
    print(json.dumps({"baseline": base, "selected": selected, "groups": len(train), "tune_queries": len(tune.queries)}))


if __name__ == "__main__":
    main()
