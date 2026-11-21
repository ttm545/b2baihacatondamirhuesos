"""Train an independent PyTorch retrieval adapter for each public corpus."""

from argparse import ArgumentParser
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from vk_mail_search.data import Dataset, load_dataset
from vk_mail_search.evaluate import score_predictions
from vk_mail_search.query_parser import parse_query

from .__main__ import binary_dataset
from .adapter import QueryAdapter
from .corpora import arte_corpus, sara_corpus, vectrix_corpus, vk_corpus, vk_trusted_negatives, w3c_corpus
from .encoder import Encoder
from .index import TinySearch
from .metric_adapter import MetricAdapter
from .student import FastMailEncoder


@dataclass(frozen=True)
class QueryRow:
    vector: np.ndarray
    rows: np.ndarray
    in_period: np.ndarray
    positive: np.ndarray
    date_only: bool


def prepare_rows(dataset: Dataset, engine: TinySearch, tower: FastMailEncoder):
    parsed = {qid: parse_query(query.query, query.query_date) for qid, query in dataset.queries.items()}
    topics = list(dict.fromkeys(item.topic for item in parsed.values() if item.topic))
    encoded = {}
    for indices, vectors in tower.teacher.batches(topics, 24):
        encoded.update({topics[index]: value for index, value in zip(indices, vectors, strict=True)})
    fallback = np.zeros(tower.dimension, dtype=np.float32)
    answer = {}
    for qid, query in dataset.queries.items():
        rows = engine.eligible(query.synthetic_user, query.query_date)
        candidates = dataset.qrels.get(qid, {})
        positive = np.asarray([candidates.get(engine.documents[row].doc_id, 0) > 0 for row in rows], dtype=bool)
        answer[qid] = QueryRow(encoded.get(parsed[qid].topic, fallback), rows,
                               engine.in_period(rows, parsed[qid]), positive, not parsed[qid].topic)
    return answer


def training_batch(ids, rows, documents, trusted_negatives=None):
    rng = np.random.default_rng(260621)
    queries, positives, negatives, weights = [], [], [], []
    pos_limit, neg_limit = 24, 32
    for qid in ids:
        row = rows[qid]
        if row.date_only:
            continue
        # Explicit dates take priority at serving time. Fit comparisons inside
        # the period when it contains all gold labels and has useful negatives.
        scope = row.in_period if np.any(row.in_period) and not np.any(row.positive & ~row.in_period) else np.ones(len(row.rows), bool)
        local = np.flatnonzero(scope)
        positive = local[row.positive[local]]
        negative = local[~row.positive[local]]
        bonus = np.asarray(trusted_negatives.get(qid, ()) if trusted_negatives else (), dtype=np.intp)
        bonus = np.setdiff1d(bonus, row.rows[local], assume_unique=True)
        if not positive.size or (not negative.size and not bonus.size):
            continue
        hard = negative[np.argsort(-(documents[row.rows[negative]] @ row.vector), kind="stable")]
        if len(bonus):
            bonus = bonus[np.argsort(-(documents[bonus] @ row.vector), kind="stable")][:8]
        remaining = 8 - len(bonus)
        sampled = rng.choice(hard[24:], min(remaining, len(hard[24:])), replace=False)
        negative = np.concatenate((row.rows[hard[:24]], row.rows[sampled], bonus))
        if len(positive) > pos_limit:
            positive = rng.choice(positive, pos_limit, replace=False)
        queries.append(row.vector)
        positives.append(documents[row.rows[positive]])
        negatives.append(documents[negative])
        weights.append(np.asarray([2.5 if index < 5 else 1.0 for index in range(len(negative))], np.float32))
    if not queries:
        raise ValueError("No labeled positive–negative comparisons in training")
    dim = documents.shape[1]
    pos = np.zeros((len(queries), pos_limit, dim), dtype=np.float32)
    neg = np.zeros((len(queries), neg_limit, dim), dtype=np.float32)
    pair_weight = np.zeros((len(queries), pos_limit, neg_limit), dtype=np.float32)
    for i, (left, right, factor) in enumerate(zip(positives, negatives, weights, strict=True)):
        pos[i, :len(left)], neg[i, :len(right)] = left, right
        pair_weight[i, :len(left), :len(right)] = factor[None, :] / (len(left) * factor.sum())
    return tuple(torch.from_numpy(array) for array in (np.stack(queries), pos, neg, pair_weight))


def train_epoch(model, optimizer, batch, penalty, rng):
    order = torch.randperm(len(batch[0]), generator=rng)
    for indices in order.split(24):
        query, pos, neg, weights = (array[indices] for array in batch)
        query_projected = model.query_tower(query)
        pos_projected, neg_projected = model.document_tower(pos), model.document_tower(neg)
        pos_scores = (pos_projected * query_projected[:, None, :]).sum(dim=-1)
        neg_scores = (neg_projected * query_projected[:, None, :]).sum(dim=-1)
        gap = pos_scores[:, :, None] - neg_scores[:, None, :]
        ranking_loss = (F.softplus(-16 * gap) * weights).sum(dim=(1, 2)).mean()
        drift = (1 - F.cosine_similarity(query_projected, query)).mean()
        if model.dual:
            pos_weight, neg_weight = weights.sum(dim=2), weights.sum(dim=1)
            drift += .5 * ((1 - F.cosine_similarity(pos_projected, pos, dim=-1)) * pos_weight).sum(dim=1).mean()
            drift += .5 * ((1 - F.cosine_similarity(neg_projected, neg, dim=-1)) * neg_weight).sum(dim=1).mean()
        optimizer.zero_grad()
        (ranking_loss + penalty * drift).backward()
        optimizer.step()


def measured(dataset: Dataset, ids, rows, engine: TinySearch, model=None, older=None):
    documents = model.documents(engine.vectors) if model is not None else engine.vectors
    output = {}
    for qid in ids:
        row = rows[qid]
        if row.date_only:
            ranked = sorted(row.rows[row.in_period], key=lambda index: (
                engine.documents[index].doc_dt, engine.documents[index].doc_id), reverse=True)[:100]
        else:
            vector = model.query(row.vector) if model is not None else older.project(row.vector) if older else row.vector
            scores = documents[row.rows] @ vector
            ranked = row.rows[np.lexsort((row.rows, -scores, ~row.in_period))[:100]]
        output[qid] = [engine.documents[index].doc_id for index in ranked]
    subset = Dataset(dataset.documents, {qid: dataset.queries[qid] for qid in ids},
                     {qid: dataset.qrels.get(qid, {}) for qid in ids})
    report = score_predictions(subset, output, (5, 10, 100))
    ideal = {qid: [doc_id for doc_id, grade in subset.qrels[qid].items() if grade > 0] for qid in ids}
    can_fill = [qid for qid in ids if len(ideal[qid]) >= 5]
    complete = sum(len(output[qid]) >= 5 and set(output[qid][:5]).issubset(ideal[qid]) for qid in can_fill)
    return {"scored": report["scored_queries"], "metrics": report["metrics"],
            "complete_top5": complete, "eligible_complete_top5": len(can_fill),
            "oracle_recall5": score_predictions(subset, ideal, (5,))["metrics"]["@5"]["recall"]}


def tune(corpus, data, rows, engine, dimension, epochs, trusted_negatives=None):
    batch = training_batch(corpus.train, rows, engine.vectors, trusted_negatives)
    best, winner = None, None
    log = []
    for dual in (False, True):
        for penalty in (8., 24.):
            torch.manual_seed(260621)
            rng = torch.Generator().manual_seed(260621)
            model = MetricAdapter(dimension, dual=dual)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.015)
            for step in range(1, epochs + 1):
                train_epoch(model, optimizer, batch, penalty, rng)
                if step not in {max(1, epochs // 4), max(1, epochs // 2), epochs}:
                    continue
                result = measured(data, corpus.validation, rows, engine, model)
                row = {"dual": dual, "penalty": penalty, "epochs": step, **result["metrics"]["@5"]}
                log.append(row)
                score = row["ndcg"], row["recall"], row["mrr"]
                if best is None or score > best:
                    best, winner = score, (deepcopy(model).eval(), row)
    model, configuration = winner
    return model, configuration, len(batch[0]), log


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--vk-data", type=Path, required=True)
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--projection", type=Path, required=True)
    parser.add_argument("--vk-index", type=Path, required=True)
    parser.add_argument("--older-adapter", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--only", nargs="+", choices=("vk", "sara", "arte", "vectrix", "w3c"),
                        default=("vk", "sara", "arte", "vectrix", "w3c"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--vk-expanded", action="store_true", help="Include all canonical VK tune templates and use labeled negatives from other tune topics")
    args = parser.parse_args()
    if args.epochs < 1 or not args.vk_index.is_file():
        parser.error("Build the VK index first and choose a positive epoch count")
    args.out.mkdir(parents=True, exist_ok=True)

    encoder = Encoder(args.model, max_tokens=80, threads=2)
    tower = FastMailEncoder(encoder, args.projection)
    paraphrases = Path("tests/paraphrases.jsonl")
    corpus_vk = vk_corpus(args.vk_data, paraphrases, expanded=args.vk_expanded)
    vk = binary_dataset(corpus_vk.data)
    engine_vk = TinySearch(vk.documents, tower, args.vk_index)
    rows_vk = prepare_rows(vk, engine_vk, tower)
    original = binary_dataset(load_dataset(args.vk_data))
    rows_public = prepare_rows(original, engine_vk, tower)
    older = QueryAdapter.load(args.older_adapter, encoder.signature, 80, encoder.dimension) if args.older_adapter else None
    print(json.dumps({"reference": "raw", "vk_test": measured(vk, corpus_vk.test, rows_vk, engine_vk),
                      "vk_public": measured(original, list(original.queries), rows_public, engine_vk)}), flush=True)
    if older is not None:
        print(json.dumps({"reference": "previous_vk_adapter", "vk_test": measured(vk, corpus_vk.test, rows_vk, engine_vk, older=older),
                          "vk_public": measured(original, list(original.queries), rows_public, engine_vk, older=older)}), flush=True)

    loaders = {"sara": sara_corpus, "arte": arte_corpus, "vectrix": vectrix_corpus, "w3c": w3c_corpus}
    for name in args.only:
        begin = perf_counter()
        corpus = corpus_vk if name == "vk" else loaders[name](args.public_root / name)
        print(json.dumps({"started": name, "train": len(corpus.train), "validation": len(corpus.validation),
                          "test": len(corpus.test), "documents": len(corpus.data.documents)}), flush=True)
        if not corpus.train or not corpus.validation or not corpus.test:
            raise ValueError(f"Invalid learning split for {name}")
        data = vk if name == "vk" else binary_dataset(corpus.data)
        engine = engine_vk if name == "vk" else TinySearch(data.documents, tower, args.out / f"index-{name}.npz")
        rows = rows_vk if name == "vk" else prepare_rows(data, engine, tower)
        selected_name = corpus.name
        negatives = vk_trusted_negatives(corpus, paraphrases, engine) if name == "vk" and args.vk_expanded else None
        model, config, useful_queries, trials = tune(corpus, data, rows, engine, encoder.dimension, args.epochs, negatives)
        model.save(args.out / f"{selected_name}.npz", tower.signature)
        report = {"corpus": selected_name, "protocol": corpus.protocol, "training_comparisons_from_queries": useful_queries,
                  "train": len(corpus.train), "validation": len(corpus.validation), "test": len(corpus.test),
                  "selected": config, "trials": trials, "seconds": round(perf_counter() - begin, 3),
                  "native_raw": measured(data, corpus.test, rows, engine),
                  "native_trained": measured(data, corpus.test, rows, engine, model),
                  "vk_validation": measured(vk, corpus_vk.validation, rows_vk, engine_vk, model),
                  "vk_test": measured(vk, corpus_vk.test, rows_vk, engine_vk, model),
                  "vk_public": measured(original, list(original.queries), rows_public, engine_vk, model)}
        (args.out / f"{selected_name}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
