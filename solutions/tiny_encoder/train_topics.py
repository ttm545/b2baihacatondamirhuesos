"""Cross-fit a VK topic head by mailbox and date, then save a model fitted on public labels."""

from argparse import ArgumentParser
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from vk_mail_search.data import Dataset, jsonl, load_dataset
from vk_mail_search.evaluate import _load_paraphrase_benchmark, _select_queries, score_predictions
from vk_mail_search.query_parser import parse_query

from .encoder import Encoder
from .index import TinySearch
from .student import FastMailEncoder
from .topic_adapter import TopicAdapter
from .train_corpora import prepare_rows


def fit_topics(ids, topics, rows, matrix):
    names = sorted({topics[qid] for qid in ids})
    name_index = {name: i for i, name in enumerate(names)}
    example_queries, labels = {}, defaultdict(dict)
    for qid in ids:
        name = topics[qid]
        row = rows[qid]
        example_queries.setdefault(name, row.vector)
        scope = row.in_period if row.in_period.any() else np.ones(len(row.rows), bool)
        for index, is_relevant in zip(row.rows[scope], row.positive[scope], strict=True):
            labels[name][int(index)] = max(int(is_relevant), labels[name].get(int(index), 0))
    positives = {name: {i for i, gain in labels[name].items() if gain} for name in names}
    distinct, targets, weights, columns = [], [], [], []
    for name in names:
        expected = labels[name]
        for other in names:
            if name != other:
                for index in sorted(positives[other]):
                    expected.setdefault(index, 0)
        count = sum(expected.values())
        nonmatches = len(expected) - count
        if not count or not nonmatches:
            continue
        for index, gain in expected.items():
            distinct.append(index)
            targets.append(gain)
            columns.append(name_index[name])
            weights.append(1 / (2 * (count if gain else nonmatches) * len(names)))
    if not distinct:
        raise ValueError("No labeled topic contrasts")

    torch.manual_seed(260621)
    classifier = torch.nn.Linear(matrix.shape[1], len(names))
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=.02, weight_decay=.1)
    batch = torch.from_numpy(matrix[distinct])
    columns = torch.tensor(columns)[:, None]
    weights = torch.tensor(weights)
    targets = torch.tensor(targets, dtype=torch.float32)
    for _ in range(100):
        logits = classifier(batch).gather(1, columns)[:, 0]
        loss = (F.binary_cross_entropy_with_logits(logits, targets, reduction="none") * weights).sum()
        optimizer.zero_grad()
        (loss + .0002 * classifier.weight.square().sum()).backward()
        optimizer.step()
    bases = torch.from_numpy(np.stack([example_queries[name] for name in names]))
    return TopicAdapter.from_pytorch(bases, classifier)


def cross_validate(original, paraphrases, sources, rows, index):
    parsed = {qid: parse_query(query.query, query.query_date) for qid, query in original.queries.items()}
    topics = {qid: result.topic for qid, result in parsed.items()}
    buckets = {qid: f"{query.synthetic_user}:{parsed[qid].period}" for qid, query in original.queries.items()}
    ordered = sorted(set(buckets.values()), key=lambda key: sha256(f"260621:{key}".encode()).digest())
    folds = {name: i % 5 for i, name in enumerate(ordered)}
    assignment = {qid: folds[name] for qid, name in buckets.items()}
    for qid in paraphrases.queries:
        assignment[qid] = assignment[sources[qid]]

    ranking = {"raw": {}, "topic": {}}
    for fold in range(5):
        train = [qid for qid in original.queries if assignment[qid] != fold]
        head = fit_topics(train, topics, rows, index.vectors)
        probabilities = head.document_probabilities(index.vectors)
        for qid in assignment:
            if assignment[qid] != fold:
                continue
            row = rows[qid]
            raw = index.vectors[row.rows] @ row.vector
            adapted = raw + head.scores(row.vector, probabilities[row.rows])
            for key, scores in (("raw", raw), ("topic", adapted)):
                positions = np.lexsort((row.rows, -scores, ~row.in_period))[:100]
                ranking[key][qid] = [index.documents[doc].doc_id for doc in row.rows[positions]]
    report = {}
    for name, dataset in (("public_crossfit", original), ("paraphrases_held_out_by_date", paraphrases)):
        report[name] = {key: score_predictions(dataset, {qid: values[qid] for qid in dataset.queries}, (5, 10, 100))
                        for key, values in ranking.items()}
    return report, topics


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--projection", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.index.is_file() or args.out.resolve() in {args.index.resolve(), args.projection.resolve()}:
        parser.error("Build the base index first and choose a separate adapter output")
    original = load_dataset(args.data_dir)
    if any(grade not in (0, 1) for labels in original.qrels.values() for grade in labels.values()):
        parser.error("The topic head expects the VK binary benchmark")
    source = Path("tests/paraphrases.jsonl")
    all_paraphrases, splits = _load_paraphrase_benchmark(original, source)
    holdout = _select_queries(all_paraphrases, splits["test"])
    sources = {row["query_id"]: row["source_query_id"] for _, row in jsonl(source)}
    encoder = Encoder(args.model, max_tokens=80, threads=2)
    tower = FastMailEncoder(encoder, args.projection)
    index = TinySearch(original.documents, tower, args.index)
    data = Dataset(original.documents, {**original.queries, **holdout.queries}, {**original.qrels, **holdout.qrels})
    rows = prepare_rows(data, index, tower)
    report, topics = cross_validate(original, holdout, sources, rows, index)
    final = fit_topics(list(original.queries), topics, rows, index.vectors)
    final.save(args.out, tower.signature)
    print(json.dumps({"topic_share": final.share, "topics_trained": len(final.bias), **report}, indent=2))


if __name__ == "__main__":
    main()
