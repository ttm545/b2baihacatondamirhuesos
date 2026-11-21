"""Fit linear period-ranker weights on tune with leave-one-topic-family-out validation."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from vk_mail_search.data import jsonl, load_dataset
from vk_mail_search.evaluate import _load_paraphrase_benchmark, _select_queries, score_predictions
from vk_mail_search.period_rank import PROFILES, PeriodRanker


def fit(rows, strength):
    vectors, weights = [], []
    for family, group in rows.items():
        for features, grades in group:
            left, right = np.where(grades[:, None] > grades)
            if not left.size:
                continue
            importance = np.exp2(grades[left]) - np.exp2(grades[right])
            vectors.extend(features[left] - features[right])
            weights.extend(importance / importance.sum() / len(group) / len(rows))

    matrix = np.array(vectors)
    weight = np.array(weights)
    prior = np.array(PROFILES["cover"]) * 6

    def objective(parameters):
        value = matrix @ parameters
        distance = parameters - prior
        loss = np.logaddexp(0, -value) @ weight + strength * (distance @ distance)
        gradient = -(matrix.T @ (weight * np.exp(-np.logaddexp(0, value)))) + 2 * strength * distance
        return loss, gradient

    result = minimize(objective, prior, jac=True, bounds=[(0, None)] * len(prior), method="L-BFGS-B")
    if not result.success:
        raise RuntimeError(result.message)
    return (result.x / result.x.sum()).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    args = parser.parse_args()
    dataset = load_dataset(args.data_dir)
    path = Path("tests/paraphrases.jsonl")
    records = {row["query_id"]: row for _, row in jsonl(path) if row["split"] == "tune"}
    paraphrases, groups = _load_paraphrase_benchmark(dataset, path)
    train = _select_queries(paraphrases, groups["tune"])
    families = {qid: qid.rsplit("-", 1)[0] for qid in records}
    engine = PeriodRanker(train.documents)
    rows = defaultdict(list)
    ranking_input = {}
    original_added = set()
    for qid, family in families.items():
        source_id = records[qid]["source_query_id"]
        queries = [train.queries[qid]]
        if source_id not in original_added:
            queries.append(dataset.queries[source_id])
            original_added.add(source_id)
        for query in queries:
            parsed, pool, tail = engine.candidates(query.query, query.synthetic_user, query.query_date, 100)
            features = engine.features(engine.topic_terms(parsed.topic), pool)
            qrels = dataset.qrels[source_id]
            if pool:
                rows[family].append((np.array(features), np.array([qrels.get(did, 0) for did, _ in pool])))
            if query.query_id == qid:
                ranking_input[qid] = (pool, tail, features)

    for strength in (.003, .015, .07):
        predictions = {}
        for family in rows:
            weights = fit({key: group for key, group in rows.items() if key != family}, strength)
            for qid, (pool, tail, features) in ranking_input.items():
                if families[qid] == family:
                    prediction = (engine.combine(pool, features, weights) + tail)[:100]
                    predictions[qid] = [doc_id for doc_id, _ in prediction]
        result = score_predictions(train, predictions, (10, 100))
        weights = fit(rows, strength)
        print(json.dumps({"strength": strength, "cv_metrics": result["metrics"], "weights": weights,
                          "families": dict(Counter(families.values()))}), flush=True)


if __name__ == "__main__":
    main()
