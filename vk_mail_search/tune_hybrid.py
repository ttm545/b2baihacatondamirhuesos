"""Grid-search fusion settings offline: retrieve once per query, then re-fuse cheaply.

python -m vk_mail_search.tune_hybrid --k 10 [--data-dir DIR] [--json out.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import load_dataset
from .evaluate import STANDARD_K, score_predictions
from .hybrid import FusionConfig, fuse
from .searchers import build_searcher

DEPTH = 200
DENSE_WEIGHTS = (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0)


def grid() -> list[FusionConfig]:
    configs = []
    for w in DENSE_WEIGHTS:
        weights = {"bm25": round(1 - w, 2), "dense": w}
        for rrf_k in (10, 30, 60):
            configs.append(FusionConfig("rrf", weights, 100, rrf_k))
        configs.append(FusionConfig("score", weights, 100))
    for depth in (20, 50, 200):
        configs.append(FusionConfig("rrf", {"bm25": 0.5, "dense": 0.5}, depth, 60))
    return configs


def label(config: FusionConfig) -> str:
    w = config.weights or {}
    extra = f",rrf_k={config.rrf_k:g}" if config.mode == "rrf" else ""
    return f"mode={config.mode},w_bm25={w.get('bm25', 1):g},w_dense={w.get('dense', 1):g},depth={config.depth}{extra}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir")
    parser.add_argument("--k", type=int, default=10, help="K, по которому сортируется таблица")
    parser.add_argument("--json", type=Path, help="сохранить все строки таблицы")
    args = parser.parse_args()

    dataset = load_dataset(args.data_dir)
    cutoffs = tuple(sorted({*STANDARD_K, args.k}))
    searchers = {name: build_searcher(name, dataset.documents) for name in ("bm25", "dense")}
    lists = {
        query.query_id: {name: s.search(query.query, query.synthetic_user, query.query_date, DEPTH)
                         for name, s in searchers.items()}
        for query in dataset.queries.values()
    }
    rows = []
    for config in grid():
        predictions = {qid: [doc for doc, _ in fuse(per_query, config, max(cutoffs))]
                       for qid, per_query in lists.items()}
        metrics = score_predictions(dataset, predictions, cutoffs)["metrics"]
        rows.append({"config": label(config), **{f"{m}@{k}": metrics[f"@{k}"][m]
                                                 for k in cutoffs for m in ("recall", "ndcg", "mrr")}})
    key = f"ndcg@{args.k}"
    rows.sort(key=lambda row: -row[key])
    print(f"{'nDCG@'+str(args.k):>9} {'Recall@'+str(args.k):>10} {'MRR@'+str(args.k):>8}  config")
    for row in rows:
        print(f"{row[key]:9.4f} {row[f'recall@{args.k}']:10.4f} {row[f'mrr@{args.k}']:8.4f}  {row['config']}")
    if args.json:
        args.json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
