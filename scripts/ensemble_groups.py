"""Compare Damir's ``ensemble`` with and without corpus-derived type groups.

Both variants share one vector index, model and config. Only the text part differs: the
grouped variant builds ``period`` (and its BM25) over letters with the group words appended
(``vk_mail_search.type_groups``). The vector index is built on the original letters, so the
existing ``.cache/tiny-fast.npz`` is reused. Group parameters are the ones chosen on ``tune``.

    python -m scripts.ensemble_groups --data-dir data --model .cache/rubert-tiny-retriever \
        --projection solutions/tiny_encoder/assets/projection-80.npz --index .cache/tiny-fast.npz \
        --adapter solutions/tiny_encoder/assets/adapter-80.npz \
        --context solutions/tiny_encoder/assets/qwen-context.npz \
        --config solutions/tiny_encoder/assets/ensemble.json
"""

from argparse import ArgumentParser
import json
from pathlib import Path
from time import perf_counter

from solutions.tiny_encoder.adapter import QueryAdapter
from solutions.tiny_encoder.encoder import Encoder
from solutions.tiny_encoder.ensemble import EnsembleOptions, EnsembleSearch
from solutions.tiny_encoder.index import TinySearch
from solutions.tiny_encoder.qwen_context import ContextModel, QwenContextSearch
from solutions.tiny_encoder.student import FastMailEncoder
from vk_mail_search.data import load_dataset
from vk_mail_search.evaluate import _load_paraphrase_benchmark, _select_queries, run_search, score_predictions
from vk_mail_search.searchers import build_searcher
from vk_mail_search.type_groups import MAIN, build_grouped, parse_config


def main():
    parser = ArgumentParser(description=__doc__)
    for name in ("data-dir", "model", "projection", "index", "context", "config"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--paraphrases", type=Path, default=Path("tests/paraphrases.jsonl"))
    parser.add_argument("--groups", default="", help="параметры групп, по умолчанию выбранные на tune")
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()
    if not args.index.is_file():
        parser.error("Сначала постройте индекс по solutions/tiny_encoder/README.md")

    public = load_dataset(args.data_dir)
    paraphrased, splits = _load_paraphrase_benchmark(public, args.paraphrases)
    sets = {"public": public, **{name: _select_queries(paraphrased, ids) for name, ids in splits.items()}}
    docs = public.documents

    encoder = Encoder(args.model, max_tokens=80, threads=1, device="cpu")
    tower = FastMailEncoder(encoder, args.projection)
    adapter = QueryAdapter.load(args.adapter, encoder.signature, 80, encoder.dimension) if args.adapter else None
    vectors = TinySearch(docs, tower, args.index, adapter=adapter)
    context = ContextModel.load(args.context, tower.signature, tower.dimension)
    options = EnsembleOptions.load(args.config)

    engines, seconds = {}, {}
    for label, make_period in (("ensemble", lambda: build_searcher("period", docs)),
                               ("ensemble+groups", lambda: build_grouped(docs, config=parse_config(args.groups)))):
        start = perf_counter()
        engines[label] = EnsembleSearch(QwenContextSearch(make_period(), vectors, context), options)
        seconds[label] = round(perf_counter() - start, 2)

    table = {}
    for name, dataset in sets.items():
        for label, engine in engines.items():
            predictions, timing = run_search(dataset, engine, 100, 10, args.warmup)
            report = score_predictions(dataset, predictions, (5, 10, 100))
            table[f"{name} · {label}"] = {**{title: report["metrics"][cut][key] for title, cut, key in MAIN},
                                          "p99_ms": timing["latency_ms"]["p99"], "queries": len(dataset.queries)}
    print(json.dumps(dict(text_index_seconds=seconds, results=table), ensure_ascii=False, indent=2))
    print("\n| Набор · поисковик | " + " | ".join(title for title, _, _ in MAIN) + " | p99, мс |")
    print("| --- |" + " ---: |" * (len(MAIN) + 1))
    for name, row in table.items():
        print(f"| {name} | " + " | ".join(f"{row[title]:.4f}" for title, _, _ in MAIN) + f" | {row['p99_ms']} |")


if __name__ == "__main__":
    main()
