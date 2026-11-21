"""Measure the experimental reranker through an isolated loopback-only API."""

import argparse
import json
from pathlib import Path
from threading import Thread
from time import monotonic, sleep

import uvicorn

from scripts.load_test import read_queries, run
from vk_mail_search import api
from vk_mail_search.rerank import MODELS, OnnxBgeScorer, RerankSearcher, TransformersScorer
from vk_mail_search.searchers import build_searcher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=[*MODELS, "bge-onnx-int8"], required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--top-n", type=int, choices=(10, 20, 30), nargs="+", default=[10, 20, 30])
    parser.add_argument("--budget-ms", type=float, default=400)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18764)
    parser.add_argument("--onnx-model")
    parser.add_argument("--onnx-tokenizer")
    args = parser.parse_args()
    if args.model == "bge-onnx-int8":
        if not args.onnx_model or not args.onnx_tokenizer:
            parser.error("both local ONNX and tokenizer files are required")
        scorer = OnnxBgeScorer(args.onnx_model, args.onnx_tokenizer)
    else:
        import torch

        torch.set_num_threads(args.threads)
        scorer = TransformersScorer(args.model, device=args.device, dtype=args.dtype)

    queries = read_queries(args.data_dir / "queries_dev.jsonl", 10)
    if not queries:
        parser.error("queries_dev.jsonl is empty")
    for top_n in args.top_n:
        current = []

        def factory(_name, docs):
            baseline = build_searcher("dated:bm25", docs)
            ranker = RerankSearcher(baseline, docs, scorer, top_n=top_n, budget_ms=args.budget_ms)
            current.append(ranker)
            return ranker

        previous = api.build_searcher
        api.build_searcher = factory
        config = uvicorn.Config(api.create_app(args.data_dir), host="127.0.0.1", port=args.port,
                                log_level="warning", access_log=False)
        server = uvicorn.Server(config)
        worker = Thread(target=server.run, name="rerank-http-probe")
        worker.start()
        try:
            deadline = monotonic() + 60
            while worker.is_alive() and not server.started and monotonic() < deadline:
                sleep(0.05)
            if not server.started:
                raise RuntimeError("Isolated API did not start")
            report = run(f"http://127.0.0.1:{args.port}", queries, 1000, 8, 2, 10)
            print(json.dumps(dict(model=args.model, device=args.device, top_n=top_n,
                                  budget_ms=args.budget_ms, outcomes=current[0].statistics, **report)),
                  flush=True)
            if report["errors"] or report["warmup_errors"]:
                return 1
        finally:
            server.should_exit = True
            worker.join()
            api.build_searcher = previous
            if current:
                current[0].close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
