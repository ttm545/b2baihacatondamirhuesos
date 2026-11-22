"""Export predictions with the same local search engine as the deployed mail API."""

import argparse
import asyncio
import json
from pathlib import Path
import sys
from time import perf_counter_ns

from vk_mail_search.data import data_dir
from vk_mail_search.predict import validate_output_path, write_ranked_queries

from .api import create_app


async def predict(queries: Path, output: Path, corpus: Path, k: int) -> dict:
    validate_output_path(queries, output, corpus)
    start = perf_counter_ns()
    app = create_app(corpus)
    async with app.router.lifespan_context(app):
        index_ms = (perf_counter_ns() - start) / 1_000_000
        return write_ranked_queries(queries, output, app.state.docs, app.state.searcher, k, index_ms,
                                    strict_dates=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--data-dir")
    parser.add_argument("--k", type=int, default=100)
    args = parser.parse_args(argv)
    if args.k < 1:
        parser.error("--k должен быть положительным.")
    try:
        report = asyncio.run(predict(args.queries, args.out, data_dir(args.data_dir), args.k))
    except (OSError, KeyError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"{exc}\n")
    print(json.dumps(report, ensure_ascii=False), file=sys.stderr)


if __name__ == "__main__":
    main()
