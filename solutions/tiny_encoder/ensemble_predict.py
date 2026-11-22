"""Export ensemble predictions for arbitrary organizer-compatible query JSONL."""

from argparse import ArgumentParser
import json
from pathlib import Path

from vk_mail_search.data import load_documents
from vk_mail_search.searchers import build_searcher

from .__main__ import export
from .adapter import QueryAdapter
from .encoder import Encoder
from .ensemble import EnsembleOptions, EnsembleSearch
from .index import TinySearch
from .qwen_context import ContextModel, QwenContextSearch
from .student import FastMailEncoder


def main():
    parser = ArgumentParser(description=__doc__)
    for name in ("data-dir", "model", "projection", "index", "context", "config", "queries", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--k", type=int, default=100)
    args = parser.parse_args()
    inputs = (args.queries, args.projection, args.index, args.adapter, args.context, args.config,
              args.data_dir / "documents.jsonl", args.data_dir / "queries_dev.jsonl", args.data_dir / "qrels_dev.jsonl")
    if (args.k <= 0 or not args.index.is_file() or args.out.resolve() in {path.resolve() for path in inputs if path}
            or args.model.resolve() in args.out.resolve().parents):
        parser.error("Build the full index and export to a separate file at positive K")
    documents = load_documents(args.data_dir)
    encoder = Encoder(args.model, max_tokens=80, threads=1, device="cpu")
    tower = FastMailEncoder(encoder, args.projection)
    adapter = QueryAdapter.load(args.adapter, encoder.signature, 80, encoder.dimension) if args.adapter else None
    vectors = TinySearch(documents, tower, args.index, adapter=adapter)
    context = ContextModel.load(args.context, tower.signature, tower.dimension)
    search = EnsembleSearch(QwenContextSearch(build_searcher("period", documents), vectors, context),
                            EnsembleOptions.load(args.config))
    print(json.dumps(export(search, args.queries, args.out, args.k)))


if __name__ == "__main__":
    main()
