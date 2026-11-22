"""Run word-vector search in a mailbox and print ranked IDs."""

from argparse import ArgumentParser
import json
from pathlib import Path

from navec import Navec

from vk_mail_search.data import load_documents
from vk_mail_search.period_rank import PeriodRanker

from .search import DEFAULT_SIMILARITY, WordVectorSearch


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--query", required=True)
    parser.add_argument("--mailbox", required=True)
    parser.add_argument("--as-of", help="ISO date for relative queries and the upper bound on received mail")
    parser.add_argument("--similarity", type=float, default=DEFAULT_SIMILARITY)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    if not 0 <= args.similarity <= 1 or args.k < 1:
        parser.error("--similarity must be between 0 and 1, and --k must be positive")
    documents = load_documents(args.data_dir)
    searcher = WordVectorSearch(PeriodRanker(documents), Navec.load(args.model), args.similarity)
    hits = searcher.search(args.query, args.mailbox, args.as_of, args.k)
    print(json.dumps([{"doc_id": doc_id, "score": score} for doc_id, score in hits], ensure_ascii=False))


if __name__ == "__main__":
    main()
