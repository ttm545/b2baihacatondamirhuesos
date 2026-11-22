"""Fuse two validated VK prediction files with the query period first."""

from argparse import ArgumentParser
from pathlib import Path
import json

from vk_mail_search.data import load_dataset
from vk_mail_search.evaluate import (_load_paraphrase_benchmark, _select_queries, load_predictions,
                                     score_predictions, validate_predictions, write_predictions)
from vk_mail_search.query_parser import parse_query


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--period-pred", type=Path, required=True)
    parser.add_argument("--tiny-pred", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--paraphrases", type=Path)
    parser.add_argument("--split", choices=("tune", "test"))
    parser.add_argument("--semantic-share", default=.25, type=float)
    args = parser.parse_args()
    if args.split and not args.paraphrases:
        parser.error("--split requires --paraphrases")
    if not 0 <= args.semantic_share <= 1:
        parser.error("Use a semantic share between 0 and 1")
    if args.out.resolve() in {args.period_pred.resolve(), args.tiny_pred.resolve()} or args.data_dir.resolve() in args.out.resolve().parents:
        parser.error("Output must be outside the input files and benchmark directory")

    dataset = load_dataset(args.data_dir)
    if args.paraphrases:
        dataset, splits = _load_paraphrase_benchmark(dataset, args.paraphrases)
        if args.split:
            dataset = _select_queries(dataset, splits[args.split])
    lexical, semantic = load_predictions(args.period_pred), load_predictions(args.tiny_pred)
    validate_predictions(dataset, lexical)
    validate_predictions(dataset, semantic)
    ranks = {}
    for qid, query in dataset.queries.items():
        period = parse_query(query.query, query.query_date).period
        base = {doc_id: i for i, doc_id in enumerate(lexical[qid][:100], start=1)}
        tiny = {doc_id: i for i, doc_id in enumerate(semantic[qid][:100], start=1)}

        def key(doc_id):
            outside = period is not None and not period.contains(dataset.documents[doc_id].doc_dt)
            lhs = (1 - args.semantic_share) / (12 + base[doc_id]) if doc_id in base else 0
            rhs = args.semantic_share / (12 + tiny[doc_id]) if doc_id in tiny else 0
            return outside, -(lhs + rhs), base.get(doc_id, 101), doc_id

        ranks[qid] = sorted(base.keys() | tiny.keys(), key=key)[:100]
    write_predictions(args.out, ranks, 100)
    print(json.dumps(score_predictions(dataset, ranks, (5, 10, 100)), indent=2))


if __name__ == "__main__":
    main()
