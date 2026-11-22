"""Fit and compare independent date and content rankers on VK and public mail datasets."""

from argparse import ArgumentParser
from dataclasses import asdict, replace
from datetime import date, timedelta
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

from vk_mail_search.data import Dataset, load_dataset
from vk_mail_search.evaluate import _load_paraphrase_benchmark, _select_queries, score_predictions
from vk_mail_search.query_parser import parse_query
from vk_mail_search.searchers import build_searcher

from .adapter import QueryAdapter
from .corpora import vk_corpus
from .encoder import Encoder
from .ensemble import EnsembleOptions, EnsembleSearch
from .index import TinySearch
from .qwen_context import ContextModel, QwenContextSearch
from .routed import RoutedSearch
from .student import FastMailEncoder


def fit_content(examples, judgments, dimension):
    pairs, weights = [], []
    for qid, row in examples:
        good = np.asarray([judgments[qid].get(doc_id, 0) > 0 for doc_id in row.ids])
        if not good.any() or good.all():
            continue
        values = row.features[:, :dimension]
        candidates = values[~good]
        negatives = candidates[np.argsort(-candidates.max(axis=1), kind="stable")[:35]]
        delta = (values[good, None] - negatives).reshape(-1, dimension)
        pairs.append(delta)
        weights.extend([1. / len(delta)] * len(delta))
    delta = torch.from_numpy(np.concatenate(pairs)).float()
    weight = torch.tensor(weights)
    prior = torch.tensor([0., 0., -1.4, -3.])[:dimension]
    parameters = torch.nn.Parameter(prior.clone())
    optimizer = torch.optim.LBFGS([parameters], lr=.7, max_iter=30, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        mix = torch.softmax(parameters, dim=0)
        loss = (F.softplus(-6 * (delta @ mix)) * weight).sum() / weight.sum()
        loss = loss + .015 * (parameters - prior).square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return tuple(torch.softmax(parameters, dim=0).detach().tolist() + [0.] * (4 - dimension))


def evaluate(dataset, ids, run):
    predictions = {qid: [doc_id for doc_id, _ in run(qid)] for qid in ids}
    return score_predictions(_select_queries(dataset, list(ids)), predictions, (5, 10, 100))["metrics"]


def masked_query(query, shift):
    return replace(query, query=parse_query(query.query, query.query_date).topic,
                   query_date=(date.fromisoformat(query.query_date) + timedelta(days=shift)).isoformat())


def call(engine, query):
    return engine.search(query.query, query.synthetic_user, query.query_date, 100)


def main():
    parser = ArgumentParser(description=__doc__)
    for name in ("data-dir", "model", "projection", "index", "context", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--paraphrases", type=Path, default=Path("tests/paraphrases.jsonl"))
    parser.add_argument("--external", type=Path)
    parser.add_argument("--external-cache", type=Path)
    args = parser.parse_args()
    if not args.index.is_file() or (args.external is None) != (args.external_cache is None):
        parser.error("Build the full index. For external evaluation supply both external paths.")
    public = load_dataset(args.data_dir)
    paraphrases, splits = _load_paraphrase_benchmark(public, args.paraphrases)
    data = Dataset(public.documents, {**public.queries, **paraphrases.queries}, {**public.qrels, **paraphrases.qrels})
    corpus = vk_corpus(args.data_dir, args.paraphrases, expanded=True)
    dev = list(dict.fromkeys(corpus.train + corpus.validation))
    canonical_dev = [qid for qid in dev if qid in public.queries]
    tune, heldout = splits["tune"], splits["test"]
    encoder = Encoder(args.model, max_tokens=80, threads=1, device="cpu")
    tower = FastMailEncoder(encoder, args.projection)
    adapter = QueryAdapter.load(args.adapter, encoder.signature, 80, encoder.dimension) if args.adapter else None
    context_model = ContextModel.load(args.context, tower.signature, tower.dimension)

    def factory(documents, index):
        vector = TinySearch(documents, tower, index, adapter=adapter)
        return EnsembleSearch(QwenContextSearch(build_searcher("period", documents), vector, context_model), None)

    search = factory(public.documents, args.index)
    dated = {
        qid: search.dated_candidates(parse_query(q.query, q.query_date), q.synthetic_user, q.query_date, 100)
        for qid, q in data.queries.items()
    }

    def date_scores(ids, options):
        return evaluate(data, ids, lambda qid: dated[qid].rank(*options, 100))

    base_date = ((1., 0., 0.), 10, 0.)
    control = date_scores(canonical_dev, base_date)["@10"]["ndcg"]
    options = [(base_date, date_scores(tune, base_date))]
    grid = ((.75, .25, 0.), (.50, .50, 0.), (.25, .75, 0.), (.65, 0., .35),
            (.45, .25, .30), (.25, .45, .30), (.35, .20, .45))
    for setting in product(grid, (5, 10, 20, 0), (.50, .90, 1.01)):
        if date_scores(canonical_dev, setting)["@10"]["ndcg"] >= control - .001:
            options.append((setting, date_scores(tune, setting)))
    date_option, dated_dev = max(options, key=lambda item:
                                 .60 * item[1]["@10"]["ndcg"] + .20 * item[1]["@5"]["ndcg"]
                                 + .20 * item[1]["@5"]["recall"])

    content = {}
    masked = {shift: Dataset(data.documents, {qid: masked_query(q, shift) for qid, q in data.queries.items()},
                             data.qrels) for shift in (0, 7, 30, 60)}
    for shift in (0, 7, 30, 60):
        ids = data.queries if shift in (0, 30) else dev
        for qid in ids:
            q = masked[shift].queries[qid]
            content[qid, shift] = search.content_candidates(q.query, q.synthetic_user, q.query_date)

    def content_scores(ids, shift, weights):
        return evaluate(masked[shift], ids, lambda qid: content[qid, shift].rank(weights, 100))

    training = [(qid, content[qid, shift]) for qid in dev for shift in (0, 7, 60)]
    contents = {"plain": (.50, .50, 0., 0.), "qwen_fixed": (.40, .40, .20, 0.),
                "trained_content": fit_content(training, data.qrels, 3),
                "trained_fresh": fit_content(training, data.qrels, 4)}

    def dev_score(mix):
        ordinary, shifted = (content_scores(tune, shift, mix) for shift in (0, 30))
        canonical = content_scores(canonical_dev, 0, mix)
        return (.40 * ordinary["@10"]["ndcg"] + .35 * shifted["@10"]["ndcg"]
                + .15 * canonical["@10"]["ndcg"] + .10 * shifted["@5"]["recall"])

    selected = max(contents, key=lambda name: dev_score(contents[name]))
    policy = EnsembleOptions(*date_option, contents[selected])
    result = {"options": asdict(policy), "development_dated": dated_dev, "selected_undated": selected,
              "content_weights": contents, "dated": {}, "masked": {}, "external": {}}
    hybrid = RoutedSearch(search.period, search.vector, fuse_undated=True)
    for name, ids in (("public", public.queries), ("heldout", heldout)):
        result["dated"][name] = {"period": date_scores(ids, base_date), "ensemble": date_scores(ids, date_option)}
        for shift in (0, 30):
            dataset = masked[shift]
            result["masked"][f"{name}+{shift}d"] = {
                "period": evaluate(dataset, ids, lambda qid: call(search.period, dataset.queries[qid])),
                "hybrid": evaluate(dataset, ids, lambda qid: call(hybrid, dataset.queries[qid])),
                **{key: content_scores(ids, shift, mix) for key, mix in contents.items()},
            }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(asdict(policy), indent=2) + "\n", encoding="utf-8")

    if args.external:
        for name in ("sara", "arte", "vectrix", "w3c"):
            external = load_dataset(args.external / name / "vk-jsonl")
            external_search = factory(external.documents, args.external_cache / f"index-{name}.npz")
            external_search.options = policy
            routes = {"period": external_search.period, "ensemble": external_search,
                      "hybrid": RoutedSearch(external_search.period, external_search.vector, fuse_undated=True)}
            result["external"][name] = {key: evaluate(external, external.queries,
                                                      lambda qid: call(engine, external.queries[qid]))
                                        for key, engine in routes.items()}
            print(f"Completed external {name}", flush=True, file=sys.stderr)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
