"""Distill Qwen into the full compact index and fit a relevance head by query family."""

from argparse import ArgumentParser
from collections import defaultdict
from copy import copy
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from vk_mail_search.data import Dataset, jsonl, load_dataset
from vk_mail_search.evaluate import _load_paraphrase_benchmark, _select_queries, score_predictions
from vk_mail_search.query_parser import parse_query
from vk_mail_search.searchers import build_searcher

from .adapter import QueryAdapter
from .corpora import vk_corpus
from .encoder import Encoder
from .index import TinySearch
from .qwen_context import ContextModel, QwenContextSearch, normalize
from .student import FastMailEncoder
from .tune import fit as fit_query_adapter, prepare as prepare_query_adapter


def read_vectors(path: Path, id_field: str):
    with np.load(path, allow_pickle=False) as file:
        info = json.loads(str(file["info"]))
        values = normalize(file["vectors"].astype(np.float32))
    return dict(zip(info[id_field], values, strict=True)), info


def append_bias(values):
    return np.column_stack((values, np.ones(len(values), dtype=np.float32)))


def ridge(inputs, targets, penalty, prior=None):
    if prior is not None:
        targets = targets - inputs @ prior
    gram = inputs @ inputs.T
    coefficients = inputs.T @ np.linalg.solve(gram + np.eye(len(gram), dtype=np.float32) * penalty, targets)
    return coefficients if prior is None else coefficients + prior


def document_projection(index, teacher, info):
    rows = {document.doc_id: i for i, document in enumerate(index.documents)}
    lower = info["fit_count"]
    fit_ids, test_ids = info["doc_ids"][:lower], info["doc_ids"][lower:]
    fit = np.asarray([teacher[doc_id] for doc_id in fit_ids])
    mean = fit.mean(axis=0)
    _, _, axes = np.linalg.svd(fit - mean, full_matrices=False)
    basis = axes[:64].T
    result = ridge(append_bias(index.vectors[[rows[doc_id] for doc_id in fit_ids]]), (fit - mean) @ basis, .15)
    target = normalize((np.asarray([teacher[doc_id] for doc_id in test_ids]) - mean) @ basis)
    predicted = normalize(append_bias(index.vectors[[rows[doc_id] for doc_id in test_ids]]) @ result)
    cosine = float((target * predicted).sum(axis=1).mean())
    return result, basis, mean, cosine


def pairwise_head(training_ids, examples, judgments, *, use_qwen):
    offsets, differences, importance = [], [], []
    for qid in training_ids:
        example = examples[qid]
        positive = np.asarray([i for i, doc_id in enumerate(example.ids) if judgments[qid].get(doc_id, 0) > 0])
        negative = np.asarray([i for i, doc_id in enumerate(example.ids) if not judgments[qid].get(doc_id, 0) > 0])
        if not positive.size or not negative.size:
            continue
        rank = np.argsort(np.argsort(-example.base_scores, kind="stable"), kind="stable")
        discounts = np.where(rank < 10, 1 / np.log2(rank + 2), 0)
        weight = .05 + np.abs(discounts[positive, None] - discounts[negative])
        offsets.append((example.prior[positive, None] - example.prior[negative]).ravel())
        differences.append((example.extra[positive, None] - example.extra[negative]).reshape(-1, 4))
        importance.append((weight / weight.sum()).ravel())
    base = torch.from_numpy(np.concatenate(offsets)).float()
    features = torch.from_numpy(np.concatenate(differences)).float()
    weights = torch.from_numpy(np.concatenate(importance)).float()
    if not use_qwen:
        features[:, 2] = 0
    model = torch.nn.Parameter(torch.zeros(4))
    optimizer = torch.optim.LBFGS([model], lr=.7, max_iter=40, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = (F.softplus(-2 * (base + features @ model)) * weights).sum() / weights.sum()
        loss = loss + .035 * model.square().sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return model.detach().numpy()


def metrics(dataset, ids, rankings):
    return score_predictions(_select_queries(dataset, list(ids)), rankings, (5, 10, 100))["metrics"]


def main():
    parser = ArgumentParser(description=__doc__)
    for name in ("data-dir", "model", "projection", "index", "teacher", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--paraphrases", type=Path, default=Path("tests/paraphrases.jsonl"))
    args = parser.parse_args()
    if not args.index.is_file():
        parser.error("Build the full compact index first")
    teacher_docs, doc_info = read_vectors(args.teacher / "documents.npz", "doc_ids")
    teacher_queries, _ = read_vectors(args.teacher / "queries.npz", "query_ids")
    original = load_dataset(args.data_dir)
    paraphrases, splits = _load_paraphrase_benchmark(original, args.paraphrases)
    all_data = Dataset(original.documents, {**original.queries, **paraphrases.queries},
                       {**original.qrels, **paraphrases.qrels})
    corpus = vk_corpus(args.data_dir, args.paraphrases, expanded=True)
    dev = list(dict.fromkeys(corpus.train + corpus.validation))
    tune_ids, test_ids = splits["tune"], splits["test"]
    canonical_dev = [qid for qid in dev if qid in original.queries]
    sources = {row["query_id"]: row["source_query_id"] for _, row in jsonl(args.paraphrases)}
    parsed = {qid: parse_query(query.query, query.query_date) for qid, query in all_data.queries.items()}
    families = {qid: parsed[sources.get(qid, qid)].topic for qid in all_data.queries}
    groups = sorted({families[qid] for qid in dev})
    if {families[qid] for qid in test_ids}.intersection(groups):
        raise ValueError("A held-out query family was used during training")
    encoder = Encoder(args.model, max_tokens=80, threads=1, device="cpu")
    tower = FastMailEncoder(encoder, args.projection)
    adapter = QueryAdapter.load(args.adapter, encoder.signature, 80, encoder.dimension) if args.adapter else None
    vector = TinySearch(original.documents, tower, args.index, adapter=adapter)
    period = build_searcher("period", original.documents)
    doc_map, basis, mean, teacher_cosine = document_projection(vector, teacher_docs, doc_info)
    raw = {}
    ids = list(all_data.queries)
    for rows, vectors in encoder.batches([parsed[qid].topic for qid in ids], 32):
        raw.update({ids[i]: value for i, value in zip(rows, vectors, strict=True)})
    contextual = {qid: (teacher_queries[qid] - mean) @ basis for qid in ids}

    def projection(selected):
        return ridge(append_bias(np.asarray([raw[qid] for qid in selected])),
                     np.asarray([contextual[qid] for qid in selected]), .35, prior=doc_map)

    def examples_for(model, selected, exact_qwen=False, fold_adapter=None):
        retriever = vector
        if fold_adapter is not None:
            retriever = copy(vector)
            retriever.adapter = fold_adapter
        searcher = QwenContextSearch(period, retriever, model)
        output = {}
        for qid in selected:
            query = all_data.queries[qid]
            output[qid] = searcher.candidates(parsed[qid], query.synthetic_user, query.query_date, 100, raw[qid],
                                              normalize(contextual[qid]) if exact_qwen else None)
        return output

    initial = ContextModel(doc_map, doc_map, np.zeros(4, dtype=np.float32), 0, 1.01)
    baseline = examples_for(initial, ids)

    def rank_examples(examples, model, subset):
        return {qid: [doc_id for doc_id, _ in examples[qid].ranked(model, 100)] for qid in subset}

    baseline_tune = {
        name: metrics(all_data, subset, rank_examples(baseline, initial, subset))
        for name, subset in (("tune", tune_ids), ("canonical_dev", canonical_dev))
    }
    settings = [(strategy, strength, threshold)
                for strategy, strengths in (("pairwise", (0, .25, .5, 1., 1.5)), ("rrf", (0, .125, .25, .40, .55)))
                for strength in strengths for threshold in (.50, .70, .90, 1.01)
                if strength or threshold == 1.01]
    cv = {(qwen, *setting): {} for qwen in (False, True) for setting in settings}
    adapter_rows = defaultdict(list)
    if adapter is not None:
        for qid in sorted(set(tune_ids) | {sources[qid] for qid in tune_ids}):
            adapter_rows[families[qid]].append(
                prepare_query_adapter(vector, all_data.queries[qid], all_data.qrels[qid], raw[qid])
            )
    for group in groups:
        fit = [qid for qid in dev if families[qid] != group]
        held = [qid for qid in dev if families[qid] == group]
        draft = replace(initial, query_map=projection(fit))
        fold_adapter = (fit_query_adapter({name: rows for name, rows in adapter_rows.items() if name != group},
                                          encoder.dimension, penalty=24., epochs=45) if adapter_rows else None)
        examples = examples_for(draft, dev, fold_adapter=fold_adapter)
        for qwen in (False, True):
            weights = pairwise_head(fit, examples, all_data.qrels, use_qwen=qwen)
            for strategy, strength, threshold in settings:
                model = replace(draft, strength=strength, threshold=threshold, weights=weights, strategy=strategy)
                cv[qwen, strategy, strength, threshold].update(rank_examples(examples, model, held))

    ranking = []
    for (qwen, strategy, strength, threshold), predictions in cv.items():
        tune_metrics = metrics(all_data, tune_ids, {qid: predictions[qid] for qid in tune_ids})
        canonical_metrics = metrics(all_data, canonical_dev, {qid: predictions[qid] for qid in canonical_dev})
        ranking.append({"qwen": qwen, "strategy": strategy, "strength": strength, "threshold": threshold,
                        "tune": tune_metrics, "canonical_dev": canonical_metrics})
    floor = baseline_tune["canonical_dev"]["@10"]["ndcg"] - .003
    def select(qwen, strategy=None):
        eligible = (row for row in ranking if row["qwen"] == qwen
                    and (strategy is None or row["strategy"] == strategy)
                    and row["canonical_dev"]["@10"]["ndcg"] >= floor)
        return max(eligible, key=lambda row: (row["tune"]["@10"]["ndcg"], row["tune"]["@5"]["recall"],
                                             row["canonical_dev"]["@10"]["ndcg"]))

    base = {name: metrics(all_data, subset, rank_examples(baseline, initial, subset))
            for name, subset in (("public", original.queries), ("heldout", test_ids))}
    results = {}
    draft = replace(initial, query_map=projection(dev))
    examples = examples_for(draft, ids)
    fitted_weights = {qwen: pairwise_head(dev, examples, all_data.qrels, use_qwen=qwen) for qwen in (False, True)}
    experiments = (("without_qwen", select(False)),
                   ("distilled_qwen", select(True)),
                   ("qwen_pairwise", select(True, "pairwise")),
                   ("qwen_rrf", select(True, "rrf")))
    for label, item in experiments:
        fitted = replace(draft, weights=fitted_weights[item["qwen"]], strength=item["strength"],
                         threshold=item["threshold"], strategy=item["strategy"])
        results[label] = {"selected": item, "weights": fitted.weights.tolist(),
                          **{name: metrics(all_data, subset, rank_examples(examples, fitted, subset))
                             for name, subset in (("public", original.queries), ("heldout", test_ids))}}
        if label == "distilled_qwen":
            exact = examples_for(fitted, test_ids, exact_qwen=True)
            results[label]["heldout_exact_query_diagnostic"] = metrics(all_data, test_ids,
                                                                       rank_examples(exact, fitted, test_ids))
            protected = replace(fitted, depth=10)
            protected.save(args.out, tower.signature)
            results["qwen_preserve_top10"] = {
                name: metrics(all_data, subset, rank_examples(examples, protected, subset))
                for name, subset in (("public", original.queries), ("heldout", test_ids))
            }
    print(json.dumps({"development_families": len(groups), "development_queries": len(dev),
                      "qwen_document_fit": doc_info["fit_count"], "qwen_document_test": doc_info["test_count"],
                      "projection_centered_cosine": teacher_cosine, "baseline": base, "cv_baseline": baseline_tune,
                      "variants": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
