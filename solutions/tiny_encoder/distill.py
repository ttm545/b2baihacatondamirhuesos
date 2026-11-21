"""Distill indexed teacher vectors without reading queries or relevance labels."""

from argparse import ArgumentParser
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from vk_mail_search.data import load_documents

from .encoder import Encoder, letter_text
from .index import TinySearch
from .student import feature_batches


@torch.inference_mode()
def fit(features, target, regularization):
    mean_x, mean_y = features.mean(dim=0), target.mean(dim=0)
    centered = features - mean_x
    scatter = centered.T @ centered
    scale = scatter.diag().mean().clamp_min(1e-6)
    scatter.diagonal().add_(regularization * scale)
    matrix = torch.linalg.solve(scatter, centered.T @ (target - mean_y))
    return matrix, mean_y - mean_x @ matrix


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--teacher-index", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=80)
    args = parser.parse_args()
    destination = args.out.resolve()
    if (not args.teacher_index.is_file() or destination == args.teacher_index.resolve()
            or any(path.resolve() in destination.parents for path in (args.model, args.data_dir))):
        parser.error("Use a built teacher index and an output outside the input directories")
    encoder = Encoder(args.model, max_tokens=args.max_tokens, threads=args.threads)
    index = TinySearch(load_documents(args.data_dir), encoder, args.teacher_index)
    start = perf_counter()
    texts = [letter_text(doc) for doc in index.documents]
    features = torch.cat([row for _, row in feature_batches(encoder, texts, 256)])
    target = F.normalize(torch.from_numpy(index.vectors), dim=1)
    rng = torch.Generator().manual_seed(260621)
    order = torch.randperm(len(features), generator=rng)
    heldout, train = order[:max(1, len(order) // 10)], order[max(1, len(order) // 10):]
    if not len(train):
        raise ValueError("Distillation requires at least two letters")
    reports = {}
    for strength in (.003, .03, .15):
        matrix, bias = fit(features[train], target[train], strength)
        prediction = F.normalize(features[heldout] @ matrix + bias, dim=1)
        reports[strength] = float((prediction * target[heldout]).sum(dim=1).mean())

    selected = max(reports, key=reports.get)
    matrix, bias = fit(features, target, selected)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as handle:
        np.savez(handle, matrix=matrix.numpy(), bias=bias.numpy(), teacher=encoder.signature, max_tokens=args.max_tokens)
    print(json.dumps({"documents": len(target), "validation_documents": len(heldout),
                      "validation_cosine": reports, "selected_strength": selected,
                      "distillation_seconds": round(perf_counter() - start, 3),
                      "projection_parameters": matrix.numel() + bias.numel(), "projection_bytes": args.out.stat().st_size}))


if __name__ == "__main__":
    main()
