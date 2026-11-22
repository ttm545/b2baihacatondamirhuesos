"""Collect exact Qwen vectors for a label-blind mail sample and VK query texts."""

from argparse import ArgumentParser
from collections import defaultdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from vk_mail_search.data import load_dataset
from vk_mail_search.evaluate import _load_paraphrase_benchmark
from vk_mail_search.query_parser import parse_query

from .encoder import letter_text
from .index import save_index
from .qwen_embed import QwenEncoder, QUERY_INSTRUCTION, index_query_vectors


def sample_letters(documents, seed: int = 260621):
    generator = np.random.default_rng(seed)
    mailboxes = defaultdict(list)
    for doc_id in sorted(documents):
        mailboxes[documents[doc_id].synthetic_user].append(doc_id)
    fit, test = [], []
    for doc_ids in sorted(mailboxes.values(), key=lambda ids: ids[0]):
        rows = generator.permutation(len(doc_ids))
        fit.extend(doc_ids[i] for i in rows[:14])
        test.extend(doc_ids[i] for i in rows[14:18])
    return fit, test


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--paraphrases", type=Path, default=Path("tests/paraphrases.jsonl"))
    parser.add_argument("--public-vectors", type=Path, help="Reuse the index-queries output for the same model and archive")
    parser.add_argument("--port", type=int, default=18767)
    args = parser.parse_args()
    start = perf_counter()
    data = load_dataset(args.data_dir)
    aliases, splits = _load_paraphrase_benchmark(data, args.paraphrases)
    encoder = QwenEncoder(args.port, 220, args.cache)
    args.output.mkdir(parents=True, exist_ok=True)

    path = args.public_vectors or args.output / "canonical.npz"
    if args.public_vectors is None:
        index_query_vectors(args.data_dir / "queries_dev.jsonl", path, encoder, batch_size=12)
    with np.load(path, allow_pickle=False) as file:
        ids = json.loads(str(file["info"]))["query_ids"]
        canonical = file["vectors"]
    if set(ids) != set(data.queries) or canonical.shape != (len(ids), encoder.dimension):
        raise ValueError("Query vectors must match this archive and Qwen model")
    alias_cache = args.output / "queries.npz"
    alias_ids = list(aliases.queries)
    texts = [parse_query(query.query, query.query_date).topic or query.query for query in aliases.queries.values()]
    if not alias_cache.is_file():
        encoded = np.concatenate([encoder.encode(texts[offset:offset + 12], QUERY_INSTRUCTION)
                                  for offset in range(0, len(texts), 12)])
        save_index(alias_cache, np.concatenate([canonical, encoded.astype(np.float16)]), encoder.signature,
                   {"query_ids": ids + alias_ids, "splits": splits})
    print(f"Teacher query vectors ready: {len(ids) + len(alias_ids)}", flush=True)

    fit, test = sample_letters(data.documents)
    docs = fit + test
    vectors = np.empty((len(docs), encoder.dimension), dtype=np.float16)
    for rows, batch in encoder.batches([letter_text(data.documents[doc_id]) for doc_id in docs], 12):
        vectors[rows] = batch
    save_index(args.output / "documents.npz", vectors, encoder.signature,
               {"doc_ids": docs, "fit_count": len(fit), "test_count": len(test)})
    print(json.dumps({"fit": len(fit), "test": len(test), "seconds": round(perf_counter() - start, 2)}),
          flush=True)


if __name__ == "__main__":
    main()
