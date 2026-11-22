"""A compact document tower distilled from the PyTorch retriever's vectors."""

from hashlib import sha256
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .encoder import Encoder


class FastMailEncoder:
    """Encode documents with tied token embeddings and one learned projection."""

    def __init__(self, teacher: Encoder, projection: Path):
        self.teacher = teacher
        self.max_tokens = teacher.max_tokens
        self.dimension = teacher.dimension
        self.parameters = teacher.parameters
        self.signature = sha256((teacher.signature + sha256(projection.read_bytes()).hexdigest()).encode()).hexdigest()
        with np.load(projection, allow_pickle=False) as data:
            if str(data["teacher"]) != teacher.signature or int(data["max_tokens"]) != self.max_tokens:
                raise ValueError("The fast document projection needs its original teacher and token limit")
            matrix, bias = data["matrix"], data["bias"]
        if (matrix.shape != (2 * self.dimension, self.dimension) or bias.shape != (self.dimension,) or
                not np.isfinite(matrix).all() or not np.isfinite(bias).all()):
            raise ValueError("Invalid document projection")
        self.matrix = torch.from_numpy(matrix).to(teacher.device)
        self.bias = torch.from_numpy(bias).to(teacher.device)

    def query(self, text: str) -> np.ndarray:
        return self.teacher.query(text)

    @torch.inference_mode()
    def batches(self, texts, batch_size):
        for indices, features in feature_batches(self.teacher, texts, batch_size):
            vectors = F.normalize(features @ self.matrix + self.bias, dim=1)
            yield indices, vectors.cpu().numpy()


@torch.inference_mode()
def feature_batches(teacher, texts, batch_size):
    tokens = teacher._tokenize(texts)
    weight = teacher.model.get_input_embeddings().weight
    device = weight.device
    for offset in range(0, len(tokens), batch_size):
        rows = [row[1:-1] or row[:1] for row in tokens[offset:offset + batch_size]]
        lengths = [len(row) for row in rows]
        prefix = [row[:18] for row in rows]

        def pool(sequences):
            ids = torch.tensor([idx for row in sequences for idx in row], device=device)
            offsets = torch.tensor(np.cumsum([0] + [len(row) for row in sequences[:-1]]), device=device)
            return F.embedding_bag(ids, weight, offsets, mode="mean")

        result = torch.cat((pool(prefix), pool(rows)), dim=1)
        yield list(range(offset, offset + len(lengths))), result
