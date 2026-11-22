"""An optional VK topic head over frozen RuBERT document and query vectors."""

from pathlib import Path

import numpy as np
import torch


class TopicAdapter:
    def __init__(self, query_bases: np.ndarray, weight: np.ndarray, bias: np.ndarray):
        self.query_bases = query_bases
        self.weight = weight
        self.bias = bias
        self.share = .30

    def document_probabilities(self, vectors: np.ndarray) -> np.ndarray:
        logits = np.clip(vectors @ self.weight.T + self.bias, -20, 20)
        return 1 / (1 + np.exp(-logits))

    def scores(self, query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        logits = 28 * (self.query_bases @ query)
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        return self.share * (candidates @ probabilities)

    def save(self, path: Path, signature: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as file:
            np.savez(file, signature=signature, query_bases=self.query_bases, weight=self.weight, bias=self.bias)

    @classmethod
    def load(cls, path: Path, signature: str, dimension: int):
        with np.load(path, allow_pickle=False) as arrays:
            if str(arrays["signature"]) != signature:
                raise ValueError("Topic adapter does not match the encoder pair")
            bases, weight, bias = arrays["query_bases"], arrays["weight"], arrays["bias"]
        if (bases.ndim != 2 or bases.shape[1] != dimension or bases.shape[0] == 0 or
                weight.shape != bases.shape or bias.shape != (bases.shape[0],) or
                not all(np.isfinite(value).all() for value in (bases, weight, bias))):
            raise ValueError("Invalid topic adapter parameters")
        return cls(bases, weight, bias)

    @classmethod
    def from_pytorch(cls, query_bases: torch.Tensor, classifier: torch.nn.Linear):
        return cls(query_bases.detach().numpy(), classifier.weight.detach().numpy(), classifier.bias.detach().numpy())
