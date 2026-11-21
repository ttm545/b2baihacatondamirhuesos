"""A small residual PyTorch projection for the query vector only."""

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class QueryAdapter(nn.Module):
    def __init__(self, dimension: int, rank: int = 12):
        super().__init__()
        self.down = nn.Linear(dimension, rank, bias=False)
        self.up = nn.Linear(rank, dimension, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, vectors):
        return F.normalize(vectors + self.up(self.down(vectors)), dim=-1)

    @torch.inference_mode()
    def project(self, vector: np.ndarray) -> np.ndarray:
        return self(torch.from_numpy(vector)).numpy()

    def save(self, path: Path, signature: str, max_tokens: int):
        with path.open("wb") as handle:
            np.savez(handle, signature=signature, max_tokens=max_tokens,
                     up=self.up.weight.detach().numpy(), down=self.down.weight.detach().numpy())

    @staticmethod
    def load(path: Path, signature: str, max_tokens: int, dimension: int):
        with np.load(path, allow_pickle=False) as data:
            if str(data["signature"]) != signature or int(data["max_tokens"]) != max_tokens:
                raise ValueError("The adapter was trained for a different model or token limit")
            up, down = data["up"], data["down"]
        if (up.ndim != 2 or down.shape != up.T.shape or up.shape[0] != dimension or
                not 0 < up.shape[1] <= dimension or not np.isfinite(up).all() or not np.isfinite(down).all()):
            raise ValueError("Invalid adapter weights")
        adapter = QueryAdapter(dimension, up.shape[1]).eval()
        with torch.no_grad():
            adapter.up.weight.copy_(torch.from_numpy(up))
            adapter.down.weight.copy_(torch.from_numpy(down))
        return adapter
