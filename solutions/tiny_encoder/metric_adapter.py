"""Small trainable projections for matching query and document vectors."""

from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class ResidualTower(nn.Module):
    def __init__(self, dimension: int, rank: int):
        super().__init__()
        self.down = nn.Linear(dimension, rank, bias=False)
        self.up = nn.Linear(rank, dimension, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, value):
        return F.normalize(value + self.up(self.down(value)), dim=-1)


class MetricAdapter(nn.Module):
    def __init__(self, dimension: int, rank: int = 16, *, dual: bool = True):
        super().__init__()
        self.rank = rank
        self.dimension = dimension
        self.dual = dual
        self.query_tower = ResidualTower(dimension, rank)
        self.document_tower = ResidualTower(dimension, rank) if dual else nn.Identity()

    @torch.inference_mode()
    def query(self, vector: np.ndarray) -> np.ndarray:
        return self.query_tower(torch.from_numpy(vector)).numpy()

    @torch.inference_mode()
    def documents(self, matrix: np.ndarray) -> np.ndarray:
        if not self.dual:
            return matrix
        output = np.empty_like(matrix)
        for start in range(0, len(matrix), 2048):
            output[start:start + 2048] = self.document_tower(torch.from_numpy(matrix[start:start + 2048])).numpy()
        return output

    def save(self, path: Path, signature: str):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with NamedTemporaryFile(dir=path.parent, prefix=".metric-", suffix=".npz", delete=False) as file:
                temporary = Path(file.name)
                state = {name: value.detach().cpu().numpy() for name, value in self.state_dict().items()}
                np.savez(file, signature=signature, rank=self.rank, dimension=self.dimension, dual=self.dual, **state)
            temporary.replace(path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: Path, signature: str, dimension: int):
        with np.load(path, allow_pickle=False) as data:
            rank = int(data["rank"])
            if str(data["signature"]) != signature or int(data["dimension"]) != dimension or not 0 < rank <= dimension:
                raise ValueError("Metric adapter does not match the document and query encoders")
            model = cls(dimension, rank, dual=bool(data["dual"]))
            state = {}
            for name, parameter in model.state_dict().items():
                value = data[name]
                if value.shape != parameter.shape or not np.isfinite(value).all():
                    raise ValueError("Invalid metric adapter parameters")
                state[name] = torch.from_numpy(value)
        model.load_state_dict(state)
        return model.eval()
