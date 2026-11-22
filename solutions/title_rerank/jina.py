"""Score a list of candidate subjects with the pinned local Jina model."""

from collections.abc import Sequence
from math import isfinite
from threading import Lock


MODEL_ID = "jinaai/jina-reranker-v3.5"
REVISION = "e8a93f33f0b22108f8c2364f8484ce3422552fbc"
MINIMUM_SCORE_SPAN = 0.25


class JinaSubjectScorer:
    def __init__(self, device: str = "cpu"):
        import torch
        from transformers import AutoModel, AutoTokenizer

        target = torch.device(device)
        dtype = torch.float32 if target.type == "cpu" else torch.float16
        self.model = AutoModel.from_pretrained(
            MODEL_ID, revision=REVISION, trust_remote_code=True, use_safetensors=True, dtype=dtype,
        ).to(target).eval()
        self.model._tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, revision=REVISION, padding_side="left", fix_mistral_regex=False,
        )
        self.lock = Lock()

    def score(self, query: str, titles: Sequence[str]) -> list[float]:
        if not titles:
            return []
        passages = [title.replace("<|", "< |") for title in titles]
        with self.lock:
            ranked = self.model.rerank(query[:1000].replace("<|", "< |"), passages)
        indices = [int(row["index"]) for row in ranked]
        if sorted(indices) != list(range(len(titles))):
            raise ValueError("Jina must return one result per input subject")
        scores = [0.] * len(titles)
        for index, row in zip(indices, ranked, strict=True):
            scores[index] = float(row["relevance_score"])
        if not all(isfinite(value) for value in scores):
            raise ValueError("Jina returned a non-finite subject score")
        lowest = min(scores)
        span = max(max(scores) - lowest, MINIMUM_SCORE_SPAN)
        return [(score - lowest) / span for score in scores]
