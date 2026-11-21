"""Optional second-stage ranking for letters inside a recognized period."""

from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, wait
import math
from threading import Lock
from time import monotonic
from typing import Protocol

from .data import Document
from .dated import DatedSearcher
from .query_parser import parse_query


MODELS = {
    "mixedbread": ("mixedbread-ai/mxbai-rerank-large-v2", "ca7e1ee484c37c0ddd8d178a9a5c33cec575c5e6"),
    "bge": ("BAAI/bge-reranker-v2-m3", "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"),
    "qwen": ("Qwen/Qwen3-Reranker-0.6B", "e61197ed45024b0ed8a2d74b80b4d909f1255473"),
}
QWEN_PREFIX = (
    "<|im_start|>system\nDecide whether the document matches the query and instruction. "
    "Answer only yes or no.<|im_end|>\n<|im_start|>user\n"
)
QWEN_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class PairScorer(Protocol):
    def score(self, query: str, passages: Sequence[str], deadline: float) -> Sequence[float]: ...


class TransformersScorer:
    """Local pair relevance scores from a pinned BGE or Qwen checkpoint."""

    def __init__(
        self, name: str, *, device: str = "cpu", dtype: str = "float32",
        max_length: int = 256, batch_size: int = 4,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

        if name not in MODELS or dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("Unknown reranker or dtype")
        if max_length < 128 or batch_size < 1:
            raise ValueError("max_length must be at least 128 and batch_size must be positive")
        model_id, revision = MODELS[name]
        self._name = name
        self._torch = torch
        self._device = device
        self._max_length = max_length
        self._batch_size = batch_size
        if name == "mixedbread":
            from sentence_transformers import CrossEncoder

            self._cross_encoder = CrossEncoder(
                model_id, revision=revision, trust_remote_code=False, device=device,
                max_length=max_length, model_kwargs={"dtype": getattr(torch, dtype), "use_safetensors": True},
            )
            self._model = self._cross_encoder.model
            return
        self._tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=False)
        factory = AutoModelForCausalLM if name == "qwen" else AutoModelForSequenceClassification
        self._model = factory.from_pretrained(
            model_id, revision=revision, trust_remote_code=False, use_safetensors=True,
            dtype=getattr(torch, dtype),
        ).eval().to(device)
        if name == "qwen":
            self._tokenizer.padding_side = "left"
            self._prefix = self._tokenizer.encode(QWEN_PREFIX, add_special_tokens=False)
            self._suffix = self._tokenizer.encode(QWEN_SUFFIX, add_special_tokens=False)
            self._yes = self._tokenizer.convert_tokens_to_ids("yes")
            self._no = self._tokenizer.convert_tokens_to_ids("no")

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self._model.parameters())

    def score(self, query: str, passages: Sequence[str], deadline: float) -> list[float]:
        result: list[float] = []
        with self._torch.inference_mode():
            for offset in range(0, len(passages), self._batch_size):
                if monotonic() >= deadline:
                    raise TimeoutError("Reranking budget exhausted")
                batch = passages[offset:offset + self._batch_size]
                result.extend(self._batch_scores(query[:1500], [passage[:12000] for passage in batch]))
        return result

    def _batch_scores(self, query: str, passages: Sequence[str]) -> list[float]:
        if self._name == "mixedbread":
            return self._cross_encoder.predict(
                [(query, passage) for passage in passages], batch_size=len(passages),
                show_progress_bar=False,
            ).reshape(-1).tolist()
        tokenizer = self._tokenizer
        if self._name == "bge":
            inputs = tokenizer(
                [query] * len(passages), list(passages), max_length=self._max_length,
                truncation=True, padding=True, return_tensors="pt",
            ).to(self._device)
            return self._model(**inputs).logits.reshape(-1).float().tolist()

        instruction = "Find mail relevant to the user's search topic."
        text = [f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {p}" for p in passages]
        limit = self._max_length - len(self._prefix) - len(self._suffix)
        encoded = tokenizer(text, truncation=True, max_length=limit, add_special_tokens=False)["input_ids"]
        inputs = tokenizer.pad(
            {"input_ids": [self._prefix + ids + self._suffix for ids in encoded]},
            padding=True, return_attention_mask=True, return_tensors="pt",
        ).to(self._device)
        logits = self._model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1, :].float()
        return (logits[:, self._yes] - logits[:, self._no]).tolist()


class OnnxBgeScorer:
    """CPU experiment with a local quantized XLM-R sequence classifier export."""

    def __init__(self, model_path: str, tokenizer_path: str, *, max_length: int = 256) -> None:
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        self._session = ort.InferenceSession(model_path, sess_options=options,
                                             providers=["CPUExecutionProvider"])
        self._input_names = [item.name for item in self._session.get_inputs()]
        self._np = np
        self._tokenizer = Tokenizer.from_file(tokenizer_path)
        self._tokenizer.enable_truncation(max_length=max_length)
        self._tokenizer.enable_padding(pad_id=1, pad_token="<pad>")

    def score(self, query: str, passages: Sequence[str], deadline: float) -> list[float]:
        result = []
        for offset in range(0, len(passages), 4):
            if monotonic() >= deadline:
                raise TimeoutError("Reranking budget exhausted")
            batch = self._tokenizer.encode_batch([
                (query[:1500], passage[:12000]) for passage in passages[offset:offset + 4]
            ])
            fields = {"input_ids": "ids", "attention_mask": "attention_mask", "token_type_ids": "type_ids"}
            inputs = {name: self._np.asarray([getattr(item, fields[name]) for item in batch], dtype="int64")
                      for name in self._input_names}
            result.extend(next(iter(self._session.run(None, inputs))).reshape(-1).tolist())
        return result


class RerankSearcher:
    """Try one bounded rerank at a time, or keep the original dated BM25 result."""

    def __init__(
        self, baseline: DatedSearcher, docs: Mapping[str, Document], scorer: PairScorer,
        *, top_n: int = 20, budget_ms: float = 400,
    ) -> None:
        if top_n not in (10, 20, 30) or not math.isfinite(budget_ms) or budget_ms <= 0:
            raise ValueError("Use top_n 10, 20 or 30 and a finite positive budget_ms")
        self.baseline = baseline
        self.inner = baseline.inner
        self._docs = docs
        self._scorer = scorer
        self._top_n = top_n
        self._budget = budget_ms / 1000
        self._lock = Lock()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mail-rerank")
        self._pending: Future | None = None
        self._statistics: Counter = Counter()
        self._closed = False

    @property
    def statistics(self) -> dict[str, int]:
        with self._lock:
            return dict(self._statistics)

    def close(self) -> None:
        with self._lock:
            self._closed = True
        self._pool.shutdown(wait=True, cancel_futures=True)

    def wait_idle(self, timeout: float = 30) -> None:
        """Drain one outstanding job between independent offline benchmark queries."""
        with self._lock:
            pending = self._pending
        if pending is not None and wait((pending,), timeout=timeout).not_done:
            raise TimeoutError("Reranker did not become idle")

    def _note(self, status: str) -> None:
        with self._lock:
            self._statistics[status] += 1

    def search(self, query: str, mailbox: str, as_of: str | None, k: int) -> list[tuple[str, float]]:
        if k <= 0:
            return []
        deadline = monotonic() + self._budget
        parsed = parse_query(query, as_of)
        if not parsed.topic:
            return self.baseline.search_parsed(parsed, mailbox, as_of, k)
        ranked = self.baseline.search_parsed(parsed, mailbox, as_of, max(k, self._top_n))
        fallback = ranked[:k]
        period = set(self.baseline.in_period(mailbox, parsed, as_of)) if parsed.period else None
        slots = [
            (pos, doc_id) for pos, (doc_id, _) in enumerate(ranked)
            if period is None or doc_id in period
        ][:self._top_n]
        if len(slots) < 2:
            self._note("too_few_candidates")
            return fallback

        with self._lock:
            if self._closed or (self._pending is not None and not self._pending.done()):
                self._statistics["busy"] += 1
                return fallback
            if monotonic() >= deadline:
                self._statistics["timeout"] += 1
                return fallback
            passages = [self._passage(self._docs[doc_id]) for _, doc_id in slots]
            future = self._pool.submit(self._scorer.score, parsed.topic, passages, deadline)
            self._pending = future

        try:
            scores = list(future.result(timeout=max(0, deadline - monotonic())))
            if len(scores) != len(slots) or not all(math.isfinite(value) for value in scores):
                self._note("invalid_scores")
                return fallback
        except TimeoutError:
            future.cancel()
            self._note("timeout")
            return fallback
        except Exception:
            self._note("error")
            return fallback

        candidates = sorted(zip(slots, scores, strict=True), key=lambda item: -item[1])
        for (position, _), ((_, doc_id), score) in zip(slots, candidates, strict=True):
            ranked[position] = (doc_id, float(score))
        if monotonic() >= deadline:
            self._note("timeout")
            return fallback
        self._note("reranked")
        return ranked[:k]

    @staticmethod
    def _passage(doc: Document) -> str:
        subject = doc.subject
        text = doc.doc[:12000]
        return text if text.startswith(subject) else f"{subject}\n{text}"
