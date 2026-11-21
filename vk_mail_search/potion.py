"""Fast local semantic scores from static multilingual embeddings."""

from pathlib import Path

from .data import Document
from .minilm import passage


class PotionScorer:
    def __init__(self, directory: Path):
        import numpy
        from safetensors import safe_open
        from tokenizers import Tokenizer

        self._numpy = numpy
        with safe_open(directory / "model.safetensors", framework="np") as model:
            self._embeddings = model.get_tensor("embeddings")
        self._tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        self._tokenizer.enable_truncation(max_length=100)
        self._unk_id = self._tokenizer.token_to_id("<unk>")

    def _encode(self, texts: list[str]):
        encoded = self._tokenizer.encode_batch(texts, add_special_tokens=False)
        vectors = self._numpy.zeros((len(texts), self._embeddings.shape[1]), dtype=self._embeddings.dtype)
        for i, item in enumerate(encoded):
            ids = [token_id for token_id in item.ids if token_id != self._unk_id]
            if ids:
                vectors[i] = self._embeddings[ids].mean(axis=0)
        norms = self._numpy.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / self._numpy.maximum(norms, 1e-8)

    def score(self, terms: list[str], docs: list[Document]) -> list[float]:
        if not terms:
            return [0.0] * len(docs)
        texts = [" ".join(terms)]
        texts.extend(doc.subject[:220] for doc in docs)
        texts.extend(passage(doc)[:700] for doc in docs)
        vectors = self._encode(texts)
        subject = vectors[1:len(docs) + 1] @ vectors[0]
        body = vectors[len(docs) + 1:] @ vectors[0]
        return (.62 * subject + .38 * body).tolist()
