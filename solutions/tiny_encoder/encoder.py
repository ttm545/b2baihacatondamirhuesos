"""A local PyTorch sentence encoder and the mail text it consumes."""

from collections.abc import Iterator, Sequence
from hashlib import sha256
import json
from pathlib import Path
from threading import Lock

import numpy as np
import torch
from torch.nn.functional import normalize

from vk_mail_search.data import Document, HEADER, HTML, SENDER
from vk_mail_search.embeddings import EmailText, prepare_email_text


TEXT_VERSION = "subject-leading-body-v1"


def letter_text(doc: Document) -> str:
    raw = doc.doc.lstrip()
    header = HEADER.search(raw)
    if header:
        subject = header[1]
        raw = raw[:header.start()] + raw[header.end():]
    else:
        subject = ""
    body = "\n".join(line for line in raw[:4000].splitlines() if not SENDER.match(line))
    return prepare_email_text(EmailText(subject[:200], body, bool(HTML.search(body))))[:1000]


def model_signature(directory: Path) -> str:
    digest = sha256()
    for name in ("config.json", "tokenizer.json", "1_Pooling/config.json", "model.safetensors"):
        digest.update(name.encode())
        with (directory / name).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


class Encoder:
    def __init__(self, path: Path, *, max_tokens: int, threads: int = 4, device: str = "cpu"):
        from transformers import AutoModel, AutoTokenizer

        if not 8 <= max_tokens <= 512 or threads < 1:
            raise ValueError("Use 8–512 tokens and at least one PyTorch thread")
        torch.set_num_threads(threads)
        self.device = torch.device(device)
        self.max_tokens = max_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        self.model = AutoModel.from_pretrained(path, local_files_only=True, trust_remote_code=False,
                                               use_safetensors=True, add_pooling_layer=False).eval().to(self.device)
        pooling = json.loads((path / "1_Pooling/config.json").read_text(encoding="utf-8"))
        self.pooling = "cls" if pooling.get("pooling_mode_cls_token") else "mean" if pooling.get("pooling_mode_mean_tokens") else None
        if self.pooling is None or self.tokenizer.pad_token_id is None:
            raise ValueError("The local model must provide CLS or mean pooling and a padding token")
        self.dimension = int(self.model.config.hidden_size)
        self.parameters = sum(parameter.numel() for parameter in self.model.parameters())
        self.signature = model_signature(path)
        self._query_lock = Lock()

    def _tokenize(self, texts: Sequence[str]):
        return self.tokenizer(list(texts), padding=False, truncation=True, max_length=self.max_tokens,
                              return_attention_mask=False, return_token_type_ids=False)["input_ids"]

    @torch.inference_mode()
    def _encode(self, tokens: Sequence[list[int]]) -> np.ndarray:
        length = max(map(len, tokens))
        pad = self.tokenizer.pad_token_id
        ids = torch.tensor([ids + [pad] * (length - len(ids)) for ids in tokens], device=self.device)
        lengths = torch.tensor([len(ids) for ids in tokens], device=self.device)
        mask = torch.arange(length, device=self.device)[None, :] < lengths[:, None]
        hidden = self.model(input_ids=ids, attention_mask=mask).last_hidden_state
        if self.pooling == "cls":
            vectors = hidden[:, 0]
        else:
            vectors = (hidden * mask[..., None]).sum(dim=1) / lengths[:, None]
        return normalize(vectors.float(), dim=1).cpu().numpy()

    def batches(self, texts: Sequence[str], batch_size: int) -> Iterator[tuple[list[int], np.ndarray]]:
        if batch_size < 1:
            raise ValueError("Batch size must be positive")
        tokens = self._tokenize(texts)
        order = sorted(range(len(tokens)), key=lambda index: len(tokens[index]))
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            yield indices, self._encode([tokens[i] for i in indices])

    def query(self, text: str) -> np.ndarray:
        with self._query_lock:
            return self._encode(self._tokenize([text[:1500]]))[0]
