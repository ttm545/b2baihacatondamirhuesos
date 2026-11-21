"""Local quantized cross-encoder. Models are supplied as files by the runner."""

from pathlib import Path
import re

from .data import HEADER, SENDER, Document


_VARIABLE = re.compile(r"https?://\S+|\b(?:[\w.-]+@)[\w.-]+|\b[a-z][a-z0-9-]*-\d[a-z0-9-]*\b|\b\d[\d.,:/-]*\b", re.I)


def passage(doc: Document) -> str:
    lines = doc.doc.splitlines()
    body = " ".join(line for line in lines if not (HEADER.match(line) or SENDER.match(line)))
    return f"{doc.subject}\n{body}"


def template_key(doc: Document) -> str:
    return " ".join(_VARIABLE.sub(" ", passage(doc).casefold()[:1200]).split())[:130]


def grouped_passages(docs: list[Document], query: str, limit: int):
    """Share inference for repeated templates unless the query contains a specific identifier."""
    specific = any(char.isdigit() for char in query)
    representatives = {}
    assignments = []
    for doc in docs:
        key = doc.doc_id if specific else template_key(doc)
        if key not in representatives and len(representatives) < limit:
            text = passage(doc)[:1200]
            representatives[key] = (len(representatives), text if specific else _VARIABLE.sub(" ", text))
        record = representatives.get(key)
        assignments.append(record[0] if record is not None else None)
    return [text for _, text in representatives.values()], assignments


class MiniLMScorer:
    def __init__(self, directory: Path, *, max_length: int = 112, threads: int = 1):
        import numpy
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self._numpy = numpy
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self._model = ort.InferenceSession(str(directory / "model_quint8_avx2.onnx"),
                                           sess_options=options, providers=["CPUExecutionProvider"])
        self._inputs = {item.name for item in self._model.get_inputs()}
        self._tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        pad_id = self._tokenizer.token_to_id("<pad>")
        if pad_id is None:
            raise ValueError("Expected the multilingual MiniLM tokenizer")
        self._tokenizer.enable_padding(pad_id=pad_id, pad_token="<pad>")
        self._tokenizer.enable_truncation(max_length=max_length, strategy="longest_first")

    def score(self, query: str, passages: list[str]) -> list[float]:
        encoded = self._tokenizer.encode_batch([(query[:480], text[:2500]) for text in passages])
        rows = {"input_ids": [item.ids for item in encoded],
                "attention_mask": [item.attention_mask for item in encoded],
                "token_type_ids": [item.type_ids for item in encoded]}
        inputs = {name: self._numpy.asarray(rows[name], dtype=self._numpy.int64) for name in self._inputs}
        logits, = self._model.run(None, inputs)
        return logits.ravel().tolist()
