"""Local embeddings shared by archive indexing and online search."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from html.parser import HTMLParser
from itertools import islice
from threading import Lock
from unicodedata import normalize


MODEL_ID = "deepvk/USER2-base"
MODEL_REVISION = "94bbff0cc696f26dc93b275fc874657727df0c29"
EMBEDDING_DIM = 768


@dataclass(frozen=True)
class EmailText:
    subject: str | None
    body: str | None
    body_is_html: bool = False


class _HTMLText(HTMLParser):
    _SKIP_TAGS = {"head", "script", "style"}
    _BLOCK_TAGS = {"br", "div", "hr", "li", "p", "td", "th", "tr", "blockquote"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._hidden: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP_TAGS:
            self._hidden.append(tag)
        elif tag in self._BLOCK_TAGS and not self._hidden:
            self.parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if self._hidden and tag == self._hidden[-1]:
            self._hidden.pop()
        elif tag in self._BLOCK_TAGS and not self._hidden:
            self.parts.append(" ")

    def handle_data(self, data: str) -> None:
        if not self._hidden:
            self.parts.append(data)


def _clean_text(value: str) -> str:
    return " ".join(normalize("NFC", value).split())


def prepare_email_text(email: EmailText) -> str:
    """Combine the cleaned subject and body, with the subject first."""
    subject = _clean_text(email.subject or "")
    body = email.body or ""
    if email.body_is_html and body:
        parser = _HTMLText()
        parser.feed(body)
        parser.close()
        body = "".join(parser.parts)
    return "\n".join(part for part in (subject, _clean_text(body)) if part)


class MailEmbedder:
    """Load one model per instance and reuse it across calls."""

    def __init__(self, *, device: str | None = None, revision: str = MODEL_REVISION) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(
            MODEL_ID, device=device, revision=revision, trust_remote_code=False
        )
        if self._model.get_sentence_embedding_dimension() != EMBEDDING_DIM:
            raise RuntimeError(f"{MODEL_ID} must return {EMBEDDING_DIM}-dimensional vectors")
        self._lock = Lock()

    def embed_emails(
        self, emails: Iterable[EmailText], *, batch_size: int = 8
    ) -> Iterator[list[float]]:
        """Yield one vector per email in input order, keeping only one batch in memory."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        texts = map(prepare_email_text, emails)
        while batch := list(islice(texts, batch_size)):
            yield from self._encode(batch, "search_document")

    def embed_query(self, query: str) -> list[float]:
        """Encode a nonempty query using the model's retrieval query prompt."""
        text = _clean_text(query)
        if not text:
            raise ValueError("query must not be blank")
        return self._encode([text], "search_query")[0]

    def _encode(self, texts: list[str], prompt_name: str) -> list[list[float]]:
        # The shared tokenizer configures padding and truncation during encode().
        with self._lock:
            vectors = self._model.encode(
                texts,
                prompt_name=prompt_name,
                batch_size=len(texts),
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        if vectors.shape != (len(texts), EMBEDDING_DIM):
            raise RuntimeError(f"Unexpected embedding shape: {vectors.shape}")
        return vectors.tolist()
