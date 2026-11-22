"""HTTP API and mailbox UI for the VK search corpus."""

import html
import re
import os
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import date
from heapq import nlargest
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.routing import APIRoute

from .bm25 import TAGS
from .data import HTML, Document, load_documents
from .query_parser import parse_query
from .searchers import DEFAULT_SEARCHER, build_searcher

WEB = Path(__file__).with_name("web")


MAIL_HEADER = re.compile(
    r"^(?:от|от кого|отправитель|кому|копия|тема|дата|from|to|cc|bcc|subject|date|reply-to)\s*:", re.I
)


SUBJECT_VARIABLE = re.compile(
    r"[\w.+%-]+@[\w.-]+|SYN-[A-Za-z0-9-]+|"
    r"\d{1,2}\s*(?:января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)"
    r"(?:\s*\d{4}(?:\s+года)?)?|\d+(?:[.:/-]\d+)*", re.I
)


def similarity_topic(subject: str) -> str:
    # Structured notifications append delivery metadata after the event title.
    title = re.split(r",\s*(?:дата|время|терминал)\b|,\s*(?:дата|время)(?=\d)", subject, maxsplit=1, flags=re.I)[0]
    title = re.sub(r"^(?:клиент|получатель)\s*:\s*", "", title, flags=re.I)
    title = re.sub(r"\b[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+){2}\b", " ", title)
    return " ".join(re.findall(r"[^\W\d_]+", SUBJECT_VARIABLE.sub(" ", title)))


def summary(doc: Document, query=""):
    lines = iter(doc.doc.strip().splitlines())
    for line in lines:
        if line.strip() and not MAIL_HEADER.match(line.strip()):
            break
    else:
        line = ""
    body = "\n".join([line, *lines])
    if HTML.search(body):
        body = html.unescape(TAGS.sub(" ", body))
    preview = " ".join(body.split())
    pos = preview.lower().find(query.lower()) if query else 0
    start = max(0, pos - 45)
    return dict(doc_id=doc.doc_id, doc_dt=doc.doc_dt, subject=doc.subject,
                snippet=("…" if start else "") + preview[start:start + 200]
                + ("…" if len(preview) > start + 200 else ""))


def search_archive(searcher, docs, boxes, query, as_of, period, k):
    # Scores use mailbox-local statistics. Preserve each mailbox's ranking,
    # compare scores within the same rank, and prefer letters in the period.
    ranked = (
        (doc_id, score, rank)
        for box in boxes
        for rank, (doc_id, score) in enumerate(searcher.search(query, box, as_of, k))
    )

    def order(hit):
        doc_id, score, rank = hit
        doc_date = docs[doc_id].doc_dt
        return (period is None or period.contains(doc_date), -rank, score, doc_date or "")

    return [(doc_id, score) for doc_id, score, _ in nlargest(k, ranked, key=order)]


class TimedRoute(APIRoute):
    def get_route_handler(self):
        handle = super().get_route_handler()

        async def timed(request):
            request.state.started = perf_counter()
            return await handle(request)

        return timed


def create_app(directory=None):
    @asynccontextmanager
    async def lifespan(app):
        docs = load_documents(directory)
        boxes = defaultdict(list)
        for doc in docs.values():
            boxes[doc.synthetic_user].append(doc)
        for letters in boxes.values():
            letters.sort(key=lambda d: d.doc_dt or "", reverse=True)
        archive_id = "__all__"
        while archive_id in boxes:
            archive_id += "_"
        app.state.archive_id = archive_id
        app.state.archive = sorted(docs.values(), key=lambda doc: doc.doc_dt or "", reverse=True)
        app.state.docs = docs
        app.state.boxes = boxes
        app.state.searcher = build_searcher(os.getenv("VK_SEARCHER", DEFAULT_SEARCHER), docs)
        yield

    app = FastAPI(title="VK · Поиск в почте", lifespan=lifespan)
    app.router.route_class = TimedRoute

    @app.get("/api/mailboxes")
    def mailboxes(include_all: bool = False):
        boxes = [dict(mailbox=name, count=len(docs), latest=docs[0].doc_dt)
                 for name, docs in sorted(app.state.boxes.items())]
        if include_all and app.state.archive:
            boxes.insert(0, dict(mailbox=app.state.archive_id, name="Все письма",
                                 count=len(app.state.archive), latest=app.state.archive[0].doc_dt))
        return boxes

    @app.get("/api/mailboxes/{mailbox}/docs")
    def list_docs(mailbox: str, offset: int = Query(0, ge=0), limit: int = Query(30, ge=1, le=200)):
        is_archive = mailbox == app.state.archive_id
        docs = app.state.archive if is_archive else app.state.boxes.get(mailbox)
        if docs is None:
            raise HTTPException(404, "Ящик не найден")
        page = docs[offset:offset + limit]
        if is_archive:
            return [dict(summary(doc), mailbox=doc.synthetic_user) for doc in page]
        return [summary(doc) for doc in page]

    @app.get("/api/docs/{doc_id}")
    def detail(doc_id: str):
        doc = app.state.docs.get(doc_id)
        if doc is None:
            raise HTTPException(404, "Письмо не найдено")
        return {**asdict(doc), "subject": doc.subject}

    @app.get("/api/docs/{doc_id}/similar")
    def similar(doc_id: str, k: int = Query(5, ge=1, le=100)):
        source = app.state.docs.get(doc_id)
        if source is None:
            raise HTTPException(404, "Письмо не найдено")
        subject = similarity_topic(source.subject)
        if not subject:
            return []
        engine = getattr(app.state.searcher, "inner", app.state.searcher)
        hits = engine.search(subject, source.synthetic_user, None, k + 1)
        return [summary(app.state.docs[other_id], subject)
                for other_id, _ in hits if other_id != doc_id][:k]

    @app.get("/api/search")
    def search(request: Request, q: str = Query(min_length=1), mailbox: str = Query(min_length=1),
               as_of: date | None = None, k: int = Query(10, ge=1, le=100)):
        query = q.strip()
        if not query:
            raise HTTPException(422, "Пустой запрос")
        is_archive = mailbox == app.state.archive_id
        if not is_archive and mailbox not in app.state.boxes:
            raise HTTPException(404, "Ящик не найден")
        parsed = parse_query(query, as_of)
        cutoff = as_of.isoformat() if as_of else None
        if is_archive:
            hits = search_archive(app.state.searcher, app.state.docs, app.state.boxes,
                                  query, cutoff, parsed.period, k)
        else:
            hits = app.state.searcher.search(query, mailbox, cutoff, k)
        rows = []
        for doc_id, score in hits:
            doc = app.state.docs[doc_id]
            row = dict(summary(doc, parsed.topic), score=float(score),
                       in_period=parsed.period.contains(doc.doc_dt) if parsed.period else None)
            if is_archive:
                row["mailbox"] = doc.synthetic_user
            rows.append(row)
        return dict(took_ms=round((perf_counter() - request.state.started) * 1000, 3),
                    period=parsed.period.label() if parsed.period else None, topic=parsed.topic, results=rows)

    @app.get("/", include_in_schema=False)
    def home():
        return FileResponse(WEB / "index.html")

    app.mount("/static", StaticFiles(directory=WEB), name="assets")
    return app


app = create_app()
