"""HTTP API and mailbox UI for the VK search corpus."""

import json
import os
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.routing import APIRoute

WEB = Path(__file__).with_name("web")


def load_documents(directory):
    with (directory / "documents.jsonl").open(encoding="utf-8-sig") as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


class MailStore:
    def __init__(self, documents):
        self.by_id = {}
        self.boxes = defaultdict(list)
        for record in documents:
            doc = dict(record, doc_id=str(record["doc_id"]), synthetic_user=str(record["synthetic_user"]))
            if doc["doc_id"] in self.by_id:
                raise ValueError(f"Duplicate doc_id: {doc['doc_id']}")
            self.by_id[doc["doc_id"]] = doc
            self.boxes[doc["synthetic_user"]].append(doc)
        for docs in self.boxes.values():
            docs.sort(key=lambda d: d.get("doc_dt") or "", reverse=True)


class SubstringSearch:
    def __init__(self, store):
        self.boxes = {
            mailbox: [(d["doc_id"], d.get("doc_dt"), d["doc"].casefold()) for d in docs]
            for mailbox, docs in store.boxes.items()
        }

    def search(self, query, mailbox, as_of, k):
        needle = query.strip().casefold()
        hits = []
        for doc_id, sent, body in self.boxes.get(mailbox, ()):
            if as_of and sent and sent > as_of:
                continue
            if count := body.count(needle):
                hits.append((doc_id, float(count)))
        hits.sort(key=lambda pair: pair[1], reverse=True)
        return hits[:k]


def summary(doc, query=""):
    body = doc["doc"]
    lines = (line.strip() for line in body.splitlines() if line.strip())
    subject = next(lines, "Без темы")
    for line in body.splitlines()[:12]:
        key, sep, value = line.partition(":")
        if sep and key.strip().lower() in {"тема", "subject"} and value.strip():
            subject = value.strip()
            break
    preview = " ".join(body.split())
    pos = preview.lower().find(query.lower()) if query else 0
    start = max(0, pos - 45)
    return dict(doc_id=doc["doc_id"], doc_dt=doc.get("doc_dt"), subject=subject[:160],
                snippet=("…" if start else "") + preview[start:start + 200]
                + ("…" if len(preview) > start + 200 else ""))


class TimedRoute(APIRoute):
    def get_route_handler(self):
        handle = super().get_route_handler()

        async def timed(request):
            request.state.started = perf_counter()
            return await handle(request)

        return timed


def create_app(directory=None, search_factory=SubstringSearch):
    @asynccontextmanager
    async def lifespan(app):
        root = directory or os.getenv("VK_DATA_DIR")
        if not root:
            raise RuntimeError("Укажите VK_DATA_DIR — каталог с documents.jsonl")
        store = MailStore(load_documents(Path(root)))
        app.state.store = store
        app.state.searcher = search_factory(store)
        yield

    app = FastAPI(title="VK · Поиск в почте", lifespan=lifespan)
    app.router.route_class = TimedRoute

    @app.get("/api/mailboxes")
    def mailboxes():
        return [dict(mailbox=name, count=len(docs)) for name, docs in sorted(app.state.store.boxes.items())]

    @app.get("/api/mailboxes/{mailbox}/docs")
    def list_docs(mailbox: str, offset: int = Query(0, ge=0), limit: int = Query(30, ge=1, le=200)):
        docs = app.state.store.boxes.get(mailbox)
        if docs is None:
            raise HTTPException(404, "Ящик не найден")
        return [summary(doc) for doc in docs[offset:offset + limit]]

    @app.get("/api/docs/{doc_id}")
    def detail(doc_id: str):
        doc = app.state.store.by_id.get(doc_id)
        if doc is None:
            raise HTTPException(404, "Письмо не найдено")
        return {**doc, "subject": summary(doc)["subject"]}

    @app.get("/api/search")
    def search(request: Request, q: str = Query(min_length=1), mailbox: str = Query(min_length=1),
               as_of: date | None = None, k: int = Query(10, ge=1)):
        query = q.strip()
        if not query:
            raise HTTPException(422, "Пустой запрос")
        if mailbox not in app.state.store.boxes:
            raise HTTPException(404, "Ящик не найден")
        hits = app.state.searcher.search(query, mailbox, as_of.isoformat() if as_of else None, k)
        rows = [dict(**summary(app.state.store.by_id[str(doc_id)], query), score=float(score))
                for doc_id, score in hits]
        return dict(took_ms=round((perf_counter() - request.state.started) * 1000, 3), results=rows)

    @app.get("/", include_in_schema=False)
    def home():
        return FileResponse(WEB / "index.html")

    app.mount("/static", StaticFiles(directory=WEB), name="assets")
    return app


app = create_app()
