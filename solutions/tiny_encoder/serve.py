"""Serve the existing mail UI and search API with the standalone vector engine."""

from contextlib import asynccontextmanager
import os
from pathlib import Path

from vk_mail_search.api import create_app

from .adapter import QueryAdapter
from .encoder import Encoder
from .index import TinySearch
from .student import FastMailEncoder


def create_tiny_app():
    model = Path(os.environ["VK_TINY_MODEL"])
    index = Path(os.environ["VK_TINY_INDEX"])
    startup = create_app()
    lifespan_base = startup.router.lifespan_context

    @asynccontextmanager
    async def lifespan(app):
        if not index.is_file():
            raise ValueError("Build the tiny vector index before starting the service")
        max_tokens = int(os.getenv("VK_TINY_TOKENS", "80"))
        threads = int(os.getenv("VK_TINY_THREADS", "1"))
        encoder = Encoder(model, max_tokens=max_tokens, threads=threads)
        projection = os.getenv("VK_TINY_PROJECTION")
        tower = FastMailEncoder(encoder, Path(projection)) if projection else encoder
        adapter_path = os.getenv("VK_TINY_ADAPTER")
        adapter = QueryAdapter.load(Path(adapter_path), encoder.signature, max_tokens, encoder.dimension) if adapter_path else None
        async with lifespan_base(app):
            app.state.searcher = TinySearch(app.state.docs, tower, index, adapter=adapter)
            yield

    startup.router.lifespan_context = lifespan
    return startup
