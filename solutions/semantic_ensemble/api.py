"""The existing mail UI with the semantic candidate ensemble."""

from contextlib import asynccontextmanager
import os

from navec import Navec

from solutions.tiny_encoder.ensemble import EnsembleSearch
from solutions.tiny_encoder.routed_api import create_routed_app

from .search import SemanticCandidateEnsemble


def create_app():
    app = create_routed_app()
    base_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(app):
        async with base_lifespan(app):
            original = app.state.searcher
            if not isinstance(original, EnsembleSearch):
                raise ValueError("The semantic candidate experiment requires VK_CONTEXT_MODE=ensemble")
            app.state.searcher = SemanticCandidateEnsemble(original, Navec.load(os.environ["VK_WORD_VECTORS"]))
            yield

    app.router.lifespan_context = lifespan
    return app
