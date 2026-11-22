"""The existing mail UI with the semantic candidate ensemble."""

from contextlib import asynccontextmanager
import os

from navec import Navec

from solutions.tiny_encoder.ensemble import EnsembleSearch
from solutions.tiny_encoder.routed_api import create_routed_app

from .search import DEFAULT_SEMANTIC_WEIGHT, SemanticCandidateEnsemble


def create_app(directory=None):
    app = create_routed_app(directory)
    base_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(app):
        async with base_lifespan(app):
            original = app.state.searcher
            if not isinstance(original, EnsembleSearch):
                raise ValueError("The semantic candidate experiment requires VK_CONTEXT_MODE=ensemble")
            weight = float(os.environ.get("VK_SEMANTIC_WEIGHT", DEFAULT_SEMANTIC_WEIGHT))
            app.state.searcher = SemanticCandidateEnsemble(original, Navec.load(os.environ["VK_WORD_VECTORS"]),
                                                            semantic_weight=weight)
            yield

    app.router.lifespan_context = lifespan
    return app
