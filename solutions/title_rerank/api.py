"""Run the ordinary VK Mail interface with experimental subject reranking."""

from contextlib import asynccontextmanager
import os

from vk_mail_search.api import create_app as create_period_app
from vk_mail_search.period_rank import PeriodRanker

from .models import MODELS
from .searcher import SubjectReranker


def create_app(directory=None):
    app = create_period_app(directory)
    period_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(app):
        async with period_lifespan(app):
            baseline = app.state.searcher
            if not isinstance(baseline, PeriodRanker) or baseline.mode != "learned":
                raise ValueError("Subject reranking requires VK_SEARCHER=period")
            name = os.environ.get("VK_SUBJECT_MODEL", "jina")
            if name not in MODELS:
                raise ValueError("VK_SUBJECT_MODEL must be jina or qwen")
            scorer, default_weight = MODELS[name]
            weight = float(os.environ.get("VK_SUBJECT_WEIGHT", default_weight))
            if not 0 <= weight <= 1:
                raise ValueError("VK_SUBJECT_WEIGHT must be between 0 and 1")
            if weight:
                model = scorer(os.environ.get("VK_SUBJECT_DEVICE", "cpu"))
                app.state.searcher = SubjectReranker(baseline, app.state.docs, model, weight)
            yield

    app.router.lifespan_context = lifespan
    return app
