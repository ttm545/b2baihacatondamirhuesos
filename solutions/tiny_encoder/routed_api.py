"""Expose period, vector, or automatic routing in the existing mail demo."""

from contextlib import asynccontextmanager
from dataclasses import replace
import os
from pathlib import Path

from vk_mail_search.api import create_app
from vk_mail_search.searchers import build_searcher

from .adapter import QueryAdapter
from .encoder import Encoder
from .ensemble import EnsembleOptions, EnsembleSearch
from .index import TinySearch
from .qwen_context import ContextModel, QwenContextSearch
from .qwen_embed import QwenEncoder
from .routed import RoutedSearch
from .semantic_bm25 import SemanticBM25, load_profiles
from .student import FastMailEncoder


def create_routed_app():
    app = create_app()
    base_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(app):
        mode = os.getenv("VK_CONTEXT_MODE", "auto")
        tower = os.getenv("VK_CONTEXT_ENCODER", "rubert")
        semantic_modes = {"semantic-bm25", "semantic-bm25-period", "ensemble-semantic-bm25"}
        if mode not in {"period", "vector", "auto", "hybrid", "context", "ensemble"} | semantic_modes:
            raise ValueError(f"Unknown VK_CONTEXT_MODE: {mode}")
        if tower not in {"rubert", "qwen"}:
            raise ValueError("Use VK_CONTEXT_ENCODER=rubert or qwen")
        if mode in {"context", "ensemble"} | semantic_modes and tower != "rubert":
            raise ValueError(f"The {mode} mode uses a compact RuBERT index")
        async with base_lifespan(app):
            period = app.state.searcher
            if os.getenv("VK_SEARCHER", "period") != "period":
                period = build_searcher("period", app.state.docs)
            if mode == "period":
                app.state.searcher = period
            else:
                index = Path(os.environ["VK_CONTEXT_INDEX"])
                if not index.is_file():
                    raise ValueError("Build the full mail index before starting the demo")
                adapter = None
                if tower == "qwen":
                    encoder = QwenEncoder(int(os.getenv("VK_QWEN_PORT", "18767")),
                                          int(os.getenv("VK_QWEN_TEXT_CHARS", "220")),
                                          index.with_suffix(".sqlite"))
                else:
                    encoder = Encoder(Path(os.environ["VK_TINY_MODEL"]), max_tokens=80,
                                      threads=int(os.getenv("VK_TINY_THREADS", "1")), device="cpu")
                    if adapter_path := os.getenv("VK_TINY_ADAPTER"):
                        adapter = QueryAdapter.load(Path(adapter_path), encoder.signature, 80, encoder.dimension)
                    encoder = FastMailEncoder(encoder, Path(os.environ["VK_TINY_PROJECTION"]))
                vector = TinySearch(app.state.docs, encoder, index, adapter=adapter)
                if mode in {"context", "ensemble", "ensemble-semantic-bm25"}:
                    model = ContextModel.load(Path(os.environ["VK_CONTEXT_ADAPTER"]), encoder.signature,
                                              encoder.dimension)
                    depth = int(os.getenv("VK_CONTEXT_DEPTH", str(model.depth)))
                    if not 0 <= depth <= 60:
                        raise ValueError("VK_CONTEXT_DEPTH must be between 0 and 60")
                    model = replace(model, depth=depth)
                    context = QwenContextSearch(period, vector, model)
                    if mode in {"ensemble", "ensemble-semantic-bm25"}:
                        options = EnsembleOptions.load(Path(os.environ.get(
                            "VK_ENSEMBLE_CONFIG", str(Path(__file__).parent / "assets" / "ensemble.json"))))
                        feedback = None
                        if mode == "ensemble-semantic-bm25":
                            config = Path(os.getenv("VK_ENSEMBLE_BM25_CONFIG", str(
                                Path(__file__).parent / "assets" / "ensemble-semantic-bm25.json")))
                            feedback = SemanticBM25(period, vector, *load_profiles(config))
                        app.state.searcher = EnsembleSearch(context, options, semantic_bm25=feedback)
                    else:
                        app.state.searcher = context
                elif mode in semantic_modes:
                    config = Path(os.getenv("VK_SEMANTIC_BM25_CONFIG",
                                            str(Path(__file__).parent / "assets" / "semantic-bm25.json")))
                    profiles = load_profiles(config)
                    app.state.searcher = SemanticBM25(period, vector, *profiles, rerank=mode.endswith("-period"))
                elif mode == "vector":
                    app.state.searcher = vector
                else:
                    app.state.searcher = RoutedSearch(period, vector, fuse_undated=mode == "hybrid")
            yield

    app.router.lifespan_context = lifespan
    return app
