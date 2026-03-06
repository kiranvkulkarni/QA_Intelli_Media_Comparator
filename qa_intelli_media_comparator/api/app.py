from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .middleware import RequestContextMiddleware
from .routes import compare_router, analyze_router, health_router, reports_router
from .dependencies import get_pipeline
from ..config import get_settings

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: preload neural models. Shutdown: no-op."""
    settings = get_settings()
    log.info("Starting QA Intelli Media Comparator on %s:%d", settings.host, settings.port)
    log.info("Device: %s | Reports dir: %s", settings.resolve_device(), settings.reports_dir)
    settings.reports_dir.mkdir(parents=True, exist_ok=True)

    pipeline = get_pipeline()
    pipeline.preload_models()
    log.info("Models ready: %s", pipeline.loaded_models())

    yield

    log.info("Shutting down.")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="QA Intelli Media Comparator",
        description=(
            "AI-based media comparison microservice for Smartphone Camera QA. "
            "Supports full-reference and no-reference image/video quality analysis "
            "with artifact detection and annotated output."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health_router)
    app.include_router(compare_router)
    app.include_router(analyze_router)
    app.include_router(reports_router)

    return app


app = create_app()
