from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ..dependencies import get_pipeline
from ...config import get_settings

router = APIRouter()


@router.get("/health", tags=["system"])
async def health() -> JSONResponse:
    settings = get_settings()
    pipeline = get_pipeline()
    device = settings.resolve_device()
    return JSONResponse({
        "status": "ok",
        "version": "1.0.0",
        "device": device,
        "models_loaded": pipeline.loaded_models(),
        "reports_dir": str(settings.reports_dir),
    })
