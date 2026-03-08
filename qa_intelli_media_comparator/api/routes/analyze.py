from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from ..dependencies import get_pipeline, get_report_store
from ...config import Settings, _request_settings
from ...models.report import ComparisonReport

router = APIRouter()

_VALID_PROFILES = {"low", "medium", "high", "critical"}


@router.post("/analyze", tags=["comparison"], response_model=ComparisonReport)
async def analyze(
    media: UploadFile = File(..., description="Single media file for no-reference analysis"),
    crop_preview: bool = Form(True, description="Auto-crop camera UI chrome from preview"),
    quality_profile: Optional[str] = Form(
        None,
        description="Per-request quality profile override: low | medium | high | critical. "
                    "Overrides server-side QIMC_QUALITY_PROFILE for this request only.",
    ),
) -> JSONResponse:
    """
    No-reference quality analysis of a single media file.
    Does not require a golden reference — useful for absolute quality assessment.

    **Quality profiles** (per-request, overrides server default):
    - `low` — unstable rig / outdoor / handheld
    - `medium` — semi-stable indoor / lightbox
    - `high` — stable lightbox + tripod (default)
    - `critical` — robotic / pixel-aligned rig
    """
    suffix = Path(media.filename or "file").suffix or ".tmp"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    media_path = Path(tmp.name)

    # Apply per-request quality profile (scoped to this async context only)
    token = None
    if quality_profile:
        profile = quality_profile.strip().lower()
        if profile not in _VALID_PROFILES:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid quality_profile '{quality_profile}'. "
                       f"Must be one of: {', '.join(sorted(_VALID_PROFILES))}",
            )
        token = _request_settings.set(Settings(quality_profile=profile))

    try:
        content = await media.read()
        tmp.write(content)
        tmp.flush()
        tmp.close()

        pipeline = get_pipeline()
        store = get_report_store()

        try:
            report = pipeline.run(
                dut_path=media_path,
                reference_path=None,
                crop_preview=crop_preview,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc

        store.save(report)

        return JSONResponse(
            content=report.model_dump(mode="json"),
            headers={"X-Report-Id": report.report_id},
        )
    finally:
        if token is not None:
            _request_settings.reset(token)
        media_path.unlink(missing_ok=True)
