from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from ..dependencies import get_pipeline, get_report_store
from ...config import Settings, _request_settings
from ...models.enums import SyncMode
from ...models.report import ComparisonReport

router = APIRouter()


async def _save_upload(upload: UploadFile, suffix: str) -> Path:
    """Save an UploadFile to a temp file and return its path."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        content = await upload.read()
        tmp.write(content)
        tmp.flush()
        return Path(tmp.name)
    finally:
        tmp.close()


_VALID_PROFILES = {"low", "medium", "high", "critical"}
_VALID_MODES    = {"functional", "quality"}


@router.post("/compare", tags=["comparison"], response_model=ComparisonReport)
async def compare(
    dut: UploadFile = File(..., description="DUT media file (image or video)"),
    reference: Optional[UploadFile] = File(None, description="Golden reference file (optional)"),
    sync_mode: str = Form("auto", description="Video sync: 'auto' or 'frame_by_frame'"),
    crop_preview: bool = Form(True, description="Auto-crop camera UI chrome"),
    force_media_type: Optional[str] = Form(None, description="Override auto-detection"),
    quality_profile: Optional[str] = Form(
        None,
        description="Per-request quality profile override: low | medium | high | critical. "
                    "Overrides server-side QIMC_QUALITY_PROFILE for this request only.",
    ),
    analysis_mode: Optional[str] = Form(
        None,
        description=(
            "Analysis depth: 'functional' (fast, ~50ms — functional checks + basic metrics, "
            "no neural IQA) or 'quality' (full IQA pipeline, default). "
            "Overrides QIMC_ANALYSIS_MODE for this request."
        ),
    ),
) -> JSONResponse:
    """
    Compare a DUT media file against an optional golden reference.

    Supports images (JPEG, PNG, HEIC, WEBP) and videos (MP4, MOV, AVI, MKV).
    Auto-detects image vs video and preview vs captured.

    **Analysis modes** (per-request):
    - `functional` — fast path (~50ms): is the camera working? black/frozen frame?
      correct scene? Basic quality metrics only, no neural IQA.
    - `quality` — full path: all IQA metrics + functional checks (default).

    **Quality profiles** (per-request, overrides server default):
    - `low` — unstable rig / outdoor / handheld
    - `medium` — semi-stable indoor / lightbox
    - `high` — stable lightbox + tripod (default)
    - `critical` — robotic / pixel-aligned rig
    """
    if analysis_mode and analysis_mode.strip().lower() not in _VALID_MODES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid analysis_mode '{analysis_mode}'. Must be: functional | quality",
        )
    dut_suffix = Path(dut.filename or "file").suffix or ".tmp"
    dut_path = await _save_upload(dut, dut_suffix)
    ref_path: Optional[Path] = None

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
        if reference is not None:
            ref_suffix = Path(reference.filename or "file").suffix or ".tmp"
            ref_path = await _save_upload(reference, ref_suffix)

        try:
            sync = SyncMode(sync_mode)
        except ValueError:
            raise HTTPException(status_code=422, detail=f"Invalid sync_mode '{sync_mode}'")

        pipeline = get_pipeline()
        store = get_report_store()

        try:
            report = pipeline.run(
                dut_path=dut_path,
                reference_path=ref_path,
                sync_mode=sync,
                crop_preview=crop_preview,
                force_media_type=force_media_type,
                analysis_mode=analysis_mode,
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc

        store.save(report)

        response = JSONResponse(
            content=report.model_dump(mode="json"),
            headers={"X-Report-Id": report.report_id},
        )
        return response

    finally:
        if token is not None:
            _request_settings.reset(token)
        dut_path.unlink(missing_ok=True)
        if ref_path:
            ref_path.unlink(missing_ok=True)
