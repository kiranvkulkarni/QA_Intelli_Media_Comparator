from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from ..dependencies import get_pipeline, get_report_store
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


@router.post("/compare", tags=["comparison"], response_model=ComparisonReport)
async def compare(
    dut: UploadFile = File(..., description="DUT media file (image or video)"),
    reference: Optional[UploadFile] = File(None, description="Golden reference file (optional)"),
    sync_mode: str = Form("auto", description="Video sync: 'auto' or 'frame_by_frame'"),
    crop_preview: bool = Form(True, description="Auto-crop camera UI chrome"),
    force_media_type: Optional[str] = Form(None, description="Override auto-detection"),
) -> JSONResponse:
    """
    Compare a DUT media file against an optional golden reference.

    Supports images (JPEG, PNG, HEIC, WEBP) and videos (MP4, MOV, AVI, MKV).
    Auto-detects image vs video and preview vs captured.
    """
    dut_suffix = Path(dut.filename or "file").suffix or ".tmp"
    dut_path = await _save_upload(dut, dut_suffix)
    ref_path: Optional[Path] = None

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
        dut_path.unlink(missing_ok=True)
        if ref_path:
            ref_path.unlink(missing_ok=True)
