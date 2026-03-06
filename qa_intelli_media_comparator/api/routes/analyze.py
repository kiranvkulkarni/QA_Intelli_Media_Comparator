from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from ..dependencies import get_pipeline, get_report_store
from ...models.report import ComparisonReport

router = APIRouter()


@router.post("/analyze", tags=["comparison"], response_model=ComparisonReport)
async def analyze(
    media: UploadFile = File(..., description="Single media file for no-reference analysis"),
    crop_preview: bool = Form(True, description="Auto-crop camera UI chrome from preview"),
) -> JSONResponse:
    """
    No-reference quality analysis of a single media file.
    Does not require a golden reference — useful for absolute quality assessment.
    """
    suffix = Path(media.filename or "file").suffix or ".tmp"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    media_path = Path(tmp.name)
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
        media_path.unlink(missing_ok=True)
