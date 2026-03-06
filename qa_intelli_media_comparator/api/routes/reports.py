from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse

from ..dependencies import get_report_store
from ...models.report import ComparisonReport

router = APIRouter()


@router.get("/report/{report_id}", tags=["reports"], response_model=ComparisonReport)
async def get_report(report_id: str) -> JSONResponse:
    """Retrieve the full JSON report for a given report ID."""
    store = get_report_store()
    report = store.load(report_id)
    if report is None:
        raise HTTPException(status_code=404, detail=f"Report '{report_id}' not found.")
    return JSONResponse(report.model_dump(mode="json"))


@router.get("/report/{report_id}/annotated", tags=["reports"])
async def get_annotated_image(report_id: str) -> FileResponse:
    """Return the artifact-annotated PNG image for a report."""
    store = get_report_store()
    path = store.get_annotated_path(report_id)
    if path is None:
        raise HTTPException(
            status_code=404,
            detail=f"Annotated image for report '{report_id}' not found.",
        )
    return FileResponse(str(path), media_type="image/png")


@router.get("/report/{report_id}/diff", tags=["reports"])
async def get_diff_image(report_id: str) -> FileResponse:
    """Return the difference heatmap PNG for a report (requires reference comparison)."""
    store = get_report_store()
    path = store.get_diff_path(report_id)
    if path is None:
        raise HTTPException(
            status_code=404,
            detail=f"Diff image for report '{report_id}' not found.",
        )
    return FileResponse(str(path), media_type="image/png")


@router.get("/reports", tags=["reports"])
async def list_reports(
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
    grade: Optional[str] = Query(None, description="Filter by grade: pass|warning|fail"),
    media_type: Optional[str] = Query(None, description="Filter by media type"),
) -> JSONResponse:
    """List stored reports with optional filtering."""
    store = get_report_store()
    rows = store.list_reports(limit=limit, offset=offset, grade=grade, media_type=media_type)
    return JSONResponse({"reports": rows, "count": len(rows), "offset": offset})
