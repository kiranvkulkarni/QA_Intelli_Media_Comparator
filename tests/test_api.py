from __future__ import annotations

"""Integration tests for the FastAPI endpoints using httpx."""

import io
from pathlib import Path

import cv2
import numpy as np
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport


@pytest.fixture(autouse=True)
def reset_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("QIMC_REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("QIMC_USE_NEURAL_NR", "false")
    from qa_intelli_media_comparator.config import get_settings
    from qa_intelli_media_comparator.api.dependencies import get_pipeline, get_report_store
    get_settings.cache_clear()
    get_pipeline.cache_clear()
    get_report_store.cache_clear()


@pytest.fixture
def app():
    from qa_intelli_media_comparator.api.app import create_app
    return create_app()


def _encode_image(img: np.ndarray) -> bytes:
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


@pytest.mark.asyncio
async def test_health_endpoint(app) -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "device" in data


@pytest.mark.asyncio
async def test_analyze_endpoint(app, sharp_bgr: np.ndarray) -> None:
    img_bytes = _encode_image(sharp_bgr)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/analyze",
            files={"media": ("test.jpg", img_bytes, "image/jpeg")},
            data={"crop_preview": "false"},
        )
    assert response.status_code == 200
    data = response.json()
    assert "report_id" in data
    assert "overall_grade" in data
    assert "quality_metrics" in data


@pytest.mark.asyncio
async def test_compare_endpoint_no_reference(app, sharp_bgr: np.ndarray) -> None:
    img_bytes = _encode_image(sharp_bgr)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/compare",
            files={"dut": ("dut.jpg", img_bytes, "image/jpeg")},
            data={"crop_preview": "false"},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["fr_scores"] is None  # no reference provided


@pytest.mark.asyncio
async def test_compare_endpoint_with_reference(
    app, sharp_bgr: np.ndarray, blurry_bgr: np.ndarray
) -> None:
    dut_bytes = _encode_image(blurry_bgr)
    ref_bytes = _encode_image(sharp_bgr)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/compare",
            files={
                "dut": ("dut.jpg", dut_bytes, "image/jpeg"),
                "reference": ("ref.jpg", ref_bytes, "image/jpeg"),
            },
            data={"crop_preview": "false"},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["fr_scores"] is not None


@pytest.mark.asyncio
async def test_report_retrieval(app, sharp_bgr: np.ndarray) -> None:
    img_bytes = _encode_image(sharp_bgr)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Create report
        r1 = await client.post(
            "/analyze",
            files={"media": ("test.jpg", img_bytes, "image/jpeg")},
            data={"crop_preview": "false"},
        )
        assert r1.status_code == 200
        report_id = r1.json()["report_id"]

        # Retrieve it
        r2 = await client.get(f"/report/{report_id}")
        assert r2.status_code == 200
        assert r2.json()["report_id"] == report_id


@pytest.mark.asyncio
async def test_reports_list(app, sharp_bgr: np.ndarray) -> None:
    img_bytes = _encode_image(sharp_bgr)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        await client.post(
            "/analyze",
            files={"media": ("test.jpg", img_bytes, "image/jpeg")},
            data={"crop_preview": "false"},
        )
        r = await client.get("/reports")
    assert r.status_code == 200
    data = r.json()
    assert "reports" in data
    assert data["count"] >= 1


@pytest.mark.asyncio
async def test_404_for_unknown_report(app) -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/report/nonexistent_id_xyz")
    assert r.status_code == 404
