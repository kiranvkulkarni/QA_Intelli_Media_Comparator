from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from qa_intelli_media_comparator.services.pipeline import ComparisonPipeline
from qa_intelli_media_comparator.models.enums import QualityGrade, SyncMode


@pytest.fixture
def pipeline(tmp_path, monkeypatch) -> ComparisonPipeline:
    # Use tmp_path as reports dir to avoid polluting project
    monkeypatch.setenv("QIMC_REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("QIMC_USE_NEURAL_NR", "false")
    # Reload settings
    from qa_intelli_media_comparator.config import get_settings
    get_settings.cache_clear()
    return ComparisonPipeline()


def test_image_nr_analysis(pipeline: ComparisonPipeline, tmp_image_file: Path) -> None:
    report = pipeline.run(dut_path=tmp_image_file, crop_preview=False)
    assert report.report_id
    assert report.media_type is not None
    assert report.quality_metrics is not None
    assert report.overall_grade in list(QualityGrade)


def test_image_fr_comparison(
    pipeline: ComparisonPipeline,
    tmp_image_file: Path,
    sharp_bgr: np.ndarray,
    tmp_path: Path,
) -> None:
    ref_path = tmp_path / "ref.jpg"
    cv2.imwrite(str(ref_path), sharp_bgr)

    report = pipeline.run(
        dut_path=tmp_image_file,
        reference_path=ref_path,
        crop_preview=False,
    )
    assert report.fr_scores is not None
    # psnr is always included in FR metrics; ssim may be omitted depending on profile
    assert report.fr_scores.psnr.value is not None


def test_identical_images_pass(
    pipeline: ComparisonPipeline,
    tmp_image_file: Path,
) -> None:
    report = pipeline.run(
        dut_path=tmp_image_file,
        reference_path=tmp_image_file,
        crop_preview=False,
    )
    # Identical images must not FAIL; WARNING is acceptable if NR metrics are advisory
    assert report.overall_grade != QualityGrade.FAIL


def test_video_analysis(pipeline: ComparisonPipeline, tmp_video_file: Path) -> None:
    report = pipeline.run(dut_path=tmp_video_file, crop_preview=False)
    assert report.video_temporal is not None
    assert report.quality_metrics is not None


def test_annotated_image_created(pipeline: ComparisonPipeline, tmp_image_file: Path) -> None:
    report = pipeline.run(dut_path=tmp_image_file, crop_preview=False)
    if report.annotated_image_path:
        assert Path(report.annotated_image_path).exists()


def test_failure_reasons_populated_for_overexposed(
    pipeline: ComparisonPipeline, overexposed_bgr: np.ndarray, tmp_path: Path
) -> None:
    p = tmp_path / "overexposed.jpg"
    cv2.imwrite(str(p), overexposed_bgr)
    report = pipeline.run(dut_path=p, crop_preview=False)
    # Overexposed image should trigger at least one failure reason
    assert len(report.failure_reasons) > 0 or report.overall_grade != QualityGrade.PASS


def test_report_is_json_serializable(
    pipeline: ComparisonPipeline, tmp_image_file: Path
) -> None:
    import json
    report = pipeline.run(dut_path=tmp_image_file, crop_preview=False)
    json_str = report.model_dump_json()
    parsed = json.loads(json_str)
    assert parsed["report_id"] == report.report_id
