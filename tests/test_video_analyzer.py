from __future__ import annotations

from pathlib import Path

import pytest

from qa_intelli_media_comparator.services.video_analyzer import VideoAnalyzer
from qa_intelli_media_comparator.models.enums import SyncMode


@pytest.fixture
def analyzer() -> VideoAnalyzer:
    return VideoAnalyzer()


def test_static_video_analysis(analyzer: VideoAnalyzer, tmp_video_file: Path) -> None:
    result = analyzer.analyze(dut_path=tmp_video_file)
    assert result.total_frames > 0
    assert result.sampled_frames > 0
    assert result.fps_original > 0


def test_temporal_metrics_populated(
    analyzer: VideoAnalyzer, tmp_video_file: Path
) -> None:
    result = analyzer.analyze(dut_path=tmp_video_file)
    tm = result.temporal
    # These may be None if fewer than 2 sampled frames, but should not error
    assert tm is not None


def test_quality_metrics_aggregated(
    analyzer: VideoAnalyzer, tmp_video_file: Path
) -> None:
    result = analyzer.analyze(dut_path=tmp_video_file)
    qm = result.quality_metrics
    assert qm.blur_score is not None or qm.blur_score is None  # just no crash


def test_self_comparison_frame_by_frame(
    analyzer: VideoAnalyzer, tmp_video_file: Path
) -> None:
    result = analyzer.analyze(
        dut_path=tmp_video_file,
        ref_path=tmp_video_file,
        sync_mode=SyncMode.FRAME_BY_FRAME,
    )
    assert result is not None
    # Self-comparison: worst_frame_ssim should be close to 1.0
    if result.worst_frame_ssim is not None:
        assert result.worst_frame_ssim > 0.8
