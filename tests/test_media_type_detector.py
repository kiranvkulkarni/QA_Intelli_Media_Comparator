from __future__ import annotations

import cv2
import numpy as np
import pytest
from pathlib import Path

from qa_intelli_media_comparator.services.media_type_detector import MediaTypeDetector
from qa_intelli_media_comparator.models.enums import MediaType


@pytest.fixture
def detector() -> MediaTypeDetector:
    return MediaTypeDetector()


def test_detects_image_captured(detector: MediaTypeDetector, tmp_image_file: Path) -> None:
    info = detector.detect(tmp_image_file)
    assert info.media_type in (MediaType.IMAGE_CAPTURED, MediaType.IMAGE_PREVIEW)
    assert info.width == 256
    assert info.height == 256


def test_detects_video(detector: MediaTypeDetector, tmp_video_file: Path) -> None:
    info = detector.detect(tmp_video_file)
    assert info.media_type in (MediaType.VIDEO_STATIC, MediaType.VIDEO_MOTION)
    assert info.fps and info.fps > 0
    assert info.frame_count and info.frame_count > 0


def test_detects_preview_screenshot(detector: MediaTypeDetector, preview_bgr: np.ndarray, tmp_path: Path) -> None:
    p = tmp_path / "preview.jpg"
    cv2.imwrite(str(p), preview_bgr)
    info = detector.detect(p)
    assert info.media_type == MediaType.IMAGE_PREVIEW


def test_file_not_found(detector: MediaTypeDetector) -> None:
    with pytest.raises(FileNotFoundError):
        detector.detect(Path("/nonexistent/file.jpg"))


def test_static_video_classification(detector: MediaTypeDetector, tmp_video_file: Path) -> None:
    """Static video with no motion should be VIDEO_STATIC."""
    info = detector.detect(tmp_video_file)
    assert info.media_type == MediaType.VIDEO_STATIC
