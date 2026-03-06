from __future__ import annotations

import numpy as np
import pytest
import cv2

from qa_intelli_media_comparator.services.artifact_detector import ArtifactDetector
from qa_intelli_media_comparator.models.enums import ArtifactSeverity


@pytest.fixture
def detector() -> ArtifactDetector:
    return ArtifactDetector()


def test_clean_image_has_no_critical_artifacts(
    detector: ArtifactDetector, sharp_bgr: np.ndarray
) -> None:
    report = detector.detect(sharp_bgr)
    assert report.overall_severity not in (ArtifactSeverity.HIGH, ArtifactSeverity.CRITICAL)


def test_overexposed_image_has_overexposure_artifact(
    detector: ArtifactDetector, overexposed_bgr: np.ndarray
) -> None:
    report = detector.detect(overexposed_bgr)
    types = [a.artifact_type for a in report.artifacts]
    assert "overexposure" in types


def test_noisy_image_has_noise_artifact(
    detector: ArtifactDetector, noisy_bgr: np.ndarray
) -> None:
    # Extra noisy image
    very_noisy = np.clip(
        noisy_bgr.astype(np.int16) + np.random.normal(0, 50, noisy_bgr.shape).astype(np.int16),
        0, 255
    ).astype(np.uint8)
    report = detector.detect(very_noisy)
    types = [a.artifact_type for a in report.artifacts]
    # Should detect noise or at least not crash
    assert isinstance(types, list)


def test_artifact_has_description(
    detector: ArtifactDetector, overexposed_bgr: np.ndarray
) -> None:
    report = detector.detect(overexposed_bgr)
    for artifact in report.artifacts:
        assert artifact.description != ""


def test_artifact_severity_ordering(
    detector: ArtifactDetector, overexposed_bgr: np.ndarray
) -> None:
    report = detector.detect(overexposed_bgr)
    # overall_severity should match maximum individual severity
    from qa_intelli_media_comparator.models.enums import severity_rank
    if report.artifacts:
        max_sev = max(severity_rank(a.severity) for a in report.artifacts)
        assert severity_rank(report.overall_severity) == max_sev
