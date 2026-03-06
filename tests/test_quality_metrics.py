from __future__ import annotations

import numpy as np
import pytest

from qa_intelli_media_comparator.services.quality_metrics import QualityMetricsExtractor
from qa_intelli_media_comparator.models.metrics import QualityMetrics


@pytest.fixture
def extractor() -> QualityMetricsExtractor:
    return QualityMetricsExtractor()


def test_sharp_image_has_high_sharpness(
    extractor: QualityMetricsExtractor, sharp_bgr: np.ndarray
) -> None:
    metrics = extractor.extract(sharp_bgr)
    assert metrics.blur_score is not None
    assert metrics.blur_score > 50  # sharp image should have non-trivial Laplacian var


def test_blurry_image_has_lower_sharpness(
    extractor: QualityMetricsExtractor,
    sharp_bgr: np.ndarray,
    blurry_bgr: np.ndarray,
) -> None:
    sharp_m = extractor.extract(sharp_bgr)
    blurry_m = extractor.extract(blurry_bgr)
    assert blurry_m.blur_score < sharp_m.blur_score


def test_noisy_image_has_higher_noise_sigma(
    extractor: QualityMetricsExtractor,
    sharp_bgr: np.ndarray,
    noisy_bgr: np.ndarray,
) -> None:
    sharp_m = extractor.extract(sharp_bgr)
    noisy_m = extractor.extract(noisy_bgr)
    assert noisy_m.noise_sigma > sharp_m.noise_sigma


def test_overexposed_has_highlight_clipping(
    extractor: QualityMetricsExtractor, overexposed_bgr: np.ndarray
) -> None:
    metrics = extractor.extract(overexposed_bgr)
    assert metrics.highlight_clipping_pct is not None
    assert metrics.highlight_clipping_pct > 1.0


def test_all_fields_populated(
    extractor: QualityMetricsExtractor, sharp_bgr: np.ndarray
) -> None:
    metrics = extractor.extract(sharp_bgr)
    assert metrics.blur_score is not None
    assert metrics.noise_sigma is not None
    assert metrics.exposure_mean is not None
    assert metrics.saturation_mean is not None
    assert metrics.dynamic_range_stops is not None


def test_gradient_image_dynamic_range(
    extractor: QualityMetricsExtractor, gradient_bgr: np.ndarray
) -> None:
    metrics = extractor.extract(gradient_bgr)
    # Gradient spans 0–255 → should have meaningful dynamic range
    assert metrics.dynamic_range_stops is not None
    assert metrics.dynamic_range_stops > 4.0
