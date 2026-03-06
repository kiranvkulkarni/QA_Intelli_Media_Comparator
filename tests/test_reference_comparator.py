from __future__ import annotations

import numpy as np
import pytest
import cv2

from qa_intelli_media_comparator.services.reference_comparator import ReferenceComparator


@pytest.fixture
def comparator() -> ReferenceComparator:
    return ReferenceComparator()


def test_identical_images_pass(
    comparator: ReferenceComparator, sharp_bgr: np.ndarray
) -> None:
    fr, diff = comparator.compare(sharp_bgr, sharp_bgr)
    if fr.ssim.value is not None:
        assert fr.ssim.value > 0.99
    if fr.psnr.value is not None:
        assert fr.psnr.value > 50


def test_different_images_lower_ssim(
    comparator: ReferenceComparator,
    sharp_bgr: np.ndarray,
    blurry_bgr: np.ndarray,
) -> None:
    fr_same, _ = comparator.compare(sharp_bgr, sharp_bgr)
    fr_diff, _ = comparator.compare(sharp_bgr, blurry_bgr)

    if fr_same.ssim.value is not None and fr_diff.ssim.value is not None:
        assert fr_diff.ssim.value < fr_same.ssim.value


def test_diff_heatmap_returned(
    comparator: ReferenceComparator,
    sharp_bgr: np.ndarray,
    blurry_bgr: np.ndarray,
) -> None:
    _, diff = comparator.compare(sharp_bgr, blurry_bgr)
    assert diff is not None
    assert diff.shape[2] == 3  # BGR heatmap


def test_same_image_diff_is_mostly_blue(
    comparator: ReferenceComparator, sharp_bgr: np.ndarray
) -> None:
    """JET colormap: zero-diff → blue."""
    _, diff = comparator.compare(sharp_bgr, sharp_bgr)
    if diff is not None:
        mean_blue = float(np.mean(diff[:, :, 0]))
        mean_red = float(np.mean(diff[:, :, 2]))
        assert mean_blue > mean_red


def test_mismatched_sizes_handled(
    comparator: ReferenceComparator, sharp_bgr: np.ndarray
) -> None:
    """Comparator should resize ref to DUT size without error."""
    large_ref = cv2.resize(sharp_bgr, (512, 512))
    fr, diff = comparator.compare(large_ref, sharp_bgr)
    assert diff is not None
