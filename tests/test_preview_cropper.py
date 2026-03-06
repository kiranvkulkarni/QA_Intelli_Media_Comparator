from __future__ import annotations

import numpy as np
import pytest
import cv2

from qa_intelli_media_comparator.services.preview_cropper import PreviewCropper


@pytest.fixture
def cropper() -> PreviewCropper:
    return PreviewCropper()


def test_crop_preview_returns_smaller_image(
    cropper: PreviewCropper, preview_bgr: np.ndarray
) -> None:
    cropped, result = cropper.crop_image(preview_bgr)
    h_orig, w_orig = preview_bgr.shape[:2]
    h_crop, w_crop = cropped.shape[:2]
    # Cropped image should be smaller than original
    assert h_crop < h_orig or w_crop < w_orig
    assert result.applied


def test_crop_returns_full_image_for_captured(
    cropper: PreviewCropper, sharp_bgr: np.ndarray
) -> None:
    """For a non-preview image, the cropper may still run but should not crash."""
    cropped, result = cropper.crop_image(sharp_bgr)
    assert cropped.shape[2] == 3  # still BGR


def test_crop_bbox_within_bounds(
    cropper: PreviewCropper, preview_bgr: np.ndarray
) -> None:
    h, w = preview_bgr.shape[:2]
    _, result = cropper.crop_image(preview_bgr)
    x, y, bw, bh = result.bbox
    assert x >= 0 and y >= 0
    assert x + bw <= w
    assert y + bh <= h


def test_apply_bbox_to_frame(cropper: PreviewCropper, sharp_bgr: np.ndarray) -> None:
    bbox = (10, 10, 100, 100)
    cropped = cropper.apply_bbox_to_frame(sharp_bgr, bbox)
    assert cropped.shape[:2] == (100, 100)
