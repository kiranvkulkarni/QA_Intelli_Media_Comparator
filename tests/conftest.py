from __future__ import annotations

"""
Shared pytest fixtures and synthetic test image/video generators.

Fixtures:
  sharp_bgr      : 256×256 sharp synthetic image
  blurry_bgr     : 256×256 blurred image
  noisy_bgr      : 256×256 noisy image
  gradient_bgr   : 256×256 smooth gradient (good for banding tests)
  overexposed_bgr: 256×256 mostly-white image
  preview_bgr    : 512×896 synthetic ADB screenshot with fake status bar + shutter button
"""

import numpy as np
import pytest
import cv2


def _make_sharp_image(size: int = 256) -> np.ndarray:
    # Mid-gray background avoids triggering underexposure/overexposure artifact detectors
    img = np.full((size, size, 3), 128, dtype=np.uint8)
    # Draw various shapes for feature-rich, edge-rich content
    cv2.rectangle(img, (20, 20), (100, 100), (60, 160, 220), -1)
    cv2.circle(img, (180, 80), 50, (80, 180, 80), -1)
    cv2.line(img, (0, size // 2), (size, size // 2), (200, 200, 200), 2)
    cv2.rectangle(img, (10, size - 60), (size - 10, size - 20), (100, 80, 160), -1)
    return img


@pytest.fixture
def sharp_bgr() -> np.ndarray:
    return _make_sharp_image(256)


@pytest.fixture
def blurry_bgr(sharp_bgr: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(sharp_bgr, (31, 31), 10)


@pytest.fixture
def noisy_bgr(sharp_bgr: np.ndarray) -> np.ndarray:
    noise = np.random.normal(0, 30, sharp_bgr.shape).astype(np.int16)
    noisy = np.clip(sharp_bgr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return noisy


@pytest.fixture
def gradient_bgr() -> np.ndarray:
    """Smooth horizontal gradient — useful for banding and exposure tests."""
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    for x in range(256):
        img[:, x, :] = x
    return img


@pytest.fixture
def overexposed_bgr() -> np.ndarray:
    img = np.full((256, 256, 3), 252, dtype=np.uint8)
    # Small dark region in corner to avoid complete whiteness
    img[:20, :20] = 100
    return img


@pytest.fixture
def preview_bgr() -> np.ndarray:
    """
    Synthetic ADB screenshot: 1080×2340 (20:9 aspect), dark status bar at top,
    full-color preview in the middle, dark camera controls at the bottom.
    """
    w, h = 1080, 2340
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # Status bar (top 80px): dark gray
    img[:80, :] = (40, 40, 40)
    # Draw fake clock/icon dots
    for i in range(5):
        cv2.circle(img, (w - 60 + i * 12 - 24, 40), 4, (255, 255, 255), -1)

    # Camera preview area (80 → 80+1440): colorful image content
    preview_h = 1440
    preview = _make_sharp_image(256)
    preview_resized = cv2.resize(preview, (w, preview_h))
    img[80: 80 + preview_h, :] = preview_resized

    # Bottom controls (1520 → 2340): dark gray
    img[1520:, :] = (30, 30, 30)
    # Draw shutter button circle
    cv2.circle(img, (w // 2, 1900), 80, (255, 255, 255), -1)
    cv2.circle(img, (w // 2, 1900), 70, (200, 200, 200), 8)

    return img


@pytest.fixture
def tmp_image_file(sharp_bgr: np.ndarray, tmp_path) -> "Path":
    import cv2
    from pathlib import Path
    p = tmp_path / "test_image.jpg"
    cv2.imwrite(str(p), sharp_bgr)
    return p


@pytest.fixture
def tmp_video_file(tmp_path) -> "Path":
    """Create a 30-frame synthetic MP4 video."""
    from pathlib import Path
    p = tmp_path / "test_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(p), fourcc, 30.0, (256, 256))
    base = _make_sharp_image(256)
    for i in range(30):
        frame = base.copy()
        cv2.putText(frame, str(i), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        writer.write(frame)
    writer.release()
    return p
