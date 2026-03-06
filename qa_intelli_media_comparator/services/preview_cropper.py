from __future__ import annotations

"""
PreviewCropper — extracts the camera viewfinder frame from ADB screen recordings
or full-screen screenshots by removing the camera app UI chrome.

Detection strategy (applied in order, first success wins):
  1. Contour-based: Canny edges → largest rect contour with valid aspect ratio
  2. Saturation-mask: camera preview has rich color while UI chrome is dark/gray
  3. Heuristic fallback: crop fixed percentage margins based on common camera app layouts

For video files: crop bbox is determined from the first stable frame and applied
uniformly to all frames.
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..models.media import CropResult

log = logging.getLogger(__name__)

# Accepted camera preview aspect ratios (w/h). We'll pick the candidate closest to one of these.
_PREVIEW_ASPECT_RATIOS = [4 / 3, 16 / 9, 1.0, 3 / 4, 9 / 16]
_ASPECT_TOLERANCE = 0.08

# Minimum fraction of total frame area that the preview region must occupy
_MIN_AREA_FRACTION = 0.25


class PreviewCropper:
    """Detects and crops the viewfinder/preview area from a phone UI screenshot or recording."""

    # ── Public ─────────────────────────────────────────────────────────────────

    def crop_image(self, img: np.ndarray) -> tuple[np.ndarray, CropResult]:
        """Crop a single BGR image to its camera preview region."""
        h, w = img.shape[:2]

        # Try each strategy in order
        bbox, method, confidence = self._try_contour(img)
        if bbox is None:
            bbox, method, confidence = self._try_saturation_mask(img)
        if bbox is None:
            bbox, method, confidence = self._heuristic_fallback(img)

        if bbox is None:
            log.warning("PreviewCropper: could not detect viewfinder; returning full frame")
            return img, CropResult(bbox=(0, 0, w, h), confidence=0.0, method="none", applied=False)

        x, y, bw, bh = bbox
        # Safety clamp
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        bw = min(bw, w - x)
        bh = min(bh, h - y)

        cropped = img[y: y + bh, x: x + bw]
        log.debug("PreviewCropper: method=%s bbox=%s confidence=%.2f", method, bbox, confidence)
        return cropped, CropResult(bbox=(x, y, bw, bh), confidence=confidence, method=method)

    def crop_video_frame(
        self, cap: cv2.VideoCapture
    ) -> tuple[Optional[tuple[int, int, int, int]], CropResult]:
        """
        Read the first stable frame from an open VideoCapture and determine crop bbox.
        Returns (bbox, CropResult) — does not advance the capture beyond frame detection.
        """
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        # Skip first 0.5s (camera app startup animation)
        skip_frames = min(int(fps * 0.5), max(0, total - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, skip_frames)

        frame = None
        for _ in range(5):  # try a few frames in case of black frames
            ret, f = cap.read()
            if ret and f is not None and np.mean(f) > 5:
                frame = f
                break

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # rewind

        if frame is None:
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            return None, CropResult(bbox=(0, 0, w, h), confidence=0.0, method="none", applied=False)

        _, crop_result = self.crop_image(frame)
        return crop_result.bbox if crop_result.applied else None, crop_result

    def apply_bbox_to_frame(
        self, frame: np.ndarray, bbox: tuple[int, int, int, int]
    ) -> np.ndarray:
        x, y, bw, bh = bbox
        h, w = frame.shape[:2]
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        bw = min(bw, w - x)
        bh = min(bh, h - y)
        return frame[y: y + bh, x: x + bw]

    # ── Strategy 1: Contour-based ──────────────────────────────────────────────

    def _try_contour(
        self, img: np.ndarray
    ) -> tuple[Optional[tuple[int, int, int, int]], str, float]:
        h, w = img.shape[:2]
        total_area = h * w

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Dilate to close small gaps in the viewfinder border
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, "contour", 0.0

        best: Optional[tuple[int, int, int, int]] = None
        best_score = 0.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < total_area * _MIN_AREA_FRACTION:
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            # Prefer quadrilaterals
            if len(approx) not in (4, 5, 6):
                rect = cv2.boundingRect(cnt)
            else:
                rect = cv2.boundingRect(approx)

            rx, ry, rw, rh = rect
            if rw == 0 or rh == 0:
                continue
            ar = rw / rh
            ar_score = self._aspect_ratio_score(ar)
            if ar_score == 0:
                continue

            coverage = (rw * rh) / total_area
            score = ar_score * coverage
            if score > best_score:
                best_score = score
                best = rect

        if best is None:
            return None, "contour", 0.0

        return best, "contour", min(1.0, best_score * 2)

    # ── Strategy 2: Saturation mask ───────────────────────────────────────────

    def _try_saturation_mask(
        self, img: np.ndarray
    ) -> tuple[Optional[tuple[int, int, int, int]], str, float]:
        """Camera preview content is colorful; UI chrome tends to be dark/gray."""
        h, w = img.shape[:2]
        total_area = h * w

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]

        # Threshold: pixels with saturation > 30 are likely part of the live preview
        _, mask = cv2.threshold(sat, 30, 255, cv2.THRESH_BINARY)

        # Morphological closing to fill gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, "saturation_mask", 0.0

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < total_area * _MIN_AREA_FRACTION:
            return None, "saturation_mask", 0.0

        rect = cv2.boundingRect(largest)
        rx, ry, rw, rh = rect
        ar = rw / rh
        ar_score = self._aspect_ratio_score(ar)
        if ar_score == 0:
            return None, "saturation_mask", 0.0

        coverage = (rw * rh) / total_area
        confidence = min(1.0, ar_score * coverage * 1.5)
        return rect, "saturation_mask", confidence

    # ── Strategy 3: Heuristic fallback ────────────────────────────────────────

    def _heuristic_fallback(
        self, img: np.ndarray
    ) -> tuple[Optional[tuple[int, int, int, int]], str, float]:
        """
        Crop fixed margins for common camera app layouts:
        - Top: 8% (status bar + top controls)
        - Bottom: 22% (shutter button + bottom controls)
        - Left/Right: 0% (full width usually)
        """
        h, w = img.shape[:2]
        top = int(h * 0.08)
        bottom = int(h * 0.22)
        bbox = (0, top, w, h - top - bottom)
        log.debug("PreviewCropper: using heuristic fallback crop")
        return bbox, "heuristic", 0.5

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _aspect_ratio_score(ar: float) -> float:
        """Return 0 if aspect ratio is unlikely for a camera preview, else 0–1."""
        best_diff = min(abs(ar - target) for target in _PREVIEW_ASPECT_RATIOS)
        if best_diff > _ASPECT_TOLERANCE:
            return 0.0
        return 1.0 - (best_diff / _ASPECT_TOLERANCE)
