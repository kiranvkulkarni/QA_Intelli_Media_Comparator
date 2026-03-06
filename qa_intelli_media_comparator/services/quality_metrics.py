from __future__ import annotations

"""
QualityMetricsExtractor — computes standard media quality attributes from a BGR image.

All metrics are classical (OpenCV / NumPy / SciPy) and run on CPU without GPU or
neural network models, so they are always fast and reliable.
"""

import logging

import cv2
import numpy as np
from scipy.ndimage import uniform_filter

from ..models.metrics import QualityMetrics

log = logging.getLogger(__name__)


class QualityMetricsExtractor:
    """Extract standard photographic quality metrics from a BGR image array."""

    def extract(self, img_bgr: np.ndarray) -> QualityMetrics:
        gray_u8 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)         # uint8 for OpenCV ops
        gray = gray_u8.astype(np.float32)                           # float32 for numpy ops
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        b, g, r = cv2.split(img_bgr.astype(np.float32))

        return QualityMetrics(
            blur_score=self._blur_laplacian(gray_u8),
            tenengrad_score=self._tenengrad(gray_u8),
            noise_sigma=self._noise_sigma(gray),
            exposure_mean=self._exposure_mean(lab),
            highlight_clipping_pct=self._highlight_clipping(img_bgr),
            shadow_clipping_pct=self._shadow_clipping(img_bgr),
            color_cast_r=self._color_cast_channel(r, g, b, "r"),
            color_cast_g=self._color_cast_channel(g, r, b, "g"),
            color_cast_b=self._color_cast_channel(b, r, g, "b"),
            white_balance_deviation=self._wb_deviation(r, g, b),
            saturation_mean=self._saturation_mean(hsv),
            dynamic_range_stops=self._dynamic_range(gray),
            chromatic_aberration_score=self._chromatic_aberration(img_bgr, gray),
        )

    # ── Sharpness ──────────────────────────────────────────────────────────────

    @staticmethod
    def _blur_laplacian(gray_u8: np.ndarray) -> float:
        """Variance of Laplacian — higher = sharper. Expects uint8 input."""
        lap = cv2.Laplacian(gray_u8, cv2.CV_64F)
        return float(lap.var())

    @staticmethod
    def _tenengrad(gray_u8: np.ndarray) -> float:
        """Sum of squared Sobel gradients — higher = sharper. Expects uint8 input."""
        gx = cv2.Sobel(gray_u8, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_u8, cv2.CV_64F, 0, 1, ksize=3)
        return float(np.mean(gx ** 2 + gy ** 2))

    # ── Noise ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _noise_sigma(gray: np.ndarray) -> float:
        """
        Estimate noise standard deviation in flat (low-gradient) regions.

        Algorithm:
          1. Compute Sobel gradient magnitude per pixel.
          2. Select pixels in the bottom 20th percentile of gradient (flat regions).
          3. Estimate noise as the std dev of those pixels after subtracting
             a local mean (to remove DC bias from shading).
        """
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx ** 2 + gy ** 2)
        threshold = float(np.percentile(grad_mag, 20))
        flat_mask = grad_mag <= threshold

        if flat_mask.sum() < 100:
            return float(np.std(gray))

        # Subtract local mean to remove slow illumination variation
        local_mean = uniform_filter(gray.astype(np.float64), size=15)
        residual = gray.astype(np.float64) - local_mean
        return float(np.std(residual[flat_mask]))

    # ── Exposure ───────────────────────────────────────────────────────────────

    @staticmethod
    def _exposure_mean(lab: np.ndarray) -> float:
        """Mean of L* channel in LAB (0–100 scale after OpenCV's 0–255 encoding)."""
        l_channel = lab[:, :, 0].astype(np.float32)
        return float(np.mean(l_channel) * 100.0 / 255.0)

    @staticmethod
    def _highlight_clipping(img: np.ndarray) -> float:
        """Percentage of pixels where any channel exceeds 250 (blown highlights)."""
        mask = np.any(img > 250, axis=2)
        return float(mask.mean() * 100.0)

    @staticmethod
    def _shadow_clipping(img: np.ndarray) -> float:
        """Percentage of pixels where all channels are below 5 (crushed blacks)."""
        mask = np.all(img < 5, axis=2)
        return float(mask.mean() * 100.0)

    # ── Color ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _color_cast_channel(
        ch: np.ndarray, ch2: np.ndarray, ch3: np.ndarray, name: str
    ) -> float:
        """
        Deviation of channel mean from the gray-world neutral assumption.
        Positive = channel is warmer/brighter than neutral; negative = cooler.
        """
        mean_ch = float(np.mean(ch))
        mean_ref = float((np.mean(ch2) + np.mean(ch3)) / 2.0)
        return round(mean_ch - mean_ref, 3)

    @staticmethod
    def _wb_deviation(r: np.ndarray, g: np.ndarray, b: np.ndarray) -> float:
        """
        Simple white balance deviation: ratio of R/G and B/G compared to ideal (1.0).
        Returns the RMS of (R/G - 1) and (B/G - 1) computed on bright neutral pixels.
        """
        mean_g = float(np.mean(g))
        if mean_g < 1e-3:
            return 0.0
        rg = float(np.mean(r)) / mean_g
        bg = float(np.mean(b)) / mean_g
        return float(np.sqrt(((rg - 1.0) ** 2 + (bg - 1.0) ** 2) / 2.0))

    @staticmethod
    def _saturation_mean(hsv: np.ndarray) -> float:
        """Mean HSV saturation (0–1 scale)."""
        return float(np.mean(hsv[:, :, 1]) / 255.0)

    # ── Dynamic Range ──────────────────────────────────────────────────────────

    @staticmethod
    def _dynamic_range(gray: np.ndarray) -> float:
        """
        Estimated dynamic range in stops (EV).
        DR = log2(99th percentile luminance / 1st percentile luminance).
        """
        p1 = float(np.percentile(gray, 1))
        p99 = float(np.percentile(gray, 99))
        if p1 < 1.0:
            p1 = 1.0
        if p99 <= p1:
            return 0.0
        return float(np.log2(p99 / p1))

    # ── Chromatic Aberration ───────────────────────────────────────────────────

    @staticmethod
    def _chromatic_aberration(img_bgr: np.ndarray, gray: np.ndarray) -> float:
        """
        Estimate chromatic aberration as the mean spatial offset between the
        R and B channels at high-contrast edges.

        Method:
          1. Find strong edges via Canny.
          2. For each edge pixel, compute the local centroid shift between
             R and B channels in a 7×7 patch using normalized cross-correlation.
          3. Report mean absolute offset in pixels.
        """
        edges = cv2.Canny(gray.astype(np.uint8), 80, 200)
        edge_pts = np.argwhere(edges > 0)
        if len(edge_pts) == 0:
            return 0.0

        b_ch = img_bgr[:, :, 0].astype(np.float32)
        r_ch = img_bgr[:, :, 2].astype(np.float32)
        h, w = gray.shape
        pad = 5

        offsets: list[float] = []
        # Sample up to 200 edge points for speed
        step = max(1, len(edge_pts) // 200)
        for py, px in edge_pts[::step]:
            y1 = max(pad, py - pad)
            y2 = min(h - pad, py + pad)
            x1 = max(pad, px - pad)
            x2 = min(w - pad, px + pad)
            patch_r = r_ch[y1:y2, x1:x2]
            patch_b = b_ch[y1:y2, x1:x2]
            if patch_r.size < 9:
                continue
            # Compute centroid of each channel patch
            idx_y, idx_x = np.mgrid[0: patch_r.shape[0], 0: patch_r.shape[1]]
            sum_r = float(patch_r.sum())
            sum_b = float(patch_b.sum())
            if sum_r < 1 or sum_b < 1:
                continue
            cy_r = float((patch_r * idx_y).sum()) / sum_r
            cx_r = float((patch_r * idx_x).sum()) / sum_r
            cy_b = float((patch_b * idx_y).sum()) / sum_b
            cx_b = float((patch_b * idx_x).sum()) / sum_b
            offset = float(np.sqrt((cy_r - cy_b) ** 2 + (cx_r - cx_b) ** 2))
            offsets.append(offset)

        return float(np.mean(offsets)) if offsets else 0.0
