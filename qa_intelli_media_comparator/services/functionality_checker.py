from __future__ import annotations

"""FunctionalityChecker — answers "is the camera working?" for automation.

## Two-mode design

The service supports two analysis depths controlled by QIMC_ANALYSIS_MODE
(or the per-request 'analysis_mode' form parameter):

  functional  — This checker runs ALONE (no neural IQA, no LPIPS/DISTS).
                Returns functional_grade + functional_reasons within ~50ms.
                Use case: camera functional test automation (is the camera on?
                is it capturing the right scene? is the preview alive?).

  quality     — This checker runs IN ADDITION to the full IQA pipeline.
                Returns both functional_grade (from here) and overall_grade
                (from IQA).  Use case: image quality benchmarking.

The checker is intentionally classical-only (OpenCV) so it never fails due
to missing GPU or model weights.

## Checks performed (priority order)

1. Black frame        — mean luma < 8 → camera cover / HW crash
2. White frame        — mean luma > 248 + low std → AE runaway
3. Uniform frame      — luma std < 3 → sensor stuck / lens cap
4. No content         — Canny edge density < 1% → blank surface / OOF
5. Absolute blur fail — Laplacian variance < 1.0 → AF completely stuck
6. Scene mismatch     — colour histogram correlation < 0.30 vs REF →
                        wrong scene captured (mis-trigger in automation)

Checks 4 & 5 produce WARNING (advisory); checks 1–3 and 6 produce FAIL.

## Video sequence checks

check_video_sequence() analyses a list of frames and returns:
  black_frame_count  — frames with near-zero luminance
  frozen_frame_count — consecutive near-identical frames (camera frozen)
"""

import logging
from typing import Optional

import cv2
import numpy as np

from ..models.enums import QualityGrade

log = logging.getLogger(__name__)

# ── Per-frame thresholds ──────────────────────────────────────────────────────
_BLACK_LUMA           = 8.0    # mean luminance below this → black frame
_BLACK_LUMA_NIGHT     = 4.0    # relaxed for night mode (intentional dark areas)
_WHITE_LUMA           = 248.0  # mean above this → overexposed
_WHITE_STD            = 15.0   # std below this → uniformly white (not just bright)
_UNIFORM_STD          = 3.0    # luminance std below this → stuck/covered sensor
_EDGE_DENSITY_WARN    = 0.010  # Canny edge ratio < 1% → almost featureless (WARNING)
_MIN_BLUR_SCORE       = 1.0    # Laplacian var < 1 → AF completely failed (WARNING)

# ── Scene match thresholds (colour histogram correlation in [-1, 1]) ──────────
_SCENE_CORR_FAIL      = 0.30   # < 0.30 → completely different scene (FAIL)
_SCENE_CORR_WARN      = 0.60   # < 0.60 → scene looks quite different (WARNING)

# ── Video freeze detection ────────────────────────────────────────────────────
_FREEZE_DIFF_RATIO    = 0.003  # mean abs diff / 255 below this → "frozen" frame pair
_FREEZE_MIN_RUN       = 3      # consecutive frozen pairs before counted as event


class FunctionalityChecker:
    """Per-frame and per-sequence camera functional validity checker."""

    # ── Single-frame check ────────────────────────────────────────────────────

    def check(
        self,
        img: np.ndarray,
        ref_img: Optional[np.ndarray] = None,
        camera_mode: str = "unknown",
    ) -> tuple[QualityGrade, list[str]]:
        """Return (functional_grade, functional_reasons) for one image frame.

        Parameters
        ----------
        img
            DUT frame as BGR uint8 ndarray.
        ref_img
            Golden reference as BGR uint8 ndarray (optional).  When provided,
            a coarse histogram-based scene-match check is added.
        camera_mode
            Detected camera mode string.  Night mode relaxes the black-frame
            threshold because intentionally dark areas are expected.
        """
        if img is None or img.size == 0:
            return QualityGrade.FAIL, [
                "Empty or null frame received — pipeline or I/O error."
            ]

        reasons: list[str] = []
        grade = QualityGrade.PASS

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        mean_lum = float(gray.mean())
        std_lum  = float(gray.std())

        # ── 1. Black frame ─────────────────────────────────────────────────
        black_thresh = _BLACK_LUMA_NIGHT if camera_mode == "night" else _BLACK_LUMA
        if mean_lum < black_thresh:
            reasons.append(
                f"Black frame detected (mean luminance {mean_lum:.1f} < {black_thresh:.0f}). "
                "Camera lens may be covered, hardware has crashed, or the "
                "camera app is non-responsive."
            )
            return QualityGrade.FAIL, reasons  # no further checks useful

        # ── 2. Completely white / blown frame ──────────────────────────────
        if mean_lum > _WHITE_LUMA and std_lum < _WHITE_STD:
            reasons.append(
                f"Sensor saturation detected (mean luma {mean_lum:.1f} > {_WHITE_LUMA:.0f}, "
                f"std {std_lum:.1f} < {_WHITE_STD:.0f}). "
                "Camera AE is running away or a flash misfired."
            )
            return QualityGrade.FAIL, reasons

        # ── 3. Uniform / featureless frame (stuck sensor / lens cap) ──────
        if std_lum < _UNIFORM_STD:
            reasons.append(
                f"Uniform frame detected (luminance std {std_lum:.2f} < {_UNIFORM_STD:.1f}). "
                "Camera lens may be capped, or the sensor/preview pipeline is stuck."
            )
            return QualityGrade.FAIL, reasons

        # ── 4. Edge density — is there any content at all? ─────────────────
        gray_u8 = gray.astype(np.uint8)
        edges = cv2.Canny(gray_u8, 50, 150)
        edge_density = float((edges > 0).sum()) / float(edges.size)
        if edge_density < _EDGE_DENSITY_WARN:
            reasons.append(
                f"Very low image content (edge density {edge_density:.4f} < "
                f"{_EDGE_DENSITY_WARN:.3f}). Camera may be pointed at a blank "
                "surface or autofocus is severely broken."
            )
            if grade == QualityGrade.PASS:
                grade = QualityGrade.WARNING

        # ── 5. Absolute blur floor (AF completely stuck) ───────────────────
        blur_score = float(cv2.Laplacian(gray_u8, cv2.CV_64F).var())
        if blur_score < _MIN_BLUR_SCORE:
            reasons.append(
                f"Autofocus completely failed (blur score {blur_score:.2f} < "
                f"{_MIN_BLUR_SCORE:.1f}). Lens defect or AF algorithm crash."
            )
            if grade == QualityGrade.PASS:
                grade = QualityGrade.WARNING

        # ── 6. Scene match vs reference ────────────────────────────────────
        if ref_img is not None and ref_img.size > 0:
            corr = self._histogram_correlation(img, ref_img)
            if corr < _SCENE_CORR_FAIL:
                reasons.append(
                    f"Scene mismatch vs reference (histogram correlation {corr:.2f} "
                    f"< {_SCENE_CORR_FAIL:.2f}). DUT is capturing a completely "
                    "different scene — possible mis-trigger or test sequence error."
                )
                grade = QualityGrade.FAIL
            elif corr < _SCENE_CORR_WARN:
                reasons.append(
                    f"Scene differs from reference (histogram correlation {corr:.2f} "
                    f"< {_SCENE_CORR_WARN:.2f}). Check camera orientation, "
                    "lighting changes, or rig positioning."
                )
                if grade == QualityGrade.PASS:
                    grade = QualityGrade.WARNING

        return grade, reasons

    # ── Video sequence checks ──────────────────────────────────────────────────

    def check_video_sequence(
        self,
        frames: list[np.ndarray],
    ) -> tuple[int, int]:
        """Return ``(black_frame_count, frozen_frame_count)`` for a frame list.

        black_frame_count  — number of frames with mean luminance < threshold.
        frozen_frame_count — number of frames that are part of a freeze run
                             (≥ _FREEZE_MIN_RUN consecutive near-identical frames).
        """
        if not frames:
            return 0, 0

        # Black frames
        black_count = sum(
            1 for f in frames
            if cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32).mean() < _BLACK_LUMA
        )

        # Frozen frame detection
        frozen_count = 0
        consecutive = 0
        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY).astype(np.float32)

        for frame in frames[1:]:
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            mean_diff = float(np.abs(curr_gray - prev_gray).mean()) / 255.0
            if mean_diff < _FREEZE_DIFF_RATIO:
                consecutive += 1
            else:
                if consecutive >= _FREEZE_MIN_RUN:
                    frozen_count += consecutive
                consecutive = 0
            prev_gray = curr_gray

        if consecutive >= _FREEZE_MIN_RUN:
            frozen_count += consecutive

        return black_count, frozen_count

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _histogram_correlation(img_a: np.ndarray, img_b: np.ndarray) -> float:
        """Colour histogram correlation in [-1, 1] between two BGR images.

        Uses a 3-channel 32-bin histogram (32 768 bins) normalised to unit
        area.  Spatially invariant — robust to minor positional shifts and
        moderate exposure differences between DUT and reference.
        """
        bins = [32, 32, 32]
        ranges = [0, 256, 0, 256, 0, 256]
        h_a = cv2.calcHist([img_a], [0, 1, 2], None, bins, ranges)
        h_b = cv2.calcHist([img_b], [0, 1, 2], None, bins, ranges)
        cv2.normalize(h_a, h_a)
        cv2.normalize(h_b, h_b)
        return float(cv2.compareHist(h_a, h_b, cv2.HISTCMP_CORREL))
