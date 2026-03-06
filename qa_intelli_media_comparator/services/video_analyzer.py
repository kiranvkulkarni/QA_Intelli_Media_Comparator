from __future__ import annotations

"""
VideoAnalyzer — temporal quality analysis for both static and motion videos.

Capabilities:
  - Frame extraction at configurable FPS
  - Auto-sync (SSIM-based scene matching) or frame-by-frame sync
  - Flicker detection via luminance FFT
  - Jitter detection via optical flow centroid tracking
  - Temporal SSIM consistency
  - Per-frame quality_metrics and artifact aggregation
  - Worst-frame identification for annotation
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim_fn

from ..models.enums import SyncMode, QualityGrade
from ..models.metrics import QualityMetrics
from ..models.artifacts import ArtifactInstance, ArtifactReport
from ..models.video import VideoTemporalMetrics, VideoAnalysisResult
from ..config import get_settings
from .quality_metrics import QualityMetricsExtractor
from .artifact_detector import ArtifactDetector

log = logging.getLogger(__name__)

# Thresholds for temporal metrics
_FLICKER_FAIL = 0.15       # normalized FFT high-freq energy
_FLICKER_WARN = 0.07
_JITTER_FAIL = 5.0         # px
_JITTER_WARN = 2.5


class VideoAnalyzer:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._qm_extractor = QualityMetricsExtractor()
        self._artifact_detector = ArtifactDetector()

    def analyze(
        self,
        dut_path: Path,
        ref_path: Optional[Path] = None,
        sync_mode: SyncMode = SyncMode.AUTO,
        bbox: Optional[tuple[int, int, int, int]] = None,
        ref_bbox: Optional[tuple[int, int, int, int]] = None,
    ) -> VideoAnalysisResult:
        dut_frames = self._extract_frames(dut_path, bbox)
        ref_frames: list[np.ndarray] = []
        if ref_path is not None:
            ref_frames = self._extract_frames(ref_path, ref_bbox)

        # Sync
        offset = 0
        if ref_frames and sync_mode == SyncMode.AUTO:
            offset = self._find_sync_offset(dut_frames, ref_frames)
            log.debug("Video auto-sync offset: %d frames", offset)

        # Temporal metrics
        temporal = self._compute_temporal(dut_frames, ref_frames, offset, sync_mode)

        # Per-frame quality (sample up to 10 frames)
        sample_step = max(1, len(dut_frames) // 10)
        sampled = dut_frames[::sample_step]

        all_qm = [self._qm_extractor.extract(f) for f in sampled]
        agg_qm = self._aggregate_quality_metrics(all_qm)

        # Artifacts: check worst frame
        worst_idx, worst_ssim = self._find_worst_frame(dut_frames, ref_frames, offset)
        artifact_report = ArtifactReport()
        if worst_idx < len(dut_frames):
            artifact_report = self._artifact_detector.detect(dut_frames[worst_idx])

        # Cap FPS info
        cap = cv2.VideoCapture(str(dut_path))
        fps_orig = cap.get(cv2.CAP_PROP_FPS) or 0.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_s = total_frames / fps_orig if fps_orig > 0 else 0.0
        cap.release()

        return VideoAnalysisResult(
            total_frames=total_frames,
            sampled_frames=len(dut_frames),
            fps_original=fps_orig,
            duration_s=duration_s,
            temporal=temporal,
            quality_metrics=agg_qm,
            artifacts=artifact_report,
            worst_frame_index=worst_idx,
            worst_frame_ssim=worst_ssim,
        )

    # ── Frame extraction ───────────────────────────────────────────────────────

    def _extract_frames(
        self, path: Path, bbox: Optional[tuple[int, int, int, int]]
    ) -> list[np.ndarray]:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")

        fps_video = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_fps = self._settings.video_sample_fps
        frame_step = max(1, int(round(fps_video / sample_fps)))

        frames: list[np.ndarray] = []
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % frame_step == 0:
                if bbox is not None:
                    x, y, w, h = bbox
                    frame = frame[y: y + h, x: x + w]
                frames.append(frame)
            idx += 1
        cap.release()
        log.debug("Extracted %d frames from %s (step=%d)", len(frames), path.name, frame_step)
        return frames

    # ── Sync ───────────────────────────────────────────────────────────────────

    def _find_sync_offset(
        self, dut_frames: list[np.ndarray], ref_frames: list[np.ndarray]
    ) -> int:
        """Find temporal offset using cross-correlation of per-frame mean luminance."""
        if not dut_frames or not ref_frames:
            return 0

        def luminance_curve(frames: list[np.ndarray]) -> np.ndarray:
            return np.array([
                float(np.mean(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)))
                for f in frames
            ])

        dut_lum = luminance_curve(dut_frames)
        ref_lum = luminance_curve(ref_frames)

        # Zero-mean
        dut_lum -= dut_lum.mean()
        ref_lum -= ref_lum.mean()

        corr = np.correlate(dut_lum, ref_lum, mode="full")
        best_lag = int(np.argmax(corr)) - (len(ref_lum) - 1)
        # Clamp to reasonable range
        max_offset = min(len(dut_frames), len(ref_frames)) // 4
        return max(-max_offset, min(max_offset, best_lag))

    # ── Temporal metrics ───────────────────────────────────────────────────────

    def _compute_temporal(
        self,
        dut_frames: list[np.ndarray],
        ref_frames: list[np.ndarray],
        offset: int,
        sync_mode: SyncMode,
    ) -> VideoTemporalMetrics:
        tm = VideoTemporalMetrics(
            sync_offset_frames=offset,
            sync_mode_used=sync_mode,
        )

        if len(dut_frames) < 2:
            return tm

        # Flicker
        tm.flicker_score, tm.flicker_grade = self._compute_flicker(dut_frames)

        # Jitter
        tm.jitter_score, tm.jitter_grade = self._compute_jitter(dut_frames)

        # Temporal SSIM
        ssim_values = self._frame_to_frame_ssim(dut_frames)
        if ssim_values:
            tm.temporal_ssim_mean = float(np.mean(ssim_values))
            tm.temporal_ssim_std = float(np.std(ssim_values))

        # Per-frame sharpness
        sharpness = [
            float(cv2.Laplacian(
                cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.CV_64F
            ).var())
            for f in dut_frames
        ]
        tm.sharpness_mean = float(np.mean(sharpness))
        tm.sharpness_std = float(np.std(sharpness))

        return tm

    def _compute_flicker(
        self, frames: list[np.ndarray]
    ) -> tuple[float, QualityGrade]:
        """
        Detect flicker via FFT of per-frame mean luminance.
        Returns (flicker_score, grade).
        flicker_score = energy in [5, 60 Hz] band / total energy.
        """
        lum = np.array([
            float(np.mean(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)))
            for f in frames
        ])
        if len(lum) < 8:
            return 0.0, QualityGrade.PASS

        sample_fps = self._settings.video_sample_fps
        fft = np.abs(np.fft.rfft(lum))
        freqs = np.fft.rfftfreq(len(lum), d=1.0 / sample_fps)
        fft[0] = 0  # remove DC

        flicker_mask = (freqs >= 5) & (freqs <= 60)
        flicker_energy = float(fft[flicker_mask].sum())
        total_energy = float(fft.sum()) + 1e-9
        score = flicker_energy / total_energy

        if score > _FLICKER_FAIL:
            grade = QualityGrade.FAIL
        elif score > _FLICKER_WARN:
            grade = QualityGrade.WARNING
        else:
            grade = QualityGrade.PASS

        return score, grade

    def _compute_jitter(
        self, frames: list[np.ndarray]
    ) -> tuple[float, QualityGrade]:
        """Measure video jitter as std dev of optical flow displacement centroids."""
        if len(frames) < 3:
            return 0.0, QualityGrade.PASS

        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        centroids_x: list[float] = []
        centroids_y: list[float] = []
        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

        for frame in frames[1:]:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = dis.calc(prev_gray, gray, None)
            cx = float(np.mean(flow[..., 0]))
            cy = float(np.mean(flow[..., 1]))
            centroids_x.append(cx)
            centroids_y.append(cy)
            prev_gray = gray

        jitter = float(np.sqrt(np.std(centroids_x) ** 2 + np.std(centroids_y) ** 2))
        if jitter > _JITTER_FAIL:
            grade = QualityGrade.FAIL
        elif jitter > _JITTER_WARN:
            grade = QualityGrade.WARNING
        else:
            grade = QualityGrade.PASS

        return jitter, grade

    def _frame_to_frame_ssim(self, frames: list[np.ndarray]) -> list[float]:
        """Compute SSIM between consecutive sampled frames."""
        values: list[float] = []
        for i in range(len(frames) - 1):
            g1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            g2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
            # Resize to max 256px for speed
            scale = min(1.0, 256 / max(g1.shape))
            if scale < 1.0:
                g1 = cv2.resize(g1, None, fx=scale, fy=scale)
                g2 = cv2.resize(g2, None, fx=scale, fy=scale)
            try:
                v = float(ssim_fn(g1, g2, data_range=255))
                values.append(v)
            except Exception:
                pass
        return values

    def _find_worst_frame(
        self,
        dut_frames: list[np.ndarray],
        ref_frames: list[np.ndarray],
        offset: int,
    ) -> tuple[int, Optional[float]]:
        """Find index of the lowest-quality DUT frame."""
        if not dut_frames:
            return 0, None

        if not ref_frames:
            # No reference: use per-frame Laplacian variance (lowest = worst)
            sharpness = [
                float(cv2.Laplacian(
                    cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.CV_64F
                ).var())
                for f in dut_frames
            ]
            worst_idx = int(np.argmin(sharpness))
            return worst_idx, None

        # With reference: find frame with lowest SSIM vs reference
        ssim_scores: list[float] = []
        for i, dut_f in enumerate(dut_frames):
            ref_idx = i + offset
            if ref_idx < 0 or ref_idx >= len(ref_frames):
                ssim_scores.append(1.0)
                continue
            g_dut = cv2.cvtColor(dut_f, cv2.COLOR_BGR2GRAY)
            g_ref = cv2.cvtColor(ref_frames[ref_idx], cv2.COLOR_BGR2GRAY)
            scale = min(1.0, 256 / max(g_dut.shape))
            if scale < 1.0:
                g_dut = cv2.resize(g_dut, None, fx=scale, fy=scale)
                g_ref = cv2.resize(g_ref, None, fx=scale, fy=scale)
            try:
                v = float(ssim_fn(g_dut, g_ref, data_range=255))
            except Exception:
                v = 1.0
            ssim_scores.append(v)

        worst_idx = int(np.argmin(ssim_scores))
        return worst_idx, ssim_scores[worst_idx]

    # ── Aggregation ────────────────────────────────────────────────────────────

    @staticmethod
    def _aggregate_quality_metrics(all_qm: list[QualityMetrics]) -> QualityMetrics:
        """Return mean of each quality metric across frames."""
        if not all_qm:
            return QualityMetrics()

        def avg(attr: str) -> Optional[float]:
            vals = [getattr(qm, attr) for qm in all_qm if getattr(qm, attr) is not None]
            return float(np.mean(vals)) if vals else None

        return QualityMetrics(
            blur_score=avg("blur_score"),
            tenengrad_score=avg("tenengrad_score"),
            noise_sigma=avg("noise_sigma"),
            exposure_mean=avg("exposure_mean"),
            highlight_clipping_pct=avg("highlight_clipping_pct"),
            shadow_clipping_pct=avg("shadow_clipping_pct"),
            color_cast_r=avg("color_cast_r"),
            color_cast_g=avg("color_cast_g"),
            color_cast_b=avg("color_cast_b"),
            white_balance_deviation=avg("white_balance_deviation"),
            saturation_mean=avg("saturation_mean"),
            dynamic_range_stops=avg("dynamic_range_stops"),
            chromatic_aberration_score=avg("chromatic_aberration_score"),
        )
