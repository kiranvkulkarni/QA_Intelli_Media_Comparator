from __future__ import annotations

"""
ComparisonPipeline — orchestrates the full comparison workflow.

Inputs:
  dut_path       : Path to DUT media file (required)
  reference_path : Path to golden reference file (optional)
  sync_mode      : SyncMode for video comparison
  crop_preview   : Whether to auto-crop preview UI chrome
  force_media_type: Override auto-detection if needed

Outputs:
  ComparisonReport with JSON metrics + paths to annotated/diff images.
"""

import logging
import time
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..config import get_settings, _request_settings
from ..models.enums import MediaType, QualityGrade, SyncMode
from ..models.metadata import MetadataComparison
from ..models.metrics import FullReferenceScores, NoReferenceScores
from ..models.report import ComparisonReport
from .camera_mode_detector import CameraModeDetector
from .functionality_checker import FunctionalityChecker
from .media_type_detector import MediaTypeDetector
from .preview_cropper import PreviewCropper
from .quality_metrics import QualityMetricsExtractor
from .artifact_detector import ArtifactDetector
from .no_reference_analyzer import NoReferenceAnalyzer
from .reference_comparator import ReferenceComparator
from .video_analyzer import VideoAnalyzer
from .annotation_renderer import AnnotationRenderer

log = logging.getLogger(__name__)


# ── Functional mode metric grade merger ───────────────────────────────────────

def _merge_functional_metric_grades(
    grade: QualityGrade,
    reasons: list[str],
    nr_scores: Optional[NoReferenceScores],
    ref_nr_scores: Optional[NoReferenceScores],
    fr_scores: Optional[FullReferenceScores],
) -> tuple[QualityGrade, list[str]]:
    """Fold BRISQUE delta + LPIPS + DISTS results into functional grade/reasons.

    Called by the functional mode branch to augment the go/no-go checks already
    produced by FunctionalityChecker with perceptual metric verdicts.
    """
    reasons = list(reasons)  # don't mutate the caller's list

    # ── BRISQUE absolute floor ─────────────────────────────────────────────────
    if nr_scores and nr_scores.brisque is not None:
        b = nr_scores.brisque
        if b > 70:
            reasons.append(
                f"BRISQUE {b:.1f} > 70 — poor perceptual quality "
                "(likely affected by distortion, blur, or noise)."
            )
            grade = QualityGrade.FAIL
        elif b > 50:
            reasons.append(
                f"BRISQUE {b:.1f} > 50 — moderate quality degradation detected."
            )
            if grade == QualityGrade.PASS:
                grade = QualityGrade.WARNING

    # ── BRISQUE delta vs reference ─────────────────────────────────────────────
    if (
        nr_scores and ref_nr_scores
        and nr_scores.brisque is not None
        and ref_nr_scores.brisque is not None
    ):
        delta = nr_scores.brisque - ref_nr_scores.brisque
        if delta > 15:
            reasons.append(
                f"Quality regression: BRISQUE DUT={nr_scores.brisque:.1f} vs "
                f"REF={ref_nr_scores.brisque:.1f} (+{delta:.1f}) — "
                "DUT quality significantly degraded vs golden reference."
            )
            grade = QualityGrade.FAIL
        elif delta > 8:
            reasons.append(
                f"Mild quality regression: BRISQUE DUT={nr_scores.brisque:.1f} vs "
                f"REF={ref_nr_scores.brisque:.1f} (+{delta:.1f}) — "
                "DUT slightly worse than golden reference."
            )
            if grade == QualityGrade.PASS:
                grade = QualityGrade.WARNING
        elif delta < -8:
            reasons.append(
                f"Quality improved vs reference: BRISQUE DUT={nr_scores.brisque:.1f} vs "
                f"REF={ref_nr_scores.brisque:.1f} ({delta:.1f}) — "
                "DUT is perceptually better than golden reference."
            )

    # ── LPIPS ──────────────────────────────────────────────────────────────────
    if fr_scores and fr_scores.lpips and fr_scores.lpips.value is not None:
        lpips_val = fr_scores.lpips.value
        lpips_thr = fr_scores.lpips.threshold or 0.15
        if not fr_scores.lpips.passed:
            reasons.append(
                f"LPIPS {lpips_val:.4f} > {lpips_thr:.4f} — "
                "significant perceptual difference from reference "
                "(detail / texture / structure loss)."
            )
            grade = QualityGrade.FAIL
        elif lpips_val > lpips_thr * 0.75:
            reasons.append(
                f"LPIPS {lpips_val:.4f} approaching threshold {lpips_thr:.4f} — "
                "noticeable perceptual difference from reference."
            )
            if grade == QualityGrade.PASS:
                grade = QualityGrade.WARNING

    # ── DISTS ──────────────────────────────────────────────────────────────────
    if fr_scores and fr_scores.dists and fr_scores.dists.value is not None:
        dists_val = fr_scores.dists.value
        dists_thr = fr_scores.dists.threshold or 0.15
        if not fr_scores.dists.passed:
            reasons.append(
                f"DISTS {dists_val:.4f} > {dists_thr:.4f} — "
                "texture/structure regression vs reference."
            )
            if grade != QualityGrade.FAIL:
                grade = QualityGrade.WARNING
        elif dists_val > dists_thr * 0.75:
            reasons.append(
                f"DISTS {dists_val:.4f} approaching threshold {dists_thr:.4f} — "
                "mild structural difference from reference."
            )

    return grade, reasons


class ComparisonPipeline:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._mode_detector = CameraModeDetector()
        self._func_checker = FunctionalityChecker()
        self._detector = MediaTypeDetector()
        self._cropper = PreviewCropper()
        self._qm_extractor = QualityMetricsExtractor()
        self._artifact_detector = ArtifactDetector()
        self._nr_analyzer = NoReferenceAnalyzer()
        self._fr_comparator = ReferenceComparator()
        self._video_analyzer = VideoAnalyzer()
        self._renderer = AnnotationRenderer()

    def preload_models(self) -> None:
        """Eagerly load all neural models (call once at startup)."""
        self._nr_analyzer.preload()
        self._fr_comparator.preload()

    def loaded_models(self) -> list[str]:
        return self._nr_analyzer.loaded_models()

    def run(
        self,
        dut_path: Path,
        reference_path: Optional[Path] = None,
        sync_mode: SyncMode = SyncMode.AUTO,
        crop_preview: bool = True,
        force_media_type: Optional[str] = None,
        analysis_mode: Optional[str] = None,
    ) -> ComparisonReport:
        """Run the full analysis pipeline.

        Parameters
        ----------
        analysis_mode
            ``'functional'`` — fast path: functional checks + basic metrics only
            (no neural IQA).  ``'quality'`` — full path (default).  If ``None``,
            the value from the active settings (QIMC_ANALYSIS_MODE) is used.
        """
        t_start = time.monotonic()
        report_id = uuid.uuid4().hex[:12]
        effective_mode = (analysis_mode or "").strip().lower() or get_settings().analysis_mode
        if effective_mode not in ("functional", "quality"):
            effective_mode = "quality"
        log.info(
            "Pipeline START report_id=%s mode=%s dut=%s ref=%s",
            report_id, effective_mode, dut_path.name,
            reference_path.name if reference_path else "none",
        )

        # ── 1. Detect media type ───────────────────────────────────────────────
        media_info = self._detector.detect(dut_path)
        media_type = media_info.media_type
        if force_media_type:
            try:
                media_type = MediaType(force_media_type)
            except ValueError:
                log.warning("Invalid force_media_type '%s'; using auto-detected.", force_media_type)

        is_image = media_type in (MediaType.IMAGE_CAPTURED, MediaType.IMAGE_PREVIEW)
        is_preview = media_type in (MediaType.IMAGE_PREVIEW,)  # video always needs crop check

        # ── 1b. Extract EXIF metadata + detect camera mode ────────────────────
        # For images: full EXIF extraction + mode inference.
        # For video:  a skeletal VideoMode object is returned immediately.
        dut_metadata = self._mode_detector.detect(dut_path)
        ref_metadata = self._mode_detector.detect(reference_path) if reference_path else None

        # Apply camera-mode threshold adjustments on top of any quality-profile
        # override already active via ContextVar (set by the API route).
        # The pipeline pushes the adjusted settings as a nested ContextVar so
        # all downstream get_settings() calls see mode-corrected thresholds.
        base_settings = get_settings()
        effective_settings = self._mode_detector.apply_mode_adjustments(
            dut_metadata.camera_mode, base_settings
        )
        mode_token = _request_settings.set(effective_settings)

        if dut_metadata.camera_mode.value not in ("unknown", "auto", "video"):
            log.info(
                "Camera mode '%s' detected (source=%s, confidence=%.0f%%) — "
                "thresholds adjusted accordingly.",
                dut_metadata.camera_mode.value,
                dut_metadata.camera_mode_source,
                dut_metadata.camera_mode_confidence * 100,
            )

        try:
            # ── 2. Load DUT image/frame ────────────────────────────────────────
            dut_img: Optional[np.ndarray] = None
            crop_result = None
            crop_result_ref = None
            dut_bbox: Optional[tuple] = None
            ref_bbox: Optional[tuple] = None

            if is_image:
                dut_img = cv2.imread(str(dut_path))
                if dut_img is None:
                    raise ValueError(f"Cannot read DUT image: {dut_path}")

                # Crop preview UI if needed
                if crop_preview and is_preview and effective_settings.preview_crop_enabled:
                    dut_img, crop_result = self._cropper.crop_image(dut_img)

            # ── 3. Load reference (image only; video handled by VideoAnalyzer) ─
            ref_img: Optional[np.ndarray] = None
            if is_image and reference_path is not None:
                ref_img = cv2.imread(str(reference_path))
                if ref_img is not None and crop_preview and effective_settings.preview_crop_enabled:
                    ref_img, crop_result_ref = self._cropper.crop_image(ref_img)

            # ── 4a. Image analysis ─────────────────────────────────────────────
            quality_metrics = None
            ref_quality_metrics = None
            nr_scores = None
            fr_scores: Optional[FullReferenceScores] = None
            artifacts = None
            diff_heatmap: Optional[np.ndarray] = None
            video_result = None
            func_grade = None
            func_reasons: list[str] = []

            if is_image and dut_img is not None:
                # ── Functional validity check (always fast, always runs) ────────
                func_grade, func_reasons = self._func_checker.check(
                    dut_img, ref_img,
                    camera_mode=dut_metadata.camera_mode.value,
                )

                quality_metrics = self._qm_extractor.extract(dut_img)
                artifacts = self._artifact_detector.detect(dut_img)

                if effective_mode == "quality":
                    # Full IQA — NR metrics + FR metrics
                    nr_scores = self._nr_analyzer.analyze(dut_img)
                    if ref_img is not None:
                        ref_quality_metrics = self._qm_extractor.extract(ref_img)
                        fr_scores, diff_heatmap = self._fr_comparator.compare(ref_img, dut_img)
                else:
                    # Functional mode — BRISQUE+NIQE on DUT, LPIPS+DISTS vs reference.
                    # These feed into functional_grade / functional_reasons via the
                    # _merge_functional_metric_grades() helper below.
                    nr_scores = self._nr_analyzer.analyze_classical(dut_img)
                    _ref_nr: Optional[NoReferenceScores] = None
                    if ref_img is not None:
                        ref_quality_metrics = self._qm_extractor.extract(ref_img)
                        _ref_nr = self._nr_analyzer.analyze_classical(ref_img)
                        fr_scores, diff_heatmap = self._fr_comparator.compare(
                            ref_img, dut_img, metrics=["lpips", "dists"]
                        )
                    func_grade, func_reasons = _merge_functional_metric_grades(
                        func_grade, func_reasons, nr_scores, _ref_nr, fr_scores
                    )

            # ── 4b. Video analysis ─────────────────────────────────────────────
            elif not is_image:
                if crop_preview and effective_settings.preview_crop_enabled:
                    cap = cv2.VideoCapture(str(dut_path))
                    dut_bbox_result = self._cropper.crop_video_frame(cap)
                    cap.release()
                    dut_bbox = dut_bbox_result[0]
                    crop_result = dut_bbox_result[1]

                    if reference_path is not None:
                        cap_ref = cv2.VideoCapture(str(reference_path))
                        ref_bbox_result = self._cropper.crop_video_frame(cap_ref)
                        cap_ref.release()
                        ref_bbox = ref_bbox_result[0]
                        crop_result_ref = ref_bbox_result[1]

                video_result = self._video_analyzer.analyze(
                    dut_path=dut_path,
                    ref_path=reference_path,
                    sync_mode=sync_mode,
                    bbox=dut_bbox,
                    ref_bbox=ref_bbox,
                )
                quality_metrics = video_result.quality_metrics
                artifacts = video_result.artifacts
                # Derive functional grade from video temporal results
                from ..models.enums import QualityGrade as _QG
                vt = video_result.temporal
                if vt.black_frame_count > 0 or vt.frozen_frame_count > 0:
                    func_grade = _QG.FAIL
                    func_reasons = vt.failure_reasons()[-2:]  # last two = black/frozen entries
                else:
                    func_grade = _QG.PASS

            # ── 5. Build report ────────────────────────────────────────────────
            from ..models.enums import QualityGrade as _QG
            from ..models.metrics import NoReferenceScores, QualityComparison
            from ..models.artifacts import ArtifactReport

            _qm = quality_metrics or quality_metrics.__class__()
            _quality_comparison: Optional[QualityComparison] = None
            if ref_quality_metrics is not None and _qm is not None:
                _quality_comparison = QualityComparison.build(_qm, ref_quality_metrics)

            _metadata_comparison: Optional[MetadataComparison] = None
            if ref_metadata is not None:
                _metadata_comparison = MetadataComparison.build(dut_metadata, ref_metadata)

            # func_grade / func_reasons initialised in image branch above;
            # for video they are set after video analysis.
            _func_grade = func_grade if func_grade is not None else _QG.PASS
            _func_reasons = func_reasons or []

            report = ComparisonReport(
                report_id=report_id,
                analysis_mode=effective_mode,
                media_type=media_type,
                dut_file=str(dut_path),
                reference_file=str(reference_path) if reference_path else None,
                crop_applied=crop_result is not None and crop_result.applied,
                crop_result=crop_result,
                crop_result_ref=crop_result_ref,
                dut_metadata=dut_metadata,
                ref_metadata=ref_metadata,
                metadata_comparison=_metadata_comparison,
                functional_grade=_func_grade,
                functional_reasons=_func_reasons,
                quality_metrics=_qm,
                ref_quality_metrics=ref_quality_metrics,
                quality_comparison=_quality_comparison,
                nr_scores=nr_scores or NoReferenceScores(),
                fr_scores=fr_scores,
                artifacts=artifacts or ArtifactReport(),
                video_temporal=video_result.temporal if video_result else None,
            )

            report.compute_overall_grade()

            # ── 6. Annotated image ─────────────────────────────────────────────
            annotated_img: Optional[np.ndarray] = None
            if is_image and dut_img is not None:
                annotated_img = self._renderer.render(report, dut_img, diff_heatmap)
            elif not is_image and video_result and video_result.worst_frame_index is not None:
                worst_frame = self._extract_worst_frame(
                    dut_path, video_result.worst_frame_index, dut_bbox
                )
                if worst_frame is not None:
                    annotated_img = self._renderer.render(report, worst_frame, None)

            if annotated_img is not None:
                ann_path = effective_settings.reports_dir / f"{report_id}_annotated.png"
                self._renderer.save(annotated_img, ann_path)
                report.annotated_image_path = str(ann_path)

            if diff_heatmap is not None:
                diff_path = effective_settings.reports_dir / f"{report_id}_diff.png"
                self._renderer.save(diff_heatmap, diff_path)
                report.diff_image_path = str(diff_path)

            # ── 7. Timing ──────────────────────────────────────────────────────
            report.processing_time_ms = int((time.monotonic() - t_start) * 1000)
            log.info(
                "Pipeline DONE report_id=%s grade=%s time=%dms",
                report_id, report.overall_grade.value, report.processing_time_ms,
            )
            return report

        finally:
            # Always reset mode-adjusted settings so the ContextVar chain stays clean
            _request_settings.reset(mode_token)

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_worst_frame(
        video_path: Path,
        frame_index: int,
        bbox: Optional[tuple],
    ) -> Optional[np.ndarray]:
        """Read a specific sampled frame from a video for annotation."""
        from ..config import get_settings
        settings = get_settings()
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        fps_video = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_step = max(1, int(round(fps_video / settings.video_sample_fps)))
        target_raw_idx = frame_index * frame_step
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_raw_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            return None
        if bbox is not None:
            x, y, w, h = bbox
            frame = frame[y: y + h, x: x + w]
        return frame
