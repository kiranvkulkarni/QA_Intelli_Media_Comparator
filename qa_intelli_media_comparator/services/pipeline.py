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

from ..config import get_settings
from ..models.enums import MediaType, SyncMode
from ..models.metrics import FullReferenceScores
from ..models.report import ComparisonReport
from .media_type_detector import MediaTypeDetector
from .preview_cropper import PreviewCropper
from .quality_metrics import QualityMetricsExtractor
from .artifact_detector import ArtifactDetector
from .no_reference_analyzer import NoReferenceAnalyzer
from .reference_comparator import ReferenceComparator
from .video_analyzer import VideoAnalyzer
from .annotation_renderer import AnnotationRenderer

log = logging.getLogger(__name__)


class ComparisonPipeline:
    def __init__(self) -> None:
        self._settings = get_settings()
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
    ) -> ComparisonReport:
        t_start = time.monotonic()
        report_id = uuid.uuid4().hex[:12]
        log.info("Pipeline START report_id=%s dut=%s ref=%s", report_id, dut_path.name,
                 reference_path.name if reference_path else "none")

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

        # ── 2. Load DUT image/frame ────────────────────────────────────────────
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
            if crop_preview and is_preview and self._settings.preview_crop_enabled:
                dut_img, crop_result = self._cropper.crop_image(dut_img)

        # ── 3. Load reference (image only; video handled by VideoAnalyzer) ─────
        ref_img: Optional[np.ndarray] = None
        if is_image and reference_path is not None:
            ref_img = cv2.imread(str(reference_path))
            if ref_img is not None and crop_preview and self._settings.preview_crop_enabled:
                ref_img, crop_result_ref = self._cropper.crop_image(ref_img)

        # ── 4a. Image analysis ─────────────────────────────────────────────────
        quality_metrics = None
        ref_quality_metrics = None
        nr_scores = None
        fr_scores: Optional[FullReferenceScores] = None
        artifacts = None
        diff_heatmap: Optional[np.ndarray] = None
        video_result = None

        if is_image and dut_img is not None:
            quality_metrics = self._qm_extractor.extract(dut_img)
            nr_scores = self._nr_analyzer.analyze(dut_img)
            artifacts = self._artifact_detector.detect(dut_img)

            if ref_img is not None:
                # Extract quality metrics from reference for side-by-side comparison
                ref_quality_metrics = self._qm_extractor.extract(ref_img)
                fr_scores, diff_heatmap = self._fr_comparator.compare(ref_img, dut_img)

        # ── 4b. Video analysis ─────────────────────────────────────────────────
        elif not is_image:
            # For video preview, detect crop bbox from first frame
            if crop_preview and self._settings.preview_crop_enabled:
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

        # ── 5. Build report ────────────────────────────────────────────────────
        from ..models.metrics import NoReferenceScores
        from ..models.artifacts import ArtifactReport

        report = ComparisonReport(
            report_id=report_id,
            media_type=media_type,
            dut_file=str(dut_path),
            reference_file=str(reference_path) if reference_path else None,
            crop_applied=crop_result is not None and crop_result.applied,
            crop_result=crop_result,
            crop_result_ref=crop_result_ref,
            quality_metrics=quality_metrics or quality_metrics.__class__(),
            ref_quality_metrics=ref_quality_metrics,
            nr_scores=nr_scores or NoReferenceScores(),
            fr_scores=fr_scores,
            artifacts=artifacts or ArtifactReport(),
            video_temporal=video_result.temporal if video_result else None,
        )

        report.compute_overall_grade()

        # ── 6. Annotated image ─────────────────────────────────────────────────
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
            ann_path = self._settings.reports_dir / f"{report_id}_annotated.png"
            self._renderer.save(annotated_img, ann_path)
            report.annotated_image_path = str(ann_path)

        if diff_heatmap is not None:
            diff_path = self._settings.reports_dir / f"{report_id}_diff.png"
            self._renderer.save(diff_heatmap, diff_path)
            report.diff_image_path = str(diff_path)

        # ── 7. Timing ──────────────────────────────────────────────────────────
        report.processing_time_ms = int((time.monotonic() - t_start) * 1000)
        log.info(
            "Pipeline DONE report_id=%s grade=%s time=%dms",
            report_id, report.overall_grade.value, report.processing_time_ms,
        )
        return report

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
