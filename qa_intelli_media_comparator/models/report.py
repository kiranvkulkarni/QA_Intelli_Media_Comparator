from __future__ import annotations

from datetime import datetime
from typing import Optional
from pathlib import Path

from pydantic import BaseModel, Field

from .enums import MediaType, QualityGrade, SyncMode
from .media import CropResult
from .metadata import MediaMetadata, MetadataComparison
from .metrics import FullReferenceScores, NoReferenceScores, QualityMetrics, QualityComparison
from .artifacts import ArtifactReport
from .video import VideoTemporalMetrics


class ComparisonReport(BaseModel):
    """Top-level output model — fully serializable to JSON."""
    report_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Input context
    media_type: MediaType
    dut_file: str
    reference_file: Optional[str] = None
    crop_applied: bool = False
    crop_result: Optional[CropResult] = None
    crop_result_ref: Optional[CropResult] = None

    # EXIF metadata + camera mode
    dut_metadata: Optional[MediaMetadata] = None       # EXIF from DUT
    ref_metadata: Optional[MediaMetadata] = None       # EXIF from reference (compare mode)
    metadata_comparison: Optional[MetadataComparison] = None  # side-by-side EXIF diff

    # Quality scores
    quality_metrics: QualityMetrics = Field(default_factory=QualityMetrics)
    ref_quality_metrics: Optional[QualityMetrics] = None  # reference metrics (compare mode only)
    quality_comparison: Optional[QualityComparison] = None  # structured DUT vs REF (compare mode only)
    nr_scores: NoReferenceScores = Field(default_factory=NoReferenceScores)
    fr_scores: Optional[FullReferenceScores] = None     # only when reference provided

    # Artifacts
    artifacts: ArtifactReport = Field(default_factory=ArtifactReport)

    # Video-specific
    video_temporal: Optional[VideoTemporalMetrics] = None

    # Verdict
    overall_grade: QualityGrade = QualityGrade.PASS
    failure_reasons: list[str] = Field(default_factory=list)

    # Outputs
    annotated_image_path: Optional[str] = None
    diff_image_path: Optional[str] = None

    # Performance
    processing_time_ms: int = 0

    def compute_overall_grade(self) -> QualityGrade:
        """Derive overall grade from sub-results and populate failure_reasons."""
        reasons: list[str] = []
        grade = QualityGrade.PASS

        # FR failures → FAIL
        if self.fr_scores and self.fr_scores.any_failed:
            grade = QualityGrade.FAIL
            reasons.extend(self.fr_scores.failure_reasons())

        # Quality metric failures
        qm_reasons = self.quality_metrics.failure_reasons()
        if qm_reasons:
            grade = QualityGrade.FAIL
            reasons.extend(qm_reasons)

        # Artifact failures
        if self.artifacts.has_failures:
            grade = QualityGrade.FAIL
            reasons.extend(self.artifacts.failure_reasons())

        # Video temporal failures
        if self.video_temporal:
            vt_reasons = self.video_temporal.failure_reasons()
            if vt_reasons:
                grade = QualityGrade.FAIL
                reasons.extend(vt_reasons)

        # NR score grading (advisory — WARNING only unless severe)
        if self.nr_scores.brisque and self.nr_scores.brisque > 60:
            if grade == QualityGrade.PASS:
                grade = QualityGrade.WARNING
            reasons.append(
                f"BRISQUE score {self.nr_scores.brisque:.1f} indicates poor perceptual quality."
            )

        # Comparative quality regression (compare mode only)
        if self.ref_quality_metrics is not None:
            reasons.extend(
                self.quality_metrics.comparison_failure_reasons(self.ref_quality_metrics)
            )

        # Camera mode notes — advisory only (never cause FAIL on their own)
        if self.dut_metadata and self.dut_metadata.mode_notes:
            for note in self.dut_metadata.mode_notes:
                reasons.append(f"[Mode] {note}")

        if self.metadata_comparison and self.metadata_comparison.notes:
            for note in self.metadata_comparison.notes:
                reasons.append(f"[Metadata] {note}")

        self.overall_grade = grade
        self.failure_reasons = reasons
        return grade
