from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from .enums import QualityGrade, SyncMode
from .metrics import QualityMetrics
from .artifacts import ArtifactReport


class VideoTemporalMetrics(BaseModel):
    """Temporal quality metrics computed across video frames."""
    # Flicker: energy in 5–60 Hz band of per-frame luminance FFT
    flicker_score: Optional[float] = None       # 0 = no flicker; higher = more flicker
    flicker_grade: QualityGrade = QualityGrade.PASS

    # Jitter: std dev of optical flow displacement centroid
    jitter_score: Optional[float] = None        # px; lower = more stable
    jitter_grade: QualityGrade = QualityGrade.PASS

    # Temporal SSIM consistency
    temporal_ssim_mean: Optional[float] = None  # mean frame-to-frame SSIM
    temporal_ssim_std: Optional[float] = None   # std; high = inconsistent quality

    # Aggregated per-frame quality
    sharpness_mean: Optional[float] = None
    sharpness_std: Optional[float] = None
    noise_mean: Optional[float] = None

    sync_offset_frames: int = 0                 # temporal offset found in AUTO sync mode
    sync_mode_used: SyncMode = SyncMode.FRAME_BY_FRAME

    def failure_reasons(self) -> list[str]:
        reasons = []
        if self.flicker_grade == QualityGrade.FAIL:
            reasons.append(
                f"Video flicker detected (score {self.flicker_score:.3f}). "
                "Check for AC lighting interference or exposure oscillation."
            )
        if self.jitter_grade == QualityGrade.FAIL:
            reasons.append(
                f"Video jitter detected (score {self.jitter_score:.2f} px). "
                "Check OIS / EIS stabilization."
            )
        if self.temporal_ssim_std and self.temporal_ssim_std > 0.05:
            reasons.append(
                f"Inconsistent frame quality (SSIM std {self.temporal_ssim_std:.3f}). "
                "Video quality varies significantly between frames."
            )
        return reasons


class VideoAnalysisResult(BaseModel):
    """Full video analysis output."""
    total_frames: int = 0
    sampled_frames: int = 0
    fps_original: float = 0.0
    duration_s: float = 0.0

    temporal: VideoTemporalMetrics = Field(default_factory=VideoTemporalMetrics)
    # Per-frame quality (aggregated)
    quality_metrics: QualityMetrics = Field(default_factory=QualityMetrics)
    artifacts: ArtifactReport = Field(default_factory=ArtifactReport)

    # Index of the worst quality frame (for annotation)
    worst_frame_index: Optional[int] = None
    worst_frame_ssim: Optional[float] = None
