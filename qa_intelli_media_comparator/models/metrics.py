from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from .enums import QualityGrade


class MetricResult(BaseModel):
    """Single metric score with pass/fail context."""
    value: Optional[float] = None
    threshold: Optional[float] = None
    passed: Optional[bool] = None        # None if metric was not computed
    higher_is_better: bool = True

    def evaluate(self) -> None:
        if self.value is not None and self.threshold is not None:
            if self.higher_is_better:
                self.passed = self.value >= self.threshold
            else:
                self.passed = self.value <= self.threshold


class FullReferenceScores(BaseModel):
    """Full-reference (FR) IQA metrics — require a reference image."""
    psnr: MetricResult = Field(default_factory=lambda: MetricResult(higher_is_better=True))
    ssim: MetricResult = Field(default_factory=lambda: MetricResult(higher_is_better=True))
    ms_ssim: MetricResult = Field(default_factory=lambda: MetricResult(higher_is_better=True))
    lpips: MetricResult = Field(default_factory=lambda: MetricResult(higher_is_better=False))
    dists: MetricResult = Field(default_factory=lambda: MetricResult(higher_is_better=False))

    @property
    def any_failed(self) -> bool:
        for field_name in self.model_fields:
            m: MetricResult = getattr(self, field_name)
            if m.passed is False:
                return True
        return False

    def failure_reasons(self) -> list[str]:
        reasons: list[str] = []
        labels = {"psnr": "PSNR", "ssim": "SSIM", "ms_ssim": "MS-SSIM",
                  "lpips": "LPIPS", "dists": "DISTS"}
        for key, label in labels.items():
            m: MetricResult = getattr(self, key)
            if m.passed is False:
                direction = "below" if m.higher_is_better else "above"
                reasons.append(
                    f"{label} {m.value:.4f} is {direction} threshold {m.threshold:.4f}"
                )
        return reasons


class NoReferenceScores(BaseModel):
    """No-reference (NR) IQA metric scores."""
    brisque: Optional[float] = None     # lower = better (0–100)
    niqe: Optional[float] = None        # lower = better
    musiq: Optional[float] = None       # higher = better (0–100)
    clip_iqa: Optional[float] = None    # higher = better (0–1)
    grade: QualityGrade = QualityGrade.PASS


class QualityMetrics(BaseModel):
    """Standard media quality attributes — always computed."""
    # Sharpness
    blur_score: Optional[float] = None          # Laplacian variance (higher = sharper)
    tenengrad_score: Optional[float] = None     # Sobel gradient magnitude

    # Noise
    noise_sigma: Optional[float] = None         # estimated noise std in flat regions

    # Exposure
    exposure_mean: Optional[float] = None       # L channel mean in LAB (0–100)
    highlight_clipping_pct: Optional[float] = None   # % pixels blown out
    shadow_clipping_pct: Optional[float] = None      # % pixels crushed to black

    # Color
    color_cast_r: Optional[float] = None        # mean R deviation from neutral
    color_cast_g: Optional[float] = None
    color_cast_b: Optional[float] = None
    white_balance_deviation: Optional[float] = None  # CCT distance from expected
    saturation_mean: Optional[float] = None          # HSV S-channel mean (0–1)

    # Dynamic range
    dynamic_range_stops: Optional[float] = None      # log2(99th/1st percentile luma)

    # Chromatic aberration
    chromatic_aberration_score: Optional[float] = None  # mean R-B channel offset at edges (px)

    @property
    def blur_grade(self) -> QualityGrade:
        if self.blur_score is None:
            return QualityGrade.PASS
        from qa_intelli_media_comparator.config import get_settings
        thresh = get_settings().blur_threshold
        if self.blur_score >= thresh:
            return QualityGrade.PASS
        if self.blur_score >= thresh * 0.5:
            return QualityGrade.WARNING
        return QualityGrade.FAIL

    @property
    def noise_grade(self) -> QualityGrade:
        if self.noise_sigma is None:
            return QualityGrade.PASS
        from qa_intelli_media_comparator.config import get_settings
        thresh = get_settings().noise_threshold
        if self.noise_sigma <= thresh:
            return QualityGrade.PASS
        if self.noise_sigma <= thresh * 1.5:
            return QualityGrade.WARNING
        return QualityGrade.FAIL

    def failure_reasons(self) -> list[str]:
        reasons: list[str] = []
        if self.blur_grade == QualityGrade.FAIL:
            reasons.append(
                f"Image is blurry: sharpness score {self.blur_score:.1f} "
                f"(threshold {100.0:.1f}). Check camera focus or motion blur."
            )
        if self.noise_grade == QualityGrade.FAIL:
            reasons.append(
                f"High noise: sigma {self.noise_sigma:.2f} px "
                f"(threshold {8.0:.2f}). Check ISO setting or lighting."
            )
        if self.highlight_clipping_pct and self.highlight_clipping_pct > 1.0:
            reasons.append(
                f"Highlight clipping: {self.highlight_clipping_pct:.2f}% pixels blown out. "
                "Reduce exposure."
            )
        if self.shadow_clipping_pct and self.shadow_clipping_pct > 1.0:
            reasons.append(
                f"Shadow clipping: {self.shadow_clipping_pct:.2f}% pixels crushed to black. "
                "Increase exposure or check HDR tone mapping."
            )
        return reasons
