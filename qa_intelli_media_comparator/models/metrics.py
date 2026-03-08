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
        from qa_intelli_media_comparator.config import get_settings
        s = get_settings()
        reasons: list[str] = []
        if self.blur_grade == QualityGrade.FAIL:
            reasons.append(
                f"Image is blurry: sharpness score {self.blur_score:.1f} "
                f"(threshold {s.blur_threshold:.1f}). Check camera focus or motion blur."
            )
        if self.noise_grade == QualityGrade.FAIL:
            reasons.append(
                f"High noise: sigma {self.noise_sigma:.2f} "
                f"(threshold {s.noise_threshold:.2f}). Check ISO setting or lighting."
            )
        if self.highlight_clipping_pct and self.highlight_clipping_pct > s.highlight_clip_threshold:
            reasons.append(
                f"Highlight clipping: {self.highlight_clipping_pct:.2f}% pixels blown out "
                f"(threshold {s.highlight_clip_threshold:.1f}%). Reduce exposure."
            )
        if self.shadow_clipping_pct and self.shadow_clipping_pct > s.shadow_clip_threshold:
            reasons.append(
                f"Shadow clipping: {self.shadow_clipping_pct:.2f}% pixels crushed to black "
                f"(threshold {s.shadow_clip_threshold:.1f}%). Increase exposure or check HDR tone mapping."
            )
        return reasons

    def build_comparison(self, ref: "QualityMetrics") -> "QualityComparison":
        """Build a structured DUT vs Reference comparison object."""
        return QualityComparison.build(self, ref)

    def comparison_failure_reasons(self, ref: QualityMetrics) -> list[str]:
        """Return failure reasons when DUT regresses significantly vs reference.

        Thresholds:
          - Sharpness: DUT < REF * 0.70  → regression (30% sharper reference)
          - Noise:     DUT > REF * 1.50  → regression (50% more noise than reference)
          - Exposure:  |DUT − REF| > 15 L* → significant exposure shift
          - Highlight clip: DUT > REF + 2%  → more blown highlights than reference
          - Shadow clip:    DUT > REF + 2%  → more crushed blacks than reference
          - WB deviation:   DUT > REF + 0.08 → noticeably worse white balance
          - CA score:       DUT > REF + 1.5 px → more chromatic aberration
        """
        reasons: list[str] = []

        if self.blur_score is not None and ref.blur_score and ref.blur_score > 0:
            ratio = self.blur_score / ref.blur_score
            if ratio < 0.70:
                reasons.append(
                    f"[vs REF] Sharpness regression: DUT {self.blur_score:.1f} vs REF "
                    f"{ref.blur_score:.1f} ({ratio*100:.0f}% of reference). "
                    "Check focus, OIS, or motion blur."
                )

        if self.noise_sigma is not None and ref.noise_sigma is not None:
            if self.noise_sigma > ref.noise_sigma * 1.50:
                reasons.append(
                    f"[vs REF] Noise regression: DUT σ={self.noise_sigma:.2f} vs REF "
                    f"σ={ref.noise_sigma:.2f} ({self.noise_sigma/ref.noise_sigma:.1f}× noisier). "
                    "Check ISO cap or NR aggressiveness."
                )

        if self.exposure_mean is not None and ref.exposure_mean is not None:
            delta = self.exposure_mean - ref.exposure_mean
            if abs(delta) > 15:
                direction = "brighter" if delta > 0 else "darker"
                reasons.append(
                    f"[vs REF] Exposure shift: DUT {self.exposure_mean:.1f} L* vs REF "
                    f"{ref.exposure_mean:.1f} L* ({delta:+.1f}, {direction}). "
                    "Check AE metering or EV compensation."
                )

        if self.highlight_clipping_pct is not None and ref.highlight_clipping_pct is not None:
            delta = self.highlight_clipping_pct - ref.highlight_clipping_pct
            if delta > 2.0:
                reasons.append(
                    f"[vs REF] More highlight clipping: DUT {self.highlight_clipping_pct:.2f}% "
                    f"vs REF {ref.highlight_clipping_pct:.2f}% (+{delta:.2f}%). "
                    "Reduce EV or enable highlight recovery."
                )

        if self.shadow_clipping_pct is not None and ref.shadow_clipping_pct is not None:
            delta = self.shadow_clipping_pct - ref.shadow_clipping_pct
            if delta > 2.0:
                reasons.append(
                    f"[vs REF] More shadow clipping: DUT {self.shadow_clipping_pct:.2f}% "
                    f"vs REF {ref.shadow_clipping_pct:.2f}% (+{delta:.2f}%). "
                    "Increase EV or check shadow lift."
                )

        if self.white_balance_deviation is not None and ref.white_balance_deviation is not None:
            delta = self.white_balance_deviation - ref.white_balance_deviation
            if delta > 0.08:
                reasons.append(
                    f"[vs REF] White balance regression: DUT deviation={self.white_balance_deviation:.3f} "
                    f"vs REF {ref.white_balance_deviation:.3f} (+{delta:.3f}). "
                    "Check AWB or apply manual WB preset."
                )

        if (self.chromatic_aberration_score is not None
                and ref.chromatic_aberration_score is not None):
            delta = self.chromatic_aberration_score - ref.chromatic_aberration_score
            if delta > 1.5:
                reasons.append(
                    f"[vs REF] Chromatic aberration increase: DUT {self.chromatic_aberration_score:.2f} px "
                    f"vs REF {ref.chromatic_aberration_score:.2f} px (+{delta:.2f} px). "
                    "Check lens correction profile or zoom alignment."
                )

        return reasons


# ---------------------------------------------------------------------------
# Structured DUT vs Reference comparison
# ---------------------------------------------------------------------------

class MetricComparison(BaseModel):
    """Per-metric DUT vs Reference comparison result."""
    dut: float
    ref: float
    delta: float                            # dut - ref (positive = DUT higher)
    delta_pct: Optional[float] = None       # (delta / |ref|) * 100
    regression: bool = False                # True if DUT regressed beyond threshold


class QualityComparison(BaseModel):
    """Structured DUT vs Reference quality comparison (compare mode only)."""
    sharpness: Optional[MetricComparison] = None
    noise: Optional[MetricComparison] = None
    exposure: Optional[MetricComparison] = None
    highlight_clipping: Optional[MetricComparison] = None
    shadow_clipping: Optional[MetricComparison] = None
    white_balance: Optional[MetricComparison] = None
    chromatic_aberration: Optional[MetricComparison] = None

    @classmethod
    def build(cls, dut: "QualityMetrics", ref: "QualityMetrics") -> "QualityComparison":
        def _cmp(dv: Optional[float], rv: Optional[float],
                 regression_check) -> Optional[MetricComparison]:
            if dv is None or rv is None:
                return None
            delta = dv - rv
            pct = (delta / abs(rv) * 100) if rv != 0 else None
            return MetricComparison(
                dut=round(dv, 4),
                ref=round(rv, 4),
                delta=round(delta, 4),
                delta_pct=round(pct, 2) if pct is not None else None,
                regression=regression_check(dv, rv),
            )

        return cls(
            sharpness=_cmp(
                dut.blur_score, ref.blur_score,
                lambda d, r: r > 0 and (d / r) < 0.70,
            ),
            noise=_cmp(
                dut.noise_sigma, ref.noise_sigma,
                lambda d, r: r > 0 and d > r * 1.50,
            ),
            exposure=_cmp(
                dut.exposure_mean, ref.exposure_mean,
                lambda d, r: abs(d - r) > 15,
            ),
            highlight_clipping=_cmp(
                dut.highlight_clipping_pct, ref.highlight_clipping_pct,
                lambda d, r: (d - r) > 2.0,
            ),
            shadow_clipping=_cmp(
                dut.shadow_clipping_pct, ref.shadow_clipping_pct,
                lambda d, r: (d - r) > 2.0,
            ),
            white_balance=_cmp(
                dut.white_balance_deviation, ref.white_balance_deviation,
                lambda d, r: (d - r) > 0.08,
            ),
            chromatic_aberration=_cmp(
                dut.chromatic_aberration_score, ref.chromatic_aberration_score,
                lambda d, r: (d - r) > 1.5,
            ),
        )
