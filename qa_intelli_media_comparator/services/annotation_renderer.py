from __future__ import annotations

"""
AnnotationRenderer — draws artifact bounding boxes, severity labels, metric
tables, and optional diff heatmap overlays onto a BGR image.

Output: PIL Image saved as PNG.

Severity color coding:
  NONE     → green   (0, 200, 0)
  LOW      → yellow  (0, 220, 220)
  MEDIUM   → orange  (0, 140, 255)
  HIGH     → red     (0, 0, 220)
  CRITICAL → magenta (200, 0, 200)
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..models.artifacts import ArtifactReport
from ..models.enums import ArtifactSeverity, QualityGrade
from ..models.metrics import FullReferenceScores, NoReferenceScores, QualityMetrics
from ..models.report import ComparisonReport

log = logging.getLogger(__name__)

# BGR severity colors
_SEVERITY_COLOR_BGR: dict[ArtifactSeverity, tuple[int, int, int]] = {
    ArtifactSeverity.NONE: (0, 200, 0),
    ArtifactSeverity.LOW: (0, 220, 220),
    ArtifactSeverity.MEDIUM: (0, 140, 255),
    ArtifactSeverity.HIGH: (0, 0, 220),
    ArtifactSeverity.CRITICAL: (200, 0, 200),
}

# Grade colors (BGR)
_GRADE_COLOR: dict[QualityGrade, tuple[int, int, int]] = {
    QualityGrade.PASS: (0, 200, 0),
    QualityGrade.WARNING: (0, 200, 255),
    QualityGrade.FAIL: (0, 0, 220),
}

_PANEL_WIDTH = 380    # width of the right-side metrics panel
_FONT_SCALE = 0.45
_FONT_THICKNESS = 1
_LINE_HEIGHT = 18


class AnnotationRenderer:
    def render(
        self,
        report: ComparisonReport,
        img_bgr: np.ndarray,
        diff_heatmap: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Render the annotated image.

        Returns a BGR numpy array ready to be saved with cv2.imwrite or PIL.
        """
        canvas = img_bgr.copy()

        # 1. Overlay diff heatmap (40% alpha) if available
        if diff_heatmap is not None:
            canvas = self._overlay_heatmap(canvas, diff_heatmap)

        # 2. Draw artifact bounding boxes
        canvas = self._draw_artifacts(canvas, report.artifacts)

        # 3. Draw overall grade banner at top
        canvas = self._draw_grade_banner(canvas, report.overall_grade)

        # 4. Append right-side metrics panel
        panel = self._build_metrics_panel(report, canvas.shape[0])
        canvas = np.hstack([canvas, panel])

        return canvas

    def save(self, img_bgr: np.ndarray, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        Image.fromarray(rgb).save(str(out_path))
        log.debug("Annotated image saved to %s", out_path)

    # ── Heatmap overlay ────────────────────────────────────────────────────────

    @staticmethod
    def _overlay_heatmap(canvas: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
        if heatmap.shape[:2] != canvas.shape[:2]:
            heatmap = cv2.resize(heatmap, (canvas.shape[1], canvas.shape[0]))
        return cv2.addWeighted(canvas, 0.6, heatmap, 0.4, 0)

    # ── Artifact boxes ─────────────────────────────────────────────────────────

    @staticmethod
    def _draw_artifacts(canvas: np.ndarray, artifacts: ArtifactReport) -> np.ndarray:
        for artifact in artifacts.artifacts:
            if artifact.bbox is None:
                continue
            x, y, w, h = artifact.bbox
            color = _SEVERITY_COLOR_BGR.get(artifact.severity, (128, 128, 128))

            # Box
            cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)

            # Label background + text
            label = f"{artifact.artifact_type} [{artifact.severity.value.upper()}]"
            (lw, lh), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, _FONT_SCALE, _FONT_THICKNESS
            )
            ty = max(y - 5, lh + baseline)
            # Dark background rectangle for readability
            cv2.rectangle(
                canvas,
                (x, ty - lh - baseline - 2),
                (x + lw + 4, ty + baseline),
                (20, 20, 20),
                cv2.FILLED,
            )
            cv2.putText(
                canvas, label, (x + 2, ty - 2),
                cv2.FONT_HERSHEY_SIMPLEX, _FONT_SCALE, color, _FONT_THICKNESS,
                cv2.LINE_AA,
            )

            # Short description below box (first 60 chars)
            short_desc = artifact.description[:70]
            cv2.putText(
                canvas, short_desc, (x + 2, y + h + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA,
            )
        return canvas

    # ── Grade banner ───────────────────────────────────────────────────────────

    @staticmethod
    def _draw_grade_banner(canvas: np.ndarray, grade: QualityGrade) -> np.ndarray:
        banner_h = 28
        color = _GRADE_COLOR.get(grade, (128, 128, 128))
        banner = np.zeros((banner_h, canvas.shape[1], 3), dtype=np.uint8)
        banner[:] = color

        label = f"  OVERALL: {grade.value.upper()}"
        cv2.putText(
            banner, label, (4, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA,
        )
        return np.vstack([banner, canvas])

    # ── Metrics panel ──────────────────────────────────────────────────────────

    def _build_metrics_panel(
        self, report: ComparisonReport, img_height: int
    ) -> np.ndarray:
        """Build a dark sidebar with all metric values and pass/fail indicators."""
        panel_h = img_height  # img_height already includes the 28px grade banner
        panel = np.full((panel_h, _PANEL_WIDTH, 3), 30, dtype=np.uint8)

        lines: list[tuple[str, Optional[tuple[int, int, int]]]] = []
        ref = report.ref_quality_metrics  # None in analyze mode, populated in compare mode

        # Header reflects mode
        if ref is not None:
            lines.append(("=== DUT vs REFERENCE ===", (200, 200, 200)))
        else:
            lines.append(("=== QUALITY METRICS ===", (200, 200, 200)))
        lines.append(("", None))

        # Standard quality metrics
        qm = report.quality_metrics
        lines.append(("-- Image Quality --", (150, 150, 150)))

        if ref is not None:
            # Compare mode: show DUT / REF / Δ for each metric
            self._add_compare_line(lines, "Sharpness", qm.blur_score, ref.blur_score, "%.0f", higher_better=True)
            self._add_compare_line(lines, "Noise Sigma", qm.noise_sigma, ref.noise_sigma, "%.2f", higher_better=False)
            self._add_compare_line(lines, "Exposure", qm.exposure_mean, ref.exposure_mean, "%.1f L*", higher_better=None)
            self._add_compare_line(lines, "Highlight Clip%", qm.highlight_clipping_pct, ref.highlight_clipping_pct, "%.2f%%", higher_better=False)
            self._add_compare_line(lines, "Shadow Clip%", qm.shadow_clipping_pct, ref.shadow_clipping_pct, "%.2f%%", higher_better=False)
            self._add_compare_line(lines, "Saturation", qm.saturation_mean, ref.saturation_mean, "%.3f", higher_better=None)
            self._add_compare_line(lines, "Dynamic Range", qm.dynamic_range_stops, ref.dynamic_range_stops, "%.1f EV", higher_better=True)
            self._add_compare_line(lines, "Color Cast R", qm.color_cast_r, ref.color_cast_r, "%+.1f", higher_better=None)
            self._add_compare_line(lines, "Color Cast G", qm.color_cast_g, ref.color_cast_g, "%+.1f", higher_better=None)
            self._add_compare_line(lines, "Color Cast B", qm.color_cast_b, ref.color_cast_b, "%+.1f", higher_better=None)
            self._add_compare_line(lines, "WB Deviation", qm.white_balance_deviation, ref.white_balance_deviation, "%.3f", higher_better=False)
            self._add_compare_line(lines, "Chrom.Aber.", qm.chromatic_aberration_score, ref.chromatic_aberration_score, "%.2f px", higher_better=False)
        else:
            # Analyze mode: show absolute values with threshold hints
            self._add_metric_line(lines, "Sharpness (Laplacian)", qm.blur_score, ">100", "%.1f")
            self._add_metric_line(lines, "Noise Sigma", qm.noise_sigma, "<8", "%.2f")
            self._add_metric_line(lines, "Exposure Mean", qm.exposure_mean, None, "%.1f L*")
            self._add_metric_line(lines, "Highlight Clip%", qm.highlight_clipping_pct, "<1%", "%.2f%%")
            self._add_metric_line(lines, "Shadow Clip%", qm.shadow_clipping_pct, "<1%", "%.2f%%")
            self._add_metric_line(lines, "Saturation Mean", qm.saturation_mean, None, "%.3f")
            self._add_metric_line(lines, "Dynamic Range", qm.dynamic_range_stops, None, "%.1f EV")
            self._add_metric_line(lines, "Color Cast R", qm.color_cast_r, None, "%+.1f")
            self._add_metric_line(lines, "Color Cast G", qm.color_cast_g, None, "%+.1f")
            self._add_metric_line(lines, "Color Cast B", qm.color_cast_b, None, "%+.1f")
            self._add_metric_line(lines, "WB Deviation", qm.white_balance_deviation, None, "%.3f")
            self._add_metric_line(lines, "Chrom. Aberration", qm.chromatic_aberration_score, "<2px", "%.2f px")
        lines.append(("", None))

        # NR scores
        nr = report.nr_scores
        lines.append(("-- NR-IQA Scores (DUT) --", (150, 150, 150)))
        self._add_metric_line(lines, "BRISQUE", nr.brisque, "<50", "%.1f")
        self._add_metric_line(lines, "NIQE", nr.niqe, "<6", "%.2f")
        self._add_metric_line(lines, "MUSIQ", nr.musiq, ">50", "%.1f")
        self._add_metric_line(lines, "CLIP-IQA", nr.clip_iqa, ">0.5", "%.3f")
        lines.append(("", None))

        # FR scores (if available)
        if report.fr_scores:
            fr = report.fr_scores
            lines.append(("-- Full-Reference IQA --", (150, 150, 150)))
            self._add_fr_line(lines, "PSNR", fr.psnr, "dB")
            self._add_fr_line(lines, "SSIM", fr.ssim, "")
            self._add_fr_line(lines, "MS-SSIM", fr.ms_ssim, "")
            self._add_fr_line(lines, "LPIPS", fr.lpips, "")
            self._add_fr_line(lines, "DISTS", fr.dists, "")
            lines.append(("", None))

        # Failure reasons
        if report.failure_reasons:
            lines.append(("-- FAILURE REASONS --", (100, 100, 255)))
            for reason in report.failure_reasons[:8]:  # cap at 8
                for chunk in self._wrap(reason, 42):
                    lines.append((f"  {chunk}", (100, 100, 255)))

        # Video temporal
        if report.video_temporal:
            vt = report.video_temporal
            lines.append(("", None))
            lines.append(("-- Video Temporal --", (150, 150, 150)))
            self._add_metric_line(lines, "Flicker Score", vt.flicker_score, "<0.07", "%.3f")
            self._add_metric_line(lines, "Jitter Score", vt.jitter_score, "<2.5px", "%.2f px")
            self._add_metric_line(lines, "SSIM Mean", vt.temporal_ssim_mean, ">0.85", "%.3f")
            self._add_metric_line(lines, "SSIM Std", vt.temporal_ssim_std, "<0.05", "%.3f")

        # Render lines onto panel
        y = _LINE_HEIGHT + 10
        for text, color in lines:
            if not text:
                y += 8
                continue
            c = color if color else (220, 220, 220)
            cv2.putText(
                panel, text, (8, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, c, 1, cv2.LINE_AA,
            )
            y += _LINE_HEIGHT
            if y > panel_h - _LINE_HEIGHT:
                break

        return panel

    # ── Helper methods ─────────────────────────────────────────────────────────

    @staticmethod
    def _add_compare_line(
        lines: list,
        label: str,
        dut_val: Optional[float],
        ref_val: Optional[float],
        fmt: str,
        higher_better: Optional[bool],
    ) -> None:
        """Add two-line DUT vs REF comparison entry with delta and color coding.

        higher_better=True  → positive delta is good (green), negative is bad (red)
        higher_better=False → negative delta is good (green), positive is bad (red)
        higher_better=None  → neutral; delta shown in white (no direction judgment)
        """
        # Line 1: label
        lines.append((f"  {label}:", (170, 170, 170)))
        if dut_val is None and ref_val is None:
            lines.append(("    DUT:N/A  REF:N/A", (100, 100, 100)))
            return

        dut_str = (fmt % dut_val) if dut_val is not None else "N/A"
        ref_str = (fmt % ref_val) if ref_val is not None else "N/A"

        # Compute delta and pick color
        delta_str = ""
        color: tuple[int, int, int] = (200, 200, 200)
        if dut_val is not None and ref_val is not None and ref_val != 0:
            delta = dut_val - ref_val
            pct = (delta / abs(ref_val)) * 100
            delta_str = f" Δ{delta:+.0f}({pct:+.0f}%)"
            if higher_better is True:
                color = (100, 220, 100) if delta >= 0 else (80, 80, 220)
            elif higher_better is False:
                color = (100, 220, 100) if delta <= 0 else (80, 80, 220)
            # higher_better=None → keep neutral color (200,200,200)

        lines.append((f"    DUT:{dut_str}  REF:{ref_str}{delta_str}", color))

    @staticmethod
    def _add_metric_line(
        lines: list,
        label: str,
        value: Optional[float],
        hint: Optional[str],
        fmt: str,
    ) -> None:
        if value is None:
            lines.append((f"  {label}: N/A", (100, 100, 100)))
            return
        text = f"  {label}: {fmt % value}"
        if hint:
            text += f" ({hint})"
        lines.append((text, (220, 220, 220)))

    @staticmethod
    def _add_fr_line(
        lines: list,
        label: str,
        metric,
        unit: str,
    ) -> None:
        if metric is None or metric.value is None:
            lines.append((f"  {label}: N/A", (100, 100, 100)))
            return
        passed = metric.passed
        color: tuple[int, int, int]
        if passed is True:
            color = (100, 220, 100)
            status = "PASS"
        elif passed is False:
            color = (100, 100, 220)
            status = "FAIL"
        else:
            color = (200, 200, 200)
            status = ""
        text = f"  {label}: {metric.value:.4f} {unit} {status}"
        lines.append((text, color))

    @staticmethod
    def _wrap(text: str, width: int) -> list[str]:
        words = text.split()
        lines: list[str] = []
        current = ""
        for word in words:
            if len(current) + len(word) + 1 <= width:
                current = (current + " " + word).strip()
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines or [text[:width]]
