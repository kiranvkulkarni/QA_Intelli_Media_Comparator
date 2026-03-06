from __future__ import annotations

"""
ArtifactDetector — detects and localizes specific camera artifacts in BGR images.

Detected artifacts:
  - noise_patch        : patches of high local variance in flat regions
  - banding            : periodic false contouring in gradient regions
  - lens_flare         : elongated bright blobs / halos
  - hot_pixel          : isolated bright/dark outlier pixels
  - chromatic_aberration: R-B channel spatial misalignment at edges > 2px
  - blurry_region      : local sharpness significantly below the global mean
  - posterization      : too few unique tones in smooth gradient areas
  - overexposure       : large blown-highlight region
  - underexposure      : large crushed-shadow region
"""

import logging
from typing import Optional

import cv2
import numpy as np

from ..models.artifacts import ArtifactInstance, ArtifactReport
from ..models.enums import ArtifactSeverity
from ..config import get_settings

log = logging.getLogger(__name__)

# Patch size for local analysis
_PATCH = 32


class ArtifactDetector:
    def detect(self, img_bgr: np.ndarray) -> ArtifactReport:
        self._settings = get_settings()
        report = ArtifactReport()
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

        self._detect_noise_patches(img_bgr, gray, report)
        self._detect_banding(gray, report)
        self._detect_lens_flare(gray, report)
        self._detect_hot_pixels(gray, report)
        self._detect_chromatic_aberration(img_bgr, gray, report)
        self._detect_blurry_regions(gray, report)
        self._detect_posterization(gray, report)
        self._detect_clipping(img_bgr, report)

        log.debug("ArtifactDetector: %d artifacts (overall=%s)",
                  len(report.artifacts), report.overall_severity)
        return report

    # ── Noise patches ──────────────────────────────────────────────────────────

    def _detect_noise_patches(
        self, img: np.ndarray, gray: np.ndarray, report: ArtifactReport
    ) -> None:
        h, w = gray.shape
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx ** 2 + gy ** 2)

        # Global noise estimate for severity scaling
        global_std = float(np.std(gray))
        noise_patches: list[tuple[int, int, int, int]] = []

        for y in range(0, h - _PATCH, _PATCH):
            for x in range(0, w - _PATCH, _PATCH):
                patch_gray = gray[y: y + _PATCH, x: x + _PATCH]
                patch_grad = grad_mag[y: y + _PATCH, x: x + _PATCH]
                # Only analyse flat regions (low gradient content)
                if float(np.mean(patch_grad)) > 15:
                    continue
                local_std = float(np.std(patch_gray))
                if local_std > max(12.0, global_std * 1.5):
                    noise_patches.append((x, y, _PATCH, _PATCH))

        if not noise_patches:
            return

        pct = len(noise_patches) / ((h // _PATCH) * (w // _PATCH) + 1)
        if pct < 0.02:
            return  # negligible

        severity = (
            ArtifactSeverity.CRITICAL if pct > 0.4
            else ArtifactSeverity.HIGH if pct > 0.2
            else ArtifactSeverity.MEDIUM if pct > 0.08
            else ArtifactSeverity.LOW
        )
        # Report the worst patch as representative bbox
        worst = noise_patches[0]
        report.add(ArtifactInstance(
            artifact_type="noise_patch",
            severity=severity,
            bbox=worst,
            confidence=min(1.0, pct * 3),
            description=(
                f"Excessive noise detected in {len(noise_patches)} flat region(s) "
                f"({pct * 100:.1f}% of image area). "
                "Check ISO, lighting level, or NR strength settings."
            ),
        ))

    # ── Banding ────────────────────────────────────────────────────────────────

    def _detect_banding(self, gray: np.ndarray, report: ArtifactReport) -> None:
        """Detect periodic false contouring (banding) using 1-D FFT on gradient."""
        # Work on L channel smoothed horizontally
        smoothed = cv2.GaussianBlur(gray, (1, 7), 0)
        # Horizontal gradient (per row)
        gy = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)
        row_energy = np.mean(np.abs(gy), axis=1).astype(np.float64)

        if len(row_energy) < 16:
            return

        # Banding requires gradient activity across many rows; isolated shape edges are not banding
        active_rows_pct = float(np.count_nonzero(row_energy > 1.0)) / len(row_energy)
        if active_rows_pct < 0.25:
            return

        fft = np.abs(np.fft.rfft(row_energy))
        fft[0] = 0  # remove DC
        h = gray.shape[0]
        freqs = np.fft.rfftfreq(h)

        # Banding frequency band: > 3 bands per 100 rows
        band_min = 3.0 / 100
        band_mask = freqs > band_min
        band_energy = float(fft[band_mask].sum())
        total_energy = float(fft.sum()) + 1e-9
        banding_ratio = band_energy / total_energy

        if banding_ratio < 0.15:
            return

        high_thresh = self._settings.artifact_banding_ratio_high
        severity = (
            ArtifactSeverity.HIGH if banding_ratio > high_thresh
            else ArtifactSeverity.MEDIUM if banding_ratio > high_thresh * 0.6
            else ArtifactSeverity.LOW
        )
        report.add(ArtifactInstance(
            artifact_type="banding",
            severity=severity,
            bbox=None,
            confidence=min(1.0, banding_ratio * 2),
            description=(
                f"Color banding / false contouring detected (ratio {banding_ratio:.2f}). "
                "Indicates insufficient bit depth, aggressive quantization, or "
                "HDR tone-mapping issues."
            ),
        ))

    # ── Lens flare ─────────────────────────────────────────────────────────────

    def _detect_lens_flare(self, gray: np.ndarray, report: ArtifactReport) -> None:
        h, w = gray.shape
        threshold = float(np.percentile(gray, 99.9))
        if threshold < 200:  # nothing is very bright
            return

        binary = (gray >= threshold).astype(np.uint8) * 255
        # Connect nearby bright pixels
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        binary = cv2.dilate(binary, kernel, iterations=2)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
        flares: list[tuple[int, int, int, int]] = []

        for i in range(1, num_labels):
            cx, cy, cw, ch, area = stats[i]
            if area < 50:
                continue
            ar = max(cw, ch) / (min(cw, ch) + 1)
            # Elongated blobs (streaks) or large round halos
            if ar > 2.5 or area > (h * w * 0.01):
                flares.append((cx, cy, cw, ch))

        if not flares:
            return

        high_count = self._settings.artifact_lens_flare_high_count
        severity = ArtifactSeverity.HIGH if len(flares) > high_count else ArtifactSeverity.MEDIUM
        report.add(ArtifactInstance(
            artifact_type="lens_flare",
            severity=severity,
            bbox=flares[0],
            confidence=0.85,
            description=(
                f"{len(flares)} lens flare / halo region(s) detected. "
                "Move light source out of frame or use a lens hood."
            ),
        ))

    # ── Hot pixels ─────────────────────────────────────────────────────────────

    def _detect_hot_pixels(self, gray: np.ndarray, report: ArtifactReport) -> None:
        """Detect isolated bright or dark outlier pixels."""
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        diff = np.abs(gray.astype(np.float32) - blurred.astype(np.float32))
        sigma = float(np.std(diff))
        hot_mask = diff > max(30.0, sigma * 4)

        hot_count = int(hot_mask.sum())
        total = gray.size
        if hot_count < 5:
            return

        pct = hot_count / total
        high_pct = self._settings.artifact_hot_pixel_high_pct
        severity = (
            ArtifactSeverity.HIGH if pct > high_pct
            else ArtifactSeverity.MEDIUM if pct > high_pct * 0.2
            else ArtifactSeverity.LOW
        )

        # Get centroid of hot pixels for bbox approximation
        ys, xs = np.where(hot_mask)
        cx, cy = int(np.median(xs)), int(np.median(ys))
        report.add(ArtifactInstance(
            artifact_type="hot_pixel",
            severity=severity,
            bbox=(max(0, cx - 20), max(0, cy - 20), 40, 40),
            confidence=0.9,
            description=(
                f"{hot_count} hot/dead pixels detected ({pct * 100:.4f}% of frame). "
                "May indicate sensor defect or extreme long-exposure heat."
            ),
        ))

    # ── Chromatic aberration ───────────────────────────────────────────────────

    def _detect_chromatic_aberration(
        self, img: np.ndarray, gray: np.ndarray, report: ArtifactReport
    ) -> None:
        edges = cv2.Canny(gray.astype(np.uint8), 80, 200)
        edge_pts = np.argwhere(edges > 0)
        if len(edge_pts) < 50:
            return

        r_ch = img[:, :, 2].astype(np.float32)
        b_ch = img[:, :, 0].astype(np.float32)
        h, w = gray.shape
        pad = 5

        offsets: list[float] = []
        high_offset_pts: list[tuple[int, int]] = []
        step = max(1, len(edge_pts) // 300)

        for py, px in edge_pts[::step]:
            y1, y2 = max(pad, py - pad), min(h - pad, py + pad)
            x1, x2 = max(pad, px - pad), min(w - pad, px + pad)
            pr = r_ch[y1:y2, x1:x2]
            pb = b_ch[y1:y2, x1:x2]
            if pr.size < 9:
                continue
            idx_y, idx_x = np.mgrid[0: pr.shape[0], 0: pr.shape[1]]
            sr, sb = float(pr.sum()), float(pb.sum())
            if sr < 1 or sb < 1:
                continue
            cy_r = float((pr * idx_y).sum()) / sr
            cx_r = float((pr * idx_x).sum()) / sr
            cy_b = float((pb * idx_y).sum()) / sb
            cx_b = float((pb * idx_x).sum()) / sb
            offset = float(np.sqrt((cy_r - cy_b) ** 2 + (cx_r - cx_b) ** 2))
            offsets.append(offset)
            if offset > 2.0:
                high_offset_pts.append((px, py))

        if not offsets:
            return

        mean_offset = float(np.mean(offsets))
        if mean_offset < 0.5:
            return

        severity = (
            ArtifactSeverity.HIGH if mean_offset > 4.0
            else ArtifactSeverity.MEDIUM if mean_offset > 2.0
            else ArtifactSeverity.LOW
        )

        bbox = None
        if high_offset_pts:
            xs = [p[0] for p in high_offset_pts]
            ys = [p[1] for p in high_offset_pts]
            bx, by = max(0, min(xs) - 10), max(0, min(ys) - 10)
            bw, bh = min(w, max(xs) + 10) - bx, min(h, max(ys) + 10) - by
            bbox = (bx, by, bw, bh)

        report.add(ArtifactInstance(
            artifact_type="chromatic_aberration",
            severity=severity,
            bbox=bbox,
            confidence=0.8,
            description=(
                f"Chromatic aberration: mean R-B edge offset {mean_offset:.2f} px. "
                "Check lens correction profile or optical zoom alignment."
            ),
        ))

    # ── Blurry regions ─────────────────────────────────────────────────────────

    def _detect_blurry_regions(self, gray: np.ndarray, report: ArtifactReport) -> None:
        h, w = gray.shape
        patch_size = 64
        sharpness_map: list[tuple[float, int, int]] = []

        for y in range(0, h - patch_size, patch_size):
            for x in range(0, w - patch_size, patch_size):
                patch = gray[y: y + patch_size, x: x + patch_size]
                lap_var = float(cv2.Laplacian(patch.astype(np.uint8), cv2.CV_64F).var())
                sharpness_map.append((lap_var, x, y))

        if len(sharpness_map) < 4:
            return

        values = [s[0] for s in sharpness_map]
        global_mean = float(np.mean(values))
        if global_mean < 10:
            return  # whole image is blurry; handled by quality metrics

        threshold = global_mean / 3.0
        blurry = [(v, x, y) for v, x, y in sharpness_map if v < threshold]

        if not blurry:
            return

        pct = len(blurry) / len(sharpness_map)
        if pct < 0.1:
            return

        high_pct = self._settings.artifact_blurry_high_pct
        severity = (
            ArtifactSeverity.HIGH if pct > high_pct
            else ArtifactSeverity.MEDIUM if pct > high_pct * 0.5
            else ArtifactSeverity.LOW
        )
        worst = min(blurry, key=lambda t: t[0])
        _, bx, by = worst
        report.add(ArtifactInstance(
            artifact_type="blurry_region",
            severity=severity,
            bbox=(bx, by, patch_size, patch_size),
            confidence=min(1.0, pct * 2),
            description=(
                f"Locally blurry regions covering ~{pct * 100:.0f}% of frame. "
                "Could be out-of-focus subject, motion blur, or OIS failure."
            ),
        ))

    # ── Posterization ──────────────────────────────────────────────────────────

    def _detect_posterization(self, gray: np.ndarray, report: ArtifactReport) -> None:
        """Detect false contouring / posterization in gradient regions."""
        # Find gradient-rich (smooth gradient) areas
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx ** 2 + gy ** 2)

        low_grad = (grad < 10).astype(np.uint8)  # smooth areas
        if low_grad.sum() < 1000:
            return

        smooth_pixels = gray[low_grad == 1]
        # Posterization only applies to areas with some tonal variation (gradients)
        # Completely flat regions are not gradients and don't exhibit posterization
        if len(smooth_pixels) < 1000 or float(smooth_pixels.std()) < 10.0:
            return
        # Count unique tonal values in smooth areas (normalized to 64 bins)
        hist, _ = np.histogram(smooth_pixels, bins=64, range=(0, 256))
        filled_bins = int(np.count_nonzero(hist))
        fill_ratio = filled_bins / 64.0

        # If fewer than 30% of tonal bins are filled in smooth regions → posterization
        if fill_ratio > 0.3:
            return

        severity = (
            ArtifactSeverity.HIGH if fill_ratio < 0.05
            else ArtifactSeverity.MEDIUM if fill_ratio < 0.15
            else ArtifactSeverity.LOW
        )
        report.add(ArtifactInstance(
            artifact_type="posterization",
            severity=severity,
            bbox=None,
            confidence=0.75,
            description=(
                f"Posterization / false contouring detected (tonal fill ratio {fill_ratio:.2f}). "
                "Caused by bit-depth reduction, aggressive sharpening, or JPEG over-compression."
            ),
        ))

    # ── Clipping artifacts ─────────────────────────────────────────────────────

    def _detect_clipping(self, img: np.ndarray, report: ArtifactReport) -> None:
        h, w = img.shape[:2]
        total = h * w

        # Overexposure: large blown region
        blown_mask = np.any(img > 250, axis=2)
        blown_pct = float(blown_mask.mean() * 100)
        if blown_pct > 5.0:
            ys, xs = np.where(blown_mask)
            bx, by = int(xs.min()), int(ys.min())
            bw = int(xs.max()) - bx
            bh = int(ys.max()) - by
            severity = ArtifactSeverity.CRITICAL if blown_pct > 20 else ArtifactSeverity.HIGH
            report.add(ArtifactInstance(
                artifact_type="overexposure",
                severity=severity,
                bbox=(bx, by, bw, bh),
                confidence=0.95,
                description=(
                    f"Overexposure: {blown_pct:.1f}% of pixels are blown out. "
                    "Reduce exposure compensation or enable highlight recovery."
                ),
            ))

        # Underexposure: large crushed region
        crushed_mask = np.all(img < 5, axis=2)
        crushed_pct = float(crushed_mask.mean() * 100)
        if crushed_pct > 5.0:
            ys, xs = np.where(crushed_mask)
            bx, by = int(xs.min()), int(ys.min())
            bw = int(xs.max()) - bx
            bh = int(ys.max()) - by
            severity = ArtifactSeverity.CRITICAL if crushed_pct > 20 else ArtifactSeverity.HIGH
            report.add(ArtifactInstance(
                artifact_type="underexposure",
                severity=severity,
                bbox=(bx, by, bw, bh),
                confidence=0.95,
                description=(
                    f"Underexposure: {crushed_pct:.1f}% of pixels are crushed to black. "
                    "Increase exposure or check shadow lifting / HDR mode."
                ),
            ))
