from __future__ import annotations

"""
ReferenceComparator — full-reference IQA between a golden reference and DUT image.

Pipeline:
  1. Spatial alignment (SIFT + homography) — warp reference to DUT space
  2. Compute FR metrics: PSNR, SSIM, MS-SSIM, LPIPS, DISTS
  3. Generate difference heatmap
  4. Evaluate each metric against configured thresholds
"""

import logging
from functools import lru_cache
from typing import Optional

import cv2
import numpy as np

from ..models.metrics import FullReferenceScores, MetricResult
from ..config import get_settings

log = logging.getLogger(__name__)


class _FRModelCache:
    def __init__(self) -> None:
        self._models: dict[str, object] = {}

    def get(self, name: str, device: str):
        key = f"{name}:{device}"
        if key not in self._models:
            try:
                import pyiqa
                log.info("Loading FR metric '%s' on device '%s'...", name, device)
                self._models[key] = pyiqa.create_metric(name, device=device)
            except Exception as exc:
                log.warning("Cannot load FR metric '%s': %s", name, exc)
                self._models[key] = None
        return self._models[key]


_fr_cache = _FRModelCache()


def _bgr_to_tensor(img_bgr: np.ndarray, device: str):
    import torch
    import torchvision.transforms.functional as TF
    from PIL import Image
    rgb = img_bgr[:, :, ::-1].copy()
    pil = Image.fromarray(rgb)
    return TF.to_tensor(pil).unsqueeze(0).to(device)


class ReferenceComparator:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._device = self._settings.resolve_device()

    def preload(self) -> None:
        for name in self._settings.fr_metrics_list:
            if name not in ("psnr", "ssim", "ms_ssim"):  # these are computed natively
                _fr_cache.get(name, self._device)

    def compare(
        self, ref_bgr: np.ndarray, dut_bgr: np.ndarray
    ) -> tuple[FullReferenceScores, Optional[np.ndarray]]:
        """
        Compare reference and DUT images.
        Returns (FullReferenceScores, diff_heatmap_bgr).
        """
        ref_aligned, dut_resized = self._align_and_resize(ref_bgr, dut_bgr)
        fr = FullReferenceScores()
        metrics = self._settings.fr_metrics_list

        # ── PSNR ──────────────────────────────────────────────────────────────
        if "psnr" in metrics:
            try:
                val = float(cv2.PSNR(ref_aligned, dut_resized))
                fr.psnr = MetricResult(
                    value=val,
                    threshold=self._settings.psnr_threshold,
                    higher_is_better=True,
                )
                fr.psnr.evaluate()
            except Exception as exc:
                log.warning("PSNR computation failed: %s", exc)

        # ── SSIM ──────────────────────────────────────────────────────────────
        if "ssim" in metrics:
            try:
                from skimage.metrics import structural_similarity as ssim_fn
                ref_gray = cv2.cvtColor(ref_aligned, cv2.COLOR_BGR2GRAY)
                dut_gray = cv2.cvtColor(dut_resized, cv2.COLOR_BGR2GRAY)
                val = float(ssim_fn(ref_gray, dut_gray, data_range=255))
                fr.ssim = MetricResult(
                    value=val,
                    threshold=self._settings.ssim_threshold,
                    higher_is_better=True,
                )
                fr.ssim.evaluate()
            except Exception as exc:
                log.warning("SSIM computation failed: %s", exc)

        # ── MS-SSIM via pyiqa ──────────────────────────────────────────────────
        if "ms_ssim" in metrics:
            val = self._run_pyiqa("ms_ssim", ref_aligned, dut_resized)
            if val is not None:
                fr.ms_ssim = MetricResult(
                    value=val,
                    threshold=self._settings.ssim_threshold,
                    higher_is_better=True,
                )
                fr.ms_ssim.evaluate()

        # ── LPIPS ─────────────────────────────────────────────────────────────
        if "lpips" in metrics:
            val = self._run_pyiqa("lpips", ref_aligned, dut_resized)
            if val is not None:
                fr.lpips = MetricResult(
                    value=val,
                    threshold=self._settings.lpips_threshold,
                    higher_is_better=False,
                )
                fr.lpips.evaluate()

        # ── DISTS ─────────────────────────────────────────────────────────────
        if "dists" in metrics:
            val = self._run_pyiqa("dists", ref_aligned, dut_resized)
            if val is not None:
                fr.dists = MetricResult(
                    value=val,
                    threshold=self._settings.dists_threshold,
                    higher_is_better=False,
                )
                fr.dists.evaluate()

        # ── Diff heatmap ───────────────────────────────────────────────────────
        diff_map = self._generate_diff_heatmap(ref_aligned, dut_resized)

        return fr, diff_map

    # ── Private ────────────────────────────────────────────────────────────────

    def _align_and_resize(
        self, ref: np.ndarray, dut: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Spatially align ref to dut via SIFT + homography, then match resolution."""
        # Ensure same size (resize ref to dut size if needed)
        if ref.shape[:2] != dut.shape[:2]:
            ref = cv2.resize(ref, (dut.shape[1], dut.shape[0]), interpolation=cv2.INTER_LANCZOS4)

        # Attempt SIFT-based alignment
        ref_gray = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        dut_gray = cv2.cvtColor(dut, cv2.COLOR_BGR2GRAY)

        sift = cv2.SIFT_create()
        kp_ref, des_ref = sift.detectAndCompute(ref_gray, None)
        kp_dut, des_dut = sift.detectAndCompute(dut_gray, None)

        if des_ref is None or des_dut is None or len(kp_ref) < 20 or len(kp_dut) < 20:
            log.debug("Not enough keypoints for alignment; skipping warp.")
            return ref, dut

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        matches = bf.knnMatch(des_ref, des_dut, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]

        if len(good) < 10:
            log.debug("Insufficient good matches (%d); skipping warp.", len(good))
            return ref, dut

        src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_dut[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if H is None:
            return ref, dut

        inliers = int(mask.sum()) if mask is not None else 0
        if inliers < 8:
            return ref, dut

        h, w = dut.shape[:2]
        ref_warped = cv2.warpPerspective(ref, H, (w, h))
        log.debug("Alignment: %d inliers; warp applied.", inliers)
        return ref_warped, dut

    def _run_pyiqa(
        self, name: str, ref: np.ndarray, dut: np.ndarray
    ) -> Optional[float]:
        model = _fr_cache.get(name, self._device)
        if model is None:
            return None
        try:
            import torch
            t_ref = _bgr_to_tensor(ref, self._device)
            t_dut = _bgr_to_tensor(dut, self._device)
            with torch.no_grad():
                score = model(t_dut, t_ref)
            return float(score.item() if hasattr(score, "item") else score)
        except Exception as exc:
            log.warning("pyiqa FR metric '%s' failed: %s", name, exc)
            return None

    @staticmethod
    def _generate_diff_heatmap(ref: np.ndarray, dut: np.ndarray) -> np.ndarray:
        """Generate a JET-colormap heatmap of absolute pixel differences."""
        diff = cv2.absdiff(ref, dut)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        # Normalize to 0-255 using 99th percentile to avoid outlier domination
        p99 = float(np.percentile(diff_gray, 99))
        if p99 > 0:
            diff_gray = np.clip(diff_gray.astype(np.float32) * 255.0 / p99, 0, 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(diff_gray, cv2.COLORMAP_JET)
        return heatmap
