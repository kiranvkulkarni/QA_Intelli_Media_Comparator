from __future__ import annotations

"""
NoReferenceAnalyzer — computes NR-IQA scores using pyiqa.

Models are lazy-loaded and cached on first use (expensive to instantiate).

Always-on (CPU-friendly):
  - BRISQUE  : natural scene statistics (lower = better)
  - NIQE     : natural image quality evaluator (lower = better)

Optional neural (GPU recommended):
  - MUSIQ    : multi-scale transformer IQA (higher = better)
  - CLIP-IQA+: CLIP-based (higher = better)
"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

from ..models.enums import QualityGrade
from ..models.metrics import NoReferenceScores
from ..config import get_settings

log = logging.getLogger(__name__)


def _bgr_to_tensor(img_bgr: np.ndarray, device: str):
    """Convert a BGR uint8 numpy array to a normalized [0,1] RGB torch tensor."""
    import torch
    import torchvision.transforms.functional as TF
    from PIL import Image

    rgb = img_bgr[:, :, ::-1].copy()
    pil = Image.fromarray(rgb)
    tensor = TF.to_tensor(pil).unsqueeze(0).to(device)  # [1, 3, H, W]
    return tensor


class _ModelCache:
    """Thread-safe lazy loader / cache for pyiqa metric instances."""

    def __init__(self) -> None:
        self._models: dict[str, object] = {}

    def get(self, name: str, device: str):
        key = f"{name}:{device}"
        if key not in self._models:
            try:
                import pyiqa
                log.info("Loading pyiqa model '%s' on device '%s'...", name, device)
                self._models[key] = pyiqa.create_metric(name, device=device)
                log.info("Model '%s' loaded.", name)
            except Exception as exc:
                log.warning("Cannot load pyiqa model '%s': %s", name, exc)
                self._models[key] = None
        return self._models[key]

    def available(self) -> list[str]:
        return [k for k, v in self._models.items() if v is not None]


_cache = _ModelCache()


class NoReferenceAnalyzer:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._device = self._settings.resolve_device()

    def preload(self) -> None:
        """Eagerly load all configured models (call at startup)."""
        for name in self._settings.nr_metrics_list:
            _cache.get(name, self._device)
        if self._settings.use_neural_nr:
            _cache.get(self._settings.neural_nr_metric, self._device)

    def analyze(self, img_bgr: np.ndarray) -> NoReferenceScores:
        scores = NoReferenceScores()

        # Classical NR metrics
        if "brisque" in self._settings.nr_metrics_list:
            scores.brisque = self._run_metric("brisque", img_bgr)

        if "niqe" in self._settings.nr_metrics_list:
            scores.niqe = self._run_metric("niqe", img_bgr)

        # Neural NR metrics (optional)
        if self._settings.use_neural_nr:
            metric_name = self._settings.neural_nr_metric
            value = self._run_metric(metric_name, img_bgr)
            if metric_name == "musiq":
                scores.musiq = value
            elif "clip" in metric_name.lower():
                scores.clip_iqa = value

        scores.grade = self._grade(scores)
        return scores

    def loaded_models(self) -> list[str]:
        return _cache.available()

    # ── Private ────────────────────────────────────────────────────────────────

    def _run_metric(self, name: str, img_bgr: np.ndarray) -> Optional[float]:
        model = _cache.get(name, self._device)
        if model is None:
            return None
        try:
            tensor = _bgr_to_tensor(img_bgr, self._device)
            import torch
            with torch.no_grad():
                score = model(tensor)
            return float(score.item() if hasattr(score, "item") else score)
        except Exception as exc:
            log.warning("pyiqa metric '%s' failed: %s", name, exc)
            return None

    @staticmethod
    def _grade(scores: NoReferenceScores) -> QualityGrade:
        # BRISQUE: 0–100; > 60 is poor quality
        if scores.brisque is not None:
            if scores.brisque > 70:
                return QualityGrade.FAIL
            if scores.brisque > 50:
                return QualityGrade.WARNING

        # NIQE: typically 0–15; > 10 is poor
        if scores.niqe is not None:
            if scores.niqe > 10:
                return QualityGrade.FAIL
            if scores.niqe > 6:
                return QualityGrade.WARNING

        # MUSIQ: 0–100; < 40 is poor
        if scores.musiq is not None:
            if scores.musiq < 30:
                return QualityGrade.FAIL
            if scores.musiq < 50:
                return QualityGrade.WARNING

        # CLIP-IQA: 0–1; < 0.4 is poor
        if scores.clip_iqa is not None:
            if scores.clip_iqa < 0.3:
                return QualityGrade.FAIL
            if scores.clip_iqa < 0.5:
                return QualityGrade.WARNING

        return QualityGrade.PASS
