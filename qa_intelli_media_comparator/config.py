from __future__ import annotations

from contextvars import ContextVar
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# Quality profile presets
# ---------------------------------------------------------------------------
# Choose via QIMC_QUALITY_PROFILE=low|medium|high|critical in .env
# When a profile is set it overrides all individual threshold settings below.
# Leave QIMC_QUALITY_PROFILE empty (or unset) to control thresholds manually.
#
# Profile guide:
#   low      — unstable rig / outdoor / handheld; catches regressions only
#   medium   — semi-stable indoor / lightbox with minor vibration
#   high     — stable lightbox + tripod; default for controlled QA
#   critical — robotic / pixel-aligned rig; professional studio
# ---------------------------------------------------------------------------
_PROFILES: dict[str, dict] = {
    "low": {
        # FR-IQA  (camera movement / rig instability tolerated)
        "ssim_threshold":                0.35,
        "psnr_threshold":               8.0, #12.0,
        "lpips_threshold":               0.60,
        "dists_threshold":               0.50,
        # Standard quality
        "blur_threshold":               40.0,
        "noise_threshold":              20.0,
        "highlight_clip_threshold":      5.0,
        "shadow_clip_threshold":         5.0,
        # Artifact detector — relaxed for outdoor / unstable capture
        "artifact_hot_pixel_high_pct":   0.05,   # 5%  before HIGH
        "artifact_lens_flare_high_count": 30,    # >30 distinct halos = HIGH
        "artifact_banding_ratio_high":   0.75,   # very strong banding = HIGH
        "artifact_blurry_high_pct":      0.80,   # 80% blurry tiles = HIGH
    },
    "medium": {
        "ssim_threshold":                0.60,
        "psnr_threshold":               20.0,
        "lpips_threshold":               0.40,
        "dists_threshold":               0.35,
        "blur_threshold":               70.0,
        "noise_threshold":              12.0,
        "highlight_clip_threshold":      2.0,
        "shadow_clip_threshold":         2.0,
        "artifact_hot_pixel_high_pct":   0.01,
        "artifact_lens_flare_high_count":  8,
        "artifact_banding_ratio_high":   0.65,
        "artifact_blurry_high_pct":      0.65,
    },
    "high": {
        "ssim_threshold":                0.80,
        "psnr_threshold":               28.0,
        "lpips_threshold":               0.20,
        "dists_threshold":               0.18,
        "blur_threshold":              100.0,
        "noise_threshold":               8.0,
        "highlight_clip_threshold":      1.0,
        "shadow_clip_threshold":         1.0,
        "artifact_hot_pixel_high_pct":   0.001,
        "artifact_lens_flare_high_count":  3,
        "artifact_banding_ratio_high":   0.50,
        "artifact_blurry_high_pct":      0.50,
    },
    "critical": {
        "ssim_threshold":                0.92,
        "psnr_threshold":               35.0,
        "lpips_threshold":               0.08,
        "dists_threshold":               0.08,
        "blur_threshold":              150.0,
        "noise_threshold":               4.0,
        "highlight_clip_threshold":      0.5,
        "shadow_clip_threshold":         0.5,
        "artifact_hot_pixel_high_pct":   0.0005,
        "artifact_lens_flare_high_count":  2,
        "artifact_banding_ratio_high":   0.40,
        "artifact_blurry_high_pct":      0.35,
    },
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="QIMC_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Analysis mode ──────────────────────────────────────────────────────
    analysis_mode: str = Field(
        "quality",
        description=(
            "Default analysis depth for all requests. "
            "'functional' — fast path (< 100ms): functional validity checks + basic "
            "quality metrics; skips all neural IQA (BRISQUE, NIQE, LPIPS, DISTS). "
            "Ideal for camera functional test automation loops. "
            "'quality' — full path (current behaviour): all IQA metrics + functional checks. "
            "Can be overridden per-request via the 'analysis_mode' form parameter."
        ),
    )

    # ── Quality profile (overrides all thresholds when set) ────────────────
    quality_profile: str = Field(
        "",
        description="Preset: low | medium | high | critical. "
                    "Leave empty to set thresholds manually.",
    )

    # ── Compute ────────────────────────────────────────────────────────────
    device: str = Field("auto", description="'auto', 'cuda', or 'cpu'")

    # ── No-Reference IQA ───────────────────────────────────────────────────
    nr_metrics: str = Field("brisque,niqe")
    use_neural_nr: bool = Field(False)
    neural_nr_metric: str = Field("musiq")

    # ── Full-Reference IQA ─────────────────────────────────────────────────
    fr_metrics: str = Field("ssim,ms_ssim,psnr,lpips")

    # ── FR Quality Thresholds ──────────────────────────────────────────────
    ssim_threshold: float = Field(0.85)
    psnr_threshold: float = Field(30.0)
    lpips_threshold: float = Field(0.15)
    dists_threshold: float = Field(0.15)

    # ── Standard Quality Thresholds ────────────────────────────────────────
    blur_threshold: float = Field(100.0, description="Laplacian variance below = blurry")
    noise_threshold: float = Field(8.0, description="Noise sigma above = noisy")
    highlight_clip_threshold: float = Field(1.0, description="% blown highlights")
    shadow_clip_threshold: float = Field(1.0, description="% crushed blacks")

    # ── Artifact Detector Thresholds ───────────────────────────────────────
    # Fraction of pixels above local mean to classify as hot pixels (HIGH severity)
    artifact_hot_pixel_high_pct: float = Field(
        0.001, description="Fraction of hot pixels before HIGH severity (default 0.1%)"
    )
    # Number of distinct lens-flare blobs before severity escalates to HIGH
    artifact_lens_flare_high_count: int = Field(
        3, description="Flare blob count for HIGH severity"
    )
    # Banding band-energy ratio above which severity is HIGH
    artifact_banding_ratio_high: float = Field(
        0.50, description="Banding FFT ratio for HIGH severity"
    )
    # Fraction of blurry tiles (vs total) for HIGH severity
    artifact_blurry_high_pct: float = Field(
        0.50, description="Fraction of blurry tiles for HIGH severity"
    )

    # ── Video ──────────────────────────────────────────────────────────────
    video_sample_fps: float = Field(2.0)
    motion_flow_threshold: float = Field(2.0, description="Optical flow px mean; below = static")

    # ── Preview Crop ───────────────────────────────────────────────────────
    preview_crop_enabled: bool = Field(True)

    # ── Storage ────────────────────────────────────────────────────────────
    reports_dir: Path = Field(Path("./data/reports"))

    # ── Server ─────────────────────────────────────────────────────────────
    host: str = Field("0.0.0.0")
    port: int = Field(8080)
    log_level: str = Field("info")

    # ── Computed helpers ───────────────────────────────────────────────────
    @field_validator("reports_dir", mode="before")
    @classmethod
    def _make_path(cls, v: str | Path) -> Path:
        return Path(v)

    def model_post_init(self, __context: object) -> None:
        """Apply profile preset after all fields are loaded from env/defaults."""
        profile = self.quality_profile.strip().lower()
        if profile and profile in _PROFILES:
            for field_name, value in _PROFILES[profile].items():
                object.__setattr__(self, field_name, value)

    @property
    def nr_metrics_list(self) -> list[str]:
        return [m.strip() for m in self.nr_metrics.split(",") if m.strip()]

    @property
    def fr_metrics_list(self) -> list[str]:
        return [m.strip() for m in self.fr_metrics.split(",") if m.strip()]

    def resolve_device(self) -> str:
        if self.device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return self.device


# Per-request settings override (used by API routes to apply a quality_profile
# for a single request without touching the global cached singleton).
_request_settings: ContextVar[Settings | None] = ContextVar("_request_settings", default=None)


@lru_cache(maxsize=1)
def _global_settings() -> Settings:
    return Settings()


def get_settings() -> Settings:
    """Return the effective settings for the current context.

    If a per-request profile override has been set via `_request_settings`,
    that is returned; otherwise the global cached singleton is used.
    """
    override = _request_settings.get()
    return override if override is not None else _global_settings()
