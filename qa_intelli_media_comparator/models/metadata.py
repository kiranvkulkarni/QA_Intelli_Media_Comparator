from __future__ import annotations

import math
from typing import Optional

from pydantic import BaseModel, Field

from .enums import CameraMode


class MediaMetadata(BaseModel):
    """EXIF / container metadata extracted from a media file.

    Populated by CameraModeDetector.  For video files, only camera_mode is
    set (to CameraMode.VIDEO); all other fields remain None since video
    containers do not carry standard EXIF data.
    """

    # ── Device identity ─────────────────────────────────────────────────────
    make: Optional[str] = None              # e.g. "Samsung", "Google"
    model: Optional[str] = None             # e.g. "SM-S926B", "Pixel 9 Pro"
    software: Optional[str] = None          # firmware / app version string

    # ── Capture timestamp ────────────────────────────────────────────────────
    datetime_original: Optional[str] = None  # from DateTimeOriginal EXIF tag

    # ── Exposure triangle ────────────────────────────────────────────────────
    exposure_time: Optional[str] = None      # human-readable, e.g. "1/100" or "1/4000"
    exposure_time_s: Optional[float] = None  # in seconds (float)
    f_number: Optional[float] = None         # e.g. 1.8, 2.2
    iso: Optional[int] = None               # ISO sensitivity value
    focal_length_mm: Optional[float] = None  # physical focal length
    focal_length_35mm: Optional[int] = None  # 35 mm-equivalent focal length

    # ── Scene-assist ─────────────────────────────────────────────────────────
    flash_fired: Optional[bool] = None
    metering_mode: Optional[int] = None

    # ── EXIF scene classification ─────────────────────────────────────────────
    scene_capture_type: Optional[int] = None  # 0=Standard,1=Landscape,2=Portrait,3=Night
    exposure_program: Optional[int] = None    # 0-8 per EXIF spec

    # ── Detected camera mode ─────────────────────────────────────────────────
    camera_mode: CameraMode = CameraMode.UNKNOWN
    camera_mode_confidence: float = Field(0.0, ge=0.0, le=1.0)
    camera_mode_source: str = "none"  # "user_comment" | "exif_scene_type" |
    #                                    "exif_exposure_program" | "heuristic" |
    #                                    "media_type" | "none"
    mode_notes: list[str] = Field(default_factory=list)

    @property
    def device_label(self) -> str:
        """Compact 'Make Model' string for display."""
        parts = [p for p in [self.make, self.model] if p]
        return " ".join(parts) if parts else "Unknown device"

    @property
    def exposure_summary(self) -> str:
        """Human-readable exposure summary, e.g. '1/100s  f/1.8  ISO 400'."""
        parts: list[str] = []
        if self.exposure_time:
            parts.append(f"{self.exposure_time}s")
        if self.f_number is not None:
            parts.append(f"f/{self.f_number:.1f}")
        if self.iso is not None:
            parts.append(f"ISO {self.iso}")
        return "  ".join(parts) if parts else "N/A"


# ─────────────────────────────────────────────────────────────────────────────
# DUT vs Reference metadata comparison
# ─────────────────────────────────────────────────────────────────────────────

class MetadataComparison(BaseModel):
    """Side-by-side comparison of EXIF metadata between DUT and Reference.

    Surfaces capture-setting differences that could explain measured quality
    gaps (e.g. different ISO, different camera mode, different zoom level).
    """

    modes_match: bool = True
    dut_mode: CameraMode = CameraMode.UNKNOWN
    ref_mode: CameraMode = CameraMode.UNKNOWN

    # Numeric deltas (DUT − REF, or None if either value was absent)
    iso_delta: Optional[int] = None               # positive → DUT is noisier
    exposure_delta_stops: Optional[float] = None  # log2(DUT_t / REF_t); positive → DUT longer
    f_number_match: Optional[bool] = None
    focal_length_match: Optional[bool] = None     # True if 35mm-equiv within ±5 mm

    notes: list[str] = Field(default_factory=list)

    @classmethod
    def build(cls, dut: MediaMetadata, ref: MediaMetadata) -> "MetadataComparison":
        notes: list[str] = []

        # Mode match — unknown modes are considered compatible (no data to compare)
        modes_match = (
            dut.camera_mode == ref.camera_mode
            or dut.camera_mode in (CameraMode.UNKNOWN, CameraMode.AUTO)
            or ref.camera_mode in (CameraMode.UNKNOWN, CameraMode.AUTO)
        )
        if not modes_match:
            notes.append(
                f"Camera mode mismatch: DUT={dut.camera_mode.value}, "
                f"REF={ref.camera_mode.value}.  Quality differences may reflect "
                "intentional mode behaviour rather than a defect."
            )

        # ISO delta
        iso_delta: Optional[int] = None
        if dut.iso is not None and ref.iso is not None:
            iso_delta = dut.iso - ref.iso
            if abs(iso_delta) >= 400:
                direction = "higher" if iso_delta > 0 else "lower"
                notes.append(
                    f"ISO difference: DUT={dut.iso}, REF={ref.iso} (Δ={iso_delta:+d}).  "
                    f"DUT uses {direction} ISO — "
                    + ("expect more noise on DUT." if iso_delta > 0 else "REF may be noisier.")
                )

        # Exposure time delta in stops
        exposure_delta_stops: Optional[float] = None
        if (
            dut.exposure_time_s is not None
            and ref.exposure_time_s is not None
            and ref.exposure_time_s > 0
            and dut.exposure_time_s > 0
        ):
            exposure_delta_stops = round(math.log2(dut.exposure_time_s / ref.exposure_time_s), 2)
            if abs(exposure_delta_stops) >= 1.0:
                notes.append(
                    f"Exposure time difference: DUT={dut.exposure_time}, "
                    f"REF={ref.exposure_time} ({exposure_delta_stops:+.1f} stops).  "
                    "May explain brightness or motion-blur differences."
                )

        # Aperture match
        f_number_match: Optional[bool] = None
        if dut.f_number is not None and ref.f_number is not None:
            f_number_match = abs(dut.f_number - ref.f_number) < 0.5
            if not f_number_match:
                notes.append(
                    f"Aperture difference: DUT f/{dut.f_number:.1f}, "
                    f"REF f/{ref.f_number:.1f}.  "
                    "Different apertures affect depth of field and exposure."
                )

        # Focal length match (35 mm-equiv)
        focal_length_match: Optional[bool] = None
        if dut.focal_length_35mm is not None and ref.focal_length_35mm is not None:
            focal_length_match = abs(dut.focal_length_35mm - ref.focal_length_35mm) <= 5
            if not focal_length_match:
                notes.append(
                    f"Focal length mismatch: DUT {dut.focal_length_35mm} mm, "
                    f"REF {ref.focal_length_35mm} mm (35 mm equiv).  "
                    "Different zoom levels — full-reference comparison scores may be low by design."
                )

        return cls(
            modes_match=modes_match,
            dut_mode=dut.camera_mode,
            ref_mode=ref.camera_mode,
            iso_delta=iso_delta,
            exposure_delta_stops=exposure_delta_stops,
            f_number_match=f_number_match,
            focal_length_match=focal_length_match,
            notes=notes,
        )
