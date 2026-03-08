from __future__ import annotations

"""CameraModeDetector — EXIF extraction, camera-mode inference, and
mode-aware threshold adjustment.

## Why this matters

Smartphone cameras apply radically different processing depending on the
shooting mode:

  Portrait  — intentional background bokeh; blur_score will be low for
               background regions even on a perfectly focused shot.
  Night     — sensor integrates for hundreds of milliseconds at ISO 3200+;
               elevated noise and slight subject motion are expected.
  Sport     — fast shutter freezes action but at the cost of higher ISO and
               sometimes aggressive noise reduction that smears fine detail.
  Macro     — very shallow depth of field; almost everything except the
               subject plane is intentionally out of focus.
  HDR       — multiple exposures merged; highlight/shadow clipping thresholds
               need to be relaxed.
  Panorama  — stitched from many frames; minor sharpness loss at seams is
               normal.

Without mode awareness, all of the above would produce false FAIL verdicts
in NR analysis mode.  In compare mode the issue is less severe (the REF is
captured in the same mode), but metadata differences (e.g. DUT uses Night
while REF was captured in Auto) still explain quality gaps and must be
surfaced.

## Detection strategy (priority order)

1. Text search in UserComment / ImageDescription / XPComment EXIF fields —
   many OEMs write the mode name here (e.g. "PORTRAIT", "Night Sight").
2. EXIF SceneCaptureType tag (0=Standard, 1=Landscape, 2=Portrait, 3=Night).
3. EXIF ExposureProgram tag (7=Portrait, 8=Landscape).
4. Heuristic from ISO + ExposureTime — high ISO + long exposure → Night.
5. Default → AUTO (no threshold adjustment).

## Threshold adjustment

Mode adjustments are applied as multiplicative factors on the currently
active settings (which may already reflect a quality profile).  The pipeline
pushes these adjusted settings into the _request_settings ContextVar so that
all downstream services (QualityMetrics.failure_reasons, ArtifactDetector,
etc.) automatically see the mode-corrected thresholds for the duration of the
analysis — without any global state mutation.
"""

import logging
from pathlib import Path
from typing import Optional

from ..models.enums import CameraMode
from ..models.metadata import MediaMetadata

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Mode → threshold multiplier map
#
# Values < 1.0 on blur_threshold / highlight/shadow thresholds make the check
# *more lenient* (because a lower minimum score is required to pass).
# Values > 1.0 on noise_threshold / clip thresholds make the check more
# lenient (because a higher sigma is allowed before failing).
# ─────────────────────────────────────────────────────────────────────────────
_MODE_THRESHOLD_ADJUSTMENTS: dict[CameraMode, dict[str, float]] = {
    CameraMode.PORTRAIT: {
        # Background bokeh is intentional — only flag extreme overall unsharpness
        "blur_threshold":              0.15,
        # Subject is usually close and lit; slight noise increase is acceptable
        "noise_threshold":             1.5,
    },
    CameraMode.NIGHT: {
        # Long-exposure subject movement or hand-held shake tolerated
        "blur_threshold":              0.30,
        # High ISO is mandatory for night; noise up to 4× normal limit accepted
        "noise_threshold":             4.0,
        # Night scene HDR often clips highlights (streetlights, neon signs)
        "highlight_clip_threshold":    3.0,
        # Deep shadows are intentional in night photography
        "shadow_clip_threshold":       2.0,
    },
    CameraMode.SPORT: {
        # Fast shutter / burst — some motion blur is acceptable
        "blur_threshold":              0.30,
        # Higher ISO to achieve fast shutter speed
        "noise_threshold":             2.0,
    },
    CameraMode.MACRO: {
        # Very shallow depth of field — background is always blurry
        "blur_threshold":              0.15,
        # Close subjects often require higher ISO (limited working distance)
        "noise_threshold":             1.3,
    },
    CameraMode.HDR: {
        # HDR merge may leave minor tone-mapping artefacts at extreme ends
        "highlight_clip_threshold":    3.0,
        "shadow_clip_threshold":       3.0,
    },
    CameraMode.PANORAMA: {
        # Stitching seams can locally reduce sharpness
        "blur_threshold":              0.50,
        # Exposure blending may increase apparent noise
        "noise_threshold":             1.5,
    },
    # AUTO, LANDSCAPE, PRO, UNKNOWN, VIDEO → no adjustment (implicit factor 1.0)
}

# Keywords to search for inside EXIF text fields (case-insensitive)
_MODE_KEYWORDS: dict[CameraMode, list[str]] = {
    CameraMode.PORTRAIT: [
        "portrait", "bokeh", "depth effect", "live focus",
        "portrait mode", "portrait lighting", "focus mode",
    ],
    CameraMode.NIGHT: [
        "night", "nightmode", "night mode", "nightsight", "night sight",
        "low light", "lowlight", "dark", "astrophoto",
    ],
    CameraMode.SPORT: [
        "sport", "sports", "action", "burst", "continuous shot",
        "fast capture",
    ],
    CameraMode.MACRO: [
        "macro", "close-up", "closeup", "close up",
    ],
    CameraMode.HDR: [
        "hdr", "high dynamic range", "rich tone",
    ],
    CameraMode.PANORAMA: [
        "panorama", "pano", "wide angle", "panoramic", "stitch",
    ],
    CameraMode.PRO: [
        "pro", "manual", "expert", "pro mode",
    ],
}

# Video file extensions — EXIF is not applicable
_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".ts", ".mts", ".webm"}


class CameraModeDetector:
    """Extract EXIF metadata from an image file and detect the camera shooting
    mode.  Also computes mode-adjusted settings for downstream analysis."""

    def detect(self, path: Path) -> MediaMetadata:
        """Return a populated MediaMetadata for *path*.

        For video files a skeletal object with camera_mode=VIDEO is returned
        immediately (video containers don't carry standard EXIF).
        """
        meta = MediaMetadata()

        if path.suffix.lower() in _VIDEO_EXTENSIONS:
            meta.camera_mode = CameraMode.VIDEO
            meta.camera_mode_source = "media_type"
            return meta

        try:
            from PIL import Image
            from PIL.ExifTags import TAGS

            with Image.open(path) as img:
                exif_raw = img._getexif()  # type: ignore[attr-defined]

            if not exif_raw:
                log.debug("No EXIF data found in %s", path.name)
                return meta

            exif: dict = {TAGS.get(tag_id, tag_id): val for tag_id, val in exif_raw.items()}
            self._populate(meta, exif)

        except Exception as exc:
            log.debug("EXIF extraction failed for %s: %s", path.name, exc)

        return meta

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _populate(self, meta: MediaMetadata, exif: dict) -> None:
        """Fill all fields of *meta* from the already-decoded EXIF dict."""

        # Device identity
        meta.make    = self._str(exif.get("Make"))
        meta.model   = self._str(exif.get("Model"))
        meta.software= self._str(exif.get("Software"))
        meta.datetime_original = self._str(exif.get("DateTimeOriginal")) or \
                                 self._str(exif.get("DateTime"))

        # Exposure time
        et = exif.get("ExposureTime")
        if et is not None:
            et_s = self._to_float(et)
            meta.exposure_time_s = et_s
            meta.exposure_time   = self._format_exposure(et_s) if et_s is not None else None

        # Aperture
        fn = exif.get("FNumber")
        if fn is not None:
            meta.f_number = self._to_float(fn)

        # ISO
        iso = exif.get("ISOSpeedRatings") or exif.get("ISO") or exif.get("PhotographicSensitivity")
        if iso is not None:
            try:
                meta.iso = int(iso)
            except (TypeError, ValueError):
                pass

        # Focal lengths
        fl = exif.get("FocalLength")
        if fl is not None:
            meta.focal_length_mm = self._to_float(fl)
        fl35 = exif.get("FocalLengthIn35mmFilm")
        if fl35 is not None:
            try:
                meta.focal_length_35mm = int(fl35)
            except (TypeError, ValueError):
                pass

        # Flash
        flash = exif.get("Flash")
        if flash is not None:
            try:
                meta.flash_fired = bool(int(flash) & 0x01)  # bit 0 = fired
            except (TypeError, ValueError):
                pass

        # Scene-type tags
        sct = exif.get("SceneCaptureType")
        if sct is not None:
            try:
                meta.scene_capture_type = int(sct)
            except (TypeError, ValueError):
                pass
        ep = exif.get("ExposureProgram")
        if ep is not None:
            try:
                meta.exposure_program = int(ep)
            except (TypeError, ValueError):
                pass
        mm = exif.get("MeteringMode")
        if mm is not None:
            try:
                meta.metering_mode = int(mm)
            except (TypeError, ValueError):
                pass

        # Mode detection
        mode, confidence, source, notes = self._detect_mode(exif, meta)
        meta.camera_mode            = mode
        meta.camera_mode_confidence = confidence
        meta.camera_mode_source     = source
        meta.mode_notes             = notes

    def _detect_mode(
        self, exif: dict, meta: MediaMetadata
    ) -> tuple[CameraMode, float, str, list[str]]:
        notes: list[str] = []

        # ── 1. Search text fields ────────────────────────────────────────────
        text_blob = " ".join([
            self._decode_user_comment(exif.get("UserComment", b"")),
            self._str(exif.get("ImageDescription")) or "",
            self._decode_xp(exif.get("XPComment", b"")),
            self._str(exif.get("XPSubject")) or "",
        ]).lower()

        for mode, keywords in _MODE_KEYWORDS.items():
            if any(kw in text_blob for kw in keywords):
                notes.append(
                    f"Mode '{mode.value}' detected from image metadata text field."
                )
                return mode, 0.90, "user_comment", notes

        # ── 2. EXIF SceneCaptureType ─────────────────────────────────────────
        _scene_map = {1: CameraMode.LANDSCAPE, 2: CameraMode.PORTRAIT, 3: CameraMode.NIGHT}
        if meta.scene_capture_type in _scene_map:
            mode = _scene_map[meta.scene_capture_type]
            notes.append(
                f"Mode '{mode.value}' from EXIF SceneCaptureType={meta.scene_capture_type}."
            )
            return mode, 0.85, "exif_scene_type", notes

        # ── 3. EXIF ExposureProgram ──────────────────────────────────────────
        _prog_map = {7: CameraMode.PORTRAIT, 8: CameraMode.LANDSCAPE}
        if meta.exposure_program in _prog_map:
            mode = _prog_map[meta.exposure_program]
            notes.append(
                f"Mode '{mode.value}' from EXIF ExposureProgram={meta.exposure_program}."
            )
            return mode, 0.80, "exif_exposure_program", notes

        # ── 4. Heuristic from ISO + exposure time ────────────────────────────
        iso = meta.iso or 0
        exp_s = meta.exposure_time_s or 0.0

        if iso >= 3200 and exp_s >= 0.125:        # >= 1/8 s
            notes.append(
                f"Night mode inferred: ISO={iso}, shutter={meta.exposure_time}s."
            )
            return CameraMode.NIGHT, 0.70, "heuristic", notes

        if iso >= 1600 and exp_s >= 1.0 / 30:     # >= 1/30 s
            notes.append(
                f"Night mode inferred (moderate): ISO={iso}, shutter={meta.exposure_time}s."
            )
            return CameraMode.NIGHT, 0.55, "heuristic", notes

        if iso >= 800 and exp_s > 0 and exp_s < 1.0 / 250:  # < 1/250 s
            notes.append(
                f"Sport/action mode inferred: fast shutter ({meta.exposure_time}s), ISO={iso}."
            )
            return CameraMode.SPORT, 0.55, "heuristic", notes

        return CameraMode.AUTO, 0.50, "default", notes

    # ─────────────────────────────────────────────────────────────────────────
    # Threshold adjustment
    # ─────────────────────────────────────────────────────────────────────────

    def apply_mode_adjustments(self, mode: CameraMode, base_settings):  # type: ignore[no-untyped-def]
        """Return a Settings copy with thresholds multiplied by mode factors.

        The base_settings object already reflects any quality-profile override
        set by the API route (via ContextVar).  We multiply the relevant
        threshold fields on top of that — so quality profile + camera mode
        compose correctly.

        If the mode has no adjustments (AUTO, LANDSCAPE, PRO, UNKNOWN, VIDEO)
        the original *base_settings* object is returned unchanged (no copy).
        """
        adjustments = _MODE_THRESHOLD_ADJUSTMENTS.get(mode)
        if not adjustments:
            return base_settings

        overrides = {
            field: getattr(base_settings, field) * factor
            for field, factor in adjustments.items()
        }
        # model_copy creates a shallow copy via model_construct — does NOT
        # re-run model_post_init, so the quality-profile preset is preserved.
        adjusted = base_settings.model_copy(update=overrides)
        log.debug(
            "Mode '%s' threshold adjustments applied: %s",
            mode.value,
            {k: f"{v:.4f}" for k, v in overrides.items()},
        )
        return adjusted

    # ─────────────────────────────────────────────────────────────────────────
    # Low-level EXIF decoding helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _to_float(val) -> Optional[float]:
        if val is None:
            return None
        try:
            return float(val)
        except (TypeError, ValueError, ZeroDivisionError):
            if isinstance(val, tuple) and len(val) == 2:
                num, den = val
                return float(num) / float(den) if den else None
        return None

    @staticmethod
    def _format_exposure(seconds: float) -> str:
        """Return a human-readable exposure string ('1/100', '2.5', etc.)."""
        if seconds <= 0:
            return "0"
        if seconds < 1.0:
            n = round(1.0 / seconds)
            return f"1/{n}"
        return f"{seconds:.2f}".rstrip("0").rstrip(".")

    @staticmethod
    def _str(val) -> Optional[str]:
        if val is None:
            return None
        s = str(val).strip()
        return s if s else None

    @staticmethod
    def _decode_user_comment(val) -> str:
        """Decode EXIF UserComment byte string (charset prefix aware)."""
        if not isinstance(val, (bytes, bytearray)):
            return str(val) if val else ""
        if len(val) > 8:
            charset = val[:8].rstrip(b"\x00").decode("ascii", errors="ignore").strip()
            content = val[8:]
            if charset == "UNICODE":
                return content.decode("utf-16", errors="ignore")
            if charset in ("ASCII", ""):
                return content.decode("ascii", errors="ignore")
            return content.decode("utf-8", errors="ignore")
        return val.decode("utf-8", errors="ignore")

    @staticmethod
    def _decode_xp(val) -> str:
        """Decode Windows XP EXIF comment (UTF-16-LE)."""
        if not isinstance(val, (bytes, bytearray)):
            return str(val) if val else ""
        return val.decode("utf-16-le", errors="ignore")
