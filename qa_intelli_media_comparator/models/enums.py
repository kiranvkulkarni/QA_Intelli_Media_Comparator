from enum import Enum


class CameraMode(str, Enum):
    """Detected smartphone camera shooting mode.

    Inferred from EXIF tags (SceneCaptureType, ExposureProgram, UserComment)
    and capture-setting heuristics (ISO, exposure time).  Used to apply
    mode-aware quality threshold adjustments so that intentional optical
    characteristics (portrait bokeh, night-mode noise, sport motion-blur)
    are not misreported as defects.
    """
    UNKNOWN   = "unknown"    # EXIF absent or unrecognised
    AUTO      = "auto"       # standard auto / program mode
    PORTRAIT  = "portrait"   # bokeh / depth-of-field mode — background blur intentional
    NIGHT     = "night"      # long exposure + high ISO — elevated noise expected
    SPORT     = "sport"      # fast shutter / burst — motion blur tolerated
    MACRO     = "macro"      # very shallow DoF — background always blurry
    HDR       = "hdr"        # HDR merge — highlight/shadow clipping relaxed
    PANORAMA  = "panorama"   # stitched panorama — localised blur acceptable
    LANDSCAPE = "landscape"  # landscape mode — standard expectations
    PRO       = "pro"        # manual / pro mode — user controlled, no adjustment
    VIDEO     = "video"      # video frame — no EXIF adjustment


class MediaType(str, Enum):
    IMAGE_CAPTURED = "image_captured"
    IMAGE_PREVIEW = "image_preview"
    VIDEO_STATIC = "video_static"
    VIDEO_MOTION = "video_motion"


class ArtifactSeverity(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class QualityGrade(str, Enum):
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"


class SyncMode(str, Enum):
    AUTO = "auto"
    FRAME_BY_FRAME = "frame_by_frame"


# Severity ordering for comparisons
_SEVERITY_ORDER = {
    ArtifactSeverity.NONE: 0,
    ArtifactSeverity.LOW: 1,
    ArtifactSeverity.MEDIUM: 2,
    ArtifactSeverity.HIGH: 3,
    ArtifactSeverity.CRITICAL: 4,
}


def severity_rank(s: ArtifactSeverity) -> int:
    return _SEVERITY_ORDER[s]


def max_severity(*severities: ArtifactSeverity) -> ArtifactSeverity:
    return max(severities, key=severity_rank)
