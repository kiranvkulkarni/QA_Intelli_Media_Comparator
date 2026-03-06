from enum import Enum


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
