from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from .enums import ArtifactSeverity, max_severity


class ArtifactInstance(BaseModel):
    """A single detected artifact with its location and human-readable context."""
    artifact_type: str                          # e.g. "noise_patch", "banding", "lens_flare"
    severity: ArtifactSeverity
    bbox: Optional[tuple[int, int, int, int]] = Field(None, exclude=True)  # used internally for annotation rendering; excluded from JSON output
    confidence: float = Field(1.0, ge=0.0, le=1.0)
    description: str = ""                       # human-readable explanation + fix hint


class ArtifactReport(BaseModel):
    """Aggregated artifact detection results for one media file."""
    artifacts: list[ArtifactInstance] = Field(default_factory=list)
    overall_severity: ArtifactSeverity = ArtifactSeverity.NONE

    def model_post_init(self, __context: object) -> None:
        self._recompute_overall()

    def _recompute_overall(self) -> None:
        if not self.artifacts:
            self.overall_severity = ArtifactSeverity.NONE
        else:
            self.overall_severity = max_severity(*[a.severity for a in self.artifacts])

    def add(self, artifact: ArtifactInstance) -> None:
        self.artifacts.append(artifact)
        self._recompute_overall()

    def failure_reasons(self) -> list[str]:
        reasons: list[str] = []
        for a in self.artifacts:
            if a.severity in (ArtifactSeverity.HIGH, ArtifactSeverity.CRITICAL):
                reasons.append(f"[{a.artifact_type.upper()}] {a.description}")
        return reasons

    @property
    def has_failures(self) -> bool:
        return self.overall_severity in (ArtifactSeverity.HIGH, ArtifactSeverity.CRITICAL)
