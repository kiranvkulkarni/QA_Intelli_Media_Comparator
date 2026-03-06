from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from .enums import MediaType


class MediaInfo(BaseModel):
    path: Path
    media_type: MediaType
    width: int
    height: int
    fps: Optional[float] = None          # video only
    frame_count: Optional[int] = None    # video only
    duration_s: Optional[float] = None   # video only
    file_size_bytes: int = 0
    detection_confidence: float = Field(1.0, ge=0.0, le=1.0)


class CropResult(BaseModel):
    bbox: tuple[int, int, int, int]      # x, y, w, h (pixel coords)
    confidence: float = Field(ge=0.0, le=1.0)
    method: str                          # "contour" | "saturation_mask" | "heuristic" | "none"
    applied: bool = True
