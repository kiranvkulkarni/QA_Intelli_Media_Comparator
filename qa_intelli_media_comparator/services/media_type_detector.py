from __future__ import annotations

"""
MediaTypeDetector — auto-detects whether media is image/video and preview/captured.

Detection chain:
  1. Extension + magic bytes → IMAGE vs VIDEO
  2. For images  → status-bar / UI-chrome heuristics → PREVIEW vs CAPTURED
  3. For videos  → optical-flow magnitude sampling   → STATIC vs MOTION
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..models.enums import MediaType
from ..models.media import MediaInfo
from ..config import get_settings

log = logging.getLogger(__name__)

# Magic byte signatures
_IMAGE_MAGIC: list[tuple[bytes, str]] = [
    (b"\xff\xd8\xff", "jpeg"),
    (b"\x89PNG", "png"),
    (b"RIFF", "webp"),   # RIFF....WEBP
    (b"\x00\x00\x00", "heic"),  # ftyp box (checked further)
    (b"GIF8", "gif"),
]
_VIDEO_MAGIC: list[tuple[bytes, str]] = [
    (b"\x00\x00\x00", "mp4/mov"),  # ftyp box — also used for HEIC; differentiated below
    (b"RIFF", "avi"),
    (b"\x1a\x45\xdf\xa3", "mkv"),
    (b"FLV\x01", "flv"),
]
_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".ts", ".mts", ".m4v", ".flv", ".webm"}
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".bmp", ".tif", ".tiff"}

# Common smartphone screen aspect ratios (width/height)
_PHONE_ASPECT_RATIOS = [
    20 / 9,   # modern flagship
    19.5 / 9,
    19 / 9,
    18.5 / 9,
    16 / 9,
]
_ASPECT_RATIO_TOLERANCE = 0.05


class MediaTypeDetector:
    def __init__(self) -> None:
        self._settings = get_settings()

    # ── Public ─────────────────────────────────────────────────────────────────

    def detect(self, path: Path) -> MediaInfo:
        """Return fully-populated MediaInfo for the given file path."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Media file not found: {path}")

        file_size = path.stat().st_size
        raw_type = self._detect_raw_type(path)

        if raw_type == "image":
            img = cv2.imread(str(path))
            if img is None:
                raise ValueError(f"cv2 cannot open image: {path}")
            h, w = img.shape[:2]
            media_type = self._classify_image(img, path)
            return MediaInfo(
                path=path,
                media_type=media_type,
                width=w,
                height=h,
                file_size_bytes=file_size,
            )

        # Video
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"cv2 cannot open video: {path}")
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration_s = total_frames / fps if fps > 0 else 0.0
            is_motion = self._classify_video_motion(cap)
        finally:
            cap.release()

        media_type = MediaType.VIDEO_MOTION if is_motion else MediaType.VIDEO_STATIC
        return MediaInfo(
            path=path,
            media_type=media_type,
            width=w,
            height=h,
            fps=fps,
            frame_count=total_frames,
            duration_s=duration_s,
            file_size_bytes=file_size,
        )

    # ── Private ────────────────────────────────────────────────────────────────

    def _detect_raw_type(self, path: Path) -> str:
        """Return 'image' or 'video' using extension + magic bytes."""
        ext = path.suffix.lower()
        if ext in _IMAGE_EXTENSIONS:
            return "image"
        if ext in _VIDEO_EXTENSIONS:
            return "video"

        # Fall back to magic bytes
        magic = path.read_bytes()[:12]
        for sig, _ in _IMAGE_MAGIC:
            if magic.startswith(sig):
                # HEIC vs MP4: both start with 0x000... ftyp box; check brand
                if sig == b"\x00\x00\x00":
                    brand = magic[8:12]
                    if brand in (b"heic", b"heix", b"mif1", b"msf1"):
                        return "image"
                    if brand in (b"mp41", b"mp42", b"isom", b"M4V ", b"qt  "):
                        return "video"
                else:
                    return "image"
        for sig, _ in _VIDEO_MAGIC:
            if magic.startswith(sig):
                return "video"

        # Last resort: try cv2.VideoCapture
        cap = cv2.VideoCapture(str(path))
        is_video = cap.isOpened() and cap.get(cv2.CAP_PROP_FRAME_COUNT) > 1
        cap.release()
        return "video" if is_video else "image"

    def _classify_image(self, img: np.ndarray, path: Path) -> MediaType:
        """Decide IMAGE_PREVIEW vs IMAGE_CAPTURED."""
        h, w = img.shape[:2]
        # Normalize to larger/smaller so portrait screenshots match the same ratios
        aspect = max(w, h) / min(w, h)

        # Heuristic 1: check if aspect ratio matches a phone screen
        matches_phone_ar = any(
            abs(aspect - ar) <= _ASPECT_RATIO_TOLERANCE for ar in _PHONE_ASPECT_RATIOS
        )

        if not matches_phone_ar:
            return MediaType.IMAGE_CAPTURED

        # Heuristic 2: detect status bar — top strip has dense small icons on uniform bg
        top_strip_h = max(1, int(h * 0.06))   # top 6%
        top_strip = img[:top_strip_h, :]
        status_bar_detected = self._has_status_bar(top_strip)

        # Heuristic 3: detect camera shutter button in bottom 25% (white/gray circle)
        bottom_crop = img[int(h * 0.75):, :]
        shutter_detected = self._has_shutter_button(bottom_crop)

        if status_bar_detected or shutter_detected:
            log.debug("Classified as IMAGE_PREVIEW (status_bar=%s, shutter=%s)",
                      status_bar_detected, shutter_detected)
            return MediaType.IMAGE_PREVIEW

        return MediaType.IMAGE_CAPTURED

    def _has_status_bar(self, strip: np.ndarray) -> bool:
        """Return True if the strip looks like an Android/iOS status bar."""
        if strip.size == 0:
            return False
        gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
        # Status bar: nearly uniform background, small high-contrast elements (icons)
        bg_std = float(np.std(gray))
        if bg_std > 30:   # too much variation — not a flat status bar background
            return False

        # Look for small bright blobs on dark background (or vice versa)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        small_blobs = [c for c in contours if 5 < cv2.contourArea(c) < 500]
        return len(small_blobs) >= 3

    def _has_shutter_button(self, region: np.ndarray) -> bool:
        """Detect a large circular shutter button in the bottom region."""
        if region.size == 0:
            return False
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        h, w = gray.shape
        min_r = max(10, w // 20)
        max_r = w // 5
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=w // 4,
            param1=50,
            param2=30,
            minRadius=min_r,
            maxRadius=max_r,
        )
        return circles is not None

    def _classify_video_motion(self, cap: cv2.VideoCapture) -> bool:
        """Sample optical flow to determine if video has significant motion."""
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_video = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n_samples = min(10, max(2, total // 30))
        step = max(1, total // (n_samples + 1))

        flow_magnitudes: list[float] = []
        prev_gray: Optional[np.ndarray] = None
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)

        for i in range(n_samples):
            frame_idx = step * (i + 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                flow = dis.calc(prev_gray, gray, None)
                mag = float(np.mean(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)))
                flow_magnitudes.append(mag)
            prev_gray = gray

        if not flow_magnitudes:
            return False

        mean_flow = float(np.mean(flow_magnitudes))
        threshold = self._settings.motion_flow_threshold
        log.debug("Video mean optical flow: %.2f px (threshold %.2f)", mean_flow, threshold)
        return mean_flow > threshold
