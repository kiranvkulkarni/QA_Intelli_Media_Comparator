# Technical Design Document
## QA Intelli Media Comparator

**Version:** 1.0.0
**Domain:** Smartphone Camera Quality Assurance
**Classification:** Internal Technical Reference

---

## Table of Contents

1. [System Context](#1-system-context)
2. [Design Principles](#2-design-principles)
3. [Domain Models](#3-domain-models)
4. [Service Layer — Detailed Algorithms](#4-service-layer--detailed-algorithms)
   - 4.0 [CameraModeDetector](#40-cameramodedetector)
   - 4.1 [MediaTypeDetector](#41-mediatypedetector)
   - 4.2 [PreviewCropper](#42-previewcropper)
   - 4.3 [QualityMetricsExtractor](#43-qualitymetricsextractor)
   - 4.4 [ArtifactDetector](#44-artifactdetector)
   - 4.5 [NoReferenceAnalyzer](#45-noreferenceanalyzer)
   - 4.6 [ReferenceComparator](#46-referencecomparator)
   - 4.7 [VideoAnalyzer](#47-videoanalyzer)
   - 4.8 [AnnotationRenderer](#48-annotationrenderer)
   - 4.9 [ComparisonPipeline](#49-comparisonpipeline)
   - 4.10 [FunctionalityChecker](#410-functionalitychecker)
5. [API Layer](#5-api-layer)
6. [Storage Layer](#6-storage-layer)
7. [Configuration System](#7-configuration-system)
8. [Neural Model Loading Strategy](#8-neural-model-loading-strategy)
9. [Data Flow Diagrams](#9-data-flow-diagrams)
10. [Error Handling Strategy](#10-error-handling-strategy)
11. [Performance Characteristics](#11-performance-characteristics)
12. [Threshold Calibration Guide](#12-threshold-calibration-guide)
13. [Extending the System](#13-extending-the-system)
14. [Known Limitations and Trade-offs](#14-known-limitations-and-trade-offs)

---

## 1. System Context

### 1.1 Physical Setup

```
┌───────────────────────────────────────────────────────────┐
│           Photography Lightbox (Godex 1m × 1m)            │
│                                                           │
│   ┌─────────────┐     CoreXY Rail     ┌──────────────┐   │
│   │ Light Panel │◄─────── Scene ─────►│ Light Panel  │   │
│   └─────────────┘                     └──────────────┘   │
│                                                           │
│            ┌──────────────────────────┐                   │
│            │    Rail Head / Carriage  │                   │
│            │   ┌──────────────────┐   │                   │
│            │   │  Smartphone Mount│   │                   │
│            │   │  (Reference or   │   │                   │
│            │   │     DUT)         │   │                   │
│            │   └──────────────────┘   │                   │
│            └──────────────────────────┘                   │
└───────────────────────────────────────────────────────────┘
                              │
                     ADB over USB / WiFi
                              │
                    ┌─────────▼──────────┐
                    │  Windows 11 Host   │
                    │  ┌──────────────┐  │
                    │  │ UIAutomator2 │  │  (Existing 60% automation)
                    │  │  Framework   │  │
                    │  └──────┬───────┘  │
                    │         │          │
                    │  ┌──────▼───────┐  │
                    │  │ QIMC Service │  │  (This project — media QA)
                    │  │  :8080       │  │
                    │  └──────────────┘  │
                    └────────────────────┘
```

### 1.2 Test Execution Model

1. **Reference run**: UIAutomator2 drives the golden reference phone through all camera test scenarios; media captured to local disk via ADB.
2. **DUT run**: Same test sequence executed on the DUT; media captured identically.
3. **Comparison**: For each test step, the automation framework calls `POST /compare` with matching reference + DUT files.
4. **Report**: QIMC returns a structured `ComparisonReport` with pass/fail verdict, metric scores, artifact locations, and human-readable failure reasons.

### 1.3 Responsibilities Out of Scope

- UI automation (covered by UIAutomator2 framework)
- ADB capture commands (covered by automation framework)
- Test case management / reporting dashboards
- Camera control parameters (ISO, shutter, WB) — QIMC only analyzes outputs

---

## 2. Design Principles

### 2.1 Reliability over Speed

Classical metrics (SSIM, Laplacian, noise estimation) always run — they do not depend on model downloads, CUDA availability, or network connectivity. Neural metrics (LPIPS, MUSIQ) are additive and optional.

### 2.2 Graceful Degradation

Every neural model call is wrapped in `try/except`. If a model fails to load or produce output, its score is `null` in the report rather than causing a pipeline failure.

### 2.3 Human-Readable Diagnostics

Every artifact and metric failure must produce a `description` or `failure_reason` string that explains:
1. What was detected / what failed
2. What it likely means in camera QA terms
3. What parameter or setting to investigate

### 2.4 Stateless Processing

Each `/compare` or `/analyze` request is fully self-contained. Models are loaded once at startup and cached in memory; all per-request state is ephemeral. Reports are written to disk only after successful completion.

### 2.5 Deterministic Comparison

For a given pair of inputs with identical configuration, the service produces identical metric scores (classical metrics) or very close scores (neural, due to floating-point device differences).

---

## 3. Domain Models

### 3.1 Model Hierarchy

```
ComparisonReport
├── MediaType (enum)
├── CropResult (optional)
├── MediaMetadata — dut_metadata (EXIF + detected camera mode)
│   ├── make, model, software
│   ├── exposure_time, f_number, iso, focal_length_mm, focal_length_35mm
│   ├── flash_fired, scene_capture_type, exposure_program
│   ├── camera_mode: CameraMode (enum)
│   ├── camera_mode_confidence: float
│   ├── camera_mode_source: str
│   └── mode_notes: List[str]
├── MediaMetadata — ref_metadata (optional, compare mode only)
├── MetadataComparison (optional, compare mode only)
│   ├── modes_match: bool
│   ├── dut_mode / ref_mode: CameraMode
│   ├── iso_delta: int
│   ├── exposure_delta_stops: float
│   ├── f_number_match: bool
│   ├── focal_length_match: bool
│   └── notes: List[str]
├── QualityMetrics
│   ├── blur_score: float
│   ├── noise_sigma: float
│   ├── exposure_mean: float
│   ├── highlight_clipping_pct: float
│   ├── shadow_clipping_pct: float
│   ├── color_cast_r/g/b: float
│   ├── white_balance_deviation: float
│   ├── saturation_mean: float
│   ├── dynamic_range_stops: float
│   └── chromatic_aberration_score: float
├── NoReferenceScores
│   ├── brisque: float
│   ├── niqe: float
│   ├── musiq: float
│   ├── clip_iqa: float
│   └── grade: QualityGrade
├── FullReferenceScores (optional)
│   ├── psnr: MetricResult
│   ├── ssim: MetricResult
│   ├── ms_ssim: MetricResult
│   ├── lpips: MetricResult
│   └── dists: MetricResult
├── ArtifactReport
│   ├── artifacts: List[ArtifactInstance]
│   │   ├── artifact_type: str
│   │   ├── severity: ArtifactSeverity
│   │   ├── bbox: (x, y, w, h)
│   │   ├── confidence: float
│   │   └── description: str
│   └── overall_severity: ArtifactSeverity
├── VideoTemporalMetrics (optional)
│   ├── flicker_score: float
│   ├── jitter_score: float
│   ├── temporal_ssim_mean: float
│   ├── temporal_ssim_std: float
│   ├── sync_offset_frames: int
│   ├── black_frame_count: int      ← frames with near-zero luminance
│   └── frozen_frame_count: int     ← frames in a consecutive freeze run
├── analysis_mode: str              ← "functional" | "quality"
├── functional_grade: QualityGrade  ← PASS/WARNING/FAIL from FunctionalityChecker only
├── functional_reasons: List[str]   ← human-readable strings for functional failures
├── overall_grade: QualityGrade     ← full IQA grade (null in functional mode)
├── failure_reasons: List[str]
├── annotated_image_path: str
└── processing_time_ms: int
```

### 3.2 MetricResult

```python
class MetricResult(BaseModel):
    value: Optional[float]          # None if metric not computed
    threshold: Optional[float]      # configurable pass/fail threshold
    passed: Optional[bool]          # None if not evaluated
    higher_is_better: bool          # True for SSIM/PSNR, False for LPIPS
```

`evaluate()` sets `passed = (value >= threshold)` if `higher_is_better`, else `(value <= threshold)`.

### 3.3 Severity Ordering

```python
ArtifactSeverity: NONE(0) < LOW(1) < MEDIUM(2) < HIGH(3) < CRITICAL(4)
```

`ArtifactReport.overall_severity` is always the maximum severity across all artifacts.

Pass/fail threshold for artifacts: `HIGH` or `CRITICAL` → report.overall_grade = FAIL.

---

## 4. Service Layer — Detailed Algorithms

### 4.0 CameraModeDetector

**File:** `services/camera_mode_detector.py`

#### 4.0.1 Purpose

Smartphone cameras produce radically different optical output depending on the shooting mode. Without mode awareness, Portrait bokeh, Night high-ISO noise, Sport motion-blur, and Macro shallow-DoF are all misreported as defects in NR analysis mode. This service:

1. Extracts standard EXIF metadata from image files using Pillow.
2. Infers the shooting mode from EXIF tags and heuristics.
3. Applies multiplicative threshold adjustments to the active Settings object so all downstream services (QualityMetrics, ArtifactDetector) automatically use mode-corrected limits.
4. In compare mode, compares DUT vs REF EXIF to surface setting differences that explain quality gaps without being genuine regressions.

#### 4.0.2 EXIF Extraction

Uses `PIL.Image._getexif()` (Pillow, already a project dependency) which returns a `{tag_id: value}` dict. Values are decoded using `PIL.ExifTags.TAGS`. Key fields extracted:

| EXIF Field | Tag name | Type | Usage |
|---|---|---|---|
| Make / Model / Software | 271, 272, 305 | string | Device identity in report |
| DateTimeOriginal | 36867 | string | Capture timestamp |
| ExposureTime | 33434 | IFDRational / tuple | Night/Sport heuristic |
| FNumber | 33437 | IFDRational / tuple | Aperture comparison |
| ISOSpeedRatings | 34855 | int | Night/Sport heuristic |
| FocalLength | 37386 | IFDRational | Focal length display |
| FocalLengthIn35mmFilm | 41989 | int | Zoom level comparison |
| Flash | 37385 | int (bit flags) | Bit 0 = fired |
| SceneCaptureType | 41990 | int | 2=Portrait, 3=Night |
| ExposureProgram | 34850 | int | 7=Portrait, 8=Landscape |
| UserComment | 37510 | bytes | Mode keyword search |
| ImageDescription | 270 | string | Mode keyword search |
| XPComment | 40092 | UTF-16-LE bytes | Mode keyword search (Windows) |

`ExposureTime` and `FNumber` are IFDRational in modern Pillow — `float(val)` converts them. The code also handles legacy `tuple(numerator, denominator)` form for compatibility with older images.

#### 4.0.3 Camera Mode Detection (Priority Order)

```
Priority 1 — Text field keyword search (confidence 0.90)
  Concatenate UserComment + ImageDescription + XPComment (decoded)
  Search for mode-specific keywords (case-insensitive):
    portrait, bokeh, depth effect, live focus → PORTRAIT
    night, night mode, nightsight, low light  → NIGHT
    sport, action, burst                      → SPORT
    macro, close-up                           → MACRO
    hdr, high dynamic range                   → HDR
    panorama, pano, stitch                    → PANORAMA
    pro, manual, expert                       → PRO

Priority 2 — EXIF SceneCaptureType tag (confidence 0.85)
  1 → LANDSCAPE, 2 → PORTRAIT, 3 → NIGHT

Priority 3 — EXIF ExposureProgram tag (confidence 0.80)
  7 → PORTRAIT, 8 → LANDSCAPE

Priority 4 — Heuristic from ISO + ExposureTime (confidence 0.55–0.70)
  ISO ≥ 3200 AND shutter ≥ 1/8 s   → NIGHT (0.70)
  ISO ≥ 1600 AND shutter ≥ 1/30 s  → NIGHT (0.55)
  ISO ≥ 800  AND shutter < 1/250 s → SPORT (0.55)

Priority 5 — Default (confidence 0.50)
  → AUTO (no threshold adjustment)
```

The `camera_mode_source` field records which priority level produced the result (`user_comment`, `exif_scene_type`, `exif_exposure_program`, `heuristic`, `default`, or `media_type` for video).

#### 4.0.4 Threshold Adjustment Mechanism

Mode adjustments are **multiplicative factors** applied on top of the currently active settings (which may already reflect a quality-profile override from the API route):

| Mode | `blur_threshold` | `noise_threshold` | `highlight_clip_threshold` | `shadow_clip_threshold` |
|---|---|---|---|---|
| **Portrait** | × 0.15 | × 1.5 | — | — |
| **Night** | × 0.30 | × 4.0 | × 3.0 | × 2.0 |
| **Sport** | × 0.30 | × 2.0 | — | — |
| **Macro** | × 0.15 | × 1.3 | — | — |
| **HDR** | — | — | × 3.0 | × 3.0 |
| **Panorama** | × 0.50 | × 1.5 | — | — |
| All others | 1.0 | 1.0 | 1.0 | 1.0 |

`apply_mode_adjustments()` calls `base_settings.model_copy(update=overrides)` — a pydantic v2 shallow copy via `model_construct` that does **not** re-run `model_post_init`, so the quality-profile preset already baked into `base_settings` is preserved in the copy.

The adjusted Settings object is pushed into `_request_settings` (ContextVar) in the pipeline via a `try/finally` block, so all downstream `get_settings()` calls within the same async task context see the mode-corrected thresholds. The token is reset in the `finally` block after analysis completes.

**Composition example:**

```
Quality profile: high   → blur_threshold = 100.0
Camera mode:     night  → blur_threshold × 0.30
Effective limit:         → blur_threshold = 30.0
```

This means a Night photo only fails the blur check if its sharpness score is below 30, rather than the 100 that would apply to a standard daytime capture.

#### 4.0.5 Metadata Comparison (compare mode)

`MetadataComparison.build(dut_meta, ref_meta)` produces a structured side-by-side comparison:

- **Mode mismatch**: if DUT mode ≠ REF mode (both non-unknown), a note is added explaining quality differences may be mode-driven.
- **ISO delta**: difference ≥ 400 ISO → note that DUT is operating at higher sensitivity.
- **Exposure stop delta**: `log₂(DUT_t / REF_t)` ≥ 1 stop → note explaining brightness/motion-blur differences.
- **Aperture mismatch**: `|DUT_f − REF_f| ≥ 0.5` → note on DoF and exposure differences.
- **Focal length mismatch**: 35 mm-equivalent differs by > 5 mm → critical note that FR-IQA scores will be intrinsically low due to different zoom levels.

All notes are surfaced in `failure_reasons` with `[Metadata]` prefix — advisory only, never causes FAIL grade.

#### 4.0.6 Video Handling

Video files (`.mp4`, `.mov`, `.avi`, `.mkv`, `.ts`, `.mts`, `.webm`) do not carry standard EXIF. A skeletal `MediaMetadata` with `camera_mode=VIDEO` and `camera_mode_source=media_type` is returned immediately. No threshold adjustments are applied for video (the video analyzer's own per-frame analysis handles quality assessment).

---

### 4.1 MediaTypeDetector

**File:** `services/media_type_detector.py`

#### Image vs Video Detection

```
Path → read 12 magic bytes
  ├── JPEG: FF D8 FF → "image"
  ├── PNG:  89 50 4E 47 → "image"
  ├── WEBP: RIFF....WEBP → "image"
  ├── ftyp box (bytes 4-8 = "ftyp"):
  │   ├── brand in {heic, heix, mif1} → "image"
  │   └── brand in {mp41, isom, qt  } → "video"
  ├── RIFF....AVI  → "video"
  └── 1A 45 DF A3  → "video" (MKV/WebM)
```

Extension lookup is tried first for performance; magic bytes are fallback.

#### Preview vs Captured (Image)

```python
aspect = width / height

# Phone screen aspect ratios (with ±5% tolerance)
PHONE_ARs = [20/9, 19.5/9, 19/9, 18.5/9, 16/9]

if aspect NOT within 5% of any phone AR:
    return IMAGE_CAPTURED

# Status bar heuristic (top 6% of image)
top_strip = img[:int(h*0.06), :]
gray_std = std(top_strip)  # low std = uniform background
if gray_std < 30:
    otsu_threshold binary_mask
    small_blobs = contours where 5 < area < 500
    status_bar = len(small_blobs) >= 3

# Shutter button heuristic (bottom 25%)
bottom = img[int(h*0.75):, :]
circles = HoughCircles(bottom, HOUGH_GRADIENT, ...)
shutter = circles is not None

if status_bar OR shutter:
    return IMAGE_PREVIEW
return IMAGE_CAPTURED
```

#### Static vs Motion Video (Video)

```python
sample 10 frame pairs evenly across video duration
for each pair:
    flow = DISOpticalFlow.calc(prev_gray, curr_gray)
    magnitude = mean(sqrt(flow_x² + flow_y²))

mean_flow = average(magnitudes)
return VIDEO_MOTION if mean_flow > QIMC_MOTION_FLOW_THRESHOLD else VIDEO_STATIC
```

`cv2.DISOpticalFlow` (Dense Inverse Search) is chosen for its excellent speed/accuracy trade-off on consumer hardware without GPU.

---

### 4.2 PreviewCropper

**File:** `services/preview_cropper.py`

The goal is to isolate the camera viewfinder content from surrounding Android/iOS camera app UI elements (status bar, shutter button, zoom controls, mode selector, etc.).

#### Strategy 1: Contour-Based (primary)

```python
gray = cvtColor(img, BGR2GRAY)
blurred = GaussianBlur(gray, (5,5), 0)
edges = Canny(blurred, 50, 150)
edges = dilate(edges, 3×3 kernel, 1 iteration)  # close gaps in border

contours = findContours(edges, RETR_EXTERNAL)

for cnt in contours:
    area = contourArea(cnt)
    if area < total_area * 0.25:  # must be at least 25% of frame
        continue
    approx = approxPolyDP(cnt, 0.02*perimeter, closed=True)
    rect = boundingRect(approx if len==4-6 else cnt)
    aspect_ratio = rect.w / rect.h
    ar_score = closest_to_preview_AR([4/3, 16/9, 1.0, 3/4, 9/16], tolerance=0.08)
    coverage = (rect.w * rect.h) / total_area
    score = ar_score * coverage
    track best

return best_rect if score > 0 else None
```

The score balances two factors: (a) how close the aspect ratio is to a known camera preview ratio, and (b) how much of the frame it covers. This prevents small UI elements from being selected as the preview area.

#### Strategy 2: Saturation Mask (fallback)

Camera previews contain real-world content with high chromatic variation. The surrounding Android UI (status bar, buttons, gesture bar) tends to be dark gray/black with minimal saturation.

```python
hsv = cvtColor(img, BGR2HSV)
sat = hsv[:,:,1]          # saturation channel

# threshold: pixels with sat > 30 are likely preview content
_, mask = threshold(sat, 30, 255, THRESH_BINARY)
mask = morphologyEx(mask, MORPH_CLOSE, 15×15 kernel)  # fill holes
mask = morphologyEx(mask, MORPH_OPEN,  15×15 kernel)  # remove specks

largest_contour = max(contours, key=contourArea)
if area(largest) < 0.25 * total_area:
    return None  # too small to be the preview
return boundingRect(largest_contour)
```

#### Strategy 3: Heuristic Fallback (last resort)

Based on empirical study of common Android camera app layouts:

```
top margin    = 8% of height (status bar ~24dp + camera mode bar)
bottom margin = 22% of height (shutter + gallery + settings)
left/right    = 0% (preview is full width)
```

This is unconditionally applied if both strategy 1 and 2 fail (e.g., full-black first frame).

#### Application to Video

For video files, the crop bounding box is determined from the **first stable frame** (skip first 0.5s for camera startup animation) and applied **uniformly** to all subsequent frames. This ensures temporal consistency in the analysis.

```python
cap.set(CAP_PROP_POS_FRAMES, int(fps * 0.5))
# Try up to 5 frames to find a non-black one
for _ in range(5):
    frame = cap.read()
    if mean(frame) > 5:
        break

bbox, result = crop_image(frame)
cap.set(CAP_PROP_POS_FRAMES, 0)  # rewind for analysis
```

---

### 4.3 QualityMetricsExtractor

**File:** `services/quality_metrics.py`

#### Sharpness — Laplacian Variance

```
L(x,y) = ∂²I/∂x² + ∂²I/∂y²

var(L) = Σ(L(x,y) - mean(L))² / N
```

High variance = many sharp edges = focused image.
Low variance = all edges are soft = blurry image.

The Laplacian is computed via `cv2.Laplacian(gray, CV_64F)`. Using 64-bit float avoids integer clipping artifacts.

**Tenengrad** (secondary sharpness):

```
T = mean(Gx² + Gy²)    where Gx, Gy = Sobel(gray, 3×3)
```

Tenengrad is more sensitive to directional edges and complements the Laplacian.

#### Noise Sigma Estimation

```python
# Step 1: find flat (low-gradient) regions
gx = Sobel(gray, CV_32F, dx=1, dy=0)
gy = Sobel(gray, CV_32F, dx=0, dy=1)
grad_mag = sqrt(gx² + gy²)
flat_mask = grad_mag < percentile(grad_mag, 20)  # bottom 20% gradient

# Step 2: remove slow DC illumination variation
local_mean = uniform_filter(gray, size=15)  # 15×15 box filter
residual = gray - local_mean

# Step 3: noise estimate = std in flat regions of residual
noise_sigma = std(residual[flat_mask])
```

Removing the local mean is critical: without it, large-scale illumination gradients (vignetting, light falloff) would inflate the noise estimate in flat image regions near edges of the frame.

#### Chromatic Aberration

```python
edges = Canny(gray, 80, 200)
for each edge pixel (py, px):
    # Extract 10×10 patch of R and B channels
    patch_r = img[py-5:py+5, px-5:px+5, R]
    patch_b = img[py-5:py+5, px-5:px+5, B]

    # Compute intensity centroid of each patch
    cy_r = sum(patch_r * idx_y) / sum(patch_r)
    cx_r = sum(patch_r * idx_x) / sum(patch_r)
    cy_b = sum(patch_b * idx_y) / sum(patch_b)
    cx_b = sum(patch_b * idx_x) / sum(patch_b)

    offset = sqrt((cy_r - cy_b)² + (cx_r - cx_b)²)

chromatic_aberration_score = mean(offset for all edge pixels)
```

A mean R-B centroid offset > 2px indicates visible chromatic aberration. At image edges (where CA is typically worst), offsets can reach 5–10px on uncorrected lenses.

#### White Balance Deviation

```python
mean_r = mean(R channel)
mean_g = mean(G channel)
mean_b = mean(B channel)

rg_ratio = mean_r / mean_g    # ideal = 1.0 for neutral gray scene
bg_ratio = mean_b / mean_g

wb_deviation = sqrt(((rg_ratio - 1)² + (bg_ratio - 1)²) / 2)
```

This is a global gray-world assumption. For accurate WB assessment, a ColorChecker or gray card should be in the scene.

---

### 4.4 ArtifactDetector

**File:** `services/artifact_detector.py`

#### Noise Patch Detection

```python
divide image into 32×32 pixel patches

for each patch:
    patch_grad_mean = mean(|Sobel(patch)|)
    if patch_grad_mean > 15:
        continue  # skip edge-rich regions (not flat)
    local_std = std(patch)
    if local_std > max(12.0, global_std * 1.5):
        mark as noisy

pct = noisy_count / total_patches
severity:
    CRITICAL if pct > 0.40
    HIGH     if pct > 0.20
    MEDIUM   if pct > 0.08
    LOW      if pct > 0.02
    skip     if pct < 0.02  # negligible
```

The `global_std * 1.5` term makes the threshold adaptive — on a uniformly textured image (low global std), even modest patch variance is flagged; on a high-detail image (high global std), only extreme patches are flagged.

#### Banding Detection

```python
# Smooth the image horizontally to isolate vertical gradients
smoothed = GaussianBlur(gray, (1, 7))
# Per-row gradient energy
gy = Sobel(smoothed, dy=1)
row_energy = mean(|gy|, axis=columns)   # shape: (height,)

# FFT of row energy to find periodic components
fft = |rfft(row_energy)|
fft[0] = 0  # remove DC

freqs = rfftfreq(height)  # cycles per pixel

# Banding: > 3 bands per 100 rows = frequency > 0.03
band_mask = freqs > 0.03
banding_ratio = sum(fft[band_mask]) / sum(fft)

severity:
    HIGH   if banding_ratio > 0.50
    MEDIUM if banding_ratio > 0.30
    LOW    if banding_ratio > 0.15
```

Banding manifests as regular horizontal stripes (false contouring in smooth gradients). The FFT isolates this periodic pattern from natural image content.

#### Lens Flare Detection

```python
# Find extreme bright regions
threshold_val = percentile(gray, 99.9)
if threshold_val < 200:
    return  # nothing is extreme-bright

binary = (gray >= threshold_val).astype(uint8)
binary = dilate(binary, 7×7 ellipse, 2 iterations)  # merge nearby blobs

for each connected component:
    aspect_ratio = max(w, h) / min(w, h)
    if aspect_ratio > 2.5:      # elongated → streak flare
        add to flares
    elif area > 0.01 * total:   # large → halo flare
        add to flares
```

Lens flares typically appear as either elongated bright streaks (from direct sun/light source) or circular halos (from internal reflections). Both patterns are detected.

#### Chromatic Aberration (Artifact)

The artifact-level detection reuses the algorithm from `QualityMetricsExtractor` but additionally:
- Collects all edge points with offset > 2px
- Computes a bounding box enclosing those high-CA regions
- Reports the severity based on mean offset across all sampled edges

#### Blurry Region Detection

```python
tile_size = 64  # pixels

for each 64×64 tile:
    lap_var = var(Laplacian(tile))
    sharpness_map.append(lap_var)

global_mean = mean(sharpness_map)
threshold = global_mean / 3.0

blurry_tiles = [t for t in sharpness_map if t.value < threshold]
pct = len(blurry_tiles) / len(sharpness_map)

severity:
    HIGH   if pct > 0.50
    MEDIUM if pct > 0.25
    LOW    if pct > 0.10
```

This detects **locally** blurry regions within an otherwise sharp image — typical of partial focus failure, motion blur in specific parts of the frame, or OIS micro-failures.

---

### 4.5 NoReferenceAnalyzer

**File:** `services/no_reference_analyzer.py`

#### Model Lifecycle

```
Service startup → preload()
    → _ModelCache.get("brisque", device)
        → pyiqa.create_metric("brisque", device=device)
        → cache[key] = model
    → _ModelCache.get("niqe", device)
    → [optional] _ModelCache.get("musiq", device)

Per request → analyze(img_bgr)
    → _bgr_to_tensor(img_bgr)  # BGR uint8 → [0,1] RGB tensor
    → model(tensor) → score
    → return NoReferenceScores
```

The `_ModelCache` is a module-level singleton (`_cache = _ModelCache()`), surviving across all requests. Models are loaded exactly once per (name, device) pair.

#### BGR to Tensor Conversion

```python
rgb = img_bgr[:, :, ::-1].copy()          # BGR → RGB channel swap
pil = Image.fromarray(rgb)                 # numpy → PIL
tensor = TF.to_tensor(pil).unsqueeze(0)   # [H,W,3] → [1,3,H,W] float32 [0,1]
tensor = tensor.to(device)
```

#### BRISQUE Scoring

BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator) works by:
1. Computing mean-subtracted contrast-normalized (MSCN) coefficients from the image
2. Fitting a generalized Gaussian distribution (GGD) to MSCN and pairwise products of adjacent MSCN coefficients
3. Computing feature vector (18D) at two scales (original and half)
4. Using a pre-trained SVM to map the 36D feature vector to a quality score

BRISQUE is completely reference-free and requires no GPU. It correlates well with human judgments for distorted natural images.

**Grade thresholds:**
- Score < 30: PASS (good quality)
- Score 30–50: PASS (acceptable)
- Score 50–60: WARNING (noticeable degradation)
- Score > 60: FAIL (poor quality)

#### NIQE Scoring

NIQE (Natural Image Quality Evaluator) computes statistics of locally normalized luminance patches and measures the distance from a multivariate Gaussian model fit to a corpus of pristine natural images.

**Grade thresholds:**
- Score < 4: PASS
- Score 4–6: PASS (marginal)
- Score 6–10: WARNING
- Score > 10: FAIL

---

### 4.6 ReferenceComparator

**File:** `services/reference_comparator.py`

#### Spatial Alignment (SIFT + Homography)

```python
sift = cv2.SIFT_create()
kp_ref, des_ref = sift.detectAndCompute(ref_gray, None)
kp_dut, des_dut = sift.detectAndCompute(dut_gray, None)

# Lowe's ratio test for good matches
bf = BFMatcher(NORM_L2, crossCheck=False)
matches = bf.knnMatch(des_ref, des_dut, k=2)
good = [m for m, n in matches if m.distance < 0.75 * n.distance]

if len(good) < 10:
    skip alignment  # not enough matches

H, mask = findHomography(src_pts, dst_pts, RANSAC, ransacReprojThreshold=5.0)

if inliers >= 8:
    ref_warped = warpPerspective(ref, H, (dut_w, dut_h))
```

**Why SIFT over ORB?** SIFT is scale- and rotation-invariant and produces higher-quality matches in complex photographic scenes. ORB is faster but less reliable for images taken at slightly different angles or distances (which can occur with the CoreXY rail system).

**When alignment is skipped:**
- Fewer than 20 keypoints detected on either image
- Fewer than 10 good matches (after ratio test)
- Homography has fewer than 8 inliers

#### SSIM Computation

```python
from skimage.metrics import structural_similarity as ssim_fn

val, ssim_map = ssim_fn(
    ref_gray, dut_gray,
    data_range=255,
    full=True        # return per-pixel SSIM map for heatmap
)
```

The `ssim_map` (values 0–1, lower = worse similarity) can be used to identify spatially localized quality degradation but is not currently used for localization — the artifact detector handles localization separately.

#### Difference Heatmap Generation

```python
diff = absdiff(ref_aligned, dut_resized)
diff_gray = cvtColor(diff, BGR2GRAY)

# Normalize to 99th percentile to avoid outlier domination
p99 = percentile(diff_gray, 99)
diff_normalized = clip(diff_gray * 255.0 / p99, 0, 255).astype(uint8)

heatmap = applyColorMap(diff_normalized, COLORMAP_JET)
# COLORMAP_JET: blue=low diff → green → yellow → red=high diff
```

The 99th percentile normalization ensures that a few extreme-outlier pixels (hot pixels, compression artifacts) do not compress the visible range of the heatmap.

---

### 4.7 VideoAnalyzer

**File:** `services/video_analyzer.py`

#### Frame Sampling

```python
fps_video = cap.get(CAP_PROP_FPS)          # e.g., 30.0
sample_fps = settings.video_sample_fps     # e.g., 2.0
frame_step = round(fps_video / sample_fps) # e.g., 15

# Extract every frame_step-th frame
frames = [frame for idx, frame in enumerate(video) if idx % frame_step == 0]
```

At `QIMC_VIDEO_SAMPLE_FPS=2.0` with a 30fps source video, 1 in 15 frames is analyzed. For a 5-second clip, this yields ~10 frames — sufficient for temporal consistency analysis without excessive processing time.

#### Flicker Detection

```python
# Compute mean luminance per sampled frame
lum_curve = [mean(cvtColor(f, BGR2GRAY)) for f in frames]

# FFT of luminance time series
fft = abs(rfft(lum_curve))
freqs = rfftfreq(n=len(lum_curve), d=1.0/sample_fps)  # Hz
fft[0] = 0  # suppress DC component

# Flicker band: 5–60 Hz (mains flicker, LED PWM)
flicker_mask = (freqs >= 5) & (freqs <= 60)
flicker_score = sum(fft[flicker_mask]) / sum(fft)
```

**Why 5–60 Hz?**
- Mains electricity flicker: 50 Hz (EU/Asia) or 60 Hz (US)
- LED PWM flicker: 50–200 Hz (varies by driver)
- Human perception threshold: ~3 Hz
- The band is wide to catch both fundamental and harmonics

**Calibration note:** At 2fps sampling, the Nyquist frequency is 1 Hz — far below the flicker band. This means flicker detection requires a higher `QIMC_VIDEO_SAMPLE_FPS` setting (≥ 120fps sampling needed for 60Hz flicker). For real-time flicker analysis, use `QIMC_VIDEO_SAMPLE_FPS=120` or higher (requires the video to be recorded at ≥120fps). For standard 30fps videos, the service detects **frame-rate aliased** flicker signatures (brightness pulsing visible in the 30fps stream).

#### Jitter Detection

```python
dis = DISOpticalFlow_create(DISOPTICAL_FLOW_PRESET_FAST)
centroids_x = []
centroids_y = []

for i in range(len(frames) - 1):
    flow = dis.calc(prev_gray, curr_gray, None)  # shape: (H, W, 2)
    cx = mean(flow[:,:,0])  # mean x-displacement
    cy = mean(flow[:,:,1])  # mean y-displacement
    centroids_x.append(cx)
    centroids_y.append(cy)

jitter = sqrt(std(centroids_x)² + std(centroids_y)²)  # px
```

The mean displacement centroid represents the global frame-to-frame camera motion. The standard deviation of this centroid over time represents instability (jitter). A stable video has consistent inter-frame motion (or zero for static), while a jittery video has random centroid fluctuation.

#### Auto-Sync (Cross-Correlation)

```python
def luminance_curve(frames):
    return [mean(cvtColor(f, BGR2GRAY)) for f in frames]

dut_lum = luminance_curve(dut_frames) - mean(dut_lum)   # zero-mean
ref_lum = luminance_curve(ref_frames) - mean(ref_lum)   # zero-mean

# Full cross-correlation
corr = correlate(dut_lum, ref_lum, mode='full')
lag = argmax(corr) - (len(ref_lum) - 1)

# Clamp to ±25% of video length
max_offset = min(len(dut_frames), len(ref_frames)) // 4
sync_offset = clamp(lag, -max_offset, +max_offset)
```

Positive `sync_offset` means the DUT video starts `offset` frames **after** the reference. Frame comparison uses: `dut_frame[i]` vs `ref_frame[i + sync_offset]`.

**When to use AUTO vs FRAME_BY_FRAME:**
- `AUTO`: When the automation framework starts recording at slightly different wall-clock times. The luminance cross-correlation handles up to ~25% timing drift.
- `FRAME_BY_FRAME`: When the automation framework uses a synchronization signal (e.g., LED flash, synchronized ADB command) to guarantee frame-aligned starts.

---

### 4.8 AnnotationRenderer

**File:** `services/annotation_renderer.py`

#### Rendering Pipeline

```
Input: ComparisonReport + DUT image (BGR) + optional diff heatmap

Step 1: overlay_heatmap (40% alpha blend if diff provided)
    result = addWeighted(img, 0.6, heatmap, 0.4, 0)

Step 2: draw_artifacts
    for each ArtifactInstance with bbox:
        color = severity_color_map[severity]
        rectangle(canvas, (x,y), (x+w, y+h), color, thickness=2)
        putText(label, position above box, with dark background)
        putText(description[:70], position below box, small font)

Step 3: draw_grade_banner
    28px banner strip at top of image
    color = PASS→green, WARNING→yellow, FAIL→red
    text = "  OVERALL: PASS"

Step 4: build_metrics_panel (380px right sidebar)
    dark background (RGB 30,30,30)
    sections:
        "=== QUALITY METRICS ===" header
        "-- Image Quality --"    (blur, noise, exposure, color, WB, DR, CA)
        "-- NR-IQA Scores --"   (BRISQUE, NIQE, MUSIQ, CLIP-IQA)
        "-- Full-Reference --"  (PSNR, SSIM PASS/FAIL in green/red)
        "-- FAILURE REASONS --" (human-readable strings)
        "-- Video Temporal --"  (flicker, jitter, SSIM consistency)

Step 5: hstack([annotated_img, panel])
```

#### Color Coding

| Severity | BGR Color | RGB Color | Visual |
|---|---|---|---|
| NONE | (0, 200, 0) | #00C800 | Green |
| LOW | (0, 220, 220) | #DCDC00 | Yellow |
| MEDIUM | (0, 140, 255) | #FF8C00 | Orange |
| HIGH | (0, 0, 220) | #DC0000 | Red |
| CRITICAL | (200, 0, 200) | #C800C8 | Magenta |

---

### 4.9 ComparisonPipeline

**File:** `services/pipeline.py`

#### Full Execution Flow

```
run(dut_path, reference_path=None, sync_mode=AUTO, crop_preview=True, analysis_mode=None)
│
│  analysis_mode = request param ?? QIMC_ANALYSIS_MODE ?? "quality"
│
├── 1. MediaTypeDetector.detect(dut_path)
│       → MediaInfo (media_type, w, h, fps, ...)
│
├── 1b. CameraModeDetector.detect(dut_path)
│       CameraModeDetector.detect(ref_path)  (if reference provided)
│       → MediaMetadata (EXIF + camera_mode enum)
│
│       CameraModeDetector.apply_mode_adjustments(camera_mode, get_settings())
│       → effective_settings (mode-adjusted thresholds)
│       _request_settings.set(effective_settings)  ← pushed to ContextVar
│       [all downstream get_settings() calls now see mode-corrected thresholds]
│
├── 2. [if IMAGE and crop_preview and is_preview]
│       PreviewCropper.crop_image(dut_img)   → (cropped_dut, CropResult)
│       PreviewCropper.crop_image(ref_img)   → (cropped_ref, CropResult)
│
├── 3a. [if IMAGE]
│       ── always run (both modes) ──────────────────────────────────────
│       FunctionalityChecker.check(dut_img, ref_img, camera_mode)
│           → functional_grade, functional_reasons
│       QualityMetricsExtractor.extract(dut_img) → QualityMetrics
│       ArtifactDetector.detect(dut_img)          → ArtifactReport
│       ── quality mode only ────────────────────────────────────────────
│       [if analysis_mode == "quality"]
│           NoReferenceAnalyzer.analyze(dut_img)     → NoReferenceScores
│           [if reference]
│               ReferenceComparator.compare(ref, dut)
│                   → (FullReferenceScores, diff_heatmap)
│
├── 3b. [if VIDEO]
│       [if crop_preview]
│           PreviewCropper.crop_video_frame(dut_cap)  → (bbox, CropResult)
│           PreviewCropper.crop_video_frame(ref_cap)  → (bbox, CropResult)
│       VideoAnalyzer.analyze(dut, ref, sync_mode, bbox, ref_bbox)
│           → VideoAnalysisResult
│               (quality_metrics, artifacts, temporal)
│       FunctionalityChecker.check_video_sequence(frames)
│           → black_frame_count, frozen_frame_count
│           → stored in VideoTemporalMetrics
│
├── 4. Build ComparisonReport
│       MetadataComparison.build(dut_meta, ref_meta)  (compare mode)
│       report.analysis_mode = effective analysis_mode
│       report.functional_grade = functional_grade
│       report.functional_reasons = functional_reasons
│       [if analysis_mode == "quality"]
│           report.compute_overall_grade()
│               → aggregates FR + QM + artifact + temporal failures
│               → appends [Mode] camera mode notes (advisory)
│               → appends [Metadata] comparison notes (advisory)
│               → populates failure_reasons list
│
├── 5. AnnotationRenderer.render(report, img, diff_heatmap) → annotated_bgr
│       AnnotationRenderer.save(annotated_bgr, reports_dir/{id}_annotated.png)
│
├── 6. [if diff_heatmap]
│       AnnotationRenderer.save(diff_heatmap, reports_dir/{id}_diff.png)
│
├── 7. Return ComparisonReport
│
└── [finally] _request_settings.reset(mode_token)  ← ContextVar cleaned up
```

#### Grade Computation Logic

```python
def compute_overall_grade(self) -> QualityGrade:
    grade = PASS
    reasons = []

    # FR metric failures → FAIL (most strict)
    if fr_scores and fr_scores.any_failed:
        grade = FAIL
        reasons += fr_scores.failure_reasons()

    # Standard quality metric failures → FAIL
    if quality_metrics.blur_grade == FAIL:
        grade = FAIL
        reasons.append("Image is blurry: ...")
    if quality_metrics.noise_grade == FAIL:
        grade = FAIL
        reasons.append("High noise: ...")
    if highlight_clipping_pct > threshold:
        grade = FAIL
        reasons.append("Highlight clipping: ...")

    # Artifact failures → FAIL
    if artifacts.overall_severity in (HIGH, CRITICAL):
        grade = FAIL
        reasons += artifacts.failure_reasons()

    # Temporal video failures → FAIL
    if video_temporal.flicker_grade == FAIL:
        grade = FAIL
    if video_temporal.jitter_grade == FAIL:
        grade = FAIL

    # NR score advisory — WARNING only (not FAIL)
    if brisque > 60 and grade == PASS:
        grade = WARNING
        reasons.append("BRISQUE score indicates poor quality")

    # Camera mode notes — advisory only, never cause FAIL  ← NEW
    for note in dut_metadata.mode_notes:
        reasons.append(f"[Mode] {note}")

    # Metadata comparison notes — advisory only           ← NEW
    for note in metadata_comparison.notes:
        reasons.append(f"[Metadata] {note}")

    return grade
```

**Note:** Thresholds used in the quality checks above (`blur_threshold`, `noise_threshold`, `highlight_clip_threshold`, `shadow_clip_threshold`) are read from `get_settings()` at evaluation time. Because the pipeline has already pushed mode-adjusted settings into the `_request_settings` ContextVar before calling `compute_overall_grade()`, the threshold values are automatically the mode-corrected ones — no special-casing needed in the grade computation itself.

---

### 4.10 FunctionalityChecker

**File:** `services/functionality_checker.py`

#### 4.10.1 Purpose

The `FunctionalityChecker` answers a single binary question at very high speed: **"Is the camera producing a valid image at all?"** It uses only classical OpenCV operations and completes in ~5 ms, making it suitable as the inner loop of a fully automated test that runs hundreds of test steps per hour.

It is always executed in both `functional` and `quality` analysis modes. In `functional` mode it is the primary (and fastest) verdict. In `quality` mode its result is surfaced as `functional_grade` alongside the full IQA `overall_grade`.

#### 4.10.2 Image Checks

```python
def check(img_bgr, ref_img=None, camera_mode="unknown") -> tuple[QualityGrade, list[str]]:
```

Checks are evaluated in order; the first FAIL or WARNING that applies is recorded, but all checks run so the full reason list is populated:

| # | Check | Condition | Grade |
|---|---|---|---|
| 1 | **Black frame** | `mean(luma) < 8` (or `< 4` if Night mode) | FAIL |
| 2 | **Blown / white frame** | `mean(luma) > 248` AND `std(luma) < 15` | FAIL |
| 3 | **Uniform frame** | `std(all pixels) < 3` | FAIL |
| 4 | **Edge density** | `count(Canny edges) / total_pixels < 0.010` | WARNING |
| 5 | **Absolute blur floor** | `var(Laplacian) < 1.0` | WARNING |
| 6 | **Scene mismatch** | `histogram_corr(dut, ref) < 0.30` | FAIL |
| 6a | _(looser)_ | `histogram_corr(dut, ref) < 0.60` | WARNING |

**Black frame threshold** is halved in Night mode to allow for intentionally dark long-exposure captures that may have very low mean luminance.

**Scene mismatch** uses colour histogram correlation (OpenCV `HISTCMP_CORREL`), computed on 3-channel 32-bin normalized histograms. This is spatially invariant — robust to minor positional shifts and moderate exposure differences between DUT and REF, but sensitive to capturing a completely different scene.

#### 4.10.3 Video Sequence Checks

```python
def check_video_sequence(frames: list[np.ndarray]) -> tuple[int, int]:
    # Returns (black_frame_count, frozen_frame_count)
```

For each consecutive pair of frames:

```
abs_diff = mean(|frame_a - frame_b|) / 255

if abs_diff < 0.003:
    consecutive_frozen += 1
    if consecutive_frozen >= 3:
        frozen_frame_count += 1   # at least 3 pairs frozen = true freeze
else:
    consecutive_frozen = 0

if mean(luma(frame)) < 8:
    black_frame_count += 1
```

The results are stored in `VideoTemporalMetrics.black_frame_count` and `VideoTemporalMetrics.frozen_frame_count` and cause FAIL/WARNING in `VideoTemporalMetrics.failure_reasons()`.

#### 4.10.4 Histogram Correlation Implementation

```python
@staticmethod
def _histogram_correlation(img_a: np.ndarray, img_b: np.ndarray) -> float:
    hists = []
    for img in (img_a, img_b):
        h = cv2.calcHist([img], [0, 1, 2], None, [32, 32, 32], [0,256,0,256,0,256])
        cv2.normalize(h, h)
        hists.append(h)
    return cv2.compareHist(hists[0], hists[1], cv2.HISTCMP_CORREL)
```

Returns a value in `[-1, 1]` where 1.0 means identical distributions. Values below 0.60 indicate the DUT is unlikely to be showing the same scene as the reference.

#### 4.10.5 Integration with analysis_mode

The pipeline always calls `FunctionalityChecker.check()` for images, regardless of `analysis_mode`:

```
pipeline.run(analysis_mode="functional" | "quality")
│
├── FunctionalityChecker.check(dut_img, ref_img, camera_mode)
│       → functional_grade, functional_reasons
│
├── [if analysis_mode == "quality"]
│       QualityMetricsExtractor.extract(...)
│       NoReferenceAnalyzer.analyze(...)     ← skipped in functional mode
│       ArtifactDetector.detect(...)
│       ReferenceComparator.compare(...)     ← skipped in functional mode
│
└── Build ComparisonReport with both functional_grade and overall_grade
    [in functional mode: overall_grade is not computed, nr_scores and fr_scores are null]
```

This design keeps the fast path genuinely fast (no neural model inference) while making both grades available in `quality` mode for independent tracking.

#### 4.10.6 Model Preload Behaviour

In `functional` mode the CLI skips `pipeline.preload_models()` entirely — neural IQA models are never loaded, saving 2–10 seconds of startup time and 200–600 MB of RAM. This makes the CLI usable in low-memory CI environments.

---

## 5. API Layer

### 5.1 FastAPI Application Factory

The `create_app()` factory pattern (rather than a module-level singleton) enables test isolation — each test can call `create_app()` with fresh state.

The `lifespan` context manager handles:
1. Creating the reports directory
2. Calling `pipeline.preload_models()` — downloads and loads all configured neural models **before** accepting requests
3. Logging startup configuration

### 5.2 File Upload Handling

Uploaded files are streamed to temporary files using Python's `tempfile.NamedTemporaryFile`:

```python
async def _save_upload(upload: UploadFile, suffix: str) -> Path:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    content = await upload.read()
    tmp.write(content)
    tmp.flush()
    return Path(tmp.name)
```

Temp files are deleted in `finally` blocks after pipeline processing completes. File extension is preserved in the suffix to allow `cv2.VideoCapture` to use format-specific decoders.

### 5.3 Request Middleware

`RequestContextMiddleware` adds to every response:
- `X-Request-Id`: UUID hex for log correlation
- `X-Response-Time-Ms`: server-side processing time

Both are logged at `INFO` level for every request.

### 5.4 Dependency Injection

```python
# dependencies.py
@lru_cache(maxsize=1)
def get_pipeline() -> ComparisonPipeline:
    return ComparisonPipeline()

@lru_cache(maxsize=1)
def get_report_store() -> ReportStore:
    return ReportStore(settings.reports_dir)
```

The `lru_cache` ensures exactly one `ComparisonPipeline` and one `ReportStore` exist per process. In tests, `get_pipeline.cache_clear()` resets the singleton between tests.

---

## 6. Storage Layer

### 6.1 SQLite Schema

```sql
CREATE TABLE reports (
    report_id       TEXT PRIMARY KEY,
    timestamp       TEXT NOT NULL,          -- ISO 8601
    media_type      TEXT NOT NULL,          -- MediaType enum value
    dut_file        TEXT NOT NULL,          -- original DUT filename
    reference_file  TEXT,                  -- original reference filename (nullable)
    overall_grade   TEXT NOT NULL,          -- QualityGrade enum value
    processing_time_ms INTEGER,
    json_path       TEXT NOT NULL,          -- absolute path to JSON report
    annotated_path  TEXT,                  -- absolute path to annotated PNG
    diff_path       TEXT                   -- absolute path to diff PNG
);

CREATE INDEX idx_reports_timestamp ON reports(timestamp DESC);
CREATE INDEX idx_reports_grade      ON reports(overall_grade);
CREATE INDEX idx_reports_media_type ON reports(media_type);
```

The SQLite index enables fast filtering by grade and media_type for the `GET /reports` endpoint without loading full JSON reports.

### 6.2 JSON Report Files

Full `ComparisonReport` is serialized using `pydantic`'s `model_dump_json(indent=2)`. All enum values are serialized as strings (e.g., `"pass"`, `"image_captured"`). `datetime` fields are ISO 8601 strings. `Path` fields are converted to `str`.

### 6.3 Storage Layout

```
data/reports/
├── index.db                              ← SQLite index
├── a3f9c1b2d4e5.json                     ← Full report JSON
├── a3f9c1b2d4e5_annotated.png            ← Annotated image
├── a3f9c1b2d4e5_diff.png                 ← Diff heatmap (if reference)
├── b7e2c3d1f0a8.json
├── b7e2c3d1f0a8_annotated.png
...
```

---

## 7. Configuration System

### 7.1 Pydantic Settings

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="QIMC_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",    # ignore unknown QIMC_* vars
    )
```

Priority (highest to lowest):
1. Environment variables set in shell
2. `.env` file in working directory
3. Pydantic `Field(default=...)` values

### 7.2 Complete Settings Reference

| Setting | Type | Default | Description |
|---|---|---|---|
| `quality_profile` | str | `""` | Preset: `low` \| `medium` \| `high` \| `critical`. When set, overrides all threshold fields below via `model_post_init`. |
| `analysis_mode` | str | `"quality"` | Default analysis depth. `"functional"` — fast path (~50 ms): `FunctionalityChecker` + `QualityMetricsExtractor` + `ArtifactDetector`, no neural IQA. `"quality"` — full pipeline (default). Overridable per-request via form param. |
| `device` | str | `"auto"` | Compute device: `"auto"` \| `"cuda"` \| `"cpu"` |
| `nr_metrics` | str | `"brisque,niqe"` | Comma-separated NR metric names (pyiqa) |
| `use_neural_nr` | bool | `False` | Enable GPU neural NR metrics |
| `neural_nr_metric` | str | `"musiq"` | Which neural NR metric: `"musiq"` or `"clipiqa+"` |
| `fr_metrics` | str | `"ssim,ms_ssim,psnr,lpips"` | FR metrics to compute when reference is provided |
| `ssim_threshold` | float | `0.85` | SSIM fail threshold (below → FAIL) |
| `psnr_threshold` | float | `30.0` | PSNR fail threshold in dB (below → FAIL) |
| `lpips_threshold` | float | `0.15` | LPIPS fail threshold (above → FAIL) |
| `dists_threshold` | float | `0.15` | DISTS fail threshold (above → FAIL) |
| `blur_threshold` | float | `100.0` | Laplacian variance below this → blurry flag |
| `noise_threshold` | float | `8.0` | Noise sigma above this → noisy flag |
| `highlight_clip_threshold` | float | `1.0` | % pixels > 250 above this → FAIL |
| `shadow_clip_threshold` | float | `1.0` | % pixels < 5 above this → FAIL |
| `artifact_hot_pixel_high_pct` | float | `0.001` | Hot-pixel frame fraction above which severity escalates to HIGH (default 0.1%) |
| `artifact_lens_flare_high_count` | int | `3` | Flare blob count above which severity is HIGH |
| `artifact_banding_ratio_high` | float | `0.50` | FFT band-energy ratio above which banding severity is HIGH |
| `artifact_blurry_high_pct` | float | `0.50` | Fraction of blurry tiles above which severity is HIGH |
| `video_sample_fps` | float | `2.0` | Frames per second to sample from video |
| `motion_flow_threshold` | float | `2.0` | Mean optical flow (px); above → VIDEO_MOTION |
| `preview_crop_enabled` | bool | `True` | Auto-crop camera UI chrome |
| `reports_dir` | Path | `./data/reports` | Report storage directory |
| `host` | str | `"0.0.0.0"` | Server bind address |
| `port` | int | `8080` | Server port |
| `log_level` | str | `"info"` | Uvicorn log level |

### 7.3 Quality Profile System

The profile system allows switching all quality thresholds with a single environment variable, avoiding the need to maintain per-environment `.env` files with many individual overrides.

**Implementation:** profiles are stored as a module-level `_PROFILES` dict in `config.py`. After pydantic loads all fields from the environment (including any individual threshold overrides), `model_post_init` applies the selected profile, unconditionally overwriting each field:

```python
_PROFILES: dict[str, dict] = {
    "low": {
        "ssim_threshold": 0.35, "psnr_threshold": 12.0,
        "lpips_threshold": 0.60, "dists_threshold": 0.50,
        "blur_threshold": 40.0, "noise_threshold": 20.0,
        "highlight_clip_threshold": 5.0, "shadow_clip_threshold": 5.0,
        "artifact_hot_pixel_high_pct": 0.05,
        "artifact_lens_flare_high_count": 30,
        "artifact_banding_ratio_high": 0.75,
        "artifact_blurry_high_pct": 0.80,
    },
    "medium": { ... },   # semi-stable indoor
    "high":   { ... },   # stable lightbox + tripod
    "critical": { ... }, # robotic / pixel-aligned
}

def model_post_init(self, __context: object) -> None:
    profile = self.quality_profile.strip().lower()
    if profile and profile in _PROFILES:
        for field_name, value in _PROFILES[profile].items():
            object.__setattr__(self, field_name, value)
```

`object.__setattr__` is required because pydantic v2 models are frozen-like after construction; this bypasses the model validator to mutate fields in-place.

**Profile selection guide:**

| Profile | Target Environment | Key Relaxation |
|---|---|---|
| `low` | CoreXY rail with vibration, outdoor, handheld | PSNR ≥ 12 dB, LPIPS ≤ 0.60, flare HIGH > 30 blobs |
| `medium` | Indoor lightbox with minor vibration | PSNR ≥ 20 dB, LPIPS ≤ 0.40 |
| `high` | Stable lightbox + tripod (default lab) | PSNR ≥ 28 dB, LPIPS ≤ 0.20 |
| `critical` | Robotic arm, pixel-aligned fixture | PSNR ≥ 35 dB, LPIPS ≤ 0.08, flare HIGH > 2 blobs |

**Artifact severity thresholds in profiles:**

The 4 artifact threshold fields control when an artifact escalates from MEDIUM to HIGH severity (HIGH triggers a failure reason in the report). The profile values reflect the false-positive rates expected in each capture environment:

- **Outdoor / unstable (`low`):** bright sky regions create dozens of "lens flare" blobs; 4K sensors produce thousands of "hot pixels" from bright highlights; sky gradients produce "banding". Thresholds are set loose enough that only extreme cases flag HIGH.
- **Controlled studio (`critical`):** a single extra flare blob or 0.05% of pixels being "hot" is significant because the rig eliminates all natural sources of these artifacts.

### 7.4 Device Resolution

```python
def resolve_device(self) -> str:
    if self.device == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return self.device  # "cuda" or "cpu" explicit override
```

Called at service startup and embedded in the `/health` response so operators can confirm the device being used.

### 7.5 Singleton Pattern

```python
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
```

Settings are parsed once per process. In tests, `get_settings.cache_clear()` + `monkeypatch.setenv()` allows per-test configuration overrides without stale cached values.

**ArtifactDetector integration:** `ArtifactDetector.detect()` calls `get_settings()` at the start of each detection run (not at construction time), so threshold changes via environment variable reload or test monkeypatching take effect without restarting the service.

---

## 8. Neural Model Loading Strategy

### 8.1 pyiqa Model Caching

```python
class _ModelCache:
    _models: dict[str, object] = {}

    def get(self, name: str, device: str):
        key = f"{name}:{device}"
        if key not in self._models:
            model = pyiqa.create_metric(name, device=device)
            self._models[key] = model
        return self._models[key]
```

Models are keyed by `(name, device)` to support running the same model on multiple devices (e.g., testing on both CPU and GPU).

### 8.2 First-Request Model Download

On first use, `pyiqa.create_metric()` downloads pre-trained weights from the internet to `~/.cache/pyiqa/`. Subsequent calls use the cached weights. This means the first request after deployment may be slow; `preload_models()` at startup mitigates this for the production server path.

### 8.3 Graceful Fallback

```python
try:
    self._models[key] = pyiqa.create_metric(name, device=device)
except Exception as exc:
    log.warning("Cannot load pyiqa model '%s': %s", name, exc)
    self._models[key] = None  # sentinel: model unavailable

# Usage:
model = _cache.get(name, device)
if model is None:
    return None  # score omitted from report, not an error
```

If a model fails to load (e.g., CUDA out of memory, network unavailable), its score is `null` in the report. The pipeline continues and reports other scores normally.

---

## 9. Data Flow Diagrams

### 9.1 Image Comparison Flow

```
POST /compare  (analysis_mode = "functional" | "quality")
    │
    ├─ Save dut + ref to temp files
    │
    ▼
MediaTypeDetector.detect(dut)
    │ MediaType = IMAGE_CAPTURED | IMAGE_PREVIEW
    ▼
CameraModeDetector.detect(dut) → dut_metadata (EXIF + camera_mode)
CameraModeDetector.detect(ref) → ref_metadata
apply_mode_adjustments() → push effective_settings into ContextVar
    │
[if IMAGE_PREVIEW and crop_preview]
PreviewCropper.crop_image(dut_img)  ────────────────── CropResult
PreviewCropper.crop_image(ref_img)  ────────────────── CropResult
    │
    ▼  ── ALWAYS run (both modes) ──────────────────────────────────────
FunctionalityChecker.check(dut_img, ref_img, camera_mode)
    │   → functional_grade, functional_reasons
    ▼
QualityMetricsExtractor.extract(dut_img)  ───────────── QualityMetrics (DUT)
    │
ArtifactDetector.detect(dut_img)  ──────────────────── ArtifactReport
    │
    ▼  ── quality mode only ─────────────────────────────────────────────
[if analysis_mode == "quality"]
    NoReferenceAnalyzer.analyze(dut_img)  ──────────── NoReferenceScores
    │
    [if reference provided]
    QualityMetricsExtractor.extract(ref_img)  ──────── QualityMetrics (REF)
    ReferenceComparator.compare(ref_img, dut_img)
        ├── align (SIFT + homography)
        ├── PSNR, SSIM, MS-SSIM, LPIPS, DISTS  ─────── FullReferenceScores
        └── absdiff + JET colormap  ───────────────── diff_heatmap (BGR)
    │
    ▼
ComparisonReport built with:
    analysis_mode, functional_grade, functional_reasons
    [quality mode] compute_overall_grade()
        Aggregates FR + QM + artifact failures
        + [Mode] and [Metadata] advisory notes
        → overall_grade + failure_reasons
    ▼
AnnotationRenderer.render(report, dut_img, diff_heatmap)
    │ Heatmap overlay + artifact boxes + grade banner + metrics panel
    ▼
Save annotated PNG  ──────────────── data/reports/{id}_annotated.png
Save diff PNG       ──────────────── data/reports/{id}_diff.png
Save JSON report    ──────────────── data/reports/{id}.json
Update SQLite index ──────────────── data/reports/index.db
    │
    ▼
Return ComparisonReport JSON
```

### 9.2 Video Analysis Flow

```
POST /compare (video)
    │
    ▼
MediaTypeDetector.detect(dut) → VIDEO_STATIC | VIDEO_MOTION
    │
[if crop_preview]
PreviewCropper.crop_video_frame(dut_cap)  ─────── (bbox, CropResult)
PreviewCropper.crop_video_frame(ref_cap)  ─────── (bbox, CropResult)
    │
    ▼
VideoAnalyzer.analyze(dut, ref, sync_mode, bbox)
    │
    ├── _extract_frames(dut, bbox)  ─────────── List[frame BGR]
    ├── _extract_frames(ref, bbox)  ─────────── List[frame BGR]  (if ref)
    │
    ├── [if AUTO sync] _find_sync_offset()
    │       luminance_curves → cross_correlate → offset (frames)
    │
    ├── _compute_temporal()
    │       ├── _compute_flicker()  ─ FFT of luminance → flicker_score
    │       ├── _compute_jitter()   ─ DISOpticalFlow centroid std → jitter_score
    │       └── _frame_to_frame_ssim() → temporal_ssim_mean, std
    │
    ├── Per-frame QualityMetrics (sampled)  → aggregated
    ├── ArtifactDetector.detect(worst_frame) → ArtifactReport
    ├── FunctionalityChecker.check_video_sequence(frames)
    │       → black_frame_count, frozen_frame_count
    │       → stored in VideoTemporalMetrics
    └── VideoAnalysisResult
            (temporal, quality_metrics, artifacts, worst_frame_index)
    │
    ▼
extract_worst_frame(dut_path, worst_frame_index, bbox)  → BGR image
    │
AnnotationRenderer.render(report, worst_frame) → annotated PNG
    │
Save + return ComparisonReport
```

---

## 10. Error Handling Strategy

### 10.1 Service Level

| Error Type | Handling |
|---|---|
| File not found | `FileNotFoundError` → HTTP 404 |
| Invalid media format | `ValueError` from cv2 → HTTP 500 with detail |
| Neural model load failure | Logged WARNING, score = `null`, pipeline continues |
| Individual metric failure | Try/except per metric, score = `null`, pipeline continues |
| Invalid `sync_mode` value | Validated at API layer → HTTP 422 |
| Reports directory not writable | `OSError` → HTTP 500 |
| Unknown `report_id` | `None` from SQLite → HTTP 404 |

### 10.2 Metric-Level Try/Except Pattern

Every metric computation is wrapped:

```python
try:
    val = some_metric(ref, dut)
    fr.ssim = MetricResult(value=val, threshold=..., ...)
    fr.ssim.evaluate()
except Exception as exc:
    log.warning("SSIM computation failed: %s", exc)
    # fr.ssim remains default MetricResult() with value=None
```

This means a broken CUDA driver will not prevent classical metrics from running.

### 10.3 Temp File Cleanup

All uploaded temp files are cleaned up in `finally` blocks regardless of pipeline success or failure:

```python
try:
    report = pipeline.run(dut_path=dut_path, ...)
    store.save(report)
    return JSONResponse(...)
finally:
    dut_path.unlink(missing_ok=True)
    if ref_path:
        ref_path.unlink(missing_ok=True)
```

---

## 11. Performance Characteristics

### 11.1 Typical Processing Times (CPU, Intel i7-12th gen)

| Operation | Time |
|---|---|
| MediaTypeDetector (image) | < 50 ms |
| PreviewCropper (image) | 50–150 ms |
| QualityMetricsExtractor (2MP image) | 200–400 ms |
| ArtifactDetector (2MP image) | 300–600 ms |
| SSIM (2MP image, scikit-image) | 100–200 ms |
| PSNR (OpenCV) | < 20 ms |
| BRISQUE (pyiqa, CPU) | 300–800 ms |
| NIQE (pyiqa, CPU) | 200–500 ms |
| LPIPS (pyiqa, CPU) | 1–3 s |
| SIFT alignment (2MP image) | 200–500 ms |
| AnnotationRenderer | 100–300 ms |
| **Total image NR analysis** | **~1–2 s** |
| **Total image FR comparison** | **~2–4 s** |
| **Total video (30s, 2fps sample)** | **~5–15 s** |
| **Functional mode (image)** | **< 100 ms** |
| FunctionalityChecker | 5–15 ms |
| QualityMetricsExtractor | 200–400 ms |
| ArtifactDetector | 300–600 ms |
| _(no neural IQA, no FR comparator)_ | |

### 11.2 GPU Acceleration (RTX 3060)

| Metric | CPU Time | GPU Time | Speedup |
|---|---|---|---|
| LPIPS | 2 s | 80 ms | ~25× |
| MUSIQ | 4 s | 150 ms | ~27× |
| CLIP-IQA+ | 3 s | 120 ms | ~25× |
| DISTS | 3 s | 100 ms | ~30× |

### 11.3 Memory Usage

| Component | RAM |
|---|---|
| BRISQUE + NIQE models | ~50 MB |
| LPIPS (AlexNet backbone) | ~250 MB |
| MUSIQ (transformer) | ~400 MB |
| CLIP-IQA+ (ViT-B/32) | ~600 MB |
| Per-request peak (2MP image) | ~200–400 MB |

### 11.4 Concurrency

FastAPI with Uvicorn runs in a single process (default). Since all image processing uses the GIL-releasing OpenCV/NumPy operations and async file I/O, the service handles concurrent requests reasonably well on CPU. For GPU-accelerated inference, requests are serialized by the GPU.

For production high-throughput, run with `--workers N` (Gunicorn with Uvicorn workers) — each worker loads its own model copy.

---

## 12. Threshold Calibration Guide

The recommended approach is to **use a quality profile** (`QIMC_QUALITY_PROFILE`) rather than tuning individual thresholds. Profiles encode calibrated threshold sets for different capture environments and reduce the risk of inconsistent threshold combinations.

### 12.1 Choosing a Profile

Run the service against 20–50 known-good captures from your environment with each profile. The right profile is the one where known-good images consistently return `PASS` and known-bad images (blurry, noisy, misaligned) consistently return `FAIL`.

```bash
# Test with each profile
QIMC_QUALITY_PROFILE=low   qimc analyze media/good_sample.jpg
QIMC_QUALITY_PROFILE=high  qimc analyze media/good_sample.jpg
```

**Starting points by environment:**

| If you see false FAILs on good images due to... | Use profile |
|---|---|
| Camera rig vibration causing PSNR/SSIM to be low | `low` |
| Bright sky causing dozens of "lens flare" detections | `low` |
| 4K sensor bright highlights causing "hot pixel" hits | `low` |
| Indoor lightbox, minor positional jitter | `medium` |
| Stable controlled lightbox, tripod-mounted | `high` |
| Robotic arm, sub-millimetre repositioning accuracy | `critical` |

### 12.2 SSIM Threshold (manual tuning)

SSIM = 1.0 is identical images. In practice:

| SSIM | Meaning |
|---|---|
| > 0.95 | Imperceptible difference |
| 0.90–0.95 | Subtle; generally acceptable for device-to-device |
| 0.85–0.90 | Noticeable but minor degradation |
| 0.80–0.85 | Visible difference; likely rig positional error or processing change |
| < 0.80 | Significant quality regression |

**Calibration procedure:** Run 50–100 known-good reference comparisons. Compute the mean and standard deviation of the SSIM distribution. Set `QIMC_SSIM_THRESHOLD = mean - 2*std` to allow 97.5% of good captures to pass.

### 12.3 PSNR Threshold (manual tuning)

PSNR depends on both pixel differences and scene dynamic range:

| PSNR | Meaning |
|---|---|
| > 40 dB | Imperceptible; near-identical captures |
| 35–40 dB | Very close; excellent device match |
| 28–35 dB | Minor difference; tolerable for unstable rigs |
| 20–28 dB | Significant pixel difference; investigate |
| < 20 dB | Large difference; likely alignment or exposure issue |

**Note:** PSNR is sensitive to minor rig repositioning. A 1-pixel shift can drop PSNR by 3–5 dB even with identical rendering. For unstable rigs, use `low` profile (PSNR ≥ 12 dB) or rely on LPIPS/SSIM instead, which are more robust to geometric variation.

### 12.4 Blur Threshold (manual tuning)

Laplacian variance is highly scene-dependent:

| Scene content | Typical blur_score |
|---|---|
| Plain wall / grey card | 10–50 |
| Indoor lightbox (moderate texture) | 100–500 |
| High-detail outdoor scene | 500–5000 |
| Macro / close-up with fine texture | 1000–10000 |

`QIMC_BLUR_THRESHOLD=100` is appropriate for typical indoor lightbox scenes. For smooth scenes (colour chart, grey card), lower to 20–40. For macro or high-detail scenes, raise to 200–500.

### 12.5 Noise Threshold (manual tuning)

Noise sigma in flat regions, rough correlation with ISO:

| Noise sigma | ISO equivalent |
|---|---|
| < 2 | ISO 50–100 (excellent) |
| 2–5 | ISO 200–400 (good) |
| 5–10 | ISO 800–1600 (acceptable) |
| 10–20 | ISO 3200–6400 (poor) |
| > 20 | ISO 12800+ (unacceptable) |

`QIMC_NOISE_THRESHOLD=8.0` flags anything above ISO ~1600 equivalent. Adjust based on the target device's ISO range.

### 12.6 LPIPS Threshold (manual tuning)

LPIPS reflects human perceptual similarity:

| LPIPS | Meaning |
|---|---|
| < 0.05 | Near-identical; imperceptible |
| 0.05–0.15 | Minor; within normal device-to-device variation |
| 0.15–0.30 | Noticeable; different noise level, sharpening, or color rendering |
| > 0.30 | Large perceptual difference; investigate rendering pipeline |

`QIMC_LPIPS_THRESHOLD=0.15` is appropriate for controlled camera regression testing. For unstable rigs, `low` profile relaxes this to 0.60.

### 12.7 Artifact Severity Threshold Calibration

Each artifact has a HIGH-severity threshold that triggers a failure reason. Below HIGH, the artifact is logged but does not cause FAIL grade.

#### Hot Pixel (`artifact_hot_pixel_high_pct`)

Measures what fraction of the frame contains isolated bright pixels (> 4σ above local 5×5 mean).

- In a dark environment or long exposure, **sensor heat** can produce hot pixels on any device. Typical rates: 0.001–0.005% under normal conditions.
- In bright outdoor photography or 4K, **highlight regions** can appear as "hot pixels" due to the local-mean comparison. Rates can reach 1–2% from natural bright areas.
- Set `low` profile (5%) for outdoor/bright captures; `high`/`critical` (0.001–0.0005%) for controlled lightbox.

#### Lens Flare (`artifact_lens_flare_high_count`)

Counts bright connected blobs in the top 0.1% of the luminance distribution.

- **Outdoor / bright sky:** even a well-exposed sky scene can produce 10–30 blob detections from bright cloud patches, sun reflection on water, etc. Use `low` profile (30 blobs).
- **Controlled lightbox:** genuine flare from direct light-source contamination typically produces 1–5 blobs. Use `high`/`critical` profile (2–3 blobs).

#### Banding (`artifact_banding_ratio_high`)

Measures the ratio of periodic frequency energy in row-gradient FFT. Genuine banding (bit-depth quantization, HDR tone-mapping) produces strong peaks at 3–10 bands per 100 pixels.

- Natural images with strong horizontal structure (horizon, shelf) can produce FFT energy ratios of 0.4–0.6 without actual banding. Use `low` profile (0.75) for outdoor scenes.
- In a lightbox with uniform illumination, ratio > 0.50 is diagnostic of real banding.

#### Blurry Region (`artifact_blurry_high_pct`)

Measures what percentage of image tiles (64×64 px) fall below `global_sharpness / 3`.

- For large-aperture / narrow-DOF shots, background blur is intentional. 60–80% blurry tiles is expected. Use `low` profile (80%).
- For QA lightbox captures (typically small aperture, deep DOF), > 50% blurry tiles is diagnostic of focus failure or motion blur.

---

## 13. Extending the System

### 13.1 Adding a New Artifact Type

1. Add detection method to `ArtifactDetector.detect()`:

```python
def _detect_my_artifact(self, img: np.ndarray, report: ArtifactReport) -> None:
    # ... detection logic ...
    if detected:
        report.add(ArtifactInstance(
            artifact_type="my_artifact",
            severity=ArtifactSeverity.MEDIUM,
            bbox=(x, y, w, h),
            confidence=0.8,
            description="Clear human-readable description + fix hint.",
        ))
```

2. Call the method from `detect()`:

```python
def detect(self, img_bgr: np.ndarray) -> ArtifactReport:
    ...
    self._detect_my_artifact(img_bgr, report)
    return report
```

No other changes needed — the artifact will automatically appear in the report JSON and annotated image.

### 13.2 Adding a New IQA Metric

For NR metrics (no-reference):

1. Add the metric name to `QIMC_NR_METRICS` or load it in `NoReferenceAnalyzer.preload()`.
2. Add a field to `NoReferenceScores`:

```python
class NoReferenceScores(BaseModel):
    ...
    my_metric: Optional[float] = None
```

3. Run and populate in `NoReferenceAnalyzer.analyze()`:

```python
if "my_metric" in self._settings.nr_metrics_list:
    scores.my_metric = self._run_metric("my_metric", img_bgr)
```

4. Add grading logic to `_grade()`.

### 13.3 Adding a New Quality Metric

1. Add computation to `QualityMetricsExtractor.extract()`:

```python
def extract(self, img_bgr: np.ndarray) -> QualityMetrics:
    ...
    return QualityMetrics(
        ...
        my_metric=self._my_metric(gray),
    )

@staticmethod
def _my_metric(gray: np.ndarray) -> float:
    # computation
    return float(result)
```

2. Add the field to `QualityMetrics` in `models/metrics.py`.
3. Add pass/fail logic to `QualityMetrics.failure_reasons()` if applicable.
4. Add the metric to `AnnotationRenderer._build_metrics_panel()` for display.

### 13.4 Adding a New API Endpoint

1. Create `api/routes/my_route.py` with a `router = APIRouter()`.
2. Register in `api/routes/__init__.py`.
3. Include in `api/app.py`:

```python
from .routes import my_router
app.include_router(my_router)
```

---

## 14. Known Limitations and Trade-offs

### 14.1 Preview Detection Reliability

The preview detector uses heuristics (aspect ratio + status bar + shutter button). It may misclassify:
- Landscape orientation screenshots (aspect ratio flipped)
- Custom camera apps with non-standard UI layouts
- Phones with in-display fingerprint sensors that affect bottom UI area

**Mitigation:** Use `force_media_type=image_captured` or `crop_preview=false` in the API call when auto-detection is unreliable for a specific device/camera app.

### 14.2 SIFT Alignment Limitations

Spatial alignment fails when:
- Images have very different exposures (saturated or dark) — few keypoints detected
- Camera app applies strong computational photography effects between shots
- Scene has very little texture (sky, plain wall)

In these cases, the comparator falls back to direct comparison without alignment. If systematic alignment errors are observed, try `crop_preview=false` (comparison without cropping avoids coordinate system mismatch).

### 14.3 Flicker Detection at Standard Frame Rates

Mains-frequency flicker (50/60 Hz) cannot be detected in 30fps video due to Nyquist sampling theorem. Detection at these frame rates only catches flicker that manifests as visible brightness variation in the recorded frames (aliased flicker at typically 1–5 Hz envelope). For accurate flicker measurement, use a high-speed camera (240fps+) or a dedicated photometer.

### 14.4 Video Sync in Very Short Clips

The auto-sync cross-correlation requires at least 5–10 sampled frames to work reliably. For clips shorter than 2–3 seconds with `QIMC_VIDEO_SAMPLE_FPS=2.0`, use `sync_mode=frame_by_frame` instead.

### 14.5 BRISQUE/NIQE as Pass/Fail Criteria

BRISQUE and NIQE were trained on natural distortion types (JPEG, Gaussian noise, blur). They may not accurately reflect perceptual quality for:
- HDR tone-mapped images
- Computational photography output (multi-frame stacking)
- Night mode images (high local contrast adaptation)

These scores are therefore used as **advisory** (WARNING at most) rather than hard FAIL criteria. Full-reference metrics (SSIM, LPIPS) and artifact detectors are the primary pass/fail signals.

### 14.6 Color Assessment Requires Neutral Scene

Color cast, white balance deviation, and chromatic aberration scores assume a scene with spectrally neutral regions. Saturated scenes (red walls, blue sky) will produce misleading color cast values.

**Recommendation:** Include a grey card or ColorChecker in the lightbox scene for accurate color assessment.

---

*Document version 1.0.0 — generated alongside code implementation.*
