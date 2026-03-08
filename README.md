# QA Intelli Media Comparator

> AI-powered media comparison microservice for Smartphone Camera QA — detects artifacts, quantifies quality, and annotates failures with human-readable diagnostics.

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Architecture at a Glance](#architecture-at-a-glance)
4. [Requirements](#requirements)
5. [Installation](#installation)
6. [Configuration](#configuration)
7. [Running the Service](#running-the-service)
8. [API Reference](#api-reference)
9. [CLI Usage](#cli-usage)
10. [Output Format](#output-format)
11. [Supported Media Types](#supported-media-types)
12. [Quality Metrics Explained](#quality-metrics-explained)
13. [Artifact Detection Catalogue](#artifact-detection-catalogue)
14. [Integration with Automation Framework](#integration-with-automation-framework)
15. [Testing](#testing)
16. [Project Structure](#project-structure)

---

## Overview

**QA Intelli Media Comparator** is a REST microservice designed for professional smartphone camera quality assurance. It is built to work alongside an existing UI automation framework (UIAutomator2-based) and a controlled lightbox test environment equipped with a CoreXY rail positioning system.

### Test Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Photography Lightbox                         │
│   ┌──────────┐   CoreXY Rail   ┌──────────┐                        │
│   │ Reference│ ◄─────────────► │   DUT    │                        │
│   │  Phone   │   (same scene)  │  Phone   │                        │
│   └────┬─────┘                 └────┬─────┘                        │
│        │ ADB capture               │ ADB capture                   │
│        ▼                           ▼                               │
│   image/video                 image/video                          │
└──────────────────────────────────────────────────────────────────┬─┘
                                                                   │
                              POST /compare                        │
                         ┌────────────────────┐                   │
                         │  Media Comparator  │◄──────────────────┘
                         │   Microservice     │
                         └─────────┬──────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼               ▼
              JSON Report   Annotated PNG    Diff Heatmap
              (all metrics)  (artifact boxes) (JET colormap)
```

The service accepts media from the **Golden Reference model** and the **Device Under Test (DUT)**, automatically detects whether inputs are images or videos, whether they are live preview recordings or captured photos, crops the camera viewfinder from ADB screen recordings, and produces a detailed quality report with localized failure annotations.

---

## Features

| Capability | Details |
|---|---|
| **Auto media detection** | Image vs video (magic bytes + extension), preview vs captured, static vs motion video |
| **Preview UI cropping** | 3-strategy cascade: contour detection → saturation mask → heuristic margin crop |
| **Camera mode detection** | Reads EXIF to detect Portrait, Night, Sport, Macro, HDR, Panorama — automatically relaxes thresholds for expected optical characteristics |
| **EXIF metadata extraction** | Make/model, ISO, shutter speed, aperture, focal length, flash, scene type — included in every report |
| **Metadata comparison** | DUT vs REF side-by-side EXIF diff; flags ISO mismatch, focal-length mismatch, mode mismatch |
| **No-reference IQA** | BRISQUE, NIQE (always); MUSIQ, CLIP-IQA+ (optional, GPU) |
| **Full-reference IQA** | PSNR, SSIM, MS-SSIM, LPIPS, DISTS with pass/fail thresholds |
| **Spatial alignment** | SIFT keypoint matching + RANSAC homography before comparison |
| **Standard quality metrics** | Sharpness, noise, exposure, color cast, white balance, dynamic range, chromatic aberration, saturation |
| **Artifact detection** | 8 artifact types with bounding box, severity, and fix recommendations |
| **Video analysis** | Flicker (FFT), jitter (optical flow), temporal SSIM, worst-frame extraction |
| **Video sync** | Auto (SSIM cross-correlation) or frame-by-frame — selectable per request |
| **Annotated output** | Artifact bboxes (color-coded severity) + metrics panel + optional diff heatmap overlay |
| **REST API** | FastAPI, async, multipart file upload, JSON response |
| **CLI** | `qimc analyze` and `qimc compare` for offline use without server |
| **Report storage** | SQLite index + JSON files + PNG images, queryable via REST |
| **GPU optional** | CUDA auto-detected; full functionality without GPU |

---

## Architecture at a Glance

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FastAPI Service                             │
│  POST /compare   POST /analyze   GET /report/*   GET /health        │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                   ComparisonPipeline
                            │
         ┌──────────────────┼─────────────────────┐
         │                  │                      │
   MediaTypeDetector   PreviewCropper         (per media type)
         │                  │                      │
         ▼                  ▼              ┌────────┴────────┐
    IMAGE / VIDEO     Crop viewfinder    IMAGE            VIDEO
                      from UI chrome      │                 │
                                    ┌─────┴────┐    VideoAnalyzer
                                    │          │         │
                             QualityMetrics  Artifact   │
                             Extractor      Detector    │
                                    │          │        │
                             NoReference   Reference    │
                             Analyzer     Comparator    │
                                    └──────────┘        │
                                         │              │
                                    AnnotationRenderer  │
                                         │◄────────────┘
                                         │
                                  ComparisonReport
                                  (JSON + PNG saved)
                                         │
                                   ReportStore
                                 (SQLite + files)
```

---

## Requirements

| Requirement | Version |
|---|---|
| Python | ≥ 3.10 |
| CUDA (optional) | ≥ 11.8 (for GPU-accelerated LPIPS/MUSIQ) |
| FFmpeg | Optional (not required; OpenCV used for video) |
| RAM | ≥ 4 GB (8 GB+ recommended) |
| Disk | ≥ 2 GB (model weights for pyiqa) |

### Key Python Dependencies

| Package | Purpose |
|---|---|
| `fastapi` + `uvicorn` | REST microservice |
| `opencv-python` | Image/video processing, optical flow, artifact detection |
| `scikit-image` | SSIM computation |
| `pyiqa` | Unified IQA metric interface (BRISQUE, NIQE, LPIPS, etc.) |
| `torch` + `torchvision` | Neural model runtime (pyiqa backend) |
| `pydantic` v2 | Data models + validation |
| `pydantic-settings` | Environment variable configuration |
| `scipy` | FFT for flicker detection, local mean filtering |
| `pillow` | Image I/O for annotated PNG output |
| `rich` | Beautiful CLI output |
| `typer` | CLI framework |

---

## Installation

### 1. Clone and install

```bash
cd e:/O_Projects/QA_Intelli_Media_Comparator
pip install -e .
```

### 2. Install with development dependencies (for testing)

```bash
pip install -e ".[dev]"
```

### 3. Copy and configure environment

```bash
cp .env.example .env
# Edit .env to match your environment
```

### 4. GPU setup (optional but recommended)

If you have a CUDA-capable GPU, install PyTorch with CUDA before installing the project:

```bash
# Example for CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e .
```

Then set in `.env`:

```env
QIMC_DEVICE=cuda
QIMC_USE_NEURAL_NR=true
QIMC_NEURAL_NR_METRIC=musiq
```

---

## Configuration

All settings use the `QIMC_` prefix and can be set via environment variables or a `.env` file in the project root.

### Quality Profiles — Quick Start

The fastest way to configure the service is to pick a **quality profile** that matches your capture environment. A single line in `.env` sets all thresholds appropriately:

```env
QIMC_QUALITY_PROFILE=low       # unstable rig / outdoor / handheld
QIMC_QUALITY_PROFILE=medium    # semi-stable indoor / lightbox with vibration
QIMC_QUALITY_PROFILE=high      # stable lightbox + tripod (default for controlled QA)
QIMC_QUALITY_PROFILE=critical  # robotic / pixel-aligned / professional studio
```

When a profile is set it **overrides all individual threshold settings** below. Leave it empty (or remove it) to set thresholds manually.

#### Profile Comparison

| Threshold | `low` | `medium` | `high` | `critical` | Notes |
|---|---|---|---|---|---|
| **SSIM** ≥ | 0.35 | 0.60 | 0.80 | 0.92 | Higher = stricter structural match |
| **PSNR** ≥ | 12 dB | 20 dB | 28 dB | 35 dB | Higher = stricter signal fidelity |
| **LPIPS** ≤ | 0.60 | 0.40 | 0.20 | 0.08 | Lower = stricter perceptual match |
| **DISTS** ≤ | 0.50 | 0.35 | 0.18 | 0.08 | Lower = stricter texture match |
| **Blur** ≥ | 40 | 70 | 100 | 150 | Higher = stricter sharpness demand |
| **Noise σ** ≤ | 20 | 12 | 8 | 4 | Lower = stricter noise tolerance |
| **Highlight clip** ≤ | 5% | 2% | 1% | 0.5% | % of blown pixels allowed |
| **Shadow clip** ≤ | 5% | 2% | 1% | 0.5% | % of crushed-black pixels allowed |
| **Hot pixel** HIGH > | 5% | 1% | 0.1% | 0.05% | Frame % before HIGH severity |
| **Lens flare** HIGH > | 30 blobs | 8 blobs | 3 blobs | 2 blobs | Blob count before HIGH severity |
| **Banding** HIGH > | 0.75 | 0.65 | 0.50 | 0.40 | FFT energy ratio before HIGH |
| **Blurry region** HIGH > | 80% | 65% | 50% | 35% | Frame area % before HIGH severity |

**Which profile to choose:**

| Environment | Profile |
|---|---|
| CoreXY rail with vibration, outdoor, handheld captures | `low` |
| Indoor lightbox, minor mechanical vibration | `medium` |
| Stable lightbox + tripod, controlled studio conditions | `high` |
| Robotic arm, pixel-aligned fixture, professional studio | `critical` |

To switch profiles: edit `QIMC_QUALITY_PROFILE=` in `.env` and restart the service (settings are cached per process).

---

### Camera Mode Awareness

The service automatically reads the EXIF metadata of every image file and detects the smartphone camera mode. This matters because many modes produce optical characteristics that are **intentional** — and should not be flagged as defects:

| Mode | Expected characteristic | Auto-adjustment |
|---|---|---|
| **Portrait** | Intentional background bokeh; low blur score in background regions | `blur_threshold × 0.15` — only extreme overall unsharpness fails |
| **Night** | High ISO + long exposure; elevated noise and slight motion blur | `noise_threshold × 4.0`, `blur_threshold × 0.30`, highlight/shadow clip × 3.0 |
| **Sport / Action** | Fast shutter but higher ISO; motion blur on fast subjects | `blur_threshold × 0.30`, `noise_threshold × 2.0` |
| **Macro** | Very shallow depth of field; background always blurry | `blur_threshold × 0.15`, `noise_threshold × 1.3` |
| **HDR** | Multiple exposures merged; extreme highlights/shadows clipped | `highlight_clip_threshold × 3.0`, `shadow_clip_threshold × 3.0` |
| **Panorama** | Stitched from many frames; minor sharpness loss at seams | `blur_threshold × 0.50`, `noise_threshold × 1.5` |
| **Auto / Landscape / Pro** | Standard expectations | No adjustment |

Mode adjustments are **multiplicative on top of the active quality profile** — so if you use `QIMC_QUALITY_PROFILE=high` (blur_threshold=100) and the image is in Night mode, the effective threshold becomes 30. This means both dimensions compose correctly.

#### Detection method (priority order)

1. **EXIF text fields** — `UserComment`, `ImageDescription`, `XPComment` are searched for mode keywords (e.g. "Portrait", "Night Sight", "Bokeh"). Confidence: 90%.
2. **EXIF SceneCaptureType** — tag values 1 (Landscape), 2 (Portrait), 3 (Night). Confidence: 85%.
3. **EXIF ExposureProgram** — tag values 7 (Portrait), 8 (Landscape). Confidence: 80%.
4. **Heuristic** — ISO ≥ 3200 + shutter ≥ 1/8 s → Night; ISO ≥ 800 + shutter < 1/250 s → Sport. Confidence: 55–70%.
5. **Default** — if no signal found → `auto` (no threshold adjustment).

#### In compare mode

When both DUT and REF are provided, the service additionally compares their EXIF metadata and surfaces differences that explain quality gaps without being defects:

- **Mode mismatch** — if DUT is in Night mode but REF was captured in Auto, the quality differences may be mode-driven, not a DUT regression.
- **ISO delta** — large ISO difference (≥ 400 ISO) means the DUT is likely noisier for a legitimate reason.
- **Exposure time delta** — ≥ 1 stop difference in shutter speed may explain brightness or motion-blur differences.
- **Aperture mismatch** — different f-numbers affect depth of field and exposure.
- **Focal length mismatch** — different zoom levels make FR-IQA comparisons (SSIM, PSNR) inherently low and should be flagged.

All metadata observations appear in `failure_reasons` prefixed with `[Mode]` or `[Metadata]` and are **advisory only** — they never cause a FAIL grade on their own.

#### What is included in the report

```json
"dut_metadata": {
  "make": "Samsung",
  "model": "SM-S926B",
  "software": "S926BXXU3BXKA",
  "datetime_original": "2026:03:08 14:23:11",
  "exposure_time": "1/2500",
  "exposure_time_s": 0.0004,
  "f_number": 1.8,
  "iso": 50,
  "focal_length_mm": 6.3,
  "focal_length_35mm": 24,
  "flash_fired": false,
  "scene_capture_type": 2,
  "camera_mode": "portrait",
  "camera_mode_confidence": 0.85,
  "camera_mode_source": "exif_scene_type",
  "mode_notes": ["Mode 'portrait' from EXIF SceneCaptureType=2."]
},
"metadata_comparison": {
  "modes_match": true,
  "dut_mode": "portrait",
  "ref_mode": "portrait",
  "iso_delta": 0,
  "exposure_delta_stops": 0.1,
  "f_number_match": true,
  "focal_length_match": true,
  "notes": []
}
```

---

### Full Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `QIMC_QUALITY_PROFILE` | _(empty)_ | Preset profile: `low` \| `medium` \| `high` \| `critical`. Overrides all thresholds below when set. |
| `QIMC_DEVICE` | `auto` | Compute device: `auto` \| `cuda` \| `cpu`. `auto` detects CUDA at runtime. |
| `QIMC_NR_METRICS` | `brisque,niqe` | Comma-separated list of no-reference metrics to compute (always CPU-compatible). |
| `QIMC_USE_NEURAL_NR` | `false` | Enable neural NR metrics (MUSIQ or CLIP-IQA). GPU strongly recommended. |
| `QIMC_NEURAL_NR_METRIC` | `musiq` | Neural NR metric: `musiq` or `clipiqa+`. |
| `QIMC_FR_METRICS` | `ssim,ms_ssim,psnr,lpips` | Full-reference metrics (used when reference file is provided). |
| **FR-IQA thresholds** | | _(ignored when `QIMC_QUALITY_PROFILE` is set)_ |
| `QIMC_SSIM_THRESHOLD` | `0.85` | SSIM score below this → FAIL. Range 0–1; higher = better. |
| `QIMC_PSNR_THRESHOLD` | `30.0` | PSNR below this (dB) → FAIL. Higher = better. |
| `QIMC_LPIPS_THRESHOLD` | `0.15` | LPIPS above this → FAIL. Range 0–1; lower = more similar. |
| `QIMC_DISTS_THRESHOLD` | `0.15` | DISTS above this → FAIL. Range 0–1; lower = better. |
| **Standard quality thresholds** | | _(ignored when `QIMC_QUALITY_PROFILE` is set)_ |
| `QIMC_BLUR_THRESHOLD` | `100.0` | Laplacian variance below this → blurry. Higher scenes need higher values. |
| `QIMC_NOISE_THRESHOLD` | `8.0` | Estimated noise sigma above this → noisy. Correlates roughly with ISO level. |
| `QIMC_HIGHLIGHT_CLIP_THRESHOLD` | `1.0` | % of blown-highlight pixels above this → FAIL. |
| `QIMC_SHADOW_CLIP_THRESHOLD` | `1.0` | % of crushed-black pixels above this → FAIL. |
| **Artifact severity thresholds** | | _(ignored when `QIMC_QUALITY_PROFILE` is set)_ |
| `QIMC_ARTIFACT_HOT_PIXEL_HIGH_PCT` | `0.001` | Fraction of frame with hot/dead pixels before HIGH severity (0.001 = 0.1%). |
| `QIMC_ARTIFACT_LENS_FLARE_HIGH_COUNT` | `3` | Number of distinct flare blobs before HIGH severity. |
| `QIMC_ARTIFACT_BANDING_RATIO_HIGH` | `0.50` | FFT band-energy ratio above which banding is HIGH severity. |
| `QIMC_ARTIFACT_BLURRY_HIGH_PCT` | `0.50` | Fraction of image covered by blurry tiles before HIGH severity. |
| **Video** | | |
| `QIMC_VIDEO_SAMPLE_FPS` | `2.0` | Frames per second to sample from video for analysis. |
| `QIMC_MOTION_FLOW_THRESHOLD` | `2.0` | Mean optical flow (px) above this → VIDEO_MOTION classification. |
| **Other** | | |
| `QIMC_PREVIEW_CROP_ENABLED` | `true` | Enable automatic preview UI chrome cropping. |
| `QIMC_REPORTS_DIR` | `./data/reports` | Directory for JSON reports and annotated images. |
| `QIMC_HOST` | `0.0.0.0` | Server bind address. |
| `QIMC_PORT` | `8080` | Server port. |
| `QIMC_LOG_LEVEL` | `info` | Log level: `debug` \| `info` \| `warning` \| `error`. |

---

## Running the Service

### Start the server

```bash
# Using Make
make run

# Using uvicorn directly
uvicorn qa_intelli_media_comparator.api.app:app --host 0.0.0.0 --port 8080

# Using the CLI with overrides
qimc serve --port 8080 --log-level debug

# Development mode (auto-reload on file change)
qimc serve --reload
```

### Verify the service is running

```bash
curl http://localhost:8080/health
```

Expected response:

```json
{
  "status": "ok",
  "version": "1.0.0",
  "device": "cpu",
  "models_loaded": ["brisque:cpu", "niqe:cpu"],
  "reports_dir": "./data/reports"
}
```

---

## API Reference

### `POST /compare`

Compare a DUT media file against an optional golden reference.

**Request** — `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `dut` | file | Yes | DUT image or video |
| `reference` | file | No | Golden reference image or video |
| `sync_mode` | string | No | `auto` (default) or `frame_by_frame` |
| `crop_preview` | bool | No | Auto-crop camera UI chrome (default: `true`) |
| `force_media_type` | string | No | Override auto-detection: `image_captured`, `image_preview`, `video_static`, `video_motion` |
| `quality_profile` | string | No | Per-request profile override: `low` \| `medium` \| `high` \| `critical`. Overrides server default for this request only. |

**Response** — `200 OK`

```json
{
  "report_id": "a3f9c1b2d4e5",
  "timestamp": "2026-03-06T10:23:45.123456",
  "media_type": "image_captured",
  "dut_file": "/tmp/tmpXYZ.jpg",
  "reference_file": "/tmp/tmpABC.jpg",
  "crop_applied": false,
  "quality_metrics": {
    "blur_score": 312.5,
    "noise_sigma": 3.2,
    "exposure_mean": 52.3,
    "highlight_clipping_pct": 0.0,
    "shadow_clipping_pct": 0.0,
    "color_cast_r": 2.1,
    "color_cast_g": -0.8,
    "color_cast_b": -1.3,
    "white_balance_deviation": 0.041,
    "saturation_mean": 0.312,
    "dynamic_range_stops": 7.2,
    "chromatic_aberration_score": 0.8
  },
  "nr_scores": {
    "brisque": 22.4,
    "niqe": 4.1,
    "musiq": null,
    "clip_iqa": null,
    "grade": "pass"
  },
  "fr_scores": {
    "psnr": {"value": 38.2, "threshold": 30.0, "passed": true, "higher_is_better": true},
    "ssim": {"value": 0.934, "threshold": 0.85, "passed": true, "higher_is_better": true},
    "ms_ssim": {"value": 0.971, "threshold": 0.85, "passed": true, "higher_is_better": true},
    "lpips": {"value": 0.082, "threshold": 0.15, "passed": true, "higher_is_better": false},
    "dists": {"value": null, "threshold": null, "passed": null, "higher_is_better": false}
  },
  "artifacts": {
    "artifacts": [],
    "overall_severity": "none"
  },
  "video_temporal": null,
  "overall_grade": "pass",
  "failure_reasons": [],
  "annotated_image_path": "./data/reports/a3f9c1b2d4e5_annotated.png",
  "diff_image_path": "./data/reports/a3f9c1b2d4e5_diff.png",
  "processing_time_ms": 1240
}
```

**Response Headers**

| Header | Description |
|---|---|
| `X-Report-Id` | Report ID for subsequent retrieval |
| `X-Request-Id` | Unique request trace ID |
| `X-Response-Time-Ms` | Server processing time |

---

### `POST /analyze`

No-reference quality analysis of a single media file.

**Request** — `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `media` | file | Yes | Image or video to analyze |
| `crop_preview` | bool | No | Auto-crop camera UI chrome (default: `true`) |
| `quality_profile` | string | No | Per-request profile override: `low` \| `medium` \| `high` \| `critical`. |

**Response** — Same `ComparisonReport` schema as `/compare` (with `fr_scores: null`).

---

### `GET /health`

Returns service status and loaded model names.

---

### `GET /report/{report_id}`

Returns the full JSON `ComparisonReport` for a stored report.

**Response** — `200 OK` — full `ComparisonReport` JSON
**Error** — `404 Not Found` if report does not exist

---

### `GET /report/{report_id}/annotated`

Returns the artifact-annotated PNG image.

**Response** — `200 OK` — `image/png`

---

### `GET /report/{report_id}/diff`

Returns the JET-colormap difference heatmap PNG (only available for reference comparisons).

**Response** — `200 OK` — `image/png`

---

### `GET /reports`

List stored reports with optional filtering.

**Query Parameters**

| Parameter | Type | Description |
|---|---|---|
| `limit` | int | Max results (default 20, max 200) |
| `offset` | int | Pagination offset (default 0) |
| `grade` | string | Filter: `pass`, `warning`, `fail` |
| `media_type` | string | Filter: `image_captured`, `image_preview`, `video_static`, `video_motion` |

**Response**

```json
{
  "reports": [
    {
      "report_id": "a3f9c1b2d4e5",
      "timestamp": "2026-03-06T10:23:45",
      "media_type": "image_captured",
      "dut_file": "dut.jpg",
      "reference_file": "ref.jpg",
      "overall_grade": "pass",
      "processing_time_ms": 1240,
      "annotated_path": "./data/reports/a3f9c1b2d4e5_annotated.png"
    }
  ],
  "count": 1,
  "offset": 0
}
```

---

## CLI Usage

The `qimc` CLI provides offline access to all analysis capabilities without running a server.

### Single file analysis

```bash
qimc analyze /path/to/photo.jpg
qimc analyze /path/to/screen_recording.mp4 --no-crop
```

### Reference comparison

```bash
qimc compare /path/to/dut.jpg /path/to/golden_ref.jpg
qimc compare /path/to/dut_video.mp4 /path/to/ref_video.mp4 --sync frame_by_frame
```

### Start the server

```bash
qimc serve
qimc serve --port 9090 --log-level debug --reload
```

### Example CLI output

```
╔══════════════════════════════════╗
║  Report a3f9c1b2d4e5            ║
║         PASS                    ║
╚══════════════════════════════════╝

┌─────────────────────────────────┬──────────────┐
│ Metric                          │ Value        │
├─────────────────────────────────┼──────────────┤
│ Sharpness (Laplacian)           │ 312.5        │
│ Noise Sigma                     │ 3.20         │
│ Exposure Mean                   │ 52.3 L*      │
│ Highlight Clip%                 │ 0.00%        │
│ Shadow Clip%                    │ 0.00%        │
│ Saturation Mean                 │ 0.312        │
│ Dynamic Range                   │ 7.2 EV       │
│ Chrom. Aberration               │ 0.80 px      │
└─────────────────────────────────┴──────────────┘
Annotated image: ./data/reports/a3f9c1b2d4e5_annotated.png
Processed in 1240 ms
```

---

## Output Format

### Annotated Image

The annotated PNG consists of:

1. **Grade banner** (top bar, 28px): green=PASS, yellow=WARNING, red=FAIL
2. **Artifact bounding boxes**: drawn on the original image
   - Color-coded: green (none) → yellow (low) → orange (medium) → red (high) → magenta (critical)
   - Label: `artifact_type [SEVERITY]`
   - Short description below the box
3. **Diff heatmap overlay** (40% alpha): JET colormap showing pixel-level differences (if reference provided)
4. **Metrics panel** (right sidebar, 380px wide): all metric values with pass/fail status and failure reasons

### Report Storage

```
data/reports/
├── {report_id}.json              ← Full ComparisonReport JSON
├── {report_id}_annotated.png     ← Artifact-annotated image
├── {report_id}_diff.png          ← Difference heatmap (if reference)
└── index.db                      ← SQLite index for fast queries
```

---

## Supported Media Types

| Format | Auto-detected | Notes |
|---|---|---|
| JPEG (`.jpg`, `.jpeg`) | Yes | Primary smartphone capture format |
| PNG (`.png`) | Yes | Lossless screenshots |
| HEIC / HEIF (`.heic`, `.heif`) | Yes | Apple/modern Android format |
| WEBP (`.webp`) | Yes | |
| BMP (`.bmp`) | Yes | |
| MP4 / MOV (`.mp4`, `.mov`) | Yes | Primary video capture format |
| AVI (`.avi`) | Yes | |
| MKV (`.mkv`) | Yes | |
| TS / MTS (`.ts`, `.mts`) | Yes | MPEG transport stream |

---

## Quality Metrics Explained

All metrics are included in the JSON report and shown in the annotated image side panel. Each metric has a **direction** (higher or lower is better), a **threshold**, and a `passed` boolean. The overall grade is `FAIL` if any hard-threshold metric fails, `WARNING` if only advisory metrics are out of range, and `PASS` otherwise.

---

### Standard Quality Metrics (always computed, no reference needed)

These measure the absolute quality of a single image or video frame regardless of any reference.

#### Sharpness — `blur_score`

**What it measures:** How sharp or in-focus the image is. Computed as the variance of the Laplacian operator applied to the grayscale image — a blurry image has low contrast between adjacent pixels, so the Laplacian (which measures second-order intensity change) returns low values.

**How to read it:**
- Higher = sharper
- Values depend heavily on scene content (a plain grey wall will score low even if perfectly focused)
- `< 40` — very blurry, likely out-of-focus or heavy motion blur
- `40–100` — soft; check OIS, focus lock, motion stability
- `100–400` — moderate; typical for indoor lightbox scenes
- `> 400` — sharp; typical for high-detail scenes with macro or fine texture

**Threshold:** set by `QIMC_BLUR_THRESHOLD`. Image is flagged blurry when `blur_score < threshold`.

#### Noise Sigma — `noise_sigma`

**What it measures:** The estimated noise level in flat (low-texture) areas. Found by locating regions with low Sobel gradient magnitude (i.e., uniform surfaces like walls, grey cards, sky) and computing the standard deviation of pixel intensities there. A noisy sensor adds random brightness variation even on uniform surfaces.

**How to read it:**
- Lower = cleaner
- `< 2` — excellent, low ISO or strong NR
- `2–5` — good (ISO 100–400 range)
- `5–10` — acceptable (ISO 800–1600 range)
- `10–20` — noisy (ISO 3200+)
- `> 20` — very noisy; check ISO cap, lighting level, NR aggressiveness

**Threshold:** set by `QIMC_NOISE_THRESHOLD`. Flagged when `noise_sigma > threshold`.

#### Exposure Mean — `exposure_mean`

**What it measures:** The average scene brightness in CIELAB L* units, which is perceptually uniform (unlike raw pixel brightness). L* = 0 is pure black, L* = 100 is pure white.

**How to read it:**
- `< 30 L*` — underexposed; increase EV or add fill light
- `30–45 L*` — slightly dark; acceptable for night or dark-theme shots
- `45–70 L*` — well-exposed; typical for daylight/lightbox
- `70–85 L*` — bright; acceptable for high-key scenes
- `> 85 L*` — overexposed risk; watch highlight clipping

**Note:** This is an advisory metric — no pass/fail threshold. Use highlight/shadow clipping for hard failures.

#### Highlight Clipping % — `highlight_clipping_pct`

**What it measures:** The percentage of pixels where any RGB channel exceeds 250 (out of 255). These pixels are "blown out" — the sensor has saturated and lost all detail at the bright end. Clipped highlights appear as pure white regions with no texture or color information.

**How to read it:**
- `0%` — no clipping, ideal
- `< 1%` — acceptable; small specular highlights (metal surfaces, water reflections) are normal
- `1–3%` — notable clipping; reduce EV compensation or enable HDR mode
- `> 5%` — heavy clipping; significant detail loss in bright areas

**Threshold:** set by `QIMC_HIGHLIGHT_CLIP_THRESHOLD`.

#### Shadow Clipping % — `shadow_clipping_pct`

**What it measures:** The percentage of pixels where all RGB channels are below 5. These pixels are "crushed to black" — no detail or color information survives. Crushed shadows appear as flat-black regions.

**How to read it:**
- `0%` — no clipping, ideal
- `< 1%` — acceptable for dark scenes
- `> 2%` — significant shadow detail lost; increase EV or apply shadow lift

**Threshold:** set by `QIMC_SHADOW_CLIP_THRESHOLD`.

#### Color Cast — `color_cast_r`, `color_cast_g`, `color_cast_b`

**What it measures:** Per-channel deviation from a neutral (equal R/G/B) gray-world assumption. Computed in regions with similar R, G, B values (neutral patches). A value of 0 means no cast; positive/negative means the channel is over/under relative to neutral.

**How to read it:**
- Values near 0 for all channels → neutral white balance
- `color_cast_r > +5` → warm/red tint
- `color_cast_b > +5` → cool/blue tint
- Large positive/negative values → white balance error; check AWB or apply manual WB preset

#### White Balance Deviation — `white_balance_deviation`

**What it measures:** RMS of the R/G and B/G channel ratios from 1.0 across neutral patches. Summarises the white balance error as a single number. A perfectly neutral scene has R=G=B everywhere, so ratios are 1.0.

**How to read it:**
- `< 0.03` — excellent WB
- `0.03–0.07` — minor tint, typically invisible to eye
- `0.07–0.15` — visible color cast; check AWB or scene illumination color
- `> 0.15` — strong cast; AWB failure or mixed lighting

#### Saturation Mean — `saturation_mean`

**What it measures:** The mean of the HSV S (saturation) channel across the image. S = 0 is pure grey/white/black; S = 1 is fully saturated color.

**How to read it:**
- `< 0.15` — very desaturated; muted or near-monochrome image
- `0.15–0.50` — natural range for most scenes
- `0.50–0.70` — vivid; may indicate aggressive color boost applied
- `> 0.70` — oversaturated; check camera picture profile

#### Dynamic Range (stops) — `dynamic_range_stops`

**What it measures:** The log₂ ratio between the 99th and 1st percentile luminance values. Expressed in EV (stops/f-stops). A wide ratio means the image captures both deep shadows and bright highlights simultaneously.

**How to read it:**
- `> 10 EV` — excellent; typical HDR or RAW-processed output
- `7–10 EV` — good; typical processed JPEG
- `5–7 EV` — moderate; some scene compression
- `< 5 EV` — compressed; strong tone-mapping or JPEG aggressive compression
- `< 3 EV` — very flat; consider HDR mode

#### Chromatic Aberration Score — `chromatic_aberration_score`

**What it measures:** The mean spatial offset (in pixels) between the red and blue channels at high-contrast edges. Lateral chromatic aberration causes the R and B lens focal planes to differ slightly, creating colored fringing at edges.

**How to read it:**
- `< 0.5 px` — excellent; no visible CA
- `0.5–1.5 px` — minor; CA barely visible at 100% zoom
- `1.5–3 px` — moderate; color fringing visible on zoomed edges
- `> 3 px` — strong; apply lens correction profile or check zoom alignment

---

### No-Reference IQA Scores (always computed)

These models score image quality without needing a reference. They are trained on large databases of human perceptual judgements on natural images. **Important:** these models assume natural photographic content — they may give poor scores to synthetic test charts, flat-color patterns, or screenshots even when those are technically correct.

| Metric | Range | Good | How to read |
|---|---|---|---|
| **BRISQUE** | 0–100 (lower = better) | < 30 | Measures naturalness via local brightness statistics. A score of 0 is a "statistically natural" image. Scores above 50 often indicate visible distortion. Trained on natural images — pure color patches score ~90 even if technically perfect. |
| **NIQE** | 0–15 (lower = better) | < 4 | Fits a multivariate Gaussian to local patches and compares to a pristine natural image model. More sensitive to noise and ringing than BRISQUE. |
| **MUSIQ** | 0–100 (higher = better) | > 60 | Neural model (multi-scale Vision Transformer). Better correlation with human MOS (mean opinion score) than classical metrics. Requires GPU for real-time use. |
| **CLIP-IQA+** | 0–1 (higher = better) | > 0.6 | Uses CLIP's vision-language model to compare the image against quality descriptors. Handles diverse content well. GPU required. |

**These are advisory** — they do not directly cause FAIL grade unless combined with other failures. They are useful for trending quality across many captures.

---

### Full-Reference IQA Scores (only when reference image is provided)

These compare the DUT image pixel-by-pixel against the golden reference. They quantify how different (or similar) the two images are. All use spatial alignment (SIFT + RANSAC homography) before measurement to compensate for minor rig positioning differences.

#### PSNR — Peak Signal-to-Noise Ratio

**What it measures:** The ratio (in decibels) between the maximum possible signal power and the mean squared error between DUT and reference pixels. A high PSNR means the pixel-level difference is small.

**Formula:** `PSNR = 10 × log₁₀(255² / MSE)`

**How to read it:**
- `∞ dB` — identical images (MSE = 0)
- `> 40 dB` — imperceptible difference; excellent match
- `35–40 dB` — very close; typical for good device-to-device comparison
- `28–35 dB` — noticeable but minor difference
- `20–28 dB` — significant pixel difference (visible artifacts or exposure mismatch)
- `< 20 dB` — large difference; likely different exposure, alignment issue, or heavy compression

**Limitation:** PSNR is purely mathematical — it treats every pixel equally and doesn't account for perceptual importance. A 1-pixel shift causes a large MSE even if the images look identical.

**Threshold:** set by `QIMC_PSNR_THRESHOLD`.

#### SSIM — Structural Similarity Index

**What it measures:** Compares patches of the two images across three dimensions: luminance (mean), contrast (variance), and structure (cross-correlation). More correlated with human perception than PSNR.

**How to read it:**
- `1.0` — identical images
- `> 0.95` — virtually indistinguishable
- `0.90–0.95` — very close; minor difference
- `0.85–0.90` — noticeable; borderline for strict camera QA
- `0.80–0.85` — visible difference; investigate exposure, OIS, or alignment
- `< 0.80` — significant degradation

**Threshold:** set by `QIMC_SSIM_THRESHOLD`.

#### MS-SSIM — Multi-Scale SSIM

**What it measures:** SSIM computed at multiple spatial scales (image pyramid levels). More robust to differences in viewing distance and resolution scaling than single-scale SSIM.

**How to read it:** Same interpretation as SSIM but generally ~0.02–0.05 higher for the same image pair. Treat `> 0.96` as excellent, `< 0.88` as concerning.

#### LPIPS — Learned Perceptual Image Patch Similarity

**What it measures:** A neural-network-based distance metric. Uses features from a pre-trained AlexNet to compare patches. Trained specifically to predict human perceptual similarity judgements — it is insensitive to pixel-perfect alignment but very sensitive to texture and semantic changes.

**How to read it:**
- Lower = more similar
- `< 0.05` — perceptually identical; minor film grain or compression difference
- `0.05–0.15` — minor perceptual difference; within normal device-to-device variation
- `0.15–0.30` — noticeable difference; likely noise level, color rendering, or sharpening difference
- `> 0.30` — large perceptual difference; investigate processing pipeline

**Why it matters:** LPIPS can flag cases where PSNR/SSIM look good (e.g., slight blur or noise change) because it reflects how humans actually see differences.

**Threshold:** set by `QIMC_LPIPS_THRESHOLD`.

#### DISTS — Deep Image Structure and Texture Similarity

**What it measures:** Similar to LPIPS but explicitly separates texture and structure similarity, making it less sensitive to minor geometric misalignment. Particularly useful when the capture rig has slight vibration between the two shots.

**How to read it:** Same scale as LPIPS (0–1, lower = better). Tends to be more forgiving than LPIPS for real-world captures with minor positional variation.

**Threshold:** set by `QIMC_DISTS_THRESHOLD`.

---

## Artifact Detection Catalogue

| Artifact | Detection Method | Severity Scale | Fix Hint |
|---|---|---|---|
| **noise_patch** | Local variance in flat (low-Sobel) 32×32 patches | % of affected patches | Lower ISO, improve lighting, tune NR |
| **banding** | FFT of row-gradient energy, high-freq component ratio | Ratio > 15% | Increase bit depth, disable aggressive quantization |
| **lens_flare** | Threshold top 0.1% brightness → elongated connected blobs | Count of flare blobs | Move light source, use lens hood |
| **hot_pixel** | Pixels > 4σ above local 5×5 Gaussian mean | Count as % of frame | Sensor defect / long-exposure heat |
| **chromatic_aberration** | R-B centroid offset at Canny edge points | Mean offset in px | Lens correction profile, zoom alignment |
| **blurry_region** | Local Laplacian variance tiles < global mean / 3 | % of image area | Check focus, OIS, motion blur |
| **posterization** | Unique tone count in smooth-gradient regions | Fill ratio | Increase JPEG quality, disable sharpening |
| **overexposure** | Connected region of pixels with any channel > 250 (> 5% area) | Area % | Reduce EV, enable highlight recovery |
| **underexposure** | Connected region of pixels with all channels < 5 (> 5% area) | Area % | Increase EV, check shadow lift |

---

## Integration with Automation Framework

### Calling from UIAutomator2 test runner (Python)

```python
import requests
from pathlib import Path

QIMC_URL = "http://localhost:8080"

def compare_camera_output(dut_path: str, ref_path: str) -> dict:
    with open(dut_path, "rb") as dut, open(ref_path, "rb") as ref:
        response = requests.post(
            f"{QIMC_URL}/compare",
            files={
                "dut": (Path(dut_path).name, dut, "image/jpeg"),
                "reference": (Path(ref_path).name, ref, "image/jpeg"),
            },
            data={"crop_preview": "false"},
        )
    response.raise_for_status()
    return response.json()


def analyze_preview_recording(recording_path: str) -> dict:
    """Analyze an ADB screen recording of the camera preview."""
    with open(recording_path, "rb") as f:
        response = requests.post(
            f"{QIMC_URL}/analyze",
            files={"media": (Path(recording_path).name, f, "video/mp4")},
            data={"crop_preview": "true"},  # auto-crop camera UI chrome
        )
    response.raise_for_status()
    return response.json()


# Example usage in a test case
def test_camera_zoom_quality():
    # Capture on reference device
    adb_shell("screencap /sdcard/ref_2x_zoom.jpg")
    adb_pull("/sdcard/ref_2x_zoom.jpg", "./captures/ref_2x_zoom.jpg")

    # Capture on DUT
    adb_shell("screencap /sdcard/dut_2x_zoom.jpg")
    adb_pull("/sdcard/dut_2x_zoom.jpg", "./captures/dut_2x_zoom.jpg")

    report = compare_camera_output(
        dut_path="./captures/dut_2x_zoom.jpg",
        ref_path="./captures/ref_2x_zoom.jpg",
    )

    assert report["overall_grade"] == "pass", (
        f"Camera 2x zoom quality FAIL:\n" +
        "\n".join(f"  - {r}" for r in report["failure_reasons"])
    )
```

### ADB screen recording workflow

```python
import subprocess
import time

def capture_preview_recording(duration_s: int = 5, output_path: str = "preview.mp4") -> str:
    """Record camera preview via ADB screenrecord."""
    subprocess.Popen(["adb", "shell", f"screenrecord --time-limit {duration_s} /sdcard/preview.mp4"])
    time.sleep(duration_s + 1)
    subprocess.run(["adb", "pull", "/sdcard/preview.mp4", output_path])
    return output_path

# Then analyze with preview cropping enabled
report = analyze_preview_recording("preview.mp4")
```

---

## Testing

```bash
# Run all tests
make test

# Run tests without neural model tests (faster)
pytest tests/ -v -m "not slow"

# Run a specific test file
pytest tests/test_artifact_detector.py -v

# Run with coverage report
pytest tests/ --cov=qa_intelli_media_comparator --cov-report=html
open htmlcov/index.html
```

### Test fixtures available

| Fixture | Description |
|---|---|
| `sharp_bgr` | 256×256 sharp synthetic image with shapes |
| `blurry_bgr` | Same image blurred with Gaussian (σ=10) |
| `noisy_bgr` | Sharp image with σ=30 Gaussian noise |
| `gradient_bgr` | 256×256 smooth horizontal gradient |
| `overexposed_bgr` | Nearly-white image (mostly channel > 252) |
| `preview_bgr` | 1080×2340 synthetic ADB screenshot with status bar + shutter button |
| `tmp_image_file` | Saved JPEG of `sharp_bgr` in `tmp_path` |
| `tmp_video_file` | 30-frame synthetic MP4 video |

---

## Project Structure

```
QA_Intelli_Media_Comparator/
├── pyproject.toml                          ← Package definition + dependencies
├── .env.example                            ← Configuration template
├── Makefile                                ← Dev workflow shortcuts
├── README.md                               ← This file
├── Technical_Document.md                   ← Deep-dive technical reference
│
├── qa_intelli_media_comparator/            ← Main Python package
│   ├── __init__.py
│   ├── main.py                             ← uvicorn entrypoint + typer CLI
│   ├── config.py                           ← pydantic-settings (QIMC_ prefix)
│   │
│   ├── api/
│   │   ├── app.py                          ← FastAPI factory + lifespan
│   │   ├── middleware.py                   ← Request ID + timing middleware
│   │   ├── dependencies.py                 ← Cached pipeline + store DI
│   │   └── routes/
│   │       ├── compare.py                  ← POST /compare
│   │       ├── analyze.py                  ← POST /analyze
│   │       ├── health.py                   ← GET /health
│   │       └── reports.py                  ← GET /report/* + GET /reports
│   │
│   ├── models/
│   │   ├── enums.py                        ← MediaType, CameraMode, ArtifactSeverity, QualityGrade, SyncMode
│   │   ├── media.py                        ← MediaInfo, CropResult
│   │   ├── metadata.py                     ← MediaMetadata, MetadataComparison
│   │   ├── metrics.py                      ← MetricResult, FullReferenceScores, NoReferenceScores, QualityMetrics
│   │   ├── artifacts.py                    ← ArtifactInstance, ArtifactReport
│   │   ├── video.py                        ← VideoTemporalMetrics, VideoAnalysisResult
│   │   └── report.py                       ← ComparisonReport
│   │
│   ├── services/
│   │   ├── camera_mode_detector.py         ← EXIF extraction, camera mode inference, threshold adjustment
│   │   ├── media_type_detector.py          ← Auto-detect media type
│   │   ├── preview_cropper.py              ← Crop camera UI chrome
│   │   ├── quality_metrics.py              ← Standard photographic metrics
│   │   ├── artifact_detector.py            ← 8-type artifact detection
│   │   ├── no_reference_analyzer.py        ← NR-IQA via pyiqa
│   │   ├── reference_comparator.py         ← FR-IQA with spatial alignment
│   │   ├── video_analyzer.py               ← Temporal video quality
│   │   ├── annotation_renderer.py          ← Annotated image output
│   │   └── pipeline.py                     ← Full orchestration
│   │
│   └── storage/
│       ├── report_store.py                 ← SQLite + JSON persistence
│       └── schema.sql                      ← Database schema
│
├── data/
│   └── reports/                            ← Generated reports (gitignored)
│
└── tests/
    ├── conftest.py                         ← Synthetic fixtures
    ├── test_media_type_detector.py
    ├── test_preview_cropper.py
    ├── test_quality_metrics.py
    ├── test_artifact_detector.py
    ├── test_reference_comparator.py
    ├── test_video_analyzer.py
    ├── test_pipeline.py
    └── test_api.py
```
