# IVQEP — Image & Video Quality Enhancement Pipeline

A frame-level image and video enhancement pipeline built with **Python**, **OpenCV**, and **PyTorch**.
It applies contrast normalisation and noise reduction to low-light video frames and evaluates visual quality using **PSNR** and **SSIM** metrics.

---

## Features

| Component | Description |
|-----------|-------------|
| **Contrast normalisation** | CLAHE (Contrast Limited Adaptive Histogram Equalisation) in the LAB colour space via OpenCV |
| **Noise reduction** | Lightweight non-local-means denoising (OpenCV) as a pre-processing step |
| **CNN denoising** | DnCNN-style residual convolutional network (PyTorch) for deep denoising |
| **Quality metrics** | Per-frame PSNR and SSIM evaluation using `scikit-image` |
| **Video pipeline** | Streaming frame-by-frame processing for large video datasets |
| **CLI** | Simple command-line interface via `main.py` |

---

## Project Structure

```
IVQEP/
├── pipeline/
│   ├── __init__.py          # Public API exports
│   ├── preprocessing.py     # OpenCV contrast normalisation & noise reduction
│   ├── denoising.py         # DnCNN PyTorch model + helpers
│   ├── metrics.py           # PSNR and SSIM metric functions
│   └── video_pipeline.py    # End-to-end video processing pipeline
├── tests/
│   ├── test_preprocessing.py
│   ├── test_denoising.py
│   ├── test_metrics.py
│   └── test_video_pipeline.py
├── main.py                  # CLI entry point
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Quick Start

### Python API

```python
import cv2
from pipeline import FramePreprocessor, load_denoising_model, evaluate_quality, VideoPipeline

# --- Preprocess a single frame ---
preprocessor = FramePreprocessor(clip_limit=2.0)
frame = cv2.imread("dark_frame.jpg")
enhanced = preprocessor.preprocess(frame)

# --- CNN denoising ---
model = load_denoising_model(channels=3)          # random weights (no checkpoint)
denoised = model.denoise_frame(enhanced)

# --- Evaluate quality ---
metrics = evaluate_quality(reference=frame, enhanced=denoised)
print(f"PSNR: {metrics['psnr']:.2f} dB  SSIM: {metrics['ssim']:.4f}")

# --- Process a whole video ---
pipeline = VideoPipeline(
    preprocessor=preprocessor,
    denoising_model=model,
    compute_metrics=True,
)
summary = pipeline.process_video(
    input_path="input.mp4",
    output_path="enhanced.mp4",
    reference_path="clean.mp4",   # optional - enables PSNR / SSIM
)
print(summary)
```

### Command-Line Interface

```bash
# Enhance a video (preprocessing only, no CNN)
python main.py input.mp4 -o enhanced.mp4

# With a DnCNN checkpoint and quality metrics
python main.py input.mp4 \
    -o enhanced.mp4 \
    --checkpoint weights/dncnn.pth \
    --channels 3 \
    --reference clean.mp4 \
    --metrics

# Process only the first 200 frames
python main.py input.mp4 -o out.mp4 --max-frames 200
```

---

## Pipeline Overview

```
Input Video
    |
    v
FramePreprocessor
  |- CLAHE (LAB L-channel)        <- contrast normalisation
  +- Non-local-means denoising    <- lightweight noise reduction
    |
    v (optional)
DnCNN
  +- Residual CNN denoising       <- deep noise removal
    |
    v
Enhanced Frame
    |
    |- Write to output video
    +- Compute PSNR / SSIM        <- quality evaluation
```

---

## Quality Metrics

| Metric | Range | Good threshold |
|--------|-------|----------------|
| PSNR   | 0 - inf dB | > 30 dB (acceptable), > 40 dB (excellent) |
| SSIM   | -1 to 1   | > 0.9 (high quality) |

---

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

---

## Technologies

- **OpenCV** - image preprocessing, video I/O, CLAHE, non-local-means denoising
- **PyTorch** - DnCNN residual CNN architecture for learned denoising
- **scikit-image** - PSNR and SSIM metric computation
- **NumPy** - array operations
- **tqdm** - progress bars for large video datasets
