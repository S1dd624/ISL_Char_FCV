# Real-Time Indian Sign Language Alphabet Recognition via a Pure Computer Vision Pipeline

### Computer Vision Laboratory Project  
Department of Computer Science and Engineering  
Manipal Institute of Technology, Manipal Academy of Higher Education

---

## Overview
This repository contains the implementation and documentation for a real-time Indian Sign Language (ISL) alphabet recognition system.  
The project demonstrates that a purely classical computer vision pipeline, without reliance on machine or deep learning, can achieve interpretable, efficient gesture recognition for static ISL alphabets on standard CPU hardware.

The approach integrates:
- YCrCb color-space segmentation with CLAHE-based luminance normalization,
- A 13-dimensional geometric feature vector (Hu Moments, contour descriptors, edge density),
- Cosine similarity–based template matching, and
- A dual-condition temporal state machine combining centroid displacement and hand-exit detection.

Measured performance achieves an average processing time of approximately **2.1 ms per frame (≈476 FPS theoretical)** on consumer-grade CPUs.

---

## Pipeline Summary
**Stage-wise workflow:**

1. Frame acquisition and ROI (centered 300×300 region).  
2. Preprocessing with RGB→YCrCb conversion and CLAHE on luminance.  
3. Skin segmentation using calibrated Cr/Cb thresholds and morphological filtering.  
4. Contour and convex-hull extraction for region isolation.  
5. Feature extraction (13-D) using Hu Moments, contour metrics, and edge density.  
6. Cosine similarity template matching for A–Z and SPACE gestures.  
7. Temporal dwell and centroid-displacement logic for stable gesture commitment.

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/ISL-PureCV-Recognizer.git
cd ISL-PureCV-Recognizer
