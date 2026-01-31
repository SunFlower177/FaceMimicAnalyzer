# FaceMimicAnalyzer

A video & image based facial expression imitation analysis system using MediaPipe blendshapes and facial landmarks to evaluate static appearance, dynamic changes, and key facial regions, with automatic visualization and report generation.

---

## 📌 Overview

**FaceMimicAnalyzer** is an end-to-end facial expression imitation analysis pipeline designed to quantitatively evaluate how well a user mimics a given facial expression.

The system supports both **videos and images**, and compares a **reference (standard) expression** with a **user imitation**, analyzing:
- Static facial appearance similarity
- Dynamic expression evolution
- Region-level facial movement differences (mouth, eyes, eyebrows, etc.)

All results are automatically visualized and summarized in a human-readable report.

---

## ✨ Key Features

- 🎥 **Video & Image Support**  
  Images are treated as single-frame videos for unified processing.

- 🧠 **Official MediaPipe Blendshapes**  
  Uses MediaPipe Face Landmarker (with blendshapes) for robust facial representation.

- 📐 **Multi-level Similarity Analysis**
  - Static appearance (SSIM)
  - Facial structure alignment (landmarks & Procrustes)
  - Temporal dynamics (DTW-based blendshape comparison)

- 🎯 **Region-level Analysis**
  Automatically identifies and analyzes the most relevant facial regions for each expression.

- 📊 **Automatic Visualization & Reports**
  Generates:
  - Keypoint overlay comparisons
  - Time-series plots
  - Text-based evaluation reports

- 🖥️ **Interactive Streamlit Demo**
  Simple UI for uploading media and running analysis.

---
