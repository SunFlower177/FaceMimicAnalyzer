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

## 🧪 Example Results
<img width="1194" height="933" alt="屏幕截图 2026-01-31 182144" src="https://github.com/user-attachments/assets/4014729a-cc34-46eb-ac56-e67d45219742" />
<img width="1207" height="1251" alt="屏幕截图 2026-01-31 182154" src="https://github.com/user-attachments/assets/bf0a9840-0c58-4b46-b814-22107858b7a1" />
<img width="1212" height="600" alt="屏幕截图 2026-01-31 182201" src="https://github.com/user-attachments/assets/55ab0d84-bec9-48d9-8ef4-63e56b7db206" />



---
## 🛠️ Pipeline Overview

Input Media (Video / Image) 
        ➡ 
Frame Extraction 
        ➡ 
Expression Segmentation (single known expression) 
        ➡ 
Blendshape & Landmark Extraction (MediaPipe)
        ➡
Keyframe Reselection (AU peak)
        ➡
Relative Feature Computation
        ➡
Similarity Analysis (Static + Dynamic + Structure)
        ➡
Visualization & Report Generation

---

## 🎯 Use Cases

Facial expression imitation evaluation

Human behavior analysis

Psychology / ASD-related research (expression consistency)

Human-computer interaction research

Expression-based training or feedback systems

---

## 🔮 Future Work

Support for automatic expression classification

Multi-expression segmentation in long videos

Deep learning–based similarity metrics

Quantitative scoring benchmarks

Model optimization for real-time use

---

## 📜 License

This project is for research and educational purposes.

