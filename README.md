# Derm-Referral AI 🩺

[![Hackathon](https://img.shields.io/badge/IIT%20Ropar-AI%20Wellness%20Hackathon-blue)](https://iitrpr.ac.in)
[![Track](https://img.shields.io/badge/Track-AI4Healthcare-green)](https://iitrpr.ac.in)

[cite_start]**Derm-Referral AI** is a computer vision and multi-modal machine learning solution developed for the **PeaceOfCode 2026 AI Wellness Hackathon**[cite: 7, 35]. [cite_start]It addresses the critical shortage of dermatologists in rural areas by providing non-specialist health workers with an automated tool to identify malignant skin conditions.

## 📌 Problem Statement
[cite_start]In rural primary health centers, workers often encounter skin lesions but lack the specialist training to distinguish benign moles from malignant ones[cite: 17]. This leads to:
* [cite_start]**Delayed Referrals:** Malignant conditions go untreated until they reach critical stages[cite: 21].
* [cite_start]**Operational Gaps:** Lack of access to expensive dermatoscopic tools and expert dermatologists[cite: 24, 26].

## 🚀 The Solution
[cite_start]Our model takes a standard photograph of a lesion and clinical metadata (age, symmetry, etc.) to output a specific referral decision[cite: 18, 43].

### Key Features
* [cite_start]**Hybrid Architecture:** Combines an **EfficientNet-B0** image backbone with a **Multi-Layer Perceptron (MLP)** for tabular metadata[cite: 39].
* **Actionable Output:** Instead of complex percentages, the tool provides a color-coded flag:
  * 🟢 **Green:** Routine monitoring.
  * 🟡 **Yellow:** Consult with a district specialist.
  * [cite_start]🔴 **Red:** Urgent referral for physical examination[cite: 18, 43].
* [cite_start]**Resource Optimized:** Designed to run on mid-range hardware (RTX 3050) and mobile devices[cite: 39, 40].

## 📊 Dataset: SLICE-3D (ISIC 2024)
[cite_start]The project utilizes the **SLICE-3D 2024 Challenge Dataset**[cite: 42]:
* [cite_start]**Data Type:** 15x15mm cropped lesion-images designed to mimic non-dermoscopic photos[cite: 14].
* [cite_start]**Imbalance Handling:** Addressed a severe **1019:1 class imbalance** using Weighted Binary Cross-Entropy loss[cite: 48].

## 🛠️ Technical Implementation
* **Framework:** PyTorch
* **Backbone:** EfficientNet-B0 (Pre-trained)
* **Optimizations:** Mixed Precision (AMP), HDF5 Data Streaming, and Weighted Sampling.

## 📈 One Key Challenge
**Domain Shift:** The model is trained on standardized crops but intended for handheld mobile photos. [cite_start]We addressed this through extensive data augmentation—simulating blur, noise, and varied lighting—to ensure robustness in real-world rural settings[cite: 44, 47, 48].

---
Developed by Team **PeaceOfCode** for IIT Ropar AI Wellness Hackathon 2026.
