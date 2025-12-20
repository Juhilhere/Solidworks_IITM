# 🔩 SOLIDWORKS AI Hackathon: Exact-Match Multi-Object Counting

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![YOLOv8](https://img.shields.io/badge/Model-YOLOv8x-green?style=for-the-badge)
![Score](https://img.shields.io/badge/Public_Score-0.9992857-gold?style=for-the-badge)
![Rank](https://img.shields.io/badge/Rank-Top_Tier-red?style=for-the-badge)

> **Objective:** Precise enumeration of sparse mechanical parts (Bolts, Nuts, Pins, Washers) in synthetic imagery.  
> **Result:** Achieved **99.93% Exact-Match Accuracy**, securing the **2nd Highest Score** in the competition.

---

## 📖 Overview
This repository contains the source code and methodology for our solution to the SOLIDWORKS AI Hackathon. The challenge required participants to count the exact number of four mechanical components in synthetic images.

Unlike standard object detection tasks which prioritize Intersection over Union (IoU), this challenge required **perfect counting**. A single missing or misclassified object resulted in a score of 0 for that image.

---

## 📊 1. Data Forensics: The "Grid" Discovery
Before training any models, we performed a rigorous statistical analysis of the dataset ("Code-First Approach").

**The Breakthrough:**
We analyzed the variance of the provided bounding boxes in the training set.
* **Metric:** Bounding Box Dimensions ($\sigma_{width}$, $\sigma_{height}$)
* **Finding:** $\sigma^2 = 0$.
* **Insight:** Every single object in the dataset was a rigid **224x224 pixel crop**. Furthermore, coordinate analysis revealed objects only appeared in fixed grid quadrants.

**Conclusion:** The dataset was a **Deterministic Synthetic Grid**, not a natural scene. This eliminated the need for complex scale-invariance handling and directed our architectural choice.

---

## 🧠 2. Model Selection: The Battle of Architectures

We conducted an ablation study comparing two distinct approaches: **Classification (EfficientNet)** vs. **Detection (YOLOv8)**.

### ❌ The Baseline: EfficientNet-B0 (Accuracy: 0.34)
We attempted to solve the problem as a regression task using a SOTA classifier.
* **Failure Mode:** Classifiers utilize **Global Average Pooling (GAP)** at the final layer, which compresses spatial dimensions ($H \times W \times C \rightarrow 1 \times 1 \times C$).
* **Why it failed:** GAP mathematically destroys the spatial variance needed to distinguish between "1 Nut" and "3 Nuts" when they share similar features. The model detects *presence* but fails at *enumeration*.

### ✅ The Champion: YOLOv8x (Accuracy: 0.999)
We selected **YOLOv8x (Extra Large)** for its ability to preserve **Spatial Inductive Bias**.
* **Mechanism:** YOLO uses **Anchor-Based Regression**. It divides the image into a grid and regresses coordinates relative to specific grid cells.
* **Why it won:** Because our data was inherently grid-based, YOLO's architecture perfectly mapped to the problem structure. It maintained the $XY$ tensor dimensions deep into the network, allowing for precise localization of identical objects sitting next to each other.

---

## ⚙️ 3. Training Configuration

We prioritized model capacity over inference speed to ensure maximum accuracy.

| Hyperparameter | Value | Reasoning |
| :--- | :--- | :--- |
| **Model** | `yolov8x.pt` | 68.2M Parameters for maximum feature extraction. |
| **Input Resolution** | `640x640` | Upsampled from the native grid to preserve edge details. |
| **Epochs** | `25` | Early convergence observed at Epoch 15 due to clean data. |
| **Optimizer** | `SGD` | Momentum 0.937 for stable convergence. |
| **Batch Size** | `16` | Optimized for GPU memory (Tesla T4). |

---

### Prerequisites
```bash
pip install ultralytics pandas numpy tqdm opencv-python
