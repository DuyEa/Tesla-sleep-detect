# 💤 Tesla Concentrate Detect — Driver Drowsiness Detection (EAR + CNN 4D)

### Overview
This project implements a **dual-mode real-time driver drowsiness detection system**, inspired by:
- Mehta et al. (2019) — Real-Time Driver Drowsiness Detection Using Eye Aspect Ratio and Eye Closure Ratio
- Jahan et al. (2023) — 4D: A Real-Time Driver Drowsiness Detector Using Deep Learning

It combines **behavioral feature extraction** (Eye Aspect Ratio – EAR, Eye Closure Ratio – ECR) with **deep CNN eye-state classification**, enabling both **machine-learning-based** and **deep-learning-based** pipelines.

---

## 👁 EAR-Based (Machine Learning)

### Short intro (how it works — full face dataset)
- **Eye Aspect Ratio (EAR)** quantifies the eye’s openness: when eyes close, EAR drops below a threshold (≈0.25).
- **Eye Closure Ratio (ECR)** is the fraction of closed frames within a time window (default 15 frames). Drowsy ⇢ ECR ≥ 0.5 or ≥ 3 consecutive closed frames.
- Uses **full face** via MediaPipe FaceMesh to locate both eyes and compute EAR/ECR robustly.

### Implementation
- `train_sleep.py` computes EAR/ECR per frame and trains a **Random Forest** on window-level features:
  - `ear_mean`, `ear_std`, `ear_min`, `ecr`, `miss_ratio`, `max_closed_run`
- `online_detect_ear_ecr.py` performs real-time detection; triggers **SLEEP** if any of:
  - RF predicts ≥ 3 consecutive drowsy windows
  - ECR(window) ≥ 0.80
  - Head pitch down ≥ 25°
  - No face/eyes detected for ≥ 20 frames

✅ **Pros** — Explainable, robust on full-face input.  
⚠️ **Cons** — Requires both eyes visible; sensitive to lighting/pose outliers.

---

## 🧠 CNN-Based 4D (Deep Learning)

### Short intro (how it works — one eye is enough)
- The **4D CNN** directly classifies a **single eye crop** (open/closed) without hand-crafted features.
- Works even if only **one eye** is clearly visible (the pipeline averages R/L when both are found).
- Architecture: 3×(Conv + BatchNorm + Dropout) → FC → sigmoid (see `model_4d.py`).

### Implementation
- `train_4d.py` trains the 4D model on eye images (24×24 grayscale).
- `detect_4d.py` uses MediaPipe FaceMesh to crop eyes, runs CNN per frame, and declares **SLEEP** if
  `p_closed ≥ 0.5 for ≥ 2 seconds` (configurable).

✅ **Pros** — High accuracy, needs only one eye, less brittle than rules.  
⚠️ **Cons** — Requires GPU for best performance and eye-only training data.

---

## 📦 Datasets Used (explicit for each style)

| Pipeline | Dataset | What images look like | Where this is used in code |
|:--|:--|:--|:--|
| **EAR/ECR + Random Forest (ML)** | **NTHU Driver Drowsiness Detection (NTHU-DDD / NTHU-DDD2, Kaggle mirror)** — *full-face frames categorized into classes like `drowsy`, `notdrowsy`* | **Full face** images; eyes are found with FaceMesh to compute EAR & ECR | `train_sleep.py` (offline labeling & training), `online_detect_ear_ecr.py` (online detection). The repo folder often referenced: `subsampled_nthuddd2/` |
| **CNN 4D (DL)** | **MRL Eye Dataset** — *single-eye crops labeled `open`/`closed`* | **Single eye** grayscale crops (L/R, with/without glasses, lighting variations) | `train_4d.py` (offline training), `detect_4d.py` (online detection) |

> Tip: Keep the folder names consistent with your paths used in the scripts, e.g.  
> - `DEFAULT_ROOT = "D:\Download\pyCharmpPro\sleep_detect\subsampled_nthuddd2"` for ML  
> - `root_dir = "D:\Download\pyCharmpPro\sleep_detect\archive\mrlEyes_2018_01"` for DL

---

## 🔍 Offline Phases (Labeling + Training)

| Phase | Method | Purpose | Output |
|:------|:--------|:--------|:--------|
| **Offline Phase 1** | `train_sleep.py` (EAR/ECR + Random Forest) | Compute EAR/ECR features on **full-face** frames; fit RF for window-level drowsiness labels | `RandFor.pkl` |
| **Offline Phase 2** | `train_4d.py` (CNN 4D) | Train deep model on **single-eye** images to classify open/closed | `model_4d.pth` |

---

## ⚡ Online Detection Modes (difference in sleep condition)

| Mode | Script | Input | Sleep Condition |
|:------|:-------|:------|:----------------|
| **EAR/ECR-based (ML)** | `online_detect_ear_ecr.py` | **Full face** (both eyes & head pose) | SLEEP if **(RF ≥ 3 consecutive drowsy windows)** OR **ECR(window) ≥ 0.80** OR **pitch ≥ 25°** OR **no face ≥ 20 frames** |
| **CNN-based (DL)** | `detect_4d.py` | **Eye region** (1 eye is enough) | SLEEP if **p_closed ≥ 0.5** for **≥ 2.0 seconds** (tunable) |

Summary: **ML online** combines model probability + ECR + head pose + missing-face rules.  
**DL online** relies on per-frame eye-closure probability held across time (consecutive seconds).

---

## ⚖️ ML vs DL (in this repo)

| Aspect | EAR/ECR + RF (ML) | CNN 4D (DL) |
|:-------|:------------------|:-------------|
| Input | Full face | Single eye |
| Features | Hand-crafted (EAR/ECR) | Learned (convolutions) |
| Labeling | Uses window rules + RF | Uses eye-state labels (open/closed) |
| Accuracy (typical) | ~84% (RF, literature) | ~97.5% (4D) |
| Hardware | CPU-friendly | GPU preferred |
| Explainability | High (features/thresholds) | Lower (NN), but CAM possible |
| Robustness | Sensitive to occlusion/pose | Tolerates one-eye visibility |

---

## 🧩 Dependencies
```
Python ≥ 3.9
torch, torchvision
opencv-python
mediapipe
scikit-learn
numpy, pandas
Pillow
```
> For GPU acceleration, install a CUDA-enabled PyTorch build matching your driver.

---

## 🧪 Run Guide

### 1) Train (EAR/ECR + Random Forest, full-face)
```bash
python train_sleep.py --root_dir "D:\Download\pyCharmpPro\sleep_detect\subsampled_nthuddd2"
```

### 2) Train (CNN 4D, single-eye)
```bash
python train_4d.py
```

### 3) Real-time Detection (EAR/ECR + rules)
```bash
python online_detect_ear_ecr.py --camera 0
```

### 4) Real-time Detection (CNN 4D)
```bash
python detect_4d.py --camera 0
```

Both detectors also support `--video your_video.mp4` and produce annotated MP4 output.



