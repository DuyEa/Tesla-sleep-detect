import os
# stop log spam
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

"""
Training script for drowsiness detector based on EAR/ECR features.

Scientific basis:
- Eye Aspect Ratio (EAR) from eyelid landmarks robustly indicates eye openness and is widely used for blink/closure detection (Soukupová & Čech, 2016).
- Windowed Eye Closure Ratio (ECR), i.e., fraction of frames with EAR below a threshold in a short window, correlates with PERCLOS-style drowsiness indicators used in driver monitoring literature.
- Landmarks are extracted with MediaPipe Face Mesh (Bazarevsky et al., 2020; MediaPipe Iris blog, 2020).

This script computes per-window statistics (mean, std, min, ECR, miss ratio, max closure run) and trains a RandomForest as in many real-time systems for interpretability and robustness.
"""
import argparse, sys, pickle
import cv2
import numpy as np
import pandas as pd
from typing import List, Tuple
from contextlib import contextmanager

# ML
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import mediapipe as mp


DEFAULT_ROOT = r"D:\Download\pyCharmpPro\sleep_detect2\archive\Eye dataset"
MODEL_PATH = "RandFor.pkl"

# eye position
RIGHT_EYE = [33, 160, 158, 133, 153, 144]  # p1..p6
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
VALID_IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

FEATURE_NAMES = ["ear_mean", "ear_std", "ear_min", "ecr", "miss_ratio", "max_closed_run"]


@contextmanager
def facemesh_ctx():
    base_kwargs = dict(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    )
    try:
        fm = mp.solutions.face_mesh.FaceMesh(model_complexity=0, **base_kwargs)
    except TypeError:
        fm = mp.solutions.face_mesh.FaceMesh(**base_kwargs)
    try:
        yield fm
    finally:
        try:
            fm.close()
        except Exception:
            pass

def compute_ear_from_landmarks(coords: np.ndarray) -> float:
    v1 = np.linalg.norm(coords[1] - coords[5])
    v2 = np.linalg.norm(coords[2] - coords[4])
    h  = np.linalg.norm(coords[0] - coords[3])
    if h <= 1e-6: return np.nan
    return (v1 + v2) / (2.0 * h)

def try_face_mesh_ear(bgr: np.ndarray, fm) -> float:
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    res = fm.process(rgb)
    if not res.multi_face_landmarks:
        return np.nan
    face = res.multi_face_landmarks[0]
    pts = np.array([[lm.x * w, lm.y * h] for lm in face.landmark], dtype=np.float32)
    ear_r = compute_ear_from_landmarks(pts[RIGHT_EYE])
    ear_l = compute_ear_from_landmarks(pts[LEFT_EYE])
    return np.nanmean([ear_r, ear_l])

def robust_load_image(path: str) -> np.ndarray:
    #read the image using cv2
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return bgr

def scan_root_dir(root_dir: str) -> pd.DataFrame:
    rows = []
    for label in os.listdir(root_dir):
        class_dir = os.path.join(root_dir, label)
        if not os.path.isdir(class_dir): continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(VALID_IMG_EXT):
                rows.append({"filepath": os.path.join(class_dir, fname), "label": label})
    if not rows:
        raise RuntimeError(f"No images found under: {root_dir}")
    return pd.DataFrame(rows)

def window_features_and_label(ears: np.ndarray, closed_flags: np.ndarray, ecr_thresh: float, consec: int):
    miss = np.isnan(ears)
    if np.all(miss):
        ears_imp = np.full_like(ears, 0.25, dtype=float)
    else:
        med = np.nanmedian(ears)
        ears_imp = np.where(miss, med, ears)
    ear_mean = float(np.mean(ears_imp))
    ear_std  = float(np.std(ears_imp))
    ear_min  = float(np.min(ears_imp))
    ecr      = float(np.mean(closed_flags))
    miss_ratio = float(np.mean(miss))

    run = max_run = 0
    for c in closed_flags:
        if c:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    y = 1 if (ecr > ecr_thresh or max_run >= consec) else 0
    X = [ear_mean, ear_std, ear_min, ecr, miss_ratio, max_run]
    return X, y

def build_windows(df: pd.DataFrame, window: int, ecr_thresh: float, consec: int) -> Tuple[np.ndarray, np.ndarray]:
    feats, targets = [], []
    buf_ear, buf_closed = [], []
    for _, r in df.sort_values("filepath").iterrows():
        buf_ear.append(r["EAR"])
        buf_closed.append(r["closed"])
        if len(buf_ear) == window:
            X, y = window_features_and_label(np.array(buf_ear), np.array(buf_closed), ecr_thresh, consec)
            feats.append(X); targets.append(y)
            buf_ear, buf_closed = [], []
    return np.array(feats, dtype=float), np.array(targets, dtype=int)

# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", default=DEFAULT_ROOT, help="Root folder containing class subfolders")
    ap.add_argument("--window", type=int, default=15, help="ECR window size (paper: 15)")
    ap.add_argument("--ear_thresh", type=float, default=0.25, help="EAR threshold for per-frame 'closed'")
    ap.add_argument("--ecr_thresh", type=float, default=0.5, help="ECR threshold for window label")
    ap.add_argument("--consec", type=int, default=3, help="Consecutive closed frames for window label")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=42)
    args = ap.parse_args()

    print(f"Scan dataset under: {args.root_dir}")
    df = scan_root_dir(args.root_dir)

    # Compute EAR per image
    print(f"Computing EAR for {len(df)} images (this may take a while)...")
    ears = []
    with facemesh_ctx() as fm:
        for i, row in df.iterrows():
            try:
                bgr = robust_load_image(row["filepath"])
                ear = try_face_mesh_ear(bgr, fm)
            except Exception:
                ear = np.nan
            ears.append(ear)
            if (i + 1) % 500 == 0:
                print(f"  processed {i+1}/{len(df)}")
    df["EAR"] = ears

    # Derive per-frame closed from EAR only (no label leakage)
    df["closed"] = (df["EAR"] < args.ear_thresh).astype(int)

    # Build windows -> features + labels
    X, y = build_windows(df, args.window, args.ecr_thresh, args.consec)
    if len(X) == 0:
        print("No full windows formed. Need at least `window` images total.", file=sys.stderr)
        sys.exit(1)

    # Train/test split
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=args.test_size,
                                          random_state=args.random_state, stratify=y)

    # Train RandomForest (as requested)
    clf = RandomForestClassifier(n_estimators=300, random_state=args.random_state)
    clf.fit(Xtr, ytr)
    yp = clf.predict(Xte)
    acc = accuracy_score(yte, yp)
    prec, rec, f1, _ = precision_recall_fscore_support(yte, yp, average="binary", zero_division=0)
    print("\n=== Random Forest (window-level) ===")
    print(f"Accuracy : {acc*100:.2f}%")
    print(f"Precision: {prec*100:.2f}%")
    print(f"Recall   : {rec*100:.2f}%")
    print(f"F1-score : {f1*100:.2f}%")

    # Save model + meta
    payload = {
        "model": clf,
        "feature_names": FEATURE_NAMES,
        "window": args.window,
        "ear_thresh": args.ear_thresh,
        "ecr_thresh": args.ecr_thresh,
        "consec": args.consec,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(payload, f)
    print(f"\nSaved model to {MODEL_PATH}")

if __name__ == "__main__":
    main()
