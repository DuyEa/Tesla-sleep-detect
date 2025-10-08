import os
# stop log spam
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse, sys, pickle, math
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
# Head pose landmarks and 3D model
POSE_LANDMARKS = [1, 152, 33, 263, 61, 291]
MODEL_3D = np.array([
    [   0.0,     0.0,    0.0],   # nose tip
    [   0.0,  -330.0,  -65.0],   # chin
    [ -225.0,   170.0, -135.0],  # left eye outer corner
    [  225.0,   170.0, -135.0],  # right eye outer corner
    [ -150.0,  -150.0, -125.0],  # left mouth corner
    [  150.0,  -150.0, -125.0],  # right mouth corner
], dtype=np.float32)

VALID_IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

FEATURE_NAMES = ["ear_mean", "ear_std", "ear_min", "ecr", "miss_ratio", "max_closed_run", "pitch_mean", "yaw_abs_mean"]


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

def extract_eye_and_pose(bgr: np.ndarray, fm) -> Tuple[float, float, float]:
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    res = fm.process(rgb)
    if not res.multi_face_landmarks:
        return np.nan, np.nan, np.nan
    face = res.multi_face_landmarks[0]
    pts = np.array([[lm.x * w, lm.y * h] for lm in face.landmark], dtype=np.float32)
    ear_r = compute_ear_from_landmarks(pts[RIGHT_EYE])
    ear_l = compute_ear_from_landmarks(pts[LEFT_EYE])
    ear = np.nanmean([ear_r, ear_l])
    lm2d = pts[POSE_LANDMARKS][:, :2].astype(np.float32)
    f = w
    cam_mtx = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]], dtype=np.float32)
    dist = np.zeros((4,1), dtype=np.float32)
    ok, rvec, tvec = cv2.solvePnP(MODEL_3D, lm2d, cam_mtx, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return ear, np.nan, np.nan
    R, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2 + 1e-9)
    pitch = math.degrees(math.atan2(-R[2,0], sy))
    yaw = math.degrees(math.atan2(-R[1,0], R[0,0]))
    return ear, pitch, yaw

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

def build_windows_by_folder_label(df: pd.DataFrame, window: int, ear_thresh: float) -> Tuple[np.ndarray, np.ndarray]:
    feats, targets = [], []
    for lbl in sorted(df["label"].unique()):
        sub = df[df["label"] == lbl].sort_values("filepath")
        ears = sub["EAR"].values.astype(float)
        pitches = sub["pitch"].values.astype(float)
        yaws = sub["yaw"].values.astype(float)
        n_full = len(ears) // window
        for i in range(n_full):
            chunk_ear = ears[i*window:(i+1)*window]
            chunk_pitch = pitches[i*window:(i+1)*window]
            chunk_yaw = yaws[i*window:(i+1)*window]
            # Impute EAR
            miss_ear = np.isnan(chunk_ear)
            if np.all(miss_ear):
                ear_imp = np.full_like(chunk_ear, 0.25, dtype=float)
            else:
                med = np.nanmedian(chunk_ear)
                ear_imp = np.where(miss_ear, med, chunk_ear)
            ear_mean = float(np.mean(ear_imp))
            ear_std  = float(np.std(ear_imp))
            ear_min  = float(np.min(ear_imp))
            closed_flags = (ear_imp < ear_thresh).astype(int)
            ecr      = float(np.mean(closed_flags))
            miss_ratio = float(np.mean(miss_ear))
            # Max closed run
            run = max_run = 0
            for c in closed_flags:
                if c:
                    run += 1
                    max_run = max(max_run, run)
                else:
                    run = 0
            # Impute pitch
            miss_pitch = np.isnan(chunk_pitch)
            if np.all(miss_pitch):
                pitch_imp = np.zeros_like(chunk_pitch, dtype=float)
            else:
                med_p = np.nanmedian(chunk_pitch)
                pitch_imp = np.where(miss_pitch, med_p, chunk_pitch)
            pitch_mean = float(np.mean(pitch_imp))
            # Impute yaw abs
            yaw_abs = np.abs(chunk_yaw)
            miss_yaw = np.isnan(yaw_abs)
            if np.all(miss_yaw):
                yaw_abs_imp = np.zeros_like(yaw_abs, dtype=float)
            else:
                med_y = np.nanmedian(yaw_abs)
                yaw_abs_imp = np.where(miss_yaw, med_y, yaw_abs)
            yaw_abs_mean = float(np.mean(yaw_abs_imp))
            X = [ear_mean, ear_std, ear_min, ecr, miss_ratio, max_run, pitch_mean, yaw_abs_mean]
            feats.append(X)
            targets.append(0 if lbl == "forward_look" else 1)
    return np.array(feats, dtype=float), np.array(targets, dtype=int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root_dir", default=DEFAULT_ROOT, help="Root folder containing class subfolders")
    ap.add_argument("--window", type=int, default=15, help="Window size")
    ap.add_argument("--ear_thresh", type=float, default=0.25, help="EAR threshold for per-frame 'closed'")
    ap.add_argument("--ecr_thresh", type=float, default=0.5, help="ECR threshold for window label (unused now)")
    ap.add_argument("--consec", type=int, default=3, help="Consecutive closed frames for window label (unused now)")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=42)
    args = ap.parse_args()

    print(f"Scan dataset under: {args.root_dir}")
    df = scan_root_dir(args.root_dir)

    # Compute EAR and head pose for each image
    print(f"Computing EAR and head pose for {len(df)} images (this may take a while)...")
    ears, pitches, yaws = [], [], []
    with facemesh_ctx() as fm:
        for i, row in df.iterrows():
            try:
                bgr = robust_load_image(row["filepath"])
                ear, pitch, yaw = extract_eye_and_pose(bgr, fm)
            except Exception:
                ear = pitch = yaw = np.nan
            ears.append(ear)
            pitches.append(pitch)
            yaws.append(yaw)
            if (i + 1) % 500 == 0:
                print(f"  processed {i+1}/{len(df)}")
    df["EAR"] = ears
    df["pitch"] = pitches
    df["yaw"] = yaws

    # Build windows -> features + labels (forward_look=0 focused, others=1 distracted)
    X, y = build_windows_by_folder_label(df, args.window, args.ear_thresh)
    if len(X) == 0:
        print("No full windows formed. Need at least `window` images total.", file=sys.stderr)
        sys.exit(1)

    # Train/test split
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=args.test_size,
                                          random_state=args.random_state, stratify=y)

    # Train RandomForest
    clf = RandomForestClassifier(n_estimators=300, random_state=args.random_state)
    clf.fit(Xtr, ytr)
    yp = clf.predict(Xte)
    acc = accuracy_score(yte, yp)
    prec, rec, f1, _ = precision_recall_fscore_support(yte, yp, average="binary", zero_division=0)
    print("\n=== Random Forest (focus detection) ===")
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