import os

from pyexpat import features

# stop log spam
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse, sys, pickle, pathlib, math
import cv2, numpy as np
from collections import deque
from contextlib import contextmanager
import mediapipe as mp

# paper based face features
# FaceMesh indices: 6-point eyes (enough for EAR + overlay polygons)
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE  = [362, 385, 387, 263, 373, 380]

# 2D–3D points for head pose (generic face model)
POSE_LANDMARKS = [1, 152, 33, 263, 61, 291]
MODEL_3D = np.array([
    [   0.0,     0.0,    0.0],   # nose tip
    [   0.0,  -330.0,  -65.0],   # chin
    [ -225.0,   170.0, -135.0],  # left eye outer corner
    [  225.0,   170.0, -135.0],  # right eye outer corner
    [ -150.0,  -150.0, -125.0],  # left mouth corner
    [  150.0,  -150.0, -125.0],  # right mouth corner
], dtype=np.float32)

# Manage FaceMesh
@contextmanager
def facemesh_ctx(static=False):
    base_kwargs = dict(
        static_image_mode=static,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    try:
        fm = mp.solutions.face_mesh.FaceMesh(model_complexity=0, **base_kwargs) #increase if needed max:2
    except TypeError:
        fm = mp.solutions.face_mesh.FaceMesh(**base_kwargs)
    try:
        yield fm
    finally:
        try: fm.close()
        except Exception: pass


def compute_ear(coords):
    # Euclidean of both eyes base on paper
    v1 = np.linalg.norm(coords[1] - coords[5])
    v2 = np.linalg.norm(coords[2] - coords[4])
    h  = np.linalg.norm(coords[0] - coords[3])
    if h <= 1e-6: return np.nan # avoid divide for 0
    return (v1 + v2) / (2.0 * h)

def extract_landmarks(frame_bgr, fm):
    """Return (ear, rpoly, lpoly, lm2d) or (nan, None, None, None) if no face."""
    h, w = frame_bgr.shape[:2] # Only take height and width of the frame(:2) dont need color ch
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = fm.process(rgb)
    if not res.multi_face_landmarks:
        return np.nan, None, None, None

    face = res.multi_face_landmarks[0] #Only have 1 because of max face
    pts = np.array([[lm.x * w, lm.y * h, lm.z] for lm in face.landmark], dtype=np.float32) #Convert h and w of frame to pixel

    r = pts[RIGHT_EYE][:, :2]
    l = pts[LEFT_EYE][:, :2]
    ear = np.nanmean([compute_ear(r), compute_ear(l)]) # Mean without nan

    lm2d = pts[POSE_LANDMARKS][:, :2].astype(np.float32)  # for head pose
    return ear, r.astype(np.int32), l.astype(np.int32), lm2d

def window_features(ear_list, ear_thresh):
    ears = np.array(ear_list, dtype=float) #deque EAR list
    miss = np.isnan(ears)
    ears_imp = np.full_like(ears, 0.25, dtype=float) if np.all(miss) else np.where(miss, np.nanmedian(ears), ears)  #ensure no nan for continue calc
    closed = (ears_imp < ear_thresh).astype(int) #decide the state (sleep/awake)

    ear_mean = float(np.mean(ears_imp))
    ear_std  = float(np.std(ears_imp))
    ear_min  = float(np.min(ears_imp))
    ecr      = float(np.mean(closed))
    miss_ratio = float(np.mean(miss))

    run = max_run = 0 #calc the time eyes closed
    for c in closed:
        if c: run += 1; max_run = max(max_run, run)
        else: run = 0

    X = np.array([ear_mean, ear_std, ear_min, ecr, miss_ratio, max_run], dtype=float)
    return X, ecr, max_run

def head_euler_degrees(lm2d, w, h):
    """
    Compute (yaw, pitch, roll) in degrees from 2D-3D correspondences using solvePnP.
    yaw(+): head turning to the RIGHT
    pitch(+): head tilting DOWN
    roll(+): right ear toward shoulder
    """
    if lm2d is None:
        return None, None, None
    f = w
    cam_mtx = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]], dtype=np.float32)
    dist = np.zeros((4,1), dtype=np.float32)
    ok, rvec, tvec = cv2.solvePnP(MODEL_3D, lm2d, cam_mtx, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None, None, None
    R, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
    yaw   = math.degrees(math.atan2(R[1,0], R[0,0]))
    pitch = math.degrees(math.atan2(-R[2,0], sy))
    roll  = math.degrees(math.atan2(R[2,1], R[2,2]))
    return yaw, pitch, roll

# banner
def draw_banner(img, text, color, bg=(0,0,0), y=60, pad=10):
    #draw
    w = img.shape[1]
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    x = int((w - tw) / 2)
    cv2.rectangle(img, (x - pad, y - th - pad), (x + tw + pad, y + pad), bg, -1)
    cv2.putText(img, text, (x, y), font, scale, color, thick, cv2.LINE_AA)

def overlay_eye_focus(img, rpoly, lpoly, state):
    # draw
    if rpoly is None or lpoly is None: return
    if state == "SLEEP":
        overlay = img.copy()
        cv2.fillPoly(overlay, [rpoly], (0,0,255))
        cv2.fillPoly(overlay, [lpoly], (0,0,255))
        cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
    else:
        cv2.polylines(img, [rpoly], True, (0,255,0), 2, cv2.LINE_AA)
        cv2.polylines(img, [lpoly], True, (0,255,0), 2, cv2.LINE_AA)

def default_out_path(in_path: str) -> str:
    p = pathlib.Path(in_path)
    return str(p.with_name(p.stem + "_detected1").with_suffix(".mp4"))

# ---------- Source selection helper ----------
def prompt_source_if_needed(args):
    """ Choose source:
      1) Video file
      2) Webcam input 0
    """
    if args.video is not None or args.camera is not None:
        return args

    while True:
        print("\nSelect input source:")
        print("  [1] Video file")
        print("  [2] Webcam")
        choice = input("Enter 1 or 2: ").strip()

        if choice == "1":
            path = input("Enter path to video file (mp4): ").strip().strip('"').strip("'")
            if path:
                args.video = path
                args.camera = None
                return args
            else:
                print("Empty path. Please try again.")
        elif choice == "2":
            idx = input("Webcam index (press Enter for 0): ").strip()
            try:
                args.camera = int(idx) if idx else 0
            except ValueError:
                print("Invalid index. Defaulting to 0.")
                args.camera = 0
            args.video = None
            return args
        else:
            print("Invalid choice. Please enter 1 or 2.")

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=False)
    src.add_argument("--video", type=str, help="Path to MP4")
    src.add_argument("--camera", type=int, help="Webcam index")
    ap.add_argument("--model", type=str, default="RandFor1.pkl", help="Trained model (pickle)")
    ap.add_argument("--out", type=str, default="", help="Output mp4 path (auto if empty)")
    # thresholds / rules
    ap.add_argument("--prob_thresh", type=float, default=0.5, help="RF probability threshold (drowsy)")
    ap.add_argument("--sleep_run_windows", type=int, default=3, help="Consecutive drowsy windows => SLEEP")
    ap.add_argument("--sleep_ecr", type=float, default=0.80, help="Or SLEEP if ECR(window) ≥ this")
    ap.add_argument("--pitch_down_deg", type=float, default=25.0, help="Head pitch (deg) to force SLEEP")
    ap.add_argument("--miss_frames", type=int, default=20, help="No face/eyes for N frames => SLEEP")
    ap.add_argument("--yaw_side_deg", type=float, default=35.0, help="|yaw| ≥ this => looking off-road")
    ap.add_argument("--forward_pitch_deg", type=float, default=15.0, help="|pitch| ≤ this counts as forward")
    ap.add_argument("--forward_yaw_deg", type=float, default=15.0, help="|yaw| ≤ this counts as forward")
    ap.add_argument("--attn_run_windows", type=int, default=2, help="Consecutive off-road windows => NOT_FORWARD")
    ap.add_argument("--offroad_ratio", type=float, default=0.6, help="Off-road ratio in window for NOT_FORWARD")
    args = ap.parse_args()

    args = prompt_source_if_needed(args)

    with open(args.model, "rb") as f:
        payload = pickle.load(f)
    clf = payload["model"]
    window = payload["window"]
    ear_thresh = payload["ear_thresh"]

    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f"Cannot open video: {args.video}", file=sys.stderr); sys.exit(1)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_path = args.out or default_out_path(args.video)
    else:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print(f"Cannot open camera {args.camera}", file=sys.stderr); sys.exit(1)
        fps = 25.0
        W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        out_path = args.out or "webcam_detected.mp4"

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    if not writer.isOpened():
        print("Warning: cannot open writer; output will not be saved.", file=sys.stderr)

    ears_win = deque(maxlen=window)
    drowsy_run = 0
    miss_ctr = 0
    last_prob = 0.0
    last_ecr, last_run = 0.0, 0
    off_win = deque(maxlen=window)
    drive_attn = "FORWARD"

    with facemesh_ctx() as fm:
        while True:
            ok, frame = cap.read()
            if not ok: break

            yaw = pitch = roll = None

            ear, rpoly, lpoly, lm2d = extract_landmarks(frame, fm)
            if lm2d is None: miss_ctr += 1
            else:            miss_ctr = 0

            ears_win.append(ear)

            state = "AWAKE"
            if len(ears_win) == window:
                X, ecr, max_run = window_features(list(ears_win), ear_thresh)
                last_ecr, last_run = ecr, max_run

                try:
                    prob = clf.predict_proba([X])[0][1]
                except Exception:
                    prob = float(clf.predict([X])[0])
                last_prob = prob
                drowsy = (prob >= args.prob_thresh)

                drowsy_run = drowsy_run + 1 if drowsy else 0
                sleeping_model = (drowsy_run >= args.sleep_run_windows) or (ecr >= args.sleep_ecr)

                yaw, pitch, roll = head_euler_degrees(lm2d, W, H) if lm2d is not None else (None, None, None)

                sleeping_pose = (pitch is not None and pitch >= args.pitch_down_deg)
                sleeping_miss = (miss_ctr >= args.miss_frames)

                sleeping = sleeping_model or sleeping_pose or sleeping_miss
                state = "SLEEP" if sleeping else "AWAKE"

                off_down = (pitch is not None and pitch >= args.pitch_down_deg)
                off_side = (yaw is not None and abs(yaw) >= args.yaw_side_deg)
                off_miss = (miss_ctr >= args.miss_frames)
                off_flag = (off_down or off_side or off_miss)
                off_win.append(1 if off_flag else 0)

                off_ratio = float(np.mean(off_win)) if len(off_win) == window else 0.0
                run = max_run_off = 0
                for v in off_win:
                    if v:
                        run += 1
                        max_run_off = max(max_run_off, run)
                    else:
                        run = 0
                not_forward = (max_run_off >= args.attn_run_windows) or (off_ratio >= args.offroad_ratio)
                drive_attn = "NOT_FORWARD" if not_forward else "FORWARD"

            overlay_eye_focus(frame, rpoly, lpoly, state)

            top_bar_h = 42
            cv2.rectangle(frame, (0,0), (W, top_bar_h), (0,0,0), -1)
            color = (0,255,0) if state == "AWAKE" else (0,0,255)
            cv2.putText(frame, f"Status: {state}    Prob: {last_prob:.2f}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            ear_disp = 0.0 if np.isnan(ear) else ear
            pitch_str = "NA" if pitch is None else f"{pitch:.1f}°"
            yaw_str   = "NA" if yaw   is None else f"{yaw:.1f}°"
            roll_str  = "NA" if roll  is None else f"{roll:.1f}°"
            info = (f"EAR: {ear_disp:.3f}   ECR(win): {last_ecr:.2f}   Run(win): {last_run}/{window}   "
                    f"Pitch: {pitch_str}  Yaw: {yaw_str}  Roll: {roll_str}   Miss: {miss_ctr}   "
                    f"ATTN: {drive_attn}")
            cv2.putText(frame, info, (10, top_bar_h + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230,230,230), 2)

            if state == "SLEEP":
                draw_banner(frame, "SLEEPY!  ALERT!", (0,0,255), bg=(0,0,0), y=top_bar_h + 70)

            cv2.imshow("Driver Anti-Sleep (EAR+ECR + Head Pose)", frame)
            if writer.isOpened():
                writer.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Saved annotated video to: {out_path}")

if __name__ == "__main__":
    main()
