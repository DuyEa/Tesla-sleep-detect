import os
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import argparse, sys, pathlib
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import mediapipe as mp
from model_4d import Model4D  # your CNN model


FPS_DEFAULT = 30.0
CLOSED_SECONDS = 2.0      # drowsy if eyes closed ≥ 2.0 s
P_CLOSED_THRESH = 0.5     # per-frame threshold


RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE  = [362, 385, 387, 263, 373, 380]

# preprocessing for 4D model
TF_4D = T.Compose([
    T.Resize((24, 24)),
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,))
])


class FaceMeshCtx:
    def __init__(self):
        self.fm = None
    def __enter__(self):
        self.fm = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        return self.fm
    def __exit__(self, exc_type, exc, tb):
        try: self.fm.close()
        except: pass

def extract_eye_polys(frame, fm):
    h, w = frame.shape[:2]
    res = fm.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        return None, None
    pts = np.array([[lm.x*w, lm.y*h] for lm in res.multi_face_landmarks[0].landmark], np.float32)
    return pts[RIGHT_EYE].astype(np.int32), pts[LEFT_EYE].astype(np.int32)

def crop_eye(gray_frame, poly, pad_scale=0.25):
    if poly is None: return None
    x, y, w, h = cv2.boundingRect(poly)
    cx, cy = x + w/2, y + h/2
    side = int(max(w, h) * (1.0 + pad_scale))
    x0, y0 = max(0, int(cx - side/2)), max(0, int(cy - side/2))
    x1, y1 = min(gray_frame.shape[1], x0 + side), min(gray_frame.shape[0], y0 + side)
    patch = gray_frame[y0:y1, x0:x1]
    return patch if patch.size else None

def to_tensor(gray):
    if gray is None: return None
    return TF_4D(Image.fromarray(gray)).unsqueeze(0)

def load_model(path, device):
    model = Model4D().to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model

def infer_closed_prob(model, device, right_gray, left_gray):
    imgs = []
    for g in (right_gray, left_gray):
        t = to_tensor(g)
        if t is not None: imgs.append(t)
    if not imgs: return None
    batch = torch.cat(imgs, dim=0).to(device)
    with torch.no_grad():
        logits = model(batch)
        p_open = torch.sigmoid(logits)
        p_closed = 1 - p_open
    return float(p_closed.mean().item())

def overlay_eye_focus(img, rpoly, lpoly, state):
    if rpoly is None or lpoly is None: return
    if state == "SLEEP":
        overlay = img.copy()
        cv2.fillPoly(overlay, [rpoly], (0,0,255))
        cv2.fillPoly(overlay, [lpoly], (0,0,255))
        cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
    else:
        cv2.polylines(img, [rpoly], True, (0,255,0), 2, cv2.LINE_AA)
        cv2.polylines(img, [lpoly], True, (0,255,0), 2, cv2.LINE_AA)

def prompt_source_if_needed(args):
    if args.video is not None or args.camera is not None:
        return args
    while True:
        print("\nSelect input source:\n  [1] Video file\n  [2] Webcam")
        choice = input("Enter 1 or 2: ").strip()
        if choice == "1":
            path = input("Enter path to video file: ").strip().strip('"').strip("'")
            if path:
                args.video = path; args.camera = None
                return args
            else:
                print("Empty path. Try again.")
        elif choice == "2":
            idx = input("Webcam index (Enter for 0): ").strip()
            try: args.camera = int(idx) if idx else 0
            except: args.camera = 0
            args.video = None
            return args
        else:
            print("Invalid choice.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=str, help="Video path")
    ap.add_argument("--camera", type=int, help="Webcam index")
    ap.add_argument("--model_path", type=str, default="model_4d.pth")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fps", type=float, default=FPS_DEFAULT)
    ap.add_argument("--closed_sec", type=float, default=CLOSED_SECONDS)
    ap.add_argument("--p_closed_thresh", type=float, default=P_CLOSED_THRESH)
    ap.add_argument("--out", type=str, default="")
    args = prompt_source_if_needed(ap.parse_args())

    # Open source
    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print("Cannot open video", file=sys.stderr); sys.exit(1)
        fps = cap.get(cv2.CAP_PROP_FPS) or args.fps
        W, H = int(cap.get(3)), int(cap.get(4))
        out_path = args.out or str(pathlib.Path(args.video).with_name(pathlib.Path(args.video).stem + "_detected.mp4"))
    else:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print("Cannot open webcam", file=sys.stderr); sys.exit(1)
        fps = args.fps
        W, H = 640, 480
        out_path = args.out or "webcam_detected.mp4"

    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    model = load_model(args.model_path, args.device)
    closed_need = int(round(args.closed_sec * fps))

    consec_closed = 0
    with FaceMeshCtx() as fm:
        while True:
            ok, frame = cap.read()
            if not ok: break
            rpoly, lpoly = extract_eye_polys(frame, fm)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rgray = crop_eye(gray, rpoly)
            lgray = crop_eye(gray, lpoly)
            p_closed = infer_closed_prob(model, args.device, rgray, lgray)
            if p_closed is None: continue

            consec_closed = consec_closed + 1 if p_closed >= args.p_closed_thresh else 0
            sleepy = consec_closed >= closed_need
            state = "SLEEP" if sleepy else "AWAKE"
            overlay_eye_focus(frame, rpoly, lpoly, state)
            cv2.rectangle(frame, (0,0), (W,42), (0,0,0), -1)
            color = (0,0,255) if sleepy else (0,255,0)
            cv2.putText(frame, f"Status: {state}  p_closed={p_closed:.2f}  closed_run={consec_closed}/{closed_need}",
                        (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            if sleepy:
                cv2.putText(frame, "SLEEPY! ALERT!", (W//2-180,100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
            if writer.isOpened(): writer.write(frame)
            cv2.imshow("4D Drowsiness Detector", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release(); writer.release(); cv2.destroyAllWindows()
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
