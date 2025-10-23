#!/usr/bin/env python3
from flask import Flask, Response, jsonify, render_template
import os, cv2, time, logging, argparse, atexit, threading
import numpy as np
import torch
from torchvision import models
from PIL import Image

# ---------------- Args ----------------
parser = argparse.ArgumentParser()
parser.add_argument("--video", default=os.path.join(os.path.dirname(__file__), "videos", "test-video.mp4"))
parser.add_argument("--port", type=int, default=5000)
parser.add_argument("--proc_width", type=int, default=640)   # resize for speed
parser.add_argument("--proc_height", type=int, default=360)
parser.add_argument("--alpha", type=float, default=0.5)      # overlay opacity
args = parser.parse_args()

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("deeplab_stream")

# ---------------- Model (Torchvision docs style) ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_enabled = (device.type == "cuda")
weights = models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT  # COCO_WITH_VOC_LABELS_V1
model = models.segmentation.deeplabv3_resnet50(weights=weights).to(device).eval()
preprocess = weights.transforms()
log.info(f"Device: {device} | Weights: {weights}")

# VOC-like palette (21 classes)
PALETTE = np.array([
    (0,0,0),(128,0,0),(0,128,0),(128,128,0),(0,0,128),
    (128,0,128),(0,128,128),(128,128,128),(64,0,0),(192,0,0),
    (64,128,0),(192,128,0),(64,0,128),(192,0,128),(64,128,128),
    (192,128,128),(0,64,0),(128,64,0),(0,192,0),(128,192,0),(0,64,128)
], dtype=np.uint8)

def colorize(mask_hw: np.ndarray) -> np.ndarray:
    c = PALETTE[np.clip(mask_hw.astype(np.int32), 0, len(PALETTE)-1)]
    return c[:, :, ::-1].copy()  # RGB->BGR for OpenCV

# ---------------- Video ----------------
if not os.path.exists(args.video):
    raise FileNotFoundError(args.video)
cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    raise RuntimeError(f"Cannot open {args.video}")
cap_lock = threading.Lock()

@atexit.register
def _release():
    with cap_lock:
        if cap.isOpened():
            cap.release()
            log.info("Released video.")

# ---------------- Metrics ----------------
frame_count = 0
inference_count = 0
total_infer_t = 0.0
last_latency = 0.0
last_fps = 0.0
last_avg_latency = 0.0
MODEL_NAME = "deeplabv3_resnet50"

def encode_jpeg(img, q=80):
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    return buf.tobytes() if ok else b""

def run_segmentation(frame_bgr):
    global inference_count, total_infer_t, last_latency, last_fps, last_avg_latency

    # Resize for speed, preprocess per weights
    small = cv2.resize(frame_bgr, (args.proc_width, args.proc_height), interpolation=cv2.INTER_LINEAR)
    pil = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
    x = preprocess(pil).unsqueeze(0).to(device)

    if gpu_enabled:
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        out = model(x)["out"]          # [1, 21, H, W]
        mask = out.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
    if gpu_enabled:
        torch.cuda.synchronize()
    dt = time.time() - t0

    inference_count += 1
    total_infer_t += dt
    last_latency = dt
    last_fps = (1.0/dt) if dt > 0 else 0.0
    last_avg_latency = total_infer_t / max(1, inference_count)

    seg_small = colorize(mask)  # BGR
    seg = cv2.resize(seg_small, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    overlay = cv2.addWeighted(frame_bgr, 1.0 - args.alpha, seg, args.alpha, 0.0)
    return overlay

# ---------------- Streams ----------------
def gen_original():
    while True:
        with cap_lock:
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + encode_jpeg(frame) + b"\r\n")

def gen_segmentation():
    global frame_count
    while True:
        with cap_lock:
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
        overlay = run_segmentation(frame)
        frame_count += 1
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + encode_jpeg(overlay) + b"\r\n")

# ---------------- Flask ----------------
app = Flask(__name__)

@app.route("/")
def index():
    # Uses your templates/index.html (kept as-is)
    return render_template("index.html")

# Match your HTML's endpoints:
@app.route("/original_video_feed")
def original_video_feed():
    return Response(gen_original(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/depth_video_feed")
def depth_video_feed():
    # “depth” in the UI actually serves the segmentation overlay
    return Response(gen_segmentation(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/metrics")
def metrics():
    avg_fps = (inference_count/total_infer_t) if total_infer_t > 0 else 0.0
    return jsonify(
        frames=frame_count,
        latency_ms=round(last_latency*1000, 2),
        fps=round(last_fps, 2),
        avg_latency_ms=round(last_avg_latency*1000, 2),
        avg_fps=round(avg_fps, 2),
        model=MODEL_NAME,
        device=str(device),
        gpu_enabled=bool(gpu_enabled)
    )

if __name__ == "__main__":
    log.info("Routes: / -> index.html | /original_video_feed | /depth_video_feed | /metrics")
    app.run(host="0.0.0.0", port=args.port, threaded=True)
