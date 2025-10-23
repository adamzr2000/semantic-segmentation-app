#!/usr/bin/env python3
import os, time, threading, colorsys
import cv2
import torch
import numpy as np
from flask import Flask, Response, render_template, jsonify

# ---- Config ----
VIDEO_PATH = os.environ.get("VIDEO", "videos/traffic_video.mp4")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_WIDTH = int(os.environ.get("WIDTH", "640"))           # e.g., 640 for faster FPS
SHOW_LEGEND = os.environ.get("SHOW_LEGEND", "1") == "1"   # set 0 to disable
MODEL = os.environ.get("MODEL", "baseline").lower()       # baseline | sol-cpu (sol-gpu future)

MODEL_ASSETS_DIR = {
    "baseline": "models/baseline_deeplabv3_resnet50",
    "sol-cpu": "models/sol_deeplabv3_resnet50/lib_cpu",
    # "sol-gpu": "models/sol_deeplabv3_resnet50/lib_gpu", # future
}.get(MODEL, "models/baseline_deeplabv3_resnet50")

# If baseline is used and weights are cached locally, let torchvision find them
if MODEL == "baseline" and os.path.isdir(MODEL_ASSETS_DIR):
    os.environ.setdefault("TORCH_HOME", MODEL_ASSETS_DIR)

# ---- Env / Versions (startup sanity logs) ----
TORCH_VER = torch.__version__
try:
    import torchvision as _tv
    TV_VER = _tv.__version__
    from torchvision.models.segmentation import (
        deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
    )
    from torchvision import transforms
except Exception:
    TV_VER = "(not installed)"
    deeplabv3_resnet50 = DeepLabV3_ResNet50_Weights = transforms = None  # type: ignore

CUDA_AVAIL = torch.cuda.is_available()
CUDNN_ENABLED = torch.backends.cudnn.enabled
try:
    CUDNN_VERSION = torch.backends.cudnn.version()
except Exception:
    CUDNN_VERSION = None

DEVICE_NAME = "cpu"
if DEVICE.startswith("cuda") and CUDA_AVAIL and MODEL == "baseline":
    DEVICE_NAME = torch.cuda.get_device_name(0)
    CC = torch.cuda.get_device_capability(0)
    print(f"[INFO] Using CUDA device: {DEVICE_NAME} (cc {CC[0]}.{CC[1]})")
else:
    print("[INFO] Using CPU")

print(f"[INFO] MODEL={MODEL}  ASSETS_DIR={MODEL_ASSETS_DIR}")
print(f"[INFO] torch={TORCH_VER}, torchvision={TV_VER}")
print(f"[INFO] CUDA available={CUDA_AVAIL}, cuDNN enabled={CUDNN_ENABLED}, cuDNN version={CUDNN_VERSION}")

# ---- Backends ----
# We unify on a small internal interface:
#   _backend_predict(frame_bgr) -> (labels_np, model_ms)
# where labels_np is HxW uint8 aligned to the *display frame* size used for overlay.

# Categories & palette (COCO-stuff/VOC-like 21 classes fallback if unknown)
CATEGORIES = None
NUM_CLASSES = 21

def make_hsv_palette(n, s=0.75, v=0.95, seed=123):
    rng = np.random.default_rng(seed)
    hues = (np.arange(n) * 0.61803398875) % 1.0
    rng.shuffle(hues)
    pal = []
    for h in hues:
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        pal.append([int(r*255), int(g*255), int(b*255)])
    return np.array(pal, dtype=np.uint8)

PALETTE = make_hsv_palette(NUM_CLASSES)

def background_id():
    if not CATEGORIES: return 0
    for i, name in enumerate(CATEGORIES):
        if str(name).lower() in ("background", "__background__", "bg"):
            return i
    return 0

BG_ID = background_id()

# ---- Baseline (torchvision) backend ----
if MODEL == "baseline":
    weights = DeepLabV3_ResNet50_Weights.DEFAULT
    preprocess = weights.transforms()
    CATEGORIES = weights.meta.get("categories", None)
    if CATEGORIES is not None:
        NUM_CLASSES = len(CATEGORIES)
        # refresh palette with true class count
        PALETTE = make_hsv_palette(NUM_CLASSES)
        BG_ID = background_id()

    _torch_model = deeplabv3_resnet50(weights=weights).eval().to(DEVICE)
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True
    print("[INFO] Baseline DeepLabv3_ResNet50 loaded (torchvision)")

    def _backend_predict(frame_bgr, out_w):
        # Prepare a display-sized frame first (maintain AR)
        h, w = frame_bgr.shape[:2]
        new_h = int(h * (out_w / float(w)))
        frame_bgr = cv2.resize(frame_bgr, (out_w, new_h), interpolation=cv2.INTER_AREA)

        # Preprocess for model
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil = transforms.functional.to_pil_image(frame_rgb)
        x = preprocess(pil).unsqueeze(0).to(DEVICE)

        # Model timing
        if DEVICE.startswith("cuda"):
            start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
            with torch.autocast(device_type="cuda", enabled=True):
                start.record(); out = _torch_model(x)["out"]; end.record()
            torch.cuda.synchronize()
            model_ms = start.elapsed_time(end)
        else:
            t0 = time.perf_counter()
            out = _torch_model(x)["out"]
            model_ms = (time.perf_counter() - t0) * 1000.0

        # logits -> labels and align to display size
        labels = out[0].argmax(0).detach().cpu().numpy().astype(np.uint8)
        labels = cv2.resize(labels, (frame_bgr.shape[1], frame_bgr.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
        return frame_bgr, labels, model_ms

# ---- SOL (CPU) backend ----
elif MODEL == "sol-cpu":
    import ctypes, os
    # Ensure our model dir is on search path
    os.environ["LD_LIBRARY_PATH"] = "/app/models/sol_deeplabv3_resnet50/lib_cpu:" + os.environ.get("LD_LIBRARY_PATH","")

    # Preload dependencies so their symbols are globally visible at dlopen time
    for lib in (
        "/lib/x86_64-linux-gnu/libsleefgnuabi.so.3",
        "/lib/x86_64-linux-gnu/libsleef.so.3",
        "/lib/x86_64-linux-gnu/libgomp.so.1",
        "/app/models/sol_deeplabv3_resnet50/lib_cpu/libsol-dnn-dnnl-deployment-0.8.0rc4-3.7.1.so",
    ):
        try:
            ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
            print(f"[INFO] Preloaded {lib} (RTLD_GLOBAL)")
        except OSError as e:
            print(f"[WARN] Could not preload {lib}: {e}")

    # Now import the SOL wrapper (which dlopens libdeeplabv3_resnet50.so)
    from importlib import import_module
    sol_mod = import_module("models.sol_deeplabv3_resnet50.lib_cpu.deeplabv3_resnet50")
    _sol = sol_mod.sol_deeplabv3_resnet50("models/sol_deeplabv3_resnet50/lib_cpu")
    _sol.init()
    print("[INFO] SOL lib_cpu backend initialized")

    def _backend_predict(frame_bgr, out_w):
        # Prepare display-sized frame (for overlay)
        h, w = frame_bgr.shape[:2]
        new_h = int(h * (out_w / float(w)))
        frame_bgr_disp = cv2.resize(frame_bgr, (out_w, new_h), interpolation=cv2.INTER_AREA)

        # Model input: RGB, 224x224, float32 [0,1], NCHW
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inp224 = cv2.resize(frame_rgb, (224, 224), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        x = np.transpose(inp224, (2, 0, 1))[None, ...]  # (1,3,224,224) NCHW float32

        # Model timing (CPU)
        t0 = time.perf_counter()
        out = _sol(x)  # returns numpy (1,21,224,224) float32 logits
        model_ms = (time.perf_counter() - t0) * 1000.0

        # logits -> labels (224x224) -> resize to display size
        logits = out  # (1,C,H,W)
        labels224 = np.argmax(logits, axis=1)[0].astype(np.uint8)  # (224,224)
        labels = cv2.resize(labels224, (frame_bgr_disp.shape[1], frame_bgr_disp.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
        return frame_bgr_disp, labels, model_ms

else:
    raise ValueError(f"Unknown MODEL '{MODEL}'. Use 'baseline' or 'sol-cpu'.")

# ---- Metrics (thread-safe) ----
_metrics = {
    "fps_pipeline_ema": 0.0,
    "latency_e2e_ms_ema": 0.0,
    "latency_model_ms_ema": 0.0,
    "frames": 0,
    "device": DEVICE_NAME,
    "torch": TORCH_VER,
    "torchvision": TV_VER,
    "model": MODEL,
}
_METRICS_LOCK = threading.Lock()
_EMA_ALPHA = 0.2
def _update_metrics(fps_pipeline: float, e2e_ms: float, model_ms: float):
    with _METRICS_LOCK:
        _metrics["fps_pipeline_ema"] = (1 - _EMA_ALPHA) * _metrics["fps_pipeline_ema"] + _EMA_ALPHA * fps_pipeline
        _metrics["latency_e2e_ms_ema"] = (1 - _EMA_ALPHA) * _metrics["latency_e2e_ms_ema"] + _EMA_ALPHA * e2e_ms
        _metrics["latency_model_ms_ema"] = (1 - _EMA_ALPHA) * _metrics["latency_model_ms_ema"] + _EMA_ALPHA * model_ms
        _metrics["frames"] += 1

# ---- Flask App ----
app = Flask(__name__, template_folder="templates")

def open_video(path):
    cap = cv2.VideoCapture(path if not str(path).isdigit() else int(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {path}")
    return cap

def colorize_mask(labels: np.ndarray) -> np.ndarray:
    return PALETTE[labels]  # RGB uint8

def overlay(frame_bgr: np.ndarray, mask_rgb: np.ndarray, labels: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    non_bg = (labels != BG_ID)
    mask_bgr = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
    out = frame_bgr.copy()
    out[non_bg] = cv2.addWeighted(frame_bgr[non_bg], 1.0, mask_bgr[non_bg], alpha, 0.0)
    return out

def draw_legend(img_bgr: np.ndarray, labels: np.ndarray, top_k=5):
    if not SHOW_LEGEND or CATEGORIES is None:
        return img_bgr
    uniq, counts = np.unique(labels, return_counts=True)
    items = []
    for i, c in zip(uniq.tolist(), counts.tolist()):
        if i == BG_ID: continue
        name = CATEGORIES[i] if i < len(CATEGORIES) else f"id{i}"
        items.append((c, i, str(name)))
    if not items: return img_bgr
    items.sort(reverse=True); items = items[:top_k]

    pad, sw, sh = 6, 18, 14
    line_h = max(22, sh + 6); box_w = 240; box_h = pad*2 + line_h*len(items)
    x0, y0 = 10, 40; x1, y1 = x0 + box_w, y0 + box_h
    ov = img_bgr.copy(); cv2.rectangle(ov, (x0,y0), (x1,y1), (0,0,0), -1)
    img_bgr = cv2.addWeighted(img_bgr, 1.0, ov, 0.4, 0.0)

    y = y0 + pad + sh
    for _, cls_id, name in items:
        rgb = PALETTE[cls_id].tolist(); bgr = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
        cv2.rectangle(img_bgr, (x0+pad, y-sh), (x0+pad+sw, y), bgr, -1)
        cv2.putText(img_bgr, name[:22], (x0+pad+sw+8, y-2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
        y += line_h
    return img_bgr

def infer_frame(frame_bgr: np.ndarray):
    t0 = time.perf_counter()

    # Backend returns a display-sized frame (BGR) and labels aligned to it
    disp_bgr, labels, model_ms = _backend_predict(frame_bgr, OUT_WIDTH)

    # Colorize + overlay + (optional) legend
    mask_rgb = colorize_mask(labels)
    out_bgr = overlay(disp_bgr, mask_rgb, labels, alpha=0.45)
    out_bgr = draw_legend(out_bgr, labels, top_k=5)

    e2e_ms = (time.perf_counter() - t0) * 1000.0
    return out_bgr, e2e_ms, model_ms

def mjpeg_gen():
    cap = open_video(VIDEO_PATH)
    try:
        while True:
            loop_t0 = time.perf_counter()

            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
                if not ok: break

            out_bgr, e2e_ms, model_ms = infer_frame(frame)

            ok, buf = cv2.imencode(".jpg", out_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ok: continue
            chunk = buf.tobytes()

            loop_dt = time.perf_counter() - loop_t0
            fps_pipeline = 1.0 / loop_dt if loop_dt > 0 else 0.0
            _update_metrics(fps_pipeline, e2e_ms, model_ms)

            yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " +
                   str(len(chunk)).encode() + b"\r\n\r\n" + chunk + b"\r\n")
    finally:
        cap.release()

# ---- Routes ----
@app.get("/")
def index():
    return render_template("index.html",
                           src=VIDEO_PATH,
                           model=MODEL,
                           device=DEVICE_NAME,
                           width=OUT_WIDTH,
                           torch_ver=TORCH_VER,
                           tv_ver=TV_VER,
                           cuda_avail=CUDA_AVAIL,
                           cudnn_enabled=CUDNN_ENABLED,
                           cudnn_version=CUDNN_VERSION)

@app.get("/video")
def video():
    return Response(mjpeg_gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.get("/metrics")
def metrics():
    with _METRICS_LOCK:
        return jsonify({
            "fps_pipeline": round(_metrics["fps_pipeline_ema"], 2),
            "latency_e2e_ms": round(_metrics["latency_e2e_ms_ema"], 2),
            "latency_model_ms": round(_metrics["latency_model_ms_ema"], 2),
            "frames": _metrics["frames"],
            "device": _metrics["device"],
            "model": _metrics["model"],
            "torch": _metrics["torch"],
            "torchvision": _metrics["torchvision"],
        })

@app.get("/health")
def health():
    return jsonify(ok=True, device=DEVICE_NAME, model=MODEL)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)

