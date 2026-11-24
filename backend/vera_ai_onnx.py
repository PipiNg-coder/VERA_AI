# -*- coding: utf-8 -*-
"""VERA AI API (bản nhẹ): CLIP chạy bằng onnxruntime, không cần torch/transformers.

Tiền xử lý và đặc trưng FFT/SRM giữ nguyên như bản gốc để kết quả khớp model đã huấn luyện.
"""
from __future__ import annotations
import io, os, threading, urllib.request
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import joblib
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image, ImageOps, UnidentifiedImageError

cv2.setNumThreads(1)
BASE_DIR = Path(__file__).resolve().parent
BUNDLE_PATH = Path(os.getenv("VERA_MODEL_BUNDLE", str(BASE_DIR / "models" / "ml_clip_bundle_final.pkl")))
ONNX_PATH = Path(os.getenv("VERA_ONNX_MODEL", str(BASE_DIR / "models" / "clip_vision.onnx")))
ONNX_URL = os.getenv("VERA_ONNX_URL", "")  # nếu file onnx quá nặng cho GitHub: link tải (vd GitHub Releases)
MAX_UPLOAD_MB = int(os.getenv("VERA_MAX_UPLOAD_MB", "15"))
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
INDEX_PATH = BASE_DIR.parent / "index.html"

CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32).reshape(1, 1, 3)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32).reshape(1, 1, 3)

_lock = threading.Lock()
_state = {}


@asynccontextmanager
async def lifespan(app):
    try:
        load_pipeline()
    except Exception as e:  # không làm sập server; /health sẽ báo lỗi
        _state["error"] = f"{type(e).__name__}: {e}"
    yield


app = FastAPI(title="VERA AI API", version="2.0.0", description="Phân loại ảnh REAL/AI: CLIP (ONNX) + FFT Ensemble", lifespan=lifespan)
origins = [x.strip() for x in os.getenv("VERA_CORS_ORIGINS", "*").split(",") if x.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=False, allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["*"])

_FREQ_GEOM = {}
def _freq_geom(h: int, w: int):
    key = (h, w)
    if key not in _FREQ_GEOM:
        cy, cx = h // 2, w // 2; y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int); max_r = min(cy, cx); bin_size = max_r / 48
        _FREQ_GEOM[key] = [np.flatnonzero(((r >= i * bin_size) & (r < (i + 1) * bin_size)).ravel()) for i in range(48)]
    return _FREQ_GEOM[key]

def extract_frequency_features(img_pil: Image.Image) -> np.ndarray:
    """Giống hệt bản gốc: 64 đặc trưng FFT/SRM."""
    img_gray = np.array(img_pil.convert('L'), dtype=np.float32); h, w = img_gray.shape
    mag_spec = np.log(np.abs(np.fft.fftshift(np.fft.fft2(img_gray))) + 1e-8)
    cy, cx = h // 2, w // 2; flat = mag_spec.ravel(); radial_prof = np.zeros(48, dtype=np.float32)
    for i, ix in enumerate(_freq_geom(h, w)):
        if ix.size: radial_prof[i] = flat[ix].mean()
    radial_prof = (radial_prof - radial_prof.mean()) / (radial_prof.std() + 1e-8)
    srm_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    noise_res = cv2.filter2D(img_gray, -1, srm_kernel); q = np.percentile(noise_res, [10, 25, 50, 75, 90]); p95 = np.percentile(mag_spec, 95)
    noise_stats = np.array([noise_res.mean(), noise_res.std(), q[0], q[1], q[2], q[3], q[4], np.var(noise_res),
        mag_spec[:cy // 2, :cx // 2].mean(), mag_spec[cy // 2:, :cx // 2].mean(), mag_spec[:cy // 2, cx // 2:].mean(), mag_spec[cy // 2:, cx // 2:].mean(),
        mag_spec[cy - 10:cy + 10, cx - 10:cx + 10].mean(), np.std(mag_spec), np.mean(mag_spec > p95),
        np.sum(mag_spec > (mag_spec.mean() + 2 * mag_spec.std())) / (h * w)], dtype=np.float32)
    return np.concatenate([radial_prof, noise_stats]).astype(np.float32)

def prepare_image(img: Image.Image, size: int = 256) -> Image.Image:
    img = ImageOps.exif_transpose(img).convert('RGB')
    if img.size != (size, size): img = img.resize((size, size), Image.Resampling.BICUBIC)
    return img

def clip_pixel_values(img256: Image.Image) -> np.ndarray:
    """Tương đương CLIPImageProcessor của openai/clip-vit-base-patch32 với ảnh vuông 256x256."""
    im = img256.resize((224, 224), Image.Resampling.BICUBIC)
    arr = (np.asarray(im, dtype=np.float32) / 255.0 - CLIP_MEAN) / CLIP_STD
    return np.ascontiguousarray(arr.transpose(2, 0, 1)[None], dtype=np.float32)

def _ensure_onnx():
    if ONNX_PATH.is_file():
        return ONNX_PATH
    if not ONNX_URL:
        raise FileNotFoundError(f"Không tìm thấy model ONNX: {ONNX_PATH}. Đặt file vào backend/models/ hoặc đặt biến VERA_ONNX_URL.")
    dst = Path("/tmp/clip_vision.onnx")
    if not dst.is_file():
        tmp = dst.with_suffix(".part")
        urllib.request.urlretrieve(ONNX_URL, tmp)
        tmp.rename(dst)
    return dst

def load_pipeline():
    with _lock:
        if "bundle" in _state:
            return _state["bundle"], _state["session"]
        if not BUNDLE_PATH.is_file():
            raise FileNotFoundError(f"Không tìm thấy model bundle: {BUNDLE_PATH}")
        bundle = joblib.load(BUNDLE_PATH)
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(os.getenv("VERA_THREADS", "2"))
        so.enable_cpu_mem_arena = False   # tiết kiệm RAM
        so.enable_mem_pattern = False
        session = ort.InferenceSession(str(_ensure_onnx()), so, providers=["CPUExecutionProvider"])
        _state.update(bundle=bundle, session=session); _state.pop("error", None)
        return bundle, session

def predict_image(img: Image.Image):
    bundle, session = load_pipeline()
    img = prepare_image(img)
    pv = clip_pixel_values(img)
    emb = session.run(None, {session.get_inputs()[0].name: pv})[0].astype(np.float32)
    emb = emb / np.linalg.norm(emb, axis=-1, keepdims=True)
    raw_freq = extract_frequency_features(img).reshape(1, -1)
    freq = bundle['scaler_freq'].transform(raw_freq)
    features = np.concatenate([emb, freq], axis=1).astype(np.float32)
    models = bundle['models']; weights = bundle.get('ensemble_weights', {})
    probabilities = {}
    for name, model in models.items():
        classes = list(model.classes_)
        if 1 not in classes: raise ValueError(f"Model {name} không có class AI (1).")
        probabilities[name] = float(model.predict_proba(features)[:, classes.index(1)][0])
    if weights and all(name in probabilities for name in weights):
        ai_prob = float(sum(float(weights[n]) * probabilities[n] for n in weights))
    else:
        best = bundle.get('best_model_name')
        if best not in probabilities: best = next(iter(probabilities))
        ai_prob = probabilities[best]
    threshold = float(bundle.get('threshold', 0.5))
    label = 'AI' if ai_prob >= threshold else 'REAL'
    confidence = ai_prob if label == 'AI' else 1.0 - ai_prob
    return {'label': label, 'ai_probability': round(ai_prob, 6), 'confidence': round(confidence, 6), 'threshold': round(threshold, 6),
            'model': 'CLIP (ONNX) + FFT Ensemble', 'device': 'cpu', 'component_probabilities': {k: round(v, 6) for k, v in probabilities.items()}}

@app.get('/', include_in_schema=False)
def root():
    if INDEX_PATH.is_file():
        return FileResponse(INDEX_PATH)
    return {'service': 'VERA AI API', 'status': 'running', 'health': '/health'}

@app.get('/health')
def health():
    out = {'status': 'ok', 'bundle_exists': BUNDLE_PATH.is_file(), 'onnx_exists': ONNX_PATH.is_file() or bool(ONNX_URL),
           'model_loaded': "session" in _state, 'device': 'cpu'}
    if "error" in _state: out['status'] = 'error'; out['error'] = _state["error"]
    try:
        import resource; out['rss_mb'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
    except Exception: pass
    return out

@app.post('/api/predict')
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES: raise HTTPException(415, 'Chỉ hỗ trợ JPG, PNG hoặc WEBP.')
    data = await file.read()
    if not data: raise HTTPException(400, 'Tệp ảnh trống.')
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024: raise HTTPException(413, f'Ảnh vượt quá {MAX_UPLOAD_MB} MB.')
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.load(); return predict_image(im.copy())
    except FileNotFoundError as e: raise HTTPException(503, str(e))
    except UnidentifiedImageError: raise HTTPException(400, 'Không đọc được định dạng ảnh.')
    except Exception as e: raise HTTPException(500, f'Lỗi xử lý mô hình: {type(e).__name__}: {e}')
