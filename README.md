---
title: VERA AI API
emoji: 🔎
colorFrom: blue
colorTo: cyan
sdk: docker
app_port: 7860
---

# VERA AI — Website + Inference API

Repository gồm giao diện HTML tĩnh và API Python để dự đoán ảnh REAL/AI bằng **CLIP + FFT/SRM + ensemble**. Mình đã đối chiếu file huấn luyện gốc: mô hình thực tế là CLIP + FFT ensemble, không phải ResNet50; vì vậy giao diện đã được sửa để tránh mô tả sai kiến trúc.

## Cấu trúc

```text
VERA-AI/
├── index.html
├── backend/
│   ├── vera_ai_final_fixed.py       # FastAPI inference server
│   ├── training_pipeline_colab.py   # file Colab gốc, giữ riêng để huấn luyện
│   └── models/
│       └── ml_clip_bundle_final.pkl # bạn cần tự chép model bundle vào đây
├── requirements.txt
└── README.md
```

## 1. Chuẩn bị model đã huấn luyện

API không tự huấn luyện lại 70.000 ảnh khi khởi động. Nó cần file `ml_clip_bundle_final.pkl` do pipeline Colab tạo ra. Trong Google Drive, file được lưu ở:

`/content/drive/MyDrive/ml_clip_features_70k_robust/ml_clip_bundle_final.pkl`

Tải file đó về máy và đặt vào `backend/models/ml_clip_bundle_final.pkl`. Không đưa ZIP 70.000 ảnh hoặc dữ liệu cá nhân lên GitHub. File model có thể lớn; nếu vượt giới hạn GitHub, lưu ở kho model riêng và cấu hình biến môi trường `VERA_MODEL_BUNDLE` trỏ đến vị trí đã tải về.

## 2. Chạy backend trên máy

Python khuyến nghị 3.10–3.12 (không khuyến nghị Python 3.14 cho bộ thư viện ML này).

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
uvicorn backend.vera_ai_final_fixed:app --host 0.0.0.0 --port 8000
```

Kiểm tra `http://127.0.0.1:8000/health`. Lần đầu, Transformers có thể tải CLIP `openai/clip-vit-base-patch32` từ Hugging Face; cần internet và đủ RAM. Sau đó mở `index.html` bằng trình duyệt. Nếu chạy frontend và backend trên cùng máy, API mặc định là `http://127.0.0.1:8000`.

## 3. Kết nối GitHub Pages với backend đã deploy

GitHub Pages chỉ chạy HTML/CSS/JavaScript, **không chạy Python hoặc mô hình ML**. Hãy deploy thư mục backend lên một dịch vụ Python hỗ trợ FastAPI, sau đó trong `index.html` sửa dòng:

```js
const VERA_API_URL = (window.VERA_API_URL || 'http://127.0.0.1:8000').replace(/\/$/, '');
```

thành URL HTTPS thật của backend, ví dụ `https://ten-dich-vu-cua-ban.example` (thay bằng URL do nhà cung cấp cấp). Đồng thời đặt biến môi trường backend `VERA_CORS_ORIGINS` thành domain GitHub Pages của repo, ví dụ `https://TEN-USER.github.io` hoặc domain Pages chính xác của bạn.

## 4. Đưa giao diện lên GitHub Pages

1. Tạo repository tên `VERA-AI` và upload `index.html`, `backend/`, `requirements.txt`, `README.md`.
2. Vào **Settings → Pages**.
3. Chọn **Deploy from a branch**, branch `main`, folder `/ (root)`, rồi Save.
4. Đợi GitHub Pages cấp URL. URL đó chỉ phục vụ giao diện; API phải được deploy riêng.

## API

- `GET /health`: kiểm tra bundle có tồn tại không.
- `POST /api/predict`: multipart form field `file`, trả `label` (`REAL`/`AI`), `ai_probability`, `confidence`, `threshold` và xác suất từng model.

## Lưu ý quan trọng

- Chưa thể dự đoán thật nếu chưa có `ml_clip_bundle_final.pkl`.
- File `training_pipeline_colab.py` là script xuất từ notebook Colab, có lệnh magic `!pip` và mount Google Drive; chạy trong Colab theo các cell, không dùng làm lệnh khởi động API.
- API hiện chỉ trả kết quả phân loại và xác suất. Grad-CAM không được giả lập; kiến trúc CLIP + FFT hiện tại cũng không phải ResNet50.
- Miễn phí 24/7 không được đảm bảo bởi GitHub Pages; backend miễn phí có thể sleep, giới hạn RAM/CPU hoặc thay đổi URL.
- Không commit khóa API, thông tin tài khoản, dataset lớn hoặc model bundle nhạy cảm vào repository công khai.
