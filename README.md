---


# VERA AI — Image Authenticity Detection

Space Gradio dùng model đã huấn luyện **CLIP + FFT/SRM + Ensemble** để phân loại ảnh REAL/AI.

## Chạy trên Hugging Face Spaces

- SDK: Gradio.
- Với tài khoản Free, chọn ZeroGPU nếu Space của bạn đủ điều kiện; CPU Basic cho Gradio Spaces hiện yêu cầu gói đăng ký phù hợp.
- File khởi chạy: `app.py`.
- Model bundle: `backend/models/ml_clip_bundle_final.pkl`.
- Lần khởi động đầu, Transformers tải CLIP vision model từ Hugging Face; cần chờ quá trình tải hoàn tất.

## Cấu trúc

```text
VERA_AI/
├── app.py
├── index.html
├── backend/
│   ├── vera_ai_final_fixed.py
│   ├── training_pipeline_colab.py
│   └── models/
│       └── ml_clip_bundle_final.pkl
├── requirements.txt
└── README.md
```

## Model và xử lý

Ứng dụng không huấn luyện lại dữ liệu khi khởi động. `app.py` nạp bundle có sẵn và gọi hàm `predict_image()` trong backend. Hàm dự đoán sử dụng CLIP embedding kết hợp đặc trưng FFT/SRM và các model ensemble đã lưu.

## Giao diện website

`index.html` là giao diện web riêng. GitHub Pages chỉ phục vụ giao diện HTML; Space Gradio hiện là ứng dụng phân tích ảnh độc lập. Chưa kết nối giao diện `index.html` với API công khai trong bước này.

## API FastAPI cục bộ

Backend FastAPI gốc vẫn được giữ tại `backend/vera_ai_final_fixed.py` và có thể chạy cục bộ bằng:

```bash
pip install -r requirements.txt
uvicorn backend.vera_ai_final_fixed:app --host 0.0.0.0 --port 8000
```

## Lưu ý

- Kết quả phụ thuộc vào model bundle và CLIP checkpoint tải được.
- ZeroGPU chỉ cấp GPU trong thời gian hàm được đánh dấu `@spaces.GPU` thực thi; lượt chạy có thể phải chờ hàng đợi và chịu giới hạn quota.
- Không đưa dữ liệu cá nhân hoặc bí mật vào Space công khai.
