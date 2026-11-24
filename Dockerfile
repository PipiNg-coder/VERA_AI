FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=2

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render/Railway tự đặt biến PORT; chạy local thì mặc định 7860.
CMD ["sh", "-c", "uvicorn backend.vera_ai_onnx:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1"]
