FROM python:3.11-slim

# System dependencies for OpenCV, audio codecs, and geospatial libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download YOLOv8 weights during build so Render never downloads weights at runtime!
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" || true

# Copy application source
COPY . .

# Ensure weights directory exists and copy model if downloaded in root
RUN mkdir -p weights && (cp yolov8n.pt weights/yolov8n.pt 2>/dev/null || true)

# Expose Render default web port (10000)
EXPOSE 10000

# Run uvicorn server (binds dynamically to Render's $PORT or defaults to 10000)
CMD ["sh", "-c", "uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT:-10000}"]
