FROM python:3.10-slim

# Dépendances système compatibles Debian trixie
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libopenblas-dev \
    v4l-utils \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Dépendances Python
RUN pip install --no-cache-dir \
    "numpy<2" \
    tflite-runtime \
    opencv-python-headless \
    requests

WORKDIR /app

COPY model.tflite .
COPY app.py .

CMD ["python", "app.py"]