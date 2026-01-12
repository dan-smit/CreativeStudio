FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # libsm6 libxext6 libxrender-dev \
    libgl1 libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only Torch
RUN pip install --no-cache-dir \
    torch==2.9.1 torchvision==0.24.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install pre-trained model weights
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Copy application
COPY . .

# Create directories
RUN mkdir -p uploads results models

EXPOSE 8000

CMD ["python", "main.py"]
