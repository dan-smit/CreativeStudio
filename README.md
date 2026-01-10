# 🎨 Creative Studio

AI-powered photo editor with selective neural style transfer. Apply artistic styles to specific objects in your images using object detection, image segmentation, and neural style transfer.

## ✨ Features

- **🎯 Object Detection**: YOLOv8-based real-time object detection
- **🎭 Selective Style Transfer**: Apply neural styles to specific objects only
- **🖼️ Multiple Styles**: Mosaic, impressionist, cubist, oil painting, watercolor, sketch, cartoon
- **☁️ Cloud Storage**: Google Cloud Storage integration for uploads and results
- **🚀 GPU Support**: Optimized for GPU processing (Vast.ai, RunPod, Colab)
- **📱 Interactive UI**: Clean Streamlit interface for easy image editing
- **🔄 Real-time Processing**: Stream results as they're generated

## 🏗️ Architecture

```
Creative Studio/
├── backend/                 # FastAPI backend
│   ├── main.py             # FastAPI app
│   ├── config.py           # Configuration & GCS setup
│   ├── app/
│   │   ├── detection.py    # YOLOv8 object detection
│   │   ├── style_transfer.py # Neural style transfer
│   │   └── storage.py      # File storage (local/GCS)
│   └── Dockerfile
├── frontend/               # Streamlit UI
│   ├── src/
│   │   └── app.py         # Streamlit interface
│   └── Dockerfile
├── requirements.txt        # Python dependencies
└── docker-compose.yml     # Multi-container setup
```

## 🚀 Quick Start

### Local Development

1. **Clone and setup**:
```bash
cd CreativeStudio
python3.11 -m venv venv
source venv/Scripts/activate  # if using windows
pip install -r requirements.txt
```

2. **Create .env file**:
```bash
python3.11 -m venv venv

# Edit .env with your preferences
```

3. **Run backend** (terminal 1):
```bash
cd backend
pymon main.py
# Backend starts at http://localhost:8000
```

4. **Run frontend** (terminal 2):
```bash
streamlit run frontend/src/app.py
# Frontend opens at http://localhost:8501
```

### Docker Deployment

```bash
docker-compose up --build
# Backend: http://localhost:8000
# Frontend: http://localhost:8501
```

## 📋 API Endpoints

### Health Check
```bash
GET /health
```

### Image Upload
```bash
POST /upload
- multipart/form-data: image file
- Returns: { file_id, filename, message }
```

### Object Detection
```bash
POST /detect-objects
- Parameters: file_id (string)
- Returns: { file_id, objects: [...], count: int }
```

### Object Info
```json
{
  "id": 0,
  "class": "person",
  "confidence": 0.95,
  "bbox": { "x1": 100, "y1": 50, "x2": 200, "y2": 300, "width": 100, "height": 250 },
  "area": 25000
}
```

### Apply Style Transfer
```bash
POST /apply-style-transfer
- Parameters:
  - file_id: string
  - style_name: string (mosaic, impressionist, cubist, oil_painting, watercolor, pencil_sketch, cartoon)
  - object_indices: list of int (indices to stylize)
  - strength: float (0.0-1.0, default 0.8)
- Returns: { file_id, result_id, style, objects_stylized, message }
```

### Download Result
```bash
GET /result/{result_id}
- Returns: PNG image
```

### Available Styles
```bash
GET /styles
- Returns: { styles: [...], description: string }
```

### Settings
```bash
GET /settings
- Returns: GPU status, GCS status, model info
```

## ⚙️ Configuration

### Environment Variables

```bash
# Debug mode
DEBUG=True

# GPU settings
USE_GPU=True
GPU_DEVICE=0

# YOLO model size (smaller = faster, larger = more accurate)
YOLO_MODEL=yolov8m  # Options: n, s, m, l, x

# Detection confidence threshold
YOLO_CONFIDENCE=0.45

# Style transfer model type
NST_MODEL_TYPE=onnx

# Image limits
MAX_IMAGE_SIZE=50  # MB
SUPPORTED_FORMATS=jpg,jpeg,png,webp

# Google Cloud Storage (optional)
USE_GCS=True
GCS_PROJECT_ID=your-project-id
GCS_BUCKET=creative-studio-bucket
GCS_CREDENTIALS_PATH=/path/to/credentials.json

# Storage directories
LOCAL_UPLOAD_DIR=./uploads
LOCAL_RESULTS_DIR=./results
LOCAL_MODELS_DIR=./models
```

## 💻 GPU Provider Setup

### 🔵 Vast.ai (Recommended)

1. Create account: https://www.vast.ai/
2. Rent GPU instance (RTX 4090 or A100 recommended)
3. SSH into instance
4. Set `USE_GPU=True` in environment
5. Run backend as above

**Typical cost**: $0.20-$0.50/hour for A6000 GPU

### 🟠 RunPod

1. Create account: https://www.runpod.io/
2. Start GPU pod (PyTorch template)
3. Set `USE_GPU=True`
4. Deploy application

**Typical cost**: $0.29-$0.44/hour for RTX 4090

### 📓 Google Colab (Free)

```python
!pip install -r requirements.txt
!pip install pyngrok

from pyngrok import ngrok
ngrok.connect(8000)

%cd backend
!python main.py
```

## 🗄️ Google Cloud Storage Setup

### 1. Create GCS Bucket

```bash
gsutil mb gs://creative-studio-bucket
```

### 2. Create Service Account

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create service account with "Storage Admin" role
3. Download JSON key
4. Set environment variables:

```bash
export USE_GCS=True
export GCS_PROJECT_ID=your-project-id
export GCS_CREDENTIALS_PATH=/path/to/service-account-key.json
```

### 3. Test Connection

```bash
python -c "from backend.config import init_gcs; init_gcs(); print('GCS connected!')"
```

## 📊 Performance Tuning

### Model Selection
- **YOLOv8n**: Fastest (5ms), lower accuracy
- **YOLOv8m**: Balanced (25ms), good accuracy
- **YOLOv8l**: Slower (40ms), high accuracy
- **YOLOv8x**: Slowest (80ms), best accuracy

### Image Processing
- Optimal size: 1024x1024 px
- Max recommended: 2048x2048 px
- Use JPEG for better compression

### GPU Optimization
- Enable mixed precision: `torch.cuda.amp`
- Use `torch.jit` for model compilation
- Batch process images for throughput

## 🐛 Troubleshooting

### CUDA Out of Memory
```bash
# Use smaller YOLO model
export YOLO_MODEL=yolov8n

# Clear cache
rm -rf ~/.cache/torch
```

### Slow Processing
- Reduce image resolution
- Use faster YOLO model (yolov8n)
- Increase style strength (less refinement)

### GCS Connection Issues
```bash
# Verify credentials
cat $GCS_CREDENTIALS_PATH

# Test access
gsutil ls gs://creative-studio-bucket

# Check permissions
gsutil iam ch serviceAccount:your-service-account@project.iam.gserviceaccount.com:roles/storage.admin gs://creative-studio-bucket
```

### Port Already in Use
```bash
# Find and kill process
lsof -i :8000  # Backend
lsof -i :8501  # Frontend
kill -9 <PID>
```

## 📦 Deployment

### Docker Compose
```bash
docker-compose up --build
```

### With GPU on RunPod
```bash
docker-compose -f docker-compose.gpu.yml up
```

**Made with ❤️ using FastAPI, YOLOv8, and PyTorch**
