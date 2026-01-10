"""
Configuration settings for Creative Studio
"""
from pydantic_settings import BaseSettings
from typing import Optional
import os
from pathlib import Path

class Settings(BaseSettings):
    """Application settings"""
    
    # App settings
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    APP_NAME: str = "Creative Studio"
    
    # GPU settings
    USE_GPU: bool = os.getenv("USE_GPU", "False").lower() == "true"
    GPU_DEVICE: int = int(os.getenv("GPU_DEVICE", "0"))
    
    # YOLO settings
    YOLO_MODEL: str = os.getenv("YOLO_MODEL", "yolov8m")  # yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
    YOLO_CONFIDENCE: float = float(os.getenv("YOLO_CONFIDENCE", "0.45"))
    
    # Neural Style Transfer settings
    NST_MODEL_TYPE: str = os.getenv("NST_MODEL_TYPE", "onnx")  # onnx or pytorch
    AVAILABLE_STYLES: list = [
        "mosaic",
        "impressionist",
        "cubist",
        "oil_painting",
        "watercolor",
        "pencil_sketch",
        "cartoon"
    ]
    
    # Image settings
    MAX_IMAGE_SIZE: int = int(os.getenv("MAX_IMAGE_SIZE", "50"))  # MB
    SUPPORTED_FORMATS: list = ["jpg", "jpeg", "png", "webp"]
    
    # Storage settings
    USE_GCS: bool = os.getenv("USE_GCS", "False").lower() == "true"
    GCS_PROJECT_ID: Optional[str] = os.getenv("GCS_PROJECT_ID")
    GCS_BUCKET: str = os.getenv("GCS_BUCKET", "creative-studio-bucket")
    GCS_CREDENTIALS_PATH: Optional[str] = os.getenv("GCS_CREDENTIALS_PATH")
    
    # Local storage
    LOCAL_UPLOAD_DIR: str = os.getenv("LOCAL_UPLOAD_DIR", "./uploads")
    LOCAL_RESULTS_DIR: str = os.getenv("LOCAL_RESULTS_DIR", "./results")
    LOCAL_MODELS_DIR: str = os.getenv("LOCAL_MODELS_DIR", "./models")
    
    # API settings
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()


def init_gcs():
    """Initialize Google Cloud Storage credentials"""
    if not settings.USE_GCS:
        return
    
    if settings.GCS_CREDENTIALS_PATH:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.GCS_CREDENTIALS_PATH
    
    try:
        from google.cloud import storage
        client = storage.Client(project=settings.GCS_PROJECT_ID)
        bucket = client.bucket(settings.GCS_BUCKET)
        
        # Verify bucket exists
        if not bucket.exists():
            print(f"Creating GCS bucket: {settings.GCS_BUCKET}")
            bucket.create()
        
        return client
    except Exception as e:
        print(f"GCS initialization error: {str(e)}")
        raise


# Create local directories if they don't exist
Path(settings.LOCAL_UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
Path(settings.LOCAL_RESULTS_DIR).mkdir(parents=True, exist_ok=True)
Path(settings.LOCAL_MODELS_DIR).mkdir(parents=True, exist_ok=True)
