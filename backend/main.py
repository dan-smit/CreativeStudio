"""
Creative Studio Backend
AI-powered photo editor with selective neural style transfer
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from pathlib import Path
from typing import Optional
import logging

# Import custom modules
from config import settings, init_gcs
from app.detection import ObjectDetector
from app.style_transfer import StyleTransfer
from app.storage import StorageManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Creative Studio API",
    description="AI-powered photo editor with selective neural style transfer",
    version="1.0.0"
)

# Add CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
detector: Optional[ObjectDetector] = None
style_transfer: Optional[StyleTransfer] = None
storage_manager: Optional[StorageManager] = None
detection_cache: dict = {}  # Cache for detections with masks (file_id -> detections)


@app.on_event("startup")
async def startup_event():
    """Initialize models and services on startup"""
    global detector, style_transfer, storage_manager
    
    logger.info("Initializing Creative Studio API...")
    
    try:
        # Initialize GCS if enabled
        if settings.USE_GCS:
            init_gcs()
        
        # Initialize storage manager
        storage_manager = StorageManager(
            use_gcs=settings.USE_GCS,
            gcs_bucket=settings.GCS_BUCKET,
            local_upload_dir=settings.LOCAL_UPLOAD_DIR
        )
        logger.info("Storage manager initialized")
        
        # Initialize object detector (YOLOv8)
        detector = ObjectDetector(
            model_name=settings.YOLO_MODEL,
            use_gpu=settings.USE_GPU
        )
        logger.info(f"Object detector loaded: {settings.YOLO_MODEL}")
        
        # Initialize style transfer
        style_transfer = StyleTransfer(
            model_type=settings.NST_MODEL_TYPE,
            use_gpu=settings.USE_GPU
        )
        logger.info(f"Style transfer model loaded: {settings.NST_MODEL_TYPE}")
        
        logger.info("Creative Studio API ready!")
        
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        raise


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "detector_ready": detector is not None,
        "style_transfer_ready": style_transfer is not None,
        "storage_ready": storage_manager is not None
    }


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """Upload an image for processing"""
    try:
        # Log file details for debugging
        logger.info(f"Upload received: filename={file.filename}, content_type={file.content_type}")
        
        # Accept any file that Streamlit sent (it filters on the frontend)
        # Let actual image processing fail if it's not a valid image
        if not file.filename:
            raise HTTPException(status_code=400, detail="File must have a filename")
        
        # Save file
        file_id = storage_manager.save_upload(file)
        
        return {
            "file_id": file_id,
            "filename": file.filename,
            "message": "Image uploaded successfully"
        }
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect-objects")
async def detect_objects(file_id: str):
    """Detect objects in uploaded image"""
    try:
        # Load image
        image_path = storage_manager.get_upload_path(file_id)
        
        # Run detection (returns detections with masks)
        detections = detector.detect(image_path)
        
        # Cache detections (with masks) for later use in style transfer
        detection_cache[file_id] = detections
        logger.info(f"Cached detections for file_id: {file_id}")
        
        # Return detections WITHOUT masks (for JSON serialization)
        detections_for_response = [
            {
                "id": d["id"],
                "class": d["class"],
                "confidence": d["confidence"],
                "bbox": d["bbox"],
                "area": d["area"]
            }
            for d in detections
        ]
        
        return {
            "file_id": file_id,
            "objects": detections_for_response,
            "count": len(detections_for_response)
        }
    except Exception as e:
        logger.error(f"Detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/visualize-objects")
async def visualize_objects(
    file_id: str = Query(...),
    object_indices: list[int] = Query(None)
):
    """
    Visualize selected objects with bounding boxes/masks on the image
    
    Args:
        file_id: ID of uploaded image
        object_indices: List of object indices to visualize (None = all)
        
    Returns:
        Image file with visualization
    """
    try:
        if file_id not in detection_cache:
            raise HTTPException(status_code=400, detail="File not detected. Run /detect-objects first.")
        
        image_path = storage_manager.get_upload_path(file_id)
        detections = detection_cache[file_id]
        
        # If no indices specified, visualize all
        if object_indices is None or len(object_indices) == 0:
            object_indices = [d["id"] for d in detections]
        
        # Create visualization
        import cv2
        import numpy as np
        
        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        overlay = image.copy()
        
        # Color map for different objects
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (255, 128, 0), (255, 0, 128), (128, 255, 0),
            (128, 0, 255), (0, 128, 255), (255, 128, 128)
        ]
        
        # Draw boxes and masks for selected objects
        for idx in object_indices:
            if idx >= len(detections):
                continue
            
            detection = detections[idx]
            color = colors[idx % len(colors)]
            
            # Draw mask if available
            if "mask" in detection and detection["mask"] is not None:
                mask = detection["mask"]
                # Resize mask to match image dimensions if needed
                if mask.shape[:2] != (h, w):
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                overlay[mask > 128] = color
            
            # Draw bounding box
            bbox = detection["bbox"]
            x1, y1, x2, y2 = int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"])
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 3)
            
            # Put text label
            label = f"{detection['class']} ({idx})"
            cv2.putText(overlay, label, (x1, max(y1 - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Blend overlay with original
        result = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
        
        # Save temporary visualization
        import uuid
        viz_filename = f"viz_{uuid.uuid4().hex[:8]}.png"
        viz_path = os.path.join(settings.LOCAL_UPLOAD_DIR, viz_filename)
        cv2.imwrite(viz_path, result)
        
        return FileResponse(
            viz_path,
            media_type="image/png",
            filename=viz_filename
        )
    except Exception as e:
        logger.error(f"Visualization error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/apply-style-transfer")
async def apply_style_transfer(
    file_id: str = Query(...),
    style_name: str = Query(...),
    object_indices: list[int] = Query(...),
    strength: float = Query(0.8)
):
    """
    Apply neural style transfer to specific objects in image
    
    Args:
        file_id: ID of uploaded image
        style_name: Name of style to apply
        object_indices: List of object indices to stylize
        strength: Style strength (0.0-1.0)
    """
    try:
        if not (0.0 <= strength <= 1.0):
            raise HTTPException(status_code=400, detail="Strength must be 0.0-1.0")
        
        # Get cached detections with masks
        if file_id not in detection_cache:
            raise HTTPException(status_code=400, detail="File not detected. Run /detect-objects first.")
        
        detections = detection_cache[file_id]
        image_path = storage_manager.get_upload_path(file_id)
        
        # Apply style transfer
        result_path = style_transfer.apply(
            image_path=image_path,
            detections=detections,
            object_indices=object_indices,
            style_name=style_name,
            strength=strength
        )
        
        # Save result
        result_id = storage_manager.save_result(file_id, result_path, style_name)
        
        return {
            "file_id": file_id,
            "result_id": result_id,
            "style": style_name,
            "objects_stylized": len(object_indices),
            "message": "Style transfer applied successfully"
        }
    except Exception as e:
        logger.error(f"Style transfer error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/styles")
async def list_styles():
    """List available styles for NST"""
    return {
        "styles": style_transfer.available_styles(),
        "description": "Available neural style transfer styles"
    }


@app.get("/result/{result_id}")
async def download_result(result_id: str):
    """Download processed image"""
    try:
        result_path = storage_manager.get_result_path(result_id)
        
        if not os.path.exists(result_path):
            raise HTTPException(status_code=404, detail="Result not found")
        
        return FileResponse(
            result_path,
            media_type="image/png",
            filename=f"result_{result_id}.png"
        )
    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/settings")
async def get_settings():
    """Get API settings and capabilities"""
    return {
        "gpu_enabled": settings.USE_GPU,
        "gcs_enabled": settings.USE_GCS,
        "yolo_model": settings.YOLO_MODEL,
        "nst_model_type": settings.NST_MODEL_TYPE,
        "max_image_size": settings.MAX_IMAGE_SIZE,
        "supported_formats": settings.SUPPORTED_FORMATS
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    )