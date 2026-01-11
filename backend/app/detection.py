"""
YOLO-based object detection and segmentation module
Uses YOLOv8-seg for instance segmentation masks
"""
import cv2
import numpy as np
from ultralytics import YOLO
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class ObjectDetector:
    """Detect and segment objects in images using YOLOv8-seg"""
    
    def __init__(self, model_name: str = "yolov8m-seg", use_gpu: bool = False):
        """
        Initialize YOLO segmentation detector
        
        Args:
            model_name: YOLO segmentation model (yolov8n-seg, yolov8s-seg, yolov8m-seg, etc.)
            use_gpu: Use GPU for inference
        """
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.device = 0 if use_gpu else "cpu"
        
        # Load segmentation model (auto-downloads if not present)
        self.model = YOLO(f"{model_name}.pt")
        self.model.to(self.device)
        
        logger.info(f"Loaded YOLO segmentation model: {model_name} on device: {self.device}")
    
    def detect(
        self,
        image_path: str,
        confidence: float = 0.5,
        max_objects: int = 50
    ) -> List[Dict]:
        """
        Detect and segment objects in image
        
        Args:
            image_path: Path to image file
            confidence: Confidence threshold
            max_objects: Maximum objects to return
            
        Returns:
            List of detected objects with bounding boxes, labels, and segmentation masks
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        h, w = image.shape[:2]
        
        # Run segmentation inference
        results = self.model.predict(image, conf=confidence, device=self.device)
        
        # Parse detections with masks
        detections = []
        for result in results:
            # Check if masks are available
            if result.masks is None:
                logger.warning("No segmentation masks found in results")
                continue
                
            for box, conf, cls_id, mask in zip(
                result.boxes.xyxy, 
                result.boxes.conf, 
                result.boxes.cls,
                result.masks.data
            ):
                if len(detections) >= max_objects:
                    break
                
                x1, y1, x2, y2 = box.cpu().numpy()
                confidence_score = float(conf.cpu().numpy())
                class_id = int(cls_id.cpu().numpy())
                class_name = result.names[class_id]
                
                # Convert mask to binary numpy array
                mask_array = mask.cpu().numpy().astype(np.uint8) * 255
                
                detection = {
                    "id": len(detections),
                    "class": class_name,
                    "confidence": round(confidence_score, 3),
                    "bbox": {
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                        "width": float(x2 - x1),
                        "height": float(y2 - y1)
                    },
                    "area": float((x2 - x1) * (y2 - y1)),
                    "mask": mask_array  # Binary mask (H, W) where 1 = object, 0 = background
                }
                
                detections.append(detection)
        
        logger.info(f"Detected and segmented {len(detections)} objects in {image_path}")
        return detections
    
    def visualize_detections(
        self,
        image_path: str,
        detections: List[Dict],
        output_path: str
    ) -> str:
        """
        Draw segmentation masks on image for visualization
        
        Args:
            image_path: Input image path
            detections: Detection results with masks
            output_path: Path to save visualization
            
        Returns:
            Path to annotated image
        """
        image = cv2.imread(image_path)
        overlay = image.copy()
        
        # Color map for different objects
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (255, 128, 0), (255, 0, 128), (128, 255, 0)
        ]
        
        for i, det in enumerate(detections):
            if "mask" not in det:
                continue
                
            mask = det["mask"]
            color = colors[i % len(colors)]
            
            # Apply colored mask overlay
            overlay[mask > 128] = color
            
            # Draw bounding box
            bbox = det["bbox"]
            x1, y1, x2, y2 = int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"])
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            
            # Put text label
            label = f"{det['class']} {det['confidence']:.2f}"
            cv2.putText(overlay, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Blend overlay with original
        result = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)
        
        cv2.imwrite(output_path, result)
        logger.info(f"Saved annotated image with masks to {output_path}")
        return output_path
