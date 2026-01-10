"""
YOLO-based object detection module
"""
import cv2
import numpy as np
from ultralytics import YOLO
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)


class ObjectDetector:
    """Detect objects in images using YOLOv8"""
    
    def __init__(self, model_name: str = "yolov8m", use_gpu: bool = False):
        """
        Initialize YOLO detector
        
        Args:
            model_name: YOLO model size (n, s, m, l, x)
            use_gpu: Use GPU for inference
        """
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.device = 0 if use_gpu else "cpu"
        
        # Load model (auto-downloads if not present)
        self.model = YOLO(f"{model_name}.pt")
        self.model.to(self.device)
        
        logger.info(f"Loaded YOLO model: {model_name} on device: {self.device}")
    
    def detect(
        self,
        image_path: str,
        confidence: float = 0.45,
        max_objects: int = 50
    ) -> List[Dict]:
        """
        Detect objects in image
        
        Args:
            image_path: Path to image file
            confidence: Confidence threshold
            max_objects: Maximum objects to return
            
        Returns:
            List of detected objects with bounding boxes and labels
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        h, w = image.shape[:2]
        
        # Run inference
        results = self.model.predict(image, conf=confidence, device=self.device)
        
        # Parse detections
        detections = []
        for result in results:
            for box, conf, cls_id in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                if len(detections) >= max_objects:
                    break
                
                x1, y1, x2, y2 = box.cpu().numpy()
                confidence_score = float(conf.cpu().numpy())
                class_id = int(cls_id.cpu().numpy())
                class_name = result.names[class_id]
                
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
                    "area": float((x2 - x1) * (y2 - y1))
                }
                
                detections.append(detection)
        
        logger.info(f"Detected {len(detections)} objects in {image_path}")
        return detections
    
    def visualize_detections(
        self,
        image_path: str,
        detections: List[Dict],
        output_path: str
    ) -> str:
        """
        Draw bounding boxes on image for visualization
        
        Args:
            image_path: Input image path
            detections: Detection results
            output_path: Path to save visualization
            
        Returns:
            Path to annotated image
        """
        image = cv2.imread(image_path)
        
        for det in detections:
            bbox = det["bbox"]
            x1, y1, x2, y2 = int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"])
            
            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Put text label
            label = f"{det['class']} {det['confidence']:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imwrite(output_path, image)
        logger.info(f"Saved annotated image to {output_path}")
        return output_path
