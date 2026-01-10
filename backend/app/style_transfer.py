"""
Neural Style Transfer module using ONNX models
Free styles available via torchvision or ONNX Hub
"""
import cv2
import numpy as np
import logging
from typing import List, Dict, Optional
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

logger = logging.getLogger(__name__)


class StyleTransfer:
    """Apply neural style transfer to images"""
    
    # Available style models (free, ONNX or PyTorch)
    STYLE_MODELS = {
        "mosaic": "mosaic_fast",
        "impressionist": "impressionist_fast",
        "cubist": "cubist_fast",
        "oil_painting": "oil_painting_fast",
        "watercolor": "watercolor_fast",
        "pencil_sketch": "pencil_sketch_fast",
        "cartoon": "cartoon_fast"
    }
    
    def __init__(self, model_type: str = "onnx", use_gpu: bool = False):
        """
        Initialize style transfer models
        
        Args:
            model_type: "onnx" or "pytorch"
            use_gpu: Use GPU for inference
        """
        self.model_type = model_type
        self.use_gpu = use_gpu
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.models = {}
        
        logger.info(f"Style Transfer initialized with {model_type} models on {self.device}")
    
    def available_styles(self) -> List[str]:
        """Get list of available styles"""
        return list(self.STYLE_MODELS.keys())
    
    def _get_or_load_model(self, style_name: str) -> nn.Module:
        """Load style model (lazy loading)"""
        if style_name in self.models:
            return self.models[style_name]
        
        if style_name not in self.STYLE_MODELS:
            raise ValueError(f"Unknown style: {style_name}")
        
        try:
            # Simple Fast Style Transfer model
            # Using a lightweight implementation
            model = self._create_lightweight_style_model(style_name)
            self.models[style_name] = model
            return model
        except Exception as e:
            logger.error(f"Error loading style model {style_name}: {str(e)}")
            raise
    
    def _create_lightweight_style_model(self, style_name: str) -> nn.Module:
        """
        Create a lightweight style transfer model
        In production, you'd use pre-trained models from torchvision or ONNX Hub
        """
        model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Tanh()
        ).to(self.device)
        
        return model
    
    def apply(
        self,
        image_path: str,
        detections: List[Dict],
        object_indices: List[int],
        style_name: str,
        strength: float = 0.8
    ) -> str:
        """
        Apply style transfer to specific objects
        
        Args:
            image_path: Path to input image
            detections: Object detection results
            object_indices: Indices of objects to stylize
            style_name: Style to apply
            strength: Blending strength (0.0-1.0)
            
        Returns:
            Path to stylized image
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB for processing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get or load model
        model = self._get_or_load_model(style_name)
        
        # Process each selected object
        result = image_rgb.copy()
        
        for idx in object_indices:
            if idx >= len(detections):
                logger.warning(f"Object index {idx} out of range")
                continue
            
            detection = detections[idx]
            bbox = detection["bbox"]
            
            # Extract region
            x1, y1, x2, y2 = (
                int(bbox["x1"]), int(bbox["y1"]),
                int(bbox["x2"]), int(bbox["y2"])
            )
            
            # Ensure valid coordinates
            x1, x2 = max(0, x1), min(image.shape[1], x2)
            y1, y2 = max(0, y1), min(image.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            # Extract object region
            region = image_rgb[y1:y2, x1:x2]
            
            # Apply style transfer to region
            stylized_region = self._stylize_region(region, model)
            
            # Blend with original
            blended_region = cv2.addWeighted(
                stylized_region, strength,
                region, 1 - strength,
                0
            )
            
            # Put back in result
            result[y1:y2, x1:x2] = blended_region
        
        # Convert back to BGR and save
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        output_path = image_path.replace(".png", f"_{style_name}.png").replace(".jpg", f"_{style_name}.jpg")
        cv2.imwrite(output_path, result_bgr)
        
        logger.info(f"Style transfer completed: {output_path}")
        return output_path
    
    def _stylize_region(self, region: np.ndarray, model: nn.Module) -> np.ndarray:
        """
        Apply style transfer to a region
        
        Args:
            region: Image region (H, W, 3)
            model: Style transfer model
            
        Returns:
            Stylized region
        """
        # Convert to tensor and normalize
        tensor = torch.from_numpy(region).float().permute(2, 0, 1) / 255.0
        tensor = tensor.unsqueeze(0).to(self.device)
        
        # Apply style transfer
        with torch.no_grad():
            stylized = model(tensor)
        
        # Convert back to numpy
        stylized = stylized.squeeze(0).permute(1, 2, 0).cpu().numpy()
        stylized = np.clip((stylized + 1) * 127.5, 0, 255).astype(np.uint8)
        
        return stylized
