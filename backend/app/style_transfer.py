"""
Neural Style Transfer module using procedural filters and PIL transformations
Provides fast, real-time style transfer effects
"""
import cv2
import numpy as np
import logging
from typing import List, Dict, Optional
from pathlib import Path
from PIL import Image, ImageFilter, ImageDraw, ImageEnhance
import torch
import uuid
import os

logger = logging.getLogger(__name__)


class StyleTransfer:
    """Apply neural style transfer to images using procedural effects"""
    
    # Available styles with procedural implementations
    STYLE_MODELS = {
        "cartoon": "cartoon",
        "oil_painting": "oil_painting",
        "watercolor": "watercolor",
        "pencil_sketch": "pencil_sketch",
        "mosaic": "mosaic",
        "impressionist": "impressionist",
        "cubist": "cubist"
    }
    
    def __init__(self, model_type: str = "procedural", use_gpu: bool = False):
        """
        Initialize style transfer with procedural effects
        
        Args:
            model_type: "procedural" (GPU not needed for procedural effects)
            use_gpu: Use GPU if available (for future ML models)
        """
        self.model_type = model_type
        self.use_gpu = use_gpu
        
        logger.info(f"Style Transfer initialized with {model_type} procedural effects")
    
    def available_styles(self) -> List[str]:
        """Get list of available styles"""
        return list(self.STYLE_MODELS.keys())
    
    def _cartoon_effect(self, image: Image.Image) -> Image.Image:
        """Apply cartoon effect"""
        # Increase contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)
        
        # Reduce colors for cartoon look
        image = image.quantize(colors=16)
        image = image.convert('RGB')
        
        # Edge detection for outlines
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edges = cv2.dilate(edges, np.ones((2, 2)), iterations=1)
        
        # Combine edges with cartoon image
        img_array[edges > 0] = np.clip(img_array[edges > 0] * 0.5, 0, 255)
        
        return Image.fromarray(img_array)
    
    def _oil_painting_effect(self, image: Image.Image) -> Image.Image:
        """Apply oil painting effect"""
        img_array = np.array(image)
        
        # Apply bilateral filter for oil painting effect
        for _ in range(2):
            img_array = cv2.bilateralFilter(img_array, 9, 75, 75)
        
        return Image.fromarray(img_array)
    
    def _watercolor_effect(self, image: Image.Image) -> Image.Image:
        """Apply watercolor effect"""
        img_array = np.array(image)
        
        # Apply median blur
        img_array = cv2.medianBlur(img_array, 5)
        
        # Apply Laplacian edge detection
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = np.uint8(np.absolute(laplacian))
        
        # Reduce colors
        for i in range(3):
            img_array[:, :, i] = cv2.medianBlur(img_array[:, :, i], 7)
        
        return Image.fromarray(img_array)
    
    def _pencil_sketch_effect(self, image: Image.Image) -> Image.Image:
        """Apply pencil sketch effect"""
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Invert gray image
        inv_gray = cv2.bitwise_not(gray)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(inv_gray, (21, 21), 0)
        
        # Blend to create sketch effect
        sketch = cv2.divide(gray, np.maximum(255 - blurred, 1), scale=256)
        sketch = np.clip(sketch, 0, 255).astype(np.uint8)
        
        # Convert to RGB
        sketch_3channel = cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)
        
        return Image.fromarray(sketch_3channel)
    
    def _mosaic_effect(self, image: Image.Image) -> Image.Image:
        """Apply mosaic effect"""
        img_array = np.array(image)
        
        # Downscale
        small = cv2.resize(img_array, (img_array.shape[1] // 10, img_array.shape[0] // 10))
        
        # Upscale to create mosaic
        mosaic = cv2.resize(small, (img_array.shape[1], img_array.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        return Image.fromarray(mosaic)
    
    def _impressionist_effect(self, image: Image.Image) -> Image.Image:
        """Apply impressionist effect"""
        img_array = np.array(image)
        
        # Multiple passes of edge-preserving smoothing
        result = img_array.copy().astype(float)
        
        for _ in range(3):
            result = cv2.pyrMeanShiftFiltering(result.astype(np.uint8), 15, 40)
        
        # Add slight blur for impressionist look
        result = cv2.GaussianBlur(result, (5, 5), 0)
        
        return Image.fromarray(result.astype(np.uint8))
    
    def _cubist_effect(self, image: Image.Image) -> Image.Image:
        """Apply cubist effect"""
        img_array = np.array(image)
        
        # Create mosaic blocks with random colors
        h, w = img_array.shape[:2]
        block_size = 20
        
        result = img_array.copy()
        
        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                y2 = min(y + block_size, h)
                x2 = min(x + block_size, w)
                
                # Get average color of block
                region = img_array[y:y2, x:x2]
                avg_color = region.reshape(-1, 3).mean(axis=0).astype(int)
                
                # Fill with average color
                result[y:y2, x:x2] = avg_color
        
        return Image.fromarray(result)
    
    def apply(
        self,
        image_path: str,
        detections: List[Dict],
        object_indices: List[int],
        style_name: str,
        strength: float = 0.8
    ) -> str:
        """
        Apply style transfer to specific objects using segmentation masks
        
        Args:
            image_path: Path to input image
            detections: Object detection results with masks
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
        
        logger.info(f"Applying {style_name} style to {len(object_indices)} objects with strength {strength}")
        
        # Convert BGR to RGB for processing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = image_rgb.copy()
        
        # Process each selected object using its mask
        for idx in object_indices:
            if idx >= len(detections):
                logger.warning(f"Object index {idx} out of range")
                continue
            
            detection = detections[idx]
            
            # Check if mask is available
            if "mask" not in detection:
                logger.warning(f"No mask found for object {idx}, skipping")
                continue
            
            mask = detection["mask"]
            if mask is None:
                logger.warning(f"Mask is None for object {idx}, skipping")
                continue
            
            logger.info(f"Stylizing object {idx} with mask (shape: {mask.shape})")
            
            # Resize mask to match image dimensions if needed
            h_img, w_img = image_rgb.shape[:2]
            if mask.shape[:2] != (h_img, w_img):
                mask = cv2.resize(mask, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
                logger.info(f"Resized mask from {detection['mask'].shape} to {mask.shape}")
            
            # Normalize mask to 0-1
            mask_normalized = mask.astype(float) / 255.0
            
            # Extract bounding box for efficiency
            bbox = detection["bbox"]
            x1, y1, x2, y2 = (
                int(bbox["x1"]), int(bbox["y1"]),
                int(bbox["x2"]), int(bbox["y2"])
            )
            
            # Ensure valid coordinates
            x1, x2 = max(0, x1), min(image.shape[1], x2)
            y1, y2 = max(0, y1), min(image.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                logger.warning(f"Invalid bbox coordinates for object {idx}")
                continue
            
            # Extract region from image and mask
            region = result[y1:y2, x1:x2].copy()
            region_mask = mask_normalized[y1:y2, x1:x2]
            
            # Apply style transfer to region
            region_pil = Image.fromarray(region)
            stylized_region_pil = self._apply_style(region_pil, style_name)
            stylized_region = np.array(stylized_region_pil).astype(np.uint8)
            
            # Ensure same dtype
            region = region.astype(np.uint8)
            
            # Blend with original using mask
            blended_region = cv2.addWeighted(
                stylized_region, strength,
                region, 1 - strength,
                0
            ).astype(np.uint8)
            
            # Apply mask to blend result (only affect masked pixels)
            region_mask_3d = np.stack([region_mask] * 3, axis=2)
            masked_blend = (blended_region * region_mask_3d + region * (1 - region_mask_3d)).astype(np.uint8)
            
            # Put back in result
            result[y1:y2, x1:x2] = masked_blend
            logger.info(f"Object {idx} stylized and masked")
        
        # Convert back to BGR and save
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        
        # Create output path with unique name
        input_dir = os.path.dirname(image_path)
        unique_id = str(uuid.uuid4())[:8]
        output_filename = f"result_{unique_id}_{style_name}.png"
        output_path = os.path.join(input_dir, output_filename)
        
        cv2.imwrite(output_path, result_bgr)
        logger.info(f"Style transfer completed: {output_path}")
        
        return output_path
    
    def _apply_style(self, image: Image.Image, style_name: str) -> Image.Image:
        """Apply selected style to image"""
        if style_name not in self.STYLE_MODELS:
            logger.warning(f"Unknown style {style_name}, returning original")
            return image
        
        try:
            if style_name == "cartoon":
                return self._cartoon_effect(image)
            elif style_name == "oil_painting":
                return self._oil_painting_effect(image)
            elif style_name == "watercolor":
                return self._watercolor_effect(image)
            elif style_name == "pencil_sketch":
                return self._pencil_sketch_effect(image)
            elif style_name == "mosaic":
                return self._mosaic_effect(image)
            elif style_name == "impressionist":
                return self._impressionist_effect(image)
            elif style_name == "cubist":
                return self._cubist_effect(image)
            else:
                return image
        except Exception as e:
            logger.error(f"Error applying style {style_name}: {str(e)}")
            return image
