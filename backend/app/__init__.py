"""
Creative Studio Backend - AI-powered photo editor with selective neural style transfer
"""

__version__ = "1.0.0"
__author__ = "Creative Studio"

from .detection import ObjectDetector
from .style_transfer import StyleTransfer
from .storage import StorageManager

__all__ = [
    "ObjectDetector",
    "StyleTransfer",
    "StorageManager"
]
