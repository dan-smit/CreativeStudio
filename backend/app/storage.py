"""
Storage management - local filesystem or Google Cloud Storage
"""
import os
import logging
from pathlib import Path
from typing import Optional
from uuid import uuid4
import shutil

logger = logging.getLogger(__name__)


class StorageManager:
    """Manage file uploads and results"""
    
    def __init__(
        self,
        use_gcs: bool = False,
        gcs_bucket: Optional[str] = None,
        local_upload_dir: str = "./uploads",
        local_results_dir: str = "./results"
    ):
        """
        Initialize storage manager
        
        Args:
            use_gcs: Use Google Cloud Storage
            gcs_bucket: GCS bucket name
            local_upload_dir: Local upload directory
            local_results_dir: Local results directory
        """
        self.use_gcs = use_gcs
        self.gcs_bucket = gcs_bucket
        self.local_upload_dir = local_upload_dir
        self.local_results_dir = local_results_dir
        
        # Create local directories
        Path(local_upload_dir).mkdir(parents=True, exist_ok=True)
        Path(local_results_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize GCS if enabled
        self.gcs_client = None
        if use_gcs:
            try:
                from google.cloud import storage
                self.gcs_client = storage.Client()
                self.gcs_bucket = self.gcs_client.bucket(gcs_bucket)
                logger.info(f"GCS storage initialized: {gcs_bucket}")
            except Exception as e:
                logger.error(f"GCS initialization failed: {str(e)}")
                self.use_gcs = False
    
    def save_upload(self, file) -> str:
        """
        Save uploaded file
        
        Args:
            file: UploadFile object
            
        Returns:
            File ID
        """
        file_id = str(uuid4())
        
        # Save locally
        local_path = os.path.join(self.local_upload_dir, f"{file_id}_{file.filename}")
        
        try:
            with open(local_path, "wb") as f:
                f.write(file.file.read())
            
            logger.info(f"Saved upload: {file_id}")
            
            # Also save to GCS if enabled
            if self.use_gcs:
                self._upload_to_gcs(local_path, f"uploads/{file_id}/{file.filename}")
            
            return file_id
        except Exception as e:
            logger.error(f"Error saving upload: {str(e)}")
            raise
    
    def save_result(self, file_id: str, image_path: str, style_name: str) -> str:
        """
        Save result image
        
        Args:
            file_id: Original file ID
            image_path: Path to result image
            style_name: Style applied
            
        Returns:
            Result ID
        """
        result_id = f"{file_id}_{style_name}_{uuid4().hex[:8]}"
        
        try:
            result_path = os.path.join(self.local_results_dir, f"{result_id}.png")
            shutil.copy(image_path, result_path)
            
            logger.info(f"Saved result: {result_id}")
            
            # Also save to GCS if enabled
            if self.use_gcs:
                self._upload_to_gcs(result_path, f"results/{result_id}.png")
            
            return result_id
        except Exception as e:
            logger.error(f"Error saving result: {str(e)}")
            raise
    
    def get_upload_path(self, file_id: str) -> str:
        """Get path to uploaded file"""
        # Find file with matching ID
        uploads_dir = Path(self.local_upload_dir)
        for file_path in uploads_dir.glob(f"{file_id}_*"):
            return str(file_path)
        
        raise FileNotFoundError(f"Upload not found: {file_id}")
    
    def get_result_path(self, result_id: str) -> str:
        """Get path to result file"""
        result_path = os.path.join(self.local_results_dir, f"{result_id}.png")
        if os.path.exists(result_path):
            return result_path
        
        raise FileNotFoundError(f"Result not found: {result_id}")
    
    def _upload_to_gcs(self, local_path: str, gcs_path: str) -> None:
        """Upload file to GCS"""
        if not self.use_gcs or not self.gcs_client:
            return
        
        try:
            blob = self.gcs_bucket.blob(gcs_path)
            blob.upload_from_filename(local_path)
            logger.info(f"Uploaded to GCS: {gcs_path}")
        except Exception as e:
            logger.error(f"GCS upload failed: {str(e)}")
    
    def cleanup(self, file_id: str) -> None:
        """Clean up old files"""
        try:
            # Clean up uploads
            uploads_dir = Path(self.local_upload_dir)
            for file_path in uploads_dir.glob(f"{file_id}_*"):
                file_path.unlink()
            
            # Clean up results
            results_dir = Path(self.local_results_dir)
            for file_path in results_dir.glob(f"{file_id}_*"):
                file_path.unlink()
            
            logger.info(f"Cleaned up files for: {file_id}")
        except Exception as e:
            logger.error(f"Cleanup error: {str(e)}")
