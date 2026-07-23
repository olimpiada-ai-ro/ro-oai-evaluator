from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
import hashlib


class FileMetadata(BaseModel):
    etag: Optional[str] = Field(None, description="S3 ETag for change detection")
    last_modified: Optional[datetime] = Field(None, description="Last modified timestamp")
    size: int = Field(..., ge=0, description="File size in bytes")
    content_hash: str = Field(..., description="SHA256 hash of file content")


class CacheEntry(BaseModel):
    cache_key: str = Field(..., description="Unique cache key")
    file_path: str = Field(..., description="Local file path in cache")
    metadata: FileMetadata = Field(..., description="File metadata")
    created_at: datetime = Field(default_factory=lambda: datetime.now(datetime.timezone.utc), description="Cache entry creation time")
    last_accessed: datetime = Field(default_factory=lambda: datetime.now(datetime.timezone.utc), description="Last access time")
    access_count: int = Field(default=0, ge=0, description="Number of times accessed")


def generate_cache_key(dataset_path: str, access_key: str) -> str:
    """
    Generate a unique cache key from dataset path and access key.
    
    Args:
        dataset_path: S3 path to the dataset
        access_key: AWS access key (for multi-tenant caching)
    
    Returns:
        Unique cache key string
    """
    # Create a hash from the dataset path and access key for uniqueness
    key_data = f"{dataset_path}:{access_key}"
    return hashlib.sha256(key_data.encode()).hexdigest()[:16]


def calculate_content_hash(content: bytes) -> str:
    """
    Calculate SHA256 hash of file content.
    
    Args:
        content: File content as bytes
    
    Returns:
        SHA256 hash string
    """
    return hashlib.sha256(content).hexdigest()