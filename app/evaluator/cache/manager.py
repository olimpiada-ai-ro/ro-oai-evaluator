import os
import json
import aiofiles
import asyncio
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime, timedelta, timezone

from ..schemas.cache import CacheEntry, FileMetadata, generate_cache_key, calculate_content_hash
from app.core.logging import get_logger, log_performance

logger = get_logger("cache")


class CacheManager:
    """
    Manages local file caching with metadata persistence and change detection.
    
    Provides async file storage/retrieval, cache validation, and LRU eviction.
    """
    
    def __init__(self, cache_dir: str = "./cache", max_size_gb: int = 10, ttl_hours: int = 24):
        """
        Initialize CacheManager with configuration.
        
        Args:
            cache_dir: Directory for cache storage
            max_size_gb: Maximum cache size in GB
            ttl_hours: Time-to-live for cache entries in hours
        """
        self.cache_dir = Path(cache_dir)
        self.metadata_dir = self.cache_dir / "metadata"
        self.files_dir = self.cache_dir / "files"
        self.max_size_bytes = max_size_gb * 1024 * 1024 * 1024
        self.ttl_hours = ttl_hours
        
        # Ensure directories exist
        self._ensure_directories()
        
        # In-memory cache for metadata (loaded on startup)
        self._metadata_cache: Dict[str, CacheEntry] = {}
        self._cache_stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0
        }
    
    def _ensure_directories(self) -> None:
        """Create cache directories if they don't exist."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.files_dir.mkdir(parents=True, exist_ok=True)
    
    async def initialize(self) -> None:
        """
        Initialize cache manager by loading existing metadata.
        Should be called once during application startup.
        """
        with log_performance(logger, "cache_initialization"):
            await self._load_metadata_cache()
        
        logger.info(
            "CacheManager initialized",
            extra={
                'operation': 'cache_initialization',
                'cached_entries': len(self._metadata_cache),
                'cache_dir': str(self.cache_dir),
                'max_size_gb': self.max_size_bytes / (1024**3),
                'ttl_hours': self.ttl_hours
            }
        )
    
    async def _load_metadata_cache(self) -> None:
        """Load all metadata files into memory cache."""
        try:
            for metadata_file in self.metadata_dir.glob("*.json"):
                try:
                    async with aiofiles.open(metadata_file, 'r') as f:
                        content = await f.read()
                        entry_data = json.loads(content)
                        entry = CacheEntry(**entry_data)
                        self._metadata_cache[entry.cache_key] = entry
                except Exception as e:
                    logger.warning(f"Failed to load metadata file {metadata_file}: {e}")
                    # Remove corrupted metadata file
                    try:
                        metadata_file.unlink()
                    except Exception:
                        pass
        except Exception as e:
            logger.error(f"Error loading metadata cache: {e}")
    
    def generate_cache_key(self, dataset_path: str, access_key: str) -> str:
        """
        Generate cache key for dataset path and access key.
        
        Args:
            dataset_path: S3 path to dataset
            access_key: AWS access key for multi-tenant support
            
        Returns:
            Unique cache key
        """
        return generate_cache_key(dataset_path, access_key)
    
    async def _save_metadata(self, entry: CacheEntry) -> None:
        """
        Save cache entry metadata to disk.
        
        Args:
            entry: Cache entry to save
        """
        metadata_file = self.metadata_dir / f"{entry.cache_key}.json"
        try:
            # Convert datetime objects to ISO format for JSON serialization
            entry_dict = entry.model_dump()
            entry_dict["created_at"] = entry.created_at.isoformat()
            entry_dict["last_accessed"] = entry.last_accessed.isoformat()
            if entry.metadata.last_modified:
                entry_dict["metadata"]["last_modified"] = entry.metadata.last_modified.isoformat()
            
            async with aiofiles.open(metadata_file, 'w') as f:
                await f.write(json.dumps(entry_dict, indent=2))
        except Exception as e:
            logger.error(f"Failed to save metadata for {entry.cache_key}: {e}")
            raise
    
    async def _get_cache_file_path(self, cache_key: str) -> Path:
        """
        Get the file path for a cache key.
        
        Args:
            cache_key: Cache key
            
        Returns:
            Path to cached file
        """
        return self.files_dir / f"{cache_key}.dat"
    
    async def get_cache_size(self) -> int:
        """
        Calculate total cache size in bytes.
        
        Returns:
            Total cache size in bytes
        """
        total_size = 0
        try:
            for file_path in self.files_dir.glob("*.dat"):
                try:
                    total_size += file_path.stat().st_size
                except Exception:
                    # File might have been deleted, skip
                    pass
        except Exception as e:
            logger.error(f"Error calculating cache size: {e}")
        
        return total_size
    
    def get_cache_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with hit/miss/eviction counts
        """
        total_requests = self._cache_stats["hits"] + self._cache_stats["misses"]
        hit_rate = (self._cache_stats["hits"] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            **self._cache_stats,
            "total_entries": len(self._metadata_cache),
            "hit_rate_percent": round(hit_rate, 2)
        }

    async def cache_file(self, cache_key: str, data: bytes, metadata: FileMetadata) -> None:
        """
        Store file data and metadata in cache.
        
        Args:
            cache_key: Unique cache key
            data: File content as bytes
            metadata: File metadata
        """
        try:
            with log_performance(logger, "cache_store", cache_key=cache_key, size_bytes=len(data)):
                # Calculate content hash
                content_hash = calculate_content_hash(data)
                metadata.content_hash = content_hash
                
                # Create cache entry
                file_path = await self._get_cache_file_path(cache_key)
                entry = CacheEntry(
                    cache_key=cache_key,
                    file_path=str(file_path),
                    metadata=metadata,
                    created_at=datetime.now(timezone.utc),
                    last_accessed=datetime.now(timezone.utc),
                    access_count=1
                )
                
                # Check if we need to make space
                await self._ensure_cache_space(len(data))
                
                # Write file data
                async with aiofiles.open(file_path, 'wb') as f:
                    await f.write(data)
                
                # Save metadata
                await self._save_metadata(entry)
                
                # Update in-memory cache
                self._metadata_cache[cache_key] = entry
            
            logger.info(
                "File cached successfully",
                extra={
                    'operation': 'cache_store',
                    'cache_key': cache_key,
                    'size_bytes': len(data),
                    'etag': metadata.etag,
                    'content_hash': content_hash
                }
            )
            
        except Exception as e:
            logger.error(
                "Failed to cache file",
                extra={
                    'operation': 'cache_store',
                    'cache_key': cache_key,
                    'size_bytes': len(data),
                    'error_type': type(e).__name__,
                    'error_message': str(e)
                },
                exc_info=True
            )
            # Clean up partial files
            try:
                file_path = await self._get_cache_file_path(cache_key)
                if file_path.exists():
                    file_path.unlink()
            except Exception:
                pass
            raise
    
    async def get_cached_file(self, cache_key: str, remote_metadata: FileMetadata) -> Optional[bytes]:
        """
        Retrieve cached file if valid, otherwise return None.
        
        Args:
            cache_key: Cache key to lookup
            remote_metadata: Remote file metadata for validation
            
        Returns:
            Cached file content if valid, None otherwise
        """
        try:
            # Check if entry exists in metadata cache
            if cache_key not in self._metadata_cache:
                self._cache_stats["misses"] += 1
                logger.debug(
                    "Cache miss - entry not found",
                    extra={
                        'operation': 'cache_lookup',
                        'cache_key': cache_key,
                        'result': 'miss',
                        'reason': 'entry_not_found'
                    }
                )
                return None
            
            entry = self._metadata_cache[cache_key]
            
            # Validate cache entry
            if not await self.is_cache_valid(cache_key, remote_metadata):
                self._cache_stats["misses"] += 1
                logger.debug(
                    "Cache miss - entry invalid",
                    extra={
                        'operation': 'cache_lookup',
                        'cache_key': cache_key,
                        'result': 'miss',
                        'reason': 'entry_invalid'
                    }
                )
                await self._remove_cache_entry(cache_key)
                return None
            
            # Read cached file
            file_path = Path(entry.file_path)
            if not file_path.exists():
                logger.warning(
                    "Cache file missing on disk",
                    extra={
                        'operation': 'cache_lookup',
                        'cache_key': cache_key,
                        'file_path': str(file_path),
                        'result': 'miss',
                        'reason': 'file_missing'
                    }
                )
                self._cache_stats["misses"] += 1
                await self._remove_cache_entry(cache_key)
                return None
            
            with log_performance(logger, "cache_read", cache_key=cache_key):
                async with aiofiles.open(file_path, 'rb') as f:
                    data = await f.read()
            
            # Update access statistics
            entry.last_accessed = datetime.now(timezone.utc)
            entry.access_count += 1
            await self._save_metadata(entry)
            
            self._cache_stats["hits"] += 1
            logger.debug(
                "Cache hit successful",
                extra={
                    'operation': 'cache_lookup',
                    'cache_key': cache_key,
                    'result': 'hit',
                    'size_bytes': len(data),
                    'access_count': entry.access_count
                }
            )
            
            return data
            
        except Exception as e:
            logger.error(
                "Error retrieving cached file",
                extra={
                    'operation': 'cache_lookup',
                    'cache_key': cache_key,
                    'result': 'error',
                    'error_type': type(e).__name__,
                    'error_message': str(e)
                },
                exc_info=True
            )
            self._cache_stats["misses"] += 1
            return None
    
    async def is_cache_valid(self, cache_key: str, remote_metadata: FileMetadata) -> bool:
        """
        Check if cached file is still valid compared to remote metadata.
        
        Args:
            cache_key: Cache key to validate
            remote_metadata: Remote file metadata
            
        Returns:
            True if cache is valid, False otherwise
        """
        try:
            if cache_key not in self._metadata_cache:
                return False
            
            entry = self._metadata_cache[cache_key]
            cached_metadata = entry.metadata
            
            # Check TTL
            if self.ttl_hours > 0:
                ttl_threshold = datetime.now(timezone.utc) - timedelta(hours=self.ttl_hours)
                if entry.created_at < ttl_threshold:
                    logger.debug(f"Cache entry {cache_key} expired (TTL)")
                    return False
            
            # Check ETag if available
            if remote_metadata.etag and cached_metadata.etag:
                if remote_metadata.etag != cached_metadata.etag:
                    logger.debug(f"Cache entry {cache_key} invalid (ETag mismatch)")
                    return False
            
            # Check last modified timestamp if available
            if remote_metadata.last_modified and cached_metadata.last_modified:
                if remote_metadata.last_modified > cached_metadata.last_modified:
                    logger.debug(f"Cache entry {cache_key} invalid (timestamp)")
                    return False
            
            # Check file size
            if remote_metadata.size != cached_metadata.size:
                logger.debug(f"Cache entry {cache_key} invalid (size mismatch)")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating cache entry {cache_key}: {e}")
            return False
    
    async def _remove_cache_entry(self, cache_key: str) -> None:
        """
        Remove cache entry and associated files.
        
        Args:
            cache_key: Cache key to remove
        """
        try:
            # Remove from memory cache
            if cache_key in self._metadata_cache:
                entry = self._metadata_cache[cache_key]
                del self._metadata_cache[cache_key]
                
                # Remove cached file
                file_path = Path(entry.file_path)
                if file_path.exists():
                    file_path.unlink()
                
                # Remove metadata file
                metadata_file = self.metadata_dir / f"{cache_key}.json"
                if metadata_file.exists():
                    metadata_file.unlink()
                
                logger.debug(f"Removed cache entry {cache_key}")
                
        except Exception as e:
            logger.error(f"Error removing cache entry {cache_key}: {e}")
    
    async def _ensure_cache_space(self, required_bytes: int) -> None:
        """
        Ensure there's enough space in cache by evicting LRU entries if needed.
        
        Args:
            required_bytes: Bytes needed for new cache entry
        """
        current_size = await self.get_cache_size()
        
        if current_size + required_bytes <= self.max_size_bytes:
            return
        
        # Calculate how much space we need to free
        space_to_free = (current_size + required_bytes) - self.max_size_bytes
        
        # Get entries sorted by last accessed (LRU first)
        entries_by_lru = sorted(
            self._metadata_cache.values(),
            key=lambda x: x.last_accessed
        )
        
        freed_space = 0
        for entry in entries_by_lru:
            if freed_space >= space_to_free:
                break
            
            try:
                # Get file size before removing
                file_path = Path(entry.file_path)
                if file_path.exists():
                    file_size = file_path.stat().st_size
                    freed_space += file_size
                
                await self._remove_cache_entry(entry.cache_key)
                self._cache_stats["evictions"] += 1
                
                logger.info(f"Evicted cache entry {entry.cache_key} (LRU)")
                
            except Exception as e:
                logger.error(f"Error evicting cache entry {entry.cache_key}: {e}")
    
    async def cleanup_expired_entries(self) -> int:
        """
        Remove all expired cache entries based on TTL.
        
        Returns:
            Number of entries removed
        """
        if self.ttl_hours <= 0:
            return 0
        
        ttl_threshold = datetime.now(timezone.utc) - timedelta(hours=self.ttl_hours)
        expired_keys = []
        
        for cache_key, entry in self._metadata_cache.items():
            if entry.created_at < ttl_threshold:
                expired_keys.append(cache_key)
        
        for cache_key in expired_keys:
            await self._remove_cache_entry(cache_key)
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
        
        return len(expired_keys)
    
    async def clear_cache(self) -> None:
        """Clear all cache entries and files."""
        try:
            # Remove all cached files
            for file_path in self.files_dir.glob("*.dat"):
                try:
                    file_path.unlink()
                except Exception:
                    pass
            
            # Remove all metadata files
            for metadata_file in self.metadata_dir.glob("*.json"):
                try:
                    metadata_file.unlink()
                except Exception:
                    pass
            
            # Clear memory cache
            self._metadata_cache.clear()
            
            # Reset stats
            self._cache_stats = {
                "hits": 0,
                "misses": 0,
                "evictions": 0
            }
            
            logger.info("Cache cleared successfully")
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            raise