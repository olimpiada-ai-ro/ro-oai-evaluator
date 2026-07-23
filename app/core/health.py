"""
Health check and metrics collection service.

Provides comprehensive health monitoring, system metrics,
and readiness checks for the AI Olympiad Evaluator.
"""

import psutil
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from pathlib import Path

from app.core.config import settings
from app.core.logging import get_logger
from app.evaluator.cache.manager import CacheManager
from app.evaluator.providers.builder import DataSourceBuilder

logger = get_logger("health")


class HealthService:
    """Service for health checks and metrics collection."""

    def __init__(self):
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
        self.evaluation_count = 0
        self.cache_manager: Optional[CacheManager] = None

    def set_cache_manager(self, cache_manager: CacheManager):
        """Set the cache manager for health checks."""
        self.cache_manager = cache_manager

    def increment_request_count(self):
        """Increment the total request counter."""
        self.request_count += 1

    def increment_error_count(self):
        """Increment the error counter."""
        self.error_count += 1

    def increment_evaluation_count(self):
        """Increment the evaluation counter."""
        self.evaluation_count += 1

    async def get_health_status(self) -> Dict[str, Any]:
        """
        Get comprehensive health status.

        Returns:
            Dictionary with health status and checks
        """
        uptime = time.time() - self.start_time

        # Perform health checks
        checks = {}
        overall_status = "healthy"

        # Cache health check
        cache_status = await self._check_cache_health()
        checks["cache"] = cache_status
        if cache_status["status"] != "healthy":
            overall_status = "degraded"

        # Disk space check
        disk_status = await self._check_disk_space()
        checks["disk_space"] = disk_status
        if disk_status["status"] != "healthy":
            overall_status = "degraded"

        # Memory check
        memory_status = await self._check_memory()
        checks["memory"] = memory_status
        if memory_status["status"] != "healthy":
            overall_status = "degraded"

        # Provider connectivity check (basic)
        provider_status = await self._check_provider_connectivity()
        checks["providers"] = provider_status
        if provider_status["status"] != "healthy":
            overall_status = "degraded"

        return {
            "status": overall_status,
            "message": f"Service is {overall_status}",
            "timestamp": datetime.now(timezone.utc),
            "version": settings.VERSION,
            "uptime_seconds": uptime,
            "checks": checks,
        }

    async def get_readiness_status(self) -> Dict[str, Any]:
        """
        Get service readiness status for deployment.

        Returns:
            Dictionary with readiness status
        """
        checks = {}
        ready = True
        messages = []

        # Check if cache directory is accessible
        try:
            cache_dir = Path(settings.CACHE_DIR)
            if not cache_dir.exists():
                cache_dir.mkdir(parents=True, exist_ok=True)

            # Test write access
            test_file = cache_dir / ".health_check"
            test_file.write_text("health_check")
            test_file.unlink()

            checks["cache_directory"] = {"status": "ready", "path": str(cache_dir)}
        except Exception as e:
            checks["cache_directory"] = {"status": "not_ready", "error": str(e)}
            ready = False
            messages.append("Cache directory not accessible")

        # Check if cache manager is initialized
        if self.cache_manager:
            checks["cache_manager"] = {
                "status": "ready",
                "entries": len(self.cache_manager._metadata_cache),
            }
        else:
            checks["cache_manager"] = {
                "status": "not_ready",
                "error": "Cache manager not initialized",
            }
            ready = False
            messages.append("Cache manager not initialized")

        return {
            "ready": ready,
            "checks": checks,
            "message": "; ".join(messages) if messages else None,
        }

    async def get_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive system and application metrics.

        Returns:
            Dictionary with various metrics
        """
        # Cache metrics
        cache_metrics = await self._get_cache_metrics()

        # System metrics
        system_metrics = await self._get_system_metrics()

        # Evaluation metrics
        evaluation_metrics = {
            "total_evaluations": self.evaluation_count,
            "total_requests": self.request_count,
            "total_errors": self.error_count,
            "error_rate_percent": (self.error_count / max(self.request_count, 1)) * 100,
            "uptime_seconds": time.time() - self.start_time,
        }

        # Request metrics
        request_metrics = {
            "total_requests": self.request_count,
            "error_count": self.error_count,
            "success_count": self.request_count - self.error_count,
            "success_rate_percent": (
                (self.request_count - self.error_count) / max(self.request_count, 1)
            )
            * 100,
        }

        return {
            "timestamp": datetime.now(timezone.utc),
            "cache": cache_metrics,
            "system": system_metrics,
            "evaluation": evaluation_metrics,
            "requests": request_metrics,
        }

    async def _check_cache_health(self) -> Dict[str, Any]:
        """Check cache system health."""
        try:
            if not self.cache_manager:
                return {"status": "unhealthy", "error": "Cache manager not initialized"}

            # Check cache directory accessibility
            cache_dir = Path(settings.CACHE_DIR)
            if not cache_dir.exists():
                return {
                    "status": "unhealthy",
                    "error": "Cache directory does not exist",
                }

            # Check cache size
            cache_size = await self.cache_manager.get_cache_size()
            max_size = self.cache_manager.max_size_bytes
            usage_percent = (cache_size / max_size) * 100 if max_size > 0 else 0

            # Get cache stats
            stats = self.cache_manager.get_cache_stats()

            status = "healthy"
            if usage_percent > 90:
                status = "degraded"
            elif usage_percent > 95:
                status = "unhealthy"

            return {
                "status": status,
                "cache_size_bytes": cache_size,
                "max_size_bytes": max_size,
                "usage_percent": round(usage_percent, 2),
                "entries": stats["total_entries"],
                "hit_rate_percent": stats["hit_rate_percent"],
            }

        except Exception as e:
            logger.error(f"Cache health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}

    async def _check_disk_space(self) -> Dict[str, Any]:
        """Check disk space availability."""
        try:
            cache_dir = Path(settings.CACHE_DIR)
            disk_usage = psutil.disk_usage(str(cache_dir.parent))

            free_percent = (disk_usage.free / disk_usage.total) * 100

            status = "healthy"
            if free_percent < 10:
                status = "unhealthy"
            elif free_percent < 20:
                status = "degraded"

            return {
                "status": status,
                "total_bytes": disk_usage.total,
                "free_bytes": disk_usage.free,
                "used_bytes": disk_usage.used,
                "free_percent": round(free_percent, 2),
            }

        except Exception as e:
            logger.error(f"Disk space check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}

    async def _check_memory(self) -> Dict[str, Any]:
        """Check memory usage."""
        try:
            memory = psutil.virtual_memory()
            process = psutil.Process()
            process_memory = process.memory_info()

            status = "healthy"
            if memory.percent > 90:
                status = "unhealthy"
            elif memory.percent > 80:
                status = "degraded"

            return {
                "status": status,
                "system_memory_percent": memory.percent,
                "system_available_bytes": memory.available,
                "process_memory_bytes": process_memory.rss,
                "process_memory_mb": round(process_memory.rss / 1024 / 1024, 2),
            }

        except Exception as e:
            logger.error(f"Memory check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}

    async def _check_provider_connectivity(self) -> Dict[str, Any]:
        """Check provider connectivity (basic check)."""
        try:
            # Report registered implementations without constructing clients or
            # introducing credential-shaped values into the service process.
            supported_providers = DataSourceBuilder.get_supported_providers()
            available_providers = [
                provider
                for provider in supported_providers
                if DataSourceBuilder.is_provider_supported(provider)
            ]

            return {
                "status": "healthy",
                "supported_providers": supported_providers,
                "available_providers": available_providers,
            }

        except Exception as e:
            logger.error(f"Provider connectivity check failed: {e}")
            return {"status": "degraded", "error": str(e)}

    async def _get_cache_metrics(self) -> Dict[str, Any]:
        """Get detailed cache metrics."""
        if not self.cache_manager:
            return {"status": "unavailable", "error": "Cache manager not initialized"}

        try:
            stats = self.cache_manager.get_cache_stats()
            cache_size = await self.cache_manager.get_cache_size()

            return {
                "total_entries": stats["total_entries"],
                "cache_hits": stats["hits"],
                "cache_misses": stats["misses"],
                "hit_rate_percent": stats["hit_rate_percent"],
                "evictions": stats["evictions"],
                "cache_size_bytes": cache_size,
                "cache_size_mb": round(cache_size / 1024 / 1024, 2),
                "max_size_bytes": self.cache_manager.max_size_bytes,
                "usage_percent": (
                    round((cache_size / self.cache_manager.max_size_bytes) * 100, 2)
                    if self.cache_manager.max_size_bytes > 0
                    else 0
                ),
            }

        except Exception as e:
            logger.error(f"Failed to get cache metrics: {e}")
            return {"status": "error", "error": str(e)}

    async def _get_system_metrics(self) -> Dict[str, Any]:
        """Get system resource metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()

            # Memory metrics
            memory = psutil.virtual_memory()

            # Process metrics
            process = psutil.Process()
            process_memory = process.memory_info()

            # Disk metrics for cache directory
            cache_dir = Path(settings.CACHE_DIR)
            disk_usage = psutil.disk_usage(str(cache_dir.parent))

            return {
                "cpu_percent": cpu_percent,
                "cpu_count": cpu_count,
                "memory_total_bytes": memory.total,
                "memory_available_bytes": memory.available,
                "memory_used_percent": memory.percent,
                "process_memory_bytes": process_memory.rss,
                "process_memory_mb": round(process_memory.rss / 1024 / 1024, 2),
                "disk_total_bytes": disk_usage.total,
                "disk_free_bytes": disk_usage.free,
                "disk_used_percent": round(
                    ((disk_usage.total - disk_usage.free) / disk_usage.total) * 100, 2
                ),
            }

        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            return {"status": "error", "error": str(e)}


# Global health service instance
health_service = HealthService()
