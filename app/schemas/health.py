from pydantic import BaseModel
from typing import Dict, Any, Optional
from datetime import datetime


class HealthResponse(BaseModel):
    status: str
    message: str
    timestamp: datetime
    version: str
    uptime_seconds: float
    checks: Dict[str, Any]


class MetricsResponse(BaseModel):
    timestamp: datetime
    cache: Dict[str, Any]
    system: Dict[str, Any]
    evaluation: Dict[str, Any]
    requests: Dict[str, Any]


class ReadinessResponse(BaseModel):
    ready: bool
    checks: Dict[str, Any]
    message: Optional[str] = None