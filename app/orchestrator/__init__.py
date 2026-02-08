"""
Multi-Website Crawler Orchestrator

Manages parallel crawling, processing, OCR, and storage across multiple websites
with CPU/GPU resource optimization and queue-based backpressure control.
"""

from app.orchestrator.config import OrchestratorConfig, get_default_config
from app.orchestrator.queues import QueueManager
from app.orchestrator.coordinator import MultiSiteOrchestrator

__all__ = [
    "OrchestratorConfig",
    "get_default_config",
    "QueueManager",
    "MultiSiteOrchestrator",
]
