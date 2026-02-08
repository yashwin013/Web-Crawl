"""
Queue Manager for Multi-Site Orchestrator

Manages asyncio queues with backpressure control, monitoring, and graceful shutdown.
"""

import asyncio
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime

from app.config import get_logger
from app.orchestrator.config import OrchestratorConfig, QueueConfig
from app.orchestrator.models.task import (
    CrawlTask, ProcessTask, PdfTask, OcrTask, StorageTask, TaskStatus
)

logger = get_logger(__name__)


@dataclass
class QueueMetrics:
    """Metrics for a single queue."""
    name: str
    current_size: int = 0
    max_size: int = 0
    total_added: int = 0
    total_removed: int = 0
    total_failed: int = 0
    avg_wait_time_ms: float = 0.0
    
    @property
    def utilization_percent(self) -> float:
        """Queue utilization percentage."""
        if self.max_size == 0:
            return 0.0
        return (self.current_size / self.max_size) * 100
    
    @property
    def is_near_full(self) -> bool:
        """Check if queue is near capacity (>80%)."""
        return self.utilization_percent > 80.0
    
    @property
    def is_empty(self) -> bool:
        """Check if queue is empty."""
        return self.current_size == 0


class QueueManager:
    """
    Manages all orchestrator queues with monitoring and backpressure.
    
    Features:
    - Automatic backpressure via maxsize limits
    - Real-time metrics tracking
    - Graceful shutdown coordination
    - Dead letter queue for failed tasks
    
    Usage:
        config = get_default_config()
        manager = QueueManager(config.queues)
        
        await manager.startup()
        await manager.put_crawl_task(task)
        task = await manager.get_crawl_task()
        await manager.shutdown()
    """
    
    def __init__(self, config: QueueConfig):
        """Initialize queue manager with configuration."""
        self.config = config
        
        # Main queues (with backpressure via maxsize)
        self.crawl_queue: Optional[asyncio.Queue] = None
        self.processing_queue: Optional[asyncio.Queue] = None
        self.pdf_queue: Optional[asyncio.Queue] = None
        self.ocr_queue: Optional[asyncio.Queue] = None
        self.storage_queue: Optional[asyncio.Queue] = None
        
        # Dead letter queue for failed tasks
        self.dead_letter_queue: Optional[asyncio.Queue] = None
        
        # Metrics tracking
        self.metrics: Dict[str, QueueMetrics] = {}
        self._start_time: Optional[datetime] = None
        self._shutdown_event: Optional[asyncio.Event] = None
        
        # Task tracking (for monitoring)
        self._active_tasks: Dict[str, Any] = {}
    
    async def startup(self):
        """Initialize all queues."""
        logger.info("Starting QueueManager...")
        
        # Create queues with maxsize for backpressure
        self.crawl_queue = asyncio.Queue(maxsize=self.config.crawl_queue_size)
        self.processing_queue = asyncio.Queue(maxsize=self.config.processing_queue_size)
        self.pdf_queue = asyncio.Queue(maxsize=self.config.pdf_queue_size)
        self.ocr_queue = asyncio.Queue(maxsize=self.config.ocr_queue_size)
        self.storage_queue = asyncio.Queue(maxsize=self.config.storage_queue_size)
        
        # Unlimited dead letter queue
        self.dead_letter_queue = asyncio.Queue()
        
        # Initialize metrics
        self.metrics = {
            "crawl": QueueMetrics("crawl", max_size=self.config.crawl_queue_size),
            "processing": QueueMetrics("processing", max_size=self.config.processing_queue_size),
            "pdf": QueueMetrics("pdf", max_size=self.config.pdf_queue_size),
            "ocr": QueueMetrics("ocr", max_size=self.config.ocr_queue_size),
            "storage": QueueMetrics("storage", max_size=self.config.storage_queue_size),
            "dead_letter": QueueMetrics("dead_letter", max_size=0),
        }
        
        self._start_time = datetime.utcnow()
        self._shutdown_event = asyncio.Event()
        
        logger.info(f"✓ Queues initialized (capacity: {self.config.total_queue_capacity})")
    
    async def shutdown(self, timeout: float = 30.0):
        """
        Gracefully shutdown queues.
        
        Args:
            timeout: Maximum time to wait for queue drain
        """
        logger.info("Shutting down QueueManager...")
        
        if self._shutdown_event:
            self._shutdown_event.set()
        
        # Wait for queues to drain or timeout
        try:
            await asyncio.wait_for(self._wait_for_empty_queues(), timeout=timeout)
            logger.info("✓ All queues drained")
        except asyncio.TimeoutError:
            logger.warning(f"Queue drain timeout after {timeout}s - forcing shutdown")
        
        # Log final metrics
        self.log_metrics()
        
        logger.info("✓ QueueManager shutdown complete")
    
    async def _wait_for_empty_queues(self):
        """Wait for all main queues to become empty."""
        while True:
            if (self.crawl_queue.empty() and
                self.processing_queue.empty() and
                self.ocr_queue.empty() and
                self.storage_queue.empty()):
                break
            await asyncio.sleep(0.5)
    
    # ==================== Crawl Queue ====================
    
    async def put_crawl_task(self, task: CrawlTask, timeout: Optional[float] = None):
        """
        Add crawl task to queue.
        
        Blocks if queue is full (backpressure).
        """
        await self._put_with_metrics(
            self.crawl_queue,
            "crawl",
            task,
            timeout
        )
    
    async def get_crawl_task(self, timeout: Optional[float] = None) -> Optional[CrawlTask]:
        """Get next crawl task from queue."""
        return await self._get_with_metrics(
            self.crawl_queue,
            "crawl",
            timeout
        )
    
    # ==================== Processing Queue ====================
    
    async def put_process_task(self, task: ProcessTask, timeout: Optional[float] = None):
        """Add process task to queue."""
        await self._put_with_metrics(
            self.processing_queue,
            "processing",
            task,
            timeout
        )
    
    async def get_process_task(self, timeout: Optional[float] = None) -> Optional[ProcessTask]:
        """Get next process task from queue."""
        return await self._get_with_metrics(
            self.processing_queue,
            "processing",
            timeout
        )
    
    # ==================== PDF Queue ====================
    
    async def put_pdf_task(self, task: PdfTask, timeout: Optional[float] = None):
        """Add PDF task to queue."""
        await self._put_with_metrics(
            self.pdf_queue,
            "pdf",
            task,
            timeout
        )
    
    async def get_pdf_task(self, timeout: Optional[float] = None) -> Optional[PdfTask]:
        """Get next PDF task from queue."""
        return await self._get_with_metrics(
            self.pdf_queue,
            "pdf",
            timeout
        )
    
    # ==================== OCR Queue ====================
    
    async def put_ocr_task(self, task: OcrTask, timeout: Optional[float] = None):
        """Add OCR task to queue."""
        await self._put_with_metrics(
            self.ocr_queue,
            "ocr",
            task,
            timeout
        )
    
    async def get_ocr_task(self, timeout: Optional[float] = None) -> Optional[OcrTask]:
        """Get next OCR task from queue."""
        return await self._get_with_metrics(
            self.ocr_queue,
            "ocr",
            timeout
        )
    
    # ==================== Storage Queue ====================
    
    async def put_storage_task(self, task: StorageTask, timeout: Optional[float] = None):
        """Add storage task to queue."""
        await self._put_with_metrics(
            self.storage_queue,
            "storage",
            task,
            timeout
        )
    
    async def get_storage_task(self, timeout: Optional[float] = None) -> Optional[StorageTask]:
        """Get next storage task from queue."""
        return await self._get_with_metrics(
            self.storage_queue,
            "storage",
            timeout
        )
    
    # ==================== Dead Letter Queue ====================
    
    async def put_dead_letter(self, task: Any, error: str):
        """Add failed task to dead letter queue."""
        task.mark_failed(error)
        await self.dead_letter_queue.put(task)
        
        metrics = self.metrics.get("dead_letter")
        if metrics:
            metrics.total_added += 1
            metrics.total_failed += 1
            metrics.current_size = self.dead_letter_queue.qsize()
        
        logger.error(f"Task {task.task_id} moved to dead letter queue: {error}")
    
    async def get_dead_letter_tasks(self) -> List[Any]:
        """Get all tasks from dead letter queue."""
        tasks = []
        while not self.dead_letter_queue.empty():
            try:
                task = self.dead_letter_queue.get_nowait()
                tasks.append(task)
            except asyncio.QueueEmpty:
                break
        return tasks
    
    # ==================== Internal Helpers ====================
    
    async def _put_with_metrics(
        self,
        queue: asyncio.Queue,
        queue_name: str,
        task: Any,
        timeout: Optional[float]
    ):
        """Put item in queue with metrics tracking."""
        start_time = datetime.utcnow()
        
        if timeout:
            await asyncio.wait_for(queue.put(task), timeout=timeout)
        else:
            await queue.put(task)
        
        # Update metrics
        metrics = self.metrics.get(queue_name)
        if metrics:
            metrics.total_added += 1
            metrics.current_size = queue.qsize()
            
            wait_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            metrics.avg_wait_time_ms = (
                (metrics.avg_wait_time_ms * (metrics.total_added - 1) + wait_time) /
                metrics.total_added
            )
    
    async def _get_with_metrics(
        self,
        queue: asyncio.Queue,
        queue_name: str,
        timeout: Optional[float]
    ) -> Optional[Any]:
        """Get item from queue with metrics tracking."""
        try:
            if timeout:
                task = await asyncio.wait_for(queue.get(), timeout=timeout)
            else:
                task = await queue.get()
            
            # Update metrics
            metrics = self.metrics.get(queue_name)
            if metrics:
                metrics.total_removed += 1
                metrics.current_size = queue.qsize()
            
            return task
            
        except asyncio.TimeoutError:
            return None
    
    # ==================== Monitoring ====================
    
    def get_metrics(self) -> Dict[str, QueueMetrics]:
        """Get current metrics for all queues."""
        # Update current sizes
        if self.crawl_queue:
            self.metrics["crawl"].current_size = self.crawl_queue.qsize()
        if self.processing_queue:
            self.metrics["processing"].current_size = self.processing_queue.qsize()
        if self.ocr_queue:
            self.metrics["ocr"].current_size = self.ocr_queue.qsize()
        if self.storage_queue:
            self.metrics["storage"].current_size = self.storage_queue.qsize()
        if self.dead_letter_queue:
            self.metrics["dead_letter"].current_size = self.dead_letter_queue.qsize()
        
        return self.metrics
    
    def log_metrics(self):
        """Log current queue metrics."""
        metrics = self.get_metrics()
        
        logger.info("=" * 60)
        logger.info("Queue Metrics:")
        for name, m in metrics.items():
            if m.max_size > 0:
                logger.info(
                    f"  {name:12s}: {m.current_size:3d}/{m.max_size:3d} "
                    f"({m.utilization_percent:5.1f}%) | "
                    f"Added: {m.total_added:4d} | Removed: {m.total_removed:4d}"
                )
            else:
                logger.info(
                    f"  {name:12s}: {m.current_size:3d} | "
                    f"Added: {m.total_added:4d} | Failed: {m.total_failed:4d}"
                )
        logger.info("=" * 60)
    
    def get_bottleneck(self) -> Optional[str]:
        """Identify which queue is the bottleneck (most full)."""
        metrics = self.get_metrics()
        
        # Exclude dead letter queue
        active_queues = {
            name: m for name, m in metrics.items()
            if name != "dead_letter" and m.max_size > 0
        }
        
        if not active_queues:
            return None
        
        # Find queue with highest utilization
        bottleneck = max(active_queues.items(), key=lambda x: x[1].utilization_percent)
        
        if bottleneck[1].is_near_full:
            return bottleneck[0]
        
        return None
    
    @property
    def is_shutdown_requested(self) -> bool:
        """Check if shutdown has been requested."""
        return self._shutdown_event and self._shutdown_event.is_set()
