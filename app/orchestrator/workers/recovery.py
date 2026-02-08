"""
Worker Recovery System

Monitors worker health and recovers from hung/stuck workers by:
- Detecting workers that are processing for too long
- Killing and restarting hung workers
- Requeuing failed tasks with retry limits
"""

import asyncio
from typing import List, Dict, Optional, Callable, Any
from datetime import datetime
from dataclasses import dataclass

from app.config import get_logger
from app.orchestrator.workers.base import BaseWorker

logger = get_logger(__name__)


@dataclass
class WorkerTimeout:
    """Timeout configuration for different worker types."""
    
    # Worker type -> timeout in seconds
    crawler: float = 300.0  # 5 minutes for crawling
    processor: float = 180.0  # 3 minutes for processing
    pdf: float = 240.0  # 4 minutes for PDF processing (was 120s, increased)
    ocr: float = 300.0  # 5 minutes for OCR
    storage: float = 120.0  # 2 minutes for storage
    
    def get_timeout(self, worker_type: str) -> float:
        """Get timeout for a worker type."""
        return getattr(self, worker_type, 300.0)


@dataclass
class RecoveryStats:
    """Statistics for worker recovery operations."""
    
    total_recoveries: int = 0
    workers_killed: int = 0
    workers_restarted: int = 0
    tasks_requeued: int = 0
    tasks_abandoned: int = 0
    
    def __str__(self) -> str:
        return (
            f"Recoveries: {self.total_recoveries} | "
            f"Killed: {self.workers_killed} | "
            f"Restarted: {self.workers_restarted} | "
            f"Requeued: {self.tasks_requeued} | "
            f"Abandoned: {self.tasks_abandoned}"
        )


class WorkerHealthMonitor:
    """
    Monitors worker health and performs recovery operations.
    
    Capabilities:
    - Detect hung workers (processing for too long)
    - Kill hung workers
    - Restart workers with new instances
    - Requeue tasks with failure tracking
    """
    
    def __init__(
        self,
        timeouts: Optional[WorkerTimeout] = None,
        max_task_retries: int = 2,
        check_interval: float = 10.0,
    ):
        """
        Initialize health monitor.
        
        Args:
            timeouts: Worker timeout configuration
            max_task_retries: Maximum retries before abandoning task
            check_interval: How often to check worker health (seconds)
        """
        self.timeouts = timeouts or WorkerTimeout()
        self.max_task_retries = max_task_retries
        self.check_interval = check_interval
        
        # Statistics
        self.stats = RecoveryStats()
        
        # Task failure tracking: task_id -> failure_count
        self.task_failures: Dict[str, int] = {}
        
        # Monitor state
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
    
    async def start_monitoring(
        self,
        get_workers_callback: Callable[[], List[BaseWorker]],
        restart_worker_callback: Callable[[BaseWorker], asyncio.Task],
        requeue_task_callback: Optional[Callable[[Any], None]] = None,
    ):
        """
        Start monitoring worker health.
        
        Args:
            get_workers_callback: Function to get current list of workers
            restart_worker_callback: Function to restart a worker (returns new worker task)
            requeue_task_callback: Optional function to requeue failed tasks
        """
        self._running = True
        
        async def monitor_loop():
            logger.info("Worker health monitor started")
            
            while self._running:
                try:
                    workers = get_workers_callback()
                    await self._check_and_recover(
                        workers,
                        restart_worker_callback,
                        requeue_task_callback,
                    )
                    await asyncio.sleep(self.check_interval)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in health monitor: {e}", exc_info=True)
                    await asyncio.sleep(self.check_interval)
            
            logger.info("Worker health monitor stopped")
        
        self._monitor_task = asyncio.create_task(monitor_loop())
    
    async def stop_monitoring(self):
        """Stop health monitoring."""
        self._running = False
        
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
    
    async def _check_and_recover(
        self,
        workers: List[BaseWorker],
        restart_callback: Callable[[BaseWorker], asyncio.Task],
        requeue_callback: Optional[Callable[[Any], None]],
    ):
        """Check all workers and recover hung ones."""
        for worker in workers:
            if not worker.is_running:
                continue
            
            # Get timeout for this worker type
            timeout = self.timeouts.get_timeout(worker.worker_type)
            
            # Check if worker is hung
            if worker.is_hung(timeout):
                await self._recover_worker(
                    worker,
                    timeout,
                    restart_callback,
                    requeue_callback,
                )
    
    async def _recover_worker(
        self,
        worker: BaseWorker,
        timeout: float,
        restart_callback: Callable[[BaseWorker], asyncio.Task],
        requeue_callback: Optional[Callable[[Any], None]],
    ):
        """Recover a hung worker."""
        self.stats.total_recoveries += 1
        
        duration = worker.current_task_duration or 0
        task = worker.current_task
        
        logger.warning(
            f"🔴 Worker {worker.worker_id} hung! "
            f"Processing for {duration:.1f}s (timeout: {timeout}s)"
        )
        
        # Log task info if available
        if task:
            task_info = self._get_task_info(task)
            logger.warning(f"   Stuck on task: {task_info}")
        
        # Kill the worker
        try:
            logger.info(f"   Killing worker {worker.worker_id}...")
            await self._kill_worker(worker)
            self.stats.workers_killed += 1
            logger.info(f"   ✓ Worker {worker.worker_id} killed")
        except Exception as e:
            logger.error(f"   Failed to kill worker {worker.worker_id}: {e}")
        
        # Restart the worker
        try:
            logger.info(f"   Restarting worker {worker.worker_id}...")
            restart_callback(worker)
            self.stats.workers_restarted += 1
            logger.info(f"   ✓ Worker {worker.worker_id} restarted")
        except Exception as e:
            logger.error(f"   Failed to restart worker {worker.worker_id}: {e}")
        
        # Handle the task
        if task and requeue_callback:
            await self._handle_failed_task(task, requeue_callback)
    
    async def _kill_worker(self, worker: BaseWorker):
        """Force kill a worker."""
        try:
            # Try graceful shutdown first
            worker._running = False
            
            # Cancel the worker task if it exists
            if worker._task and not worker._task.done():
                worker._task.cancel()
                try:
                    await asyncio.wait_for(worker._task, timeout=2.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
            
            # Force cleanup
            await worker.shutdown()
            
        except Exception as e:
            logger.error(f"Error killing worker: {e}")
    
    async def _handle_failed_task(
        self,
        task: Any,
        requeue_callback: Callable[[Any], None],
    ):
        """Handle a failed task - requeue or abandon."""
        # Get task identifier
        task_id = self._get_task_id(task)
        
        # Track failures
        self.task_failures[task_id] = self.task_failures.get(task_id, 0) + 1
        failure_count = self.task_failures[task_id]
        
        if failure_count <= self.max_task_retries:
            # Requeue for retry
            logger.info(
                f"   Requeuing task (attempt {failure_count}/{self.max_task_retries}): "
                f"{self._get_task_info(task)}"
            )
            try:
                await requeue_callback(task)
                self.stats.tasks_requeued += 1
            except Exception as e:
                logger.error(f"   Failed to requeue task: {e}")
        else:
            # Abandon task after max retries
            logger.error(
                f"   ❌ Abandoning task after {failure_count} failures: "
                f"{self._get_task_info(task)}"
            )
            self.stats.tasks_abandoned += 1
    
    def _get_task_id(self, task: Any) -> str:
        """Extract task ID from task object."""
        # Try common task ID attributes
        for attr in ['task_id', 'id', 'url', 'file_path']:
            if hasattr(task, attr):
                value = getattr(task, attr)
                if value:
                    return str(value)
        
        # Fallback to string representation
        return str(task)
    
    def _get_task_info(self, task: Any) -> str:
        """Get human-readable task info."""
        # Try to extract useful info
        if hasattr(task, 'url'):
            return f"URL: {task.url}"
        elif hasattr(task, 'file_path'):
            return f"File: {task.file_path}"
        elif hasattr(task, 'task_id'):
            return f"ID: {task.task_id}"
        else:
            return str(task)[:100]
    
    def get_stats(self) -> RecoveryStats:
        """Get recovery statistics."""
        return self.stats
    
    def reset_stats(self):
        """Reset recovery statistics."""
        self.stats = RecoveryStats()
        self.task_failures.clear()
