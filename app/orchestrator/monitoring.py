"""
Enhanced Monitoring System

Provides real-time metrics, health checks, and progress tracking for the orchestrator.
"""

import asyncio
import psutil
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

try:
    import GPUtil  # type: ignore
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

from app.config import get_logger

logger = get_logger(__name__)


class HealthStatus(str, Enum):
    """Health status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class SystemMetrics:
    """System resource metrics."""
    
    # CPU
    cpu_percent: float = 0.0
    cpu_count: int = 0
    
    # Memory
    memory_used_mb: float = 0.0
    memory_total_mb: float = 0.0
    memory_percent: float = 0.0
    
    # GPU (if available)
    gpu_memory_used_mb: float = 0.0
    gpu_memory_total_mb: float = 0.0
    gpu_utilization: float = 0.0
    
    # Disk
    disk_used_gb: float = 0.0
    disk_total_gb: float = 0.0
    disk_percent: float = 0.0
    
    # Timestamp
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def memory_available_mb(self) -> float:
        """Available memory in MB."""
        return self.memory_total_mb - self.memory_used_mb
    
    @property
    def is_cpu_overloaded(self) -> bool:
        """Check if CPU is overloaded (>90%)."""
        return self.cpu_percent > 90.0
    
    @property
    def is_memory_critical(self) -> bool:
        """Check if memory is critical (>95%)."""
        return self.memory_percent > 95.0


@dataclass
class QueueHealth:
    """Queue health metrics."""
    
    name: str
    current_size: int
    max_size: int
    utilization_percent: float
    total_added: int
    total_removed: int
    avg_wait_time_ms: float
    
    # Health indicators
    is_full: bool = False
    is_empty: bool = False
    is_stalled: bool = False  # No movement in N seconds
    
    @property
    def throughput(self) -> float:
        """Items processed per second (approximate)."""
        if self.total_removed == 0:
            return 0.0
        # This is a simple approximation
        return float(self.total_removed)
    
    @property
    def health_status(self) -> HealthStatus:
        """Overall health status."""
        if self.is_stalled:
            return HealthStatus.UNHEALTHY
        elif self.utilization_percent > 90.0:
            return HealthStatus.DEGRADED
        elif self.is_full:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY


@dataclass
class WorkerHealth:
    """Worker health metrics."""
    
    worker_id: str
    worker_type: str
    is_running: bool
    tasks_processed: int
    tasks_failed: int
    uptime_seconds: float
    
    # Performance metrics
    avg_task_time_seconds: float = 0.0
    last_task_time: Optional[datetime] = None
    
    @property
    def success_rate(self) -> float:
        """Task success rate percentage."""
        total = self.tasks_processed + self.tasks_failed
        if total == 0:
            return 100.0
        return (self.tasks_processed / total) * 100.0
    
    @property
    def tasks_per_minute(self) -> float:
        """Tasks processed per minute."""
        if self.uptime_seconds == 0:
            return 0.0
        return (self.tasks_processed / self.uptime_seconds) * 60.0
    
    @property
    def health_status(self) -> HealthStatus:
        """Overall health status."""
        if not self.is_running:
            return HealthStatus.UNHEALTHY
        elif self.success_rate < 50.0:
            return HealthStatus.UNHEALTHY
        elif self.success_rate < 80.0:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY


@dataclass
class OrchestratorHealth:
    """Overall orchestrator health."""
    
    status: HealthStatus
    system_metrics: SystemMetrics
    queue_health: List[QueueHealth]
    worker_health: List[WorkerHealth]
    
    # Alerts
    alerts: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def is_healthy(self) -> bool:
        """Check if orchestrator is healthy."""
        return self.status == HealthStatus.HEALTHY
    
    @property
    def total_workers(self) -> int:
        """Total number of workers."""
        return len(self.worker_health)
    
    @property
    def healthy_workers(self) -> int:
        """Number of healthy workers."""
        return sum(
            1 for w in self.worker_health
            if w.health_status == HealthStatus.HEALTHY
        )
    
    @property
    def unhealthy_queues(self) -> List[str]:
        """List of unhealthy queue names."""
        return [
            q.name for q in self.queue_health
            if q.health_status == HealthStatus.UNHEALTHY
        ]


class PerformanceMonitor:
    """
    Monitors system performance and resource usage.
    
    Tracks CPU, memory, GPU, and disk usage.
    """
    
    def __init__(self):
        """Initialize performance monitor."""
        self.process = psutil.Process()
        self._last_check = datetime.utcnow()
    
    def get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics."""
        metrics = SystemMetrics()
        
        try:
            # CPU metrics
            metrics.cpu_percent = psutil.cpu_percent(interval=0.1)
            metrics.cpu_count = psutil.cpu_count()
            
            # Memory metrics
            mem = psutil.virtual_memory()
            metrics.memory_total_mb = mem.total / (1024 * 1024)
            metrics.memory_used_mb = mem.used / (1024 * 1024)
            metrics.memory_percent = mem.percent
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            metrics.disk_total_gb = disk.total / (1024 * 1024 * 1024)
            metrics.disk_used_gb = disk.used / (1024 * 1024 * 1024)
            metrics.disk_percent = disk.percent
            
            # GPU metrics (if available)
            if GPU_AVAILABLE:
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]
                        metrics.gpu_memory_used_mb = gpu.memoryUsed
                        metrics.gpu_memory_total_mb = gpu.memoryTotal
                        metrics.gpu_utilization = gpu.load * 100
                except Exception:
                    # GPU monitoring failed
                    pass
            
        except Exception as e:
            logger.warning(f"Failed to collect system metrics: {e}")
        
        metrics.timestamp = datetime.utcnow()
        return metrics
    
    def check_resource_limits(
        self,
        metrics: SystemMetrics,
        max_cpu: float = 80.0,
        max_memory: float = 90.0,
    ) -> List[str]:
        """Check if resource limits are exceeded."""
        alerts = []
        
        if metrics.cpu_percent > max_cpu:
            alerts.append(
                f"CPU usage high: {metrics.cpu_percent:.1f}% (limit: {max_cpu}%)"
            )
        
        if metrics.memory_percent > max_memory:
            alerts.append(
                f"Memory usage high: {metrics.memory_percent:.1f}% "
                f"(limit: {max_memory}%)"
            )
        
        if metrics.is_memory_critical:
            alerts.append(
                f"CRITICAL: Memory usage at {metrics.memory_percent:.1f}%"
            )
        
        return alerts


class HealthMonitor:
    """
    Monitors health of orchestrator components.
    
    Checks queue health, worker health, and overall system health.
    """
    
    def __init__(self, orchestrator):
        """
        Initialize health monitor.
        
        Args:
            orchestrator: MultiSiteOrchestrator instance
        """
        self.orchestrator = orchestrator
        self.performance_monitor = PerformanceMonitor()
        
        # Health check history
        self.health_history: List[OrchestratorHealth] = []
        self.max_history_size = 100
    
    def check_queue_health(self) -> List[QueueHealth]:
        """Check health of all queues."""
        queue_health_list = []
        
        if not self.orchestrator.queue_manager:
            return queue_health_list
        
        metrics = self.orchestrator.queue_manager.get_metrics()
        
        for name, metric in metrics.items():
            if name == "dead_letter":
                continue
            
            health = QueueHealth(
                name=name,
                current_size=metric.current_size,
                max_size=metric.max_size,
                utilization_percent=metric.utilization_percent,
                total_added=metric.total_added,
                total_removed=metric.total_removed,
                avg_wait_time_ms=metric.avg_wait_time_ms,
                is_full=(metric.current_size >= metric.max_size),
                is_empty=metric.is_empty,
            )
            
            queue_health_list.append(health)
        
        return queue_health_list
    
    def check_worker_health(self) -> List[WorkerHealth]:
        """Check health of all workers."""
        worker_health_list = []
        
        all_workers = (
            self.orchestrator.crawler_workers +
            self.orchestrator.processor_workers +
            self.orchestrator.ocr_workers +
            self.orchestrator.storage_workers
        )
        
        for worker in all_workers:
            stats = worker.stats
            
            health = WorkerHealth(
                worker_id=worker.worker_id,
                worker_type=worker.worker_type,
                is_running=worker.is_running,
                tasks_processed=worker.tasks_processed,
                tasks_failed=worker.tasks_failed,
                uptime_seconds=stats.get("uptime_seconds", 0.0),
            )
            
            worker_health_list.append(health)
        
        return worker_health_list
    
    def get_overall_health(self) -> OrchestratorHealth:
        """Get overall orchestrator health."""
        # Collect metrics
        system_metrics = self.performance_monitor.get_system_metrics()
        queue_health = self.check_queue_health()
        worker_health = self.check_worker_health()
        
        # Determine overall status
        alerts = []
        warnings = []
        
        # Check system resources
        resource_alerts = self.performance_monitor.check_resource_limits(
            system_metrics,
            max_cpu=self.orchestrator.config.limits.max_cpu_percent,
            max_memory=90.0,  # 90% memory threshold
        )
        alerts.extend(resource_alerts)
        
        # Check queue health
        unhealthy_queues = [
            q for q in queue_health
            if q.health_status == HealthStatus.UNHEALTHY
        ]
        if unhealthy_queues:
            alerts.append(
                f"Unhealthy queues: {', '.join(q.name for q in unhealthy_queues)}"
            )
        
        degraded_queues = [
            q for q in queue_health
            if q.health_status == HealthStatus.DEGRADED
        ]
        if degraded_queues:
            warnings.append(
                f"Degraded queues: {', '.join(q.name for q in degraded_queues)}"
            )
        
        # Check worker health
        unhealthy_workers = [
            w for w in worker_health
            if w.health_status == HealthStatus.UNHEALTHY
        ]
        if unhealthy_workers:
            alerts.append(
                f"Unhealthy workers: {len(unhealthy_workers)}/{len(worker_health)}"
            )
        
        # Determine overall status
        if alerts:
            status = HealthStatus.UNHEALTHY
        elif warnings:
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.HEALTHY
        
        health = OrchestratorHealth(
            status=status,
            system_metrics=system_metrics,
            queue_health=queue_health,
            worker_health=worker_health,
            alerts=alerts,
            warnings=warnings,
        )
        
        # Store in history
        self.health_history.append(health)
        if len(self.health_history) > self.max_history_size:
            self.health_history.pop(0)
        
        return health
    
    def log_health_report(self, health: OrchestratorHealth):
        """Log health report."""
        logger.info("="*70)
        logger.info(f"Health Report - {health.status.upper()}")
        logger.info("-"*70)
        
        # System metrics
        logger.info("System Resources:")
        logger.info(
            f"  CPU: {health.system_metrics.cpu_percent:.1f}% "
            f"({health.system_metrics.cpu_count} cores)"
        )
        logger.info(
            f"  Memory: {health.system_metrics.memory_percent:.1f}% "
            f"({health.system_metrics.memory_used_mb:.0f}/"
            f"{health.system_metrics.memory_total_mb:.0f} MB)"
        )
        
        if health.system_metrics.gpu_memory_total_mb > 0:
            logger.info(
                f"  GPU: {health.system_metrics.gpu_utilization:.1f}% "
                f"({health.system_metrics.gpu_memory_used_mb:.0f}/"
                f"{health.system_metrics.gpu_memory_total_mb:.0f} MB)"
            )
        
        # Queue health
        logger.info("\nQueue Health:")
        for q in health.queue_health:
            status_icon = {
                HealthStatus.HEALTHY: "✓",
                HealthStatus.DEGRADED: "⚠",
                HealthStatus.UNHEALTHY: "✗",
            }.get(q.health_status, "?")
            
            logger.info(
                f"  {status_icon} {q.name:12s}: {q.current_size:3d}/{q.max_size:3d} "
                f"({q.utilization_percent:5.1f}%)"
            )
        
        # Worker health
        logger.info("\nWorker Health:")
        logger.info(
            f"  Healthy: {health.healthy_workers}/{health.total_workers} workers"
        )
        
        # Alerts and warnings
        if health.alerts:
            logger.warning("\nALERTS:")
            for alert in health.alerts:
                logger.warning(f"  ⚠ {alert}")
        
        if health.warnings:
            logger.info("\nWARNINGS:")
            for warning in health.warnings:
                logger.info(f"  ⚠ {warning}")
        
        logger.info("="*70)


class ProgressTracker:
    """
    Tracks progress of multi-website crawling.
    
    Provides detailed progress reports and ETAs.
    """
    
    def __init__(self, orchestrator):
        """
        Initialize progress tracker.
        
        Args:
            orchestrator: MultiSiteOrchestrator instance
        """
        self.orchestrator = orchestrator
        self.start_time = datetime.utcnow()
        
        # Progress history
        self.progress_snapshots: List[Dict] = []
    
    def get_progress_report(self) -> Dict[str, Any]:
        """Get detailed progress report."""
        elapsed = (datetime.utcnow() - self.start_time).total_seconds()
        
        # Get queue metrics
        queue_metrics = {}
        if self.orchestrator.queue_manager:
            metrics = self.orchestrator.queue_manager.get_metrics()
            for name, metric in metrics.items():
                queue_metrics[name] = {
                    "current": metric.current_size,
                    "max": metric.max_size,
                    "utilization": metric.utilization_percent,
                    "added": metric.total_added,
                    "removed": metric.total_removed,
                }
        
        # Get worker stats
        all_workers = (
            self.orchestrator.crawler_workers +
            self.orchestrator.processor_workers +
            self.orchestrator.ocr_workers +
            self.orchestrator.storage_workers
        )
        
        total_processed = sum(w.tasks_processed for w in all_workers)
        total_failed = sum(w.tasks_failed for w in all_workers)
        
        # Calculate throughput
        throughput = total_processed / elapsed if elapsed > 0 else 0
        
        report = {
            "elapsed_seconds": elapsed,
            "elapsed_formatted": str(timedelta(seconds=int(elapsed))),
            "websites": {
                "total": self.orchestrator.stats.total_websites,
                "completed": self.orchestrator.stats.websites_completed,
                "failed": self.orchestrator.stats.websites_failed,
                "in_progress": self.orchestrator.stats.websites_in_progress,
                "remaining": self.orchestrator.stats.websites_remaining,
            },
            "tasks": {
                "processed": total_processed,
                "failed": total_failed,
                "success_rate": (
                    (total_processed / (total_processed + total_failed) * 100)
                    if (total_processed + total_failed) > 0 else 0
                ),
            },
            "performance": {
                "throughput_per_second": throughput,
                "throughput_per_minute": throughput * 60,
            },
            "queues": queue_metrics,
            "workers": {
                "total": len(all_workers),
                "active": sum(1 for w in all_workers if w.is_running),
                "by_type": {
                    "crawlers": len(self.orchestrator.crawler_workers),
                    "processors": len(self.orchestrator.processor_workers),
                    "ocr": len(self.orchestrator.ocr_workers),
                    "storage": len(self.orchestrator.storage_workers),
                },
            },
        }
        
        # Store snapshot
        self.progress_snapshots.append({
            "timestamp": datetime.utcnow(),
            "elapsed": elapsed,
            "processed": total_processed,
        })
        
        return report
    
    def log_progress_report(self, report: Dict[str, Any]):
        """Log progress report."""
        logger.info("\n" + "="*70)
        logger.info(
            f"Progress Report - Elapsed: {report['elapsed_formatted']}"
        )
        logger.info("-"*70)
        
        # Websites
        logger.info("Websites:")
        logger.info(
            f"  Total: {report['websites']['total']} | "
            f"Completed: {report['websites']['completed']} | "
            f"Failed: {report['websites']['failed']} | "
            f"In Progress: {report['websites']['in_progress']}"
        )
        
        # Tasks
        logger.info("\nTasks:")
        logger.info(
            f"  Processed: {report['tasks']['processed']} | "
            f"Failed: {report['tasks']['failed']} | "
            f"Success Rate: {report['tasks']['success_rate']:.1f}%"
        )
        
        # Performance
        logger.info("\nPerformance:")
        logger.info(
            f"  Throughput: {report['performance']['throughput_per_second']:.2f} tasks/sec "
            f"({report['performance']['throughput_per_minute']:.1f} tasks/min)"
        )
        
        # Queues
        logger.info("\nQueues:")
        for name, metrics in report['queues'].items():
            if name == "dead_letter":
                logger.info(
                    f"  {name:12s}: {metrics['current']} failed"
                )
            else:
                logger.info(
                    f"  {name:12s}: {metrics['current']:3d}/{metrics['max']:3d} "
                    f"({metrics['utilization']:5.1f}%) | "
                    f"Added: {metrics['added']:4d} | "
                    f"Removed: {metrics['removed']:4d}"
                )
        
        logger.info("="*70)
