"""
Crawler Worker

Crawls websites and pushes discovered pages to the processing queue.
"""

import uuid
from typing import Optional
from datetime import datetime

from app.config import get_logger
from app.orchestrator.workers.base import BaseWorker
from app.orchestrator.queues import QueueManager
from app.orchestrator.models.task import CrawlTask, ProcessTask, TaskPriority
from app.crawling.stages.crawler import CrawlerStage
from app.crawling.models.document import Document
from app.crawling.models.config import PipelineConfig

logger = get_logger(__name__)


class CrawlerWorker(BaseWorker):
    """
    Crawler worker - handles website crawling.
    
    Responsibilities:
    - Pull CrawlTask from crawl_queue
    - Crawl website using CrawlerStage
    - Push individual pages to processing_queue
    - Update MongoDB with crawl status
    """
    
    def __init__(
        self,
        worker_id: str,
        queue_manager: QueueManager,
        config: Optional[PipelineConfig] = None,
    ):
        super().__init__(worker_id, queue_manager)
        self.config = config or PipelineConfig()
        self.crawler_stage = CrawlerStage(self.config)
    
    @property
    def worker_type(self) -> str:
        return "crawler"
    
    async def startup(self):
        """Initialize crawler resources."""
        await super().startup()
        await self.crawler_stage.setup()
        logger.info(f"[{self.worker_id}] Crawler initialized")
    
    async def shutdown(self):
        """Cleanup crawler resources."""
        await self.crawler_stage.teardown()
        await super().shutdown()
    
    async def get_next_task(self) -> Optional[CrawlTask]:
        """Get next crawl task from queue."""
        return await self.queue_manager.get_crawl_task(timeout=1.0)
    
    async def process_task(self, task: CrawlTask) -> bool:
        """
        Crawl a website and push pages to processing queue.
        
        Args:
            task: CrawlTask with website URL
            
        Returns:
            True if successful
        """
        task.mark_started(self.worker_id)
        
        logger.info(
            f"[{self.worker_id}] Crawling {task.website_url} "
            f"(max_pages={task.max_pages}, max_depth={task.max_depth})"
        )
        
        try:
            # Create document for crawling
            document = Document(
                start_url=task.website_url,
                output_dir=self.config.output_dir,
                crawl_depth=task.max_depth,
                max_pages=task.max_pages,
            )
            
            # Run crawler (this will discover and download pages)
            document = await self.crawler_stage.process(document)
            
            # Update task progress
            task.pages_discovered = len(document.pages)
            task.pages_crawled = len([p for p in document.pages if p.html_content or p.pdf_path])
            
            logger.info(
                f"[{self.worker_id}] Crawled {task.pages_crawled}/{task.pages_discovered} pages "
                f"from {task.website_url}"
            )
            
            # Push each page to processing queue
            pages_queued = 0
            for page in document.pages:
                # Only queue pages with content
                if page.html_content or page.pdf_path or page.dom_text:
                    process_task = ProcessTask(
                        task_id=f"{task.task_id}_page_{page.url_hash}",
                        page=page,
                        website_url=task.website_url,
                        crawl_session_id=task.crawl_session_id,
                        priority=task.priority,
                    )
                    
                    await self.queue_manager.put_process_task(process_task)
                    pages_queued += 1
            
            logger.info(
                f"[{self.worker_id}] Queued {pages_queued} pages for processing"
            )
            
            # Mark task complete
            task.mark_completed()
            return True
            
        except Exception as e:
            logger.error(
                f"[{self.worker_id}] Failed to crawl {task.website_url}: {e}",
                exc_info=True
            )
            
            # Move to dead letter queue if max retries exceeded
            if task.retry_count >= task.max_retries:
                await self.queue_manager.put_dead_letter(
                    task,
                    f"Max retries exceeded: {e}"
                )
            else:
                # Requeue for retry
                task.mark_failed(str(e))
                await self.queue_manager.put_crawl_task(task)
                logger.info(f"[{self.worker_id}] Requeued task (retry {task.retry_count}/{task.max_retries})")
            
            return False
