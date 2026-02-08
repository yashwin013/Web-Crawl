"""
Crawler stage - handles web crawling and PDF download.
"""

import datetime
import aiohttp
import aiofiles
import uuid
import asyncio
import hashlib
import time
from pathlib import Path
from typing import Optional, Set, TYPE_CHECKING
from urllib.parse import urlparse, urljoin

if TYPE_CHECKING:
    from playwright.async_api import Browser, BrowserContext

from app.crawling.stages.base import PipelineStage
from app.crawling.models.document import Document, Page
from app.crawling.utils.rate_limiter import RateLimiter
from app.crawling.utils.content_filter import ContentFilter
from app.crawling.utils.robots import RobotsRules, parse_robots_txt, parse_sitemap

from app.config import get_logger

logger = get_logger(__name__)


class CrawlerStage(PipelineStage):
    """
    Web crawler stage.
    
    Crawls a website starting from a URL, downloads PDFs,
    and saves HTML pages for further processing.
    
    Features:
    - Sitemap-based discovery
    - Robots.txt compliance
    - Rate limiting with backoff
    - Duplicate content detection
    - PDF detection and download
    """
    
    def __init__(self, config=None):
        super().__init__(config)
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._rate_limiter: Optional[RateLimiter] = None
        self._content_filter: Optional[ContentFilter] = None
        self._robots: Optional[RobotsRules] = None
    
    @property
    def name(self) -> str:
        return "crawler"
    
    async def setup(self) -> None:
        """Initialize browser and helpers."""
        crawl_config = self.config.crawl if self.config else None
        
        # Initialize rate limiter
        self._rate_limiter = RateLimiter(
            base_delay=crawl_config.request_delay if crawl_config else 1.0,
            max_delay=crawl_config.max_delay if crawl_config else 30.0,
        )
        
        # Initialize content filter
        self._content_filter = ContentFilter(
            skip_404=crawl_config.skip_404 if crawl_config else True,
            skip_login_pages=crawl_config.skip_login_pages if crawl_config else True,
            skip_duplicates=crawl_config.skip_duplicates if crawl_config else True,
            include_patterns=crawl_config.include_patterns if crawl_config else [],
            exclude_patterns=crawl_config.exclude_patterns if crawl_config else [],
        )
    
    async def teardown(self) -> None:
        """
        Cleanup resources.
        
        Browser is closed automatically by the context manager in process(),
        so we don't need to do it here.
        """
        pass
    
    async def process(self, document: Document) -> Document:
        """
        Crawl website and populate document with pages.
        
        Args:
            document: Document with start_url set
            
        Returns:
            Document populated with discovered pages
        """
        start_url = document.start_url
        output_dir = document.output_dir or self.config.output_dir
        output_dir = output_dir.resolve()  # Convert to absolute path
        output_dir.mkdir(parents=True, exist_ok=True)
        
        crawl_config = self.config.crawl if self.config else None
        max_depth = crawl_config.max_depth if crawl_config else 3
        max_pages = crawl_config.max_pages if crawl_config else 50
        
        base_domain = self._get_base_domain(start_url)
        document.start_time = time.time()
        
        # Set crawl session ID for this crawl
        self._crawl_session_id = str(uuid.uuid4())
        logger.info(f"Starting crawl session: {self._crawl_session_id}")
        
        # Parse robots.txt
        if crawl_config and crawl_config.respect_robots:
            logger.info(f"Respecting robots.txt for {base_domain}")
            self._robots = parse_robots_txt(start_url)
            if self._robots.crawl_delay > 0:
                self._rate_limiter.base_delay = max(
                    self._rate_limiter.base_delay,
                    self._robots.crawl_delay
                )
        else:
            logger.info(f"Ignoring robots.txt for {base_domain} (respect_robots=False)")
        
        # Discover URLs from sitemap
        discovered_urls: Set[str] = set()
        if crawl_config and crawl_config.use_sitemap:
            discovered_urls = parse_sitemap(start_url)
        
        # Initialize crawl queue
        queue: asyncio.Queue = asyncio.Queue()
        await queue.put((start_url, 0))  # (url, depth)
        
        for url in discovered_urls:
            if self._is_same_domain(url, base_domain):
                await queue.put((url, 1))
        
        logger.info(f"Crawl queue initialized with {queue.qsize()} URLs (start_url + {len(discovered_urls)} from sitemap)")
        
        visited: Set[str] = set()
        pdf_mapping: dict = {}  # Maps pdf_url -> (referring_page_url, depth)
        
        # Lazy import playwright
        from playwright.async_api import async_playwright
        
        async with async_playwright() as p:
            self._browser = await p.chromium.launch(headless=True)
            self._context = await self._browser.new_context(
                user_agent="Mozilla/5.0 (compatible; DoclingBot/1.0)"
            )
            
            while not queue.empty() and len(visited) < max_pages:
                url, depth = await queue.get()
                
                logger.info(f"Processing URL from queue: {url} (depth={depth})")
                
                # Skip if already visited
                if url in visited:
                    logger.info(f"Already visited, skipping: {url}")
                    continue
                
                # Normalize URL
                url = self._normalize_url(url)
                
                # Check content filter
                should_skip, reason = self._content_filter.should_skip_url(url)
                if should_skip:
                    logger.info(f"Skipping {url}: {reason}")
                    document.pages_skipped += 1
                    continue
                
                # Check robots.txt
                if self._robots and not self._robots.can_fetch(url):
                    logger.info(f"Blocked by robots.txt: {url}")
                    document.pages_skipped += 1
                    continue
                
                visited.add(url)
                
                # Handle PDFs separately
                if self._is_pdf(url):
                    # Store PDF with None as referring page (directly visited)
                    if url not in pdf_mapping:
                        pdf_mapping[url] = (None, depth)
                    continue
                
                # Rate limit
                await self._rate_limiter.wait()
                
                try:
                    logger.info(f"Attempting to crawl: {url}")
                    page = await self._crawl_single_page(url, depth, output_dir)
                    document.add_page(page)
                    self._rate_limiter.success()
                    logger.info(f"Successfully crawled: {url} (found {len(page.html_content) if page.html_content else 0} chars)")
                    
                    # Extract links for further crawling
                    if depth < max_depth and page.html_content:
                        links = await self._extract_links_from_html(
                            page.html_content, url, base_domain
                        )
                        logger.info(f"Found {len(links)} links on {url}")
                        for link in links:
                            if link not in visited:
                                # Check if link is a PDF and track the referring page
                                if self._is_pdf(link):
                                    if link not in pdf_mapping:
                                        pdf_mapping[link] = (url, depth + 1)  # Store referring page
                                else:
                                    await queue.put((link, depth + 1))
                                
                except Exception as e:
                    logger.error(f"Failed to crawl {url}: {e}", exc_info=True)
                    self._rate_limiter.failure()
                    document.pages_failed += 1
            
            # Download PDFs with their referring pages
            for pdf_url, (referring_page, depth) in pdf_mapping.items():
                try:
                    # Pass the referring page URL so PDF is nested under that page in MongoDB
                    pdf_page = await self._download_pdf(pdf_url, output_dir, current_page_url=referring_page, crawl_depth=depth)
                    document.add_page(pdf_page)
                    logger.info(f"Downloaded PDF from page: {referring_page or 'direct'} -> {pdf_url}")
                except Exception as e:
                    logger.error(f"Failed to download PDF {pdf_url}: {e}")
                    document.pages_failed += 1
        
        document.end_time = time.time()
        logger.info(
            f"Crawl complete: {document.pages_scraped} pages, "
            f"{document.pdfs_downloaded} PDFs, "
            f"{document.pages_failed} failed"
        )
        
        return document
    
    async def _crawl_single_page(
        self, url: str, depth: int, output_dir: Path
    ) -> Page:
        """Crawl a single page and save content."""
        start_time = time.time()
        
        page_obj = await self._context.new_page()
        try:
            timeout = self.config.crawl.page_timeout_ms if self.config else 30000
            await page_obj.goto(url, timeout=timeout, wait_until="networkidle")
            
            # Get HTML content
            html_content = await page_obj.content()
            
            # Extract content text and images
            # We get rendered dimensions to make better OCR decisions
            eval_script = """
            () => {
                const images = Array.from(document.querySelectorAll('img'));
                return {
                    text: document.body.innerText,
                    images: images.map(img => ({
                        src: img.src,
                        width: img.width,
                        height: img.height,
                        naturalWidth: img.naturalWidth,
                        naturalHeight: img.naturalHeight,
                        alt: img.alt
                    }))
                };
            }
            """
            data = await page_obj.evaluate(eval_script)
            dom_text = data["text"]
            scraped_images = data["images"]
            
            # Generate content hash
            content_hash = hashlib.md5(html_content.encode()).hexdigest()
            
            # Check for duplicate
            if self._content_filter.is_duplicate_content(content_hash):
                logger.info(f"Duplicate content detected: {url}")
                return Page(
                    url=url,
                    depth=depth,
                    status_code=200,
                    content_hash=content_hash,
                )
            
            # Save HTML
            filename = str(uuid.uuid4()).replace('-', '')
            html_path = output_dir / f"{filename}.html"
            async with aiofiles.open(html_path, 'w', encoding="utf-8") as f:
                await f.write(html_content)
            
            # Save DOM text
            dom_path = output_dir / f"{filename}.txt"
            async with aiofiles.open(dom_path, 'w', encoding="utf-8") as f:
                await f.write(dom_text)
            
            # Print as PDF
            pdf_path = output_dir / f"{filename}.pdf"
            await page_obj.pdf(path=str(pdf_path))
            
            # --- VECTOR PIPELINE DISABLED IN ORCHESTRATOR MODE ---
            # Note: In orchestrator mode, PDFs are automatically queued for processing
            # by the PDF processor workers. This legacy integration causes memory exhaustion
            # and blocks crawlers. Keep this code disabled during orchestration.
            # --- END VECTOR PIPELINE INTEGRATION ---
            
            elapsed = (time.time() - start_time) * 1000
            
            return Page(
                url=url,
                depth=depth,
                pdf_path=pdf_path,
                html_path=html_path,
                html_content=html_content,
                dom_text=dom_text,
                scraped_images=scraped_images,
                status_code=200,
                content_hash=content_hash,
                processing_time_ms=elapsed,
            )
            
        finally:
            await page_obj.close()
    
    async def _download_pdf(self, url: str, output_dir: Path, current_page_url: Optional[str] = None, crawl_depth: int = 0) -> Page:
        """Download a PDF file and save metadata to MongoDB using nested structure."""
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                content = await response.read()
                
                # Generate unique file ID
                file_id = f"{uuid.uuid4()}.pdf"
                
                # Get original filename from URL
                parsed_url = urlparse(url)
                original_filename = parsed_url.path.split("/")[-1] or "document.pdf"
                
                # Save to configured PDF storage path
                from app.config import UPLOAD_DIR
                pdf_storage = Path(UPLOAD_DIR)
                pdf_storage.mkdir(parents=True, exist_ok=True)
                
                pdf_path = pdf_storage / file_id
                async with aiofiles.open(pdf_path, 'wb') as f:
                    await f.write(content)
                
                # Also save to output_dir for pipeline compatibility
                output_path = output_dir / f"{self._sanitize_filename(url)}.pdf"
                async with aiofiles.open(output_path, 'wb') as f:
                    await f.write(content)
                
                # --- VECTOR PIPELINE DISABLED IN ORCHESTRATOR MODE ---
                # Note: PDF processing handled by PDF processor workers in orchestrator mode
                # Legacy vector pipeline integration disabled to prevent memory exhaustion
                logger.info(f"PDF downloaded (orchestrator will process): {file_id}")
                # --- END VECTOR PIPELINE INTEGRATION ---

                # Save metadata to MongoDB using nested website structure
                try:
                    from app.services.document_store import DocumentStore
                    from app.schemas.document import DocumentStatus, PdfDocument
                    store = DocumentStore.from_config()
                    
                    # Get crawl session ID from document metadata or generate one
                    crawl_session_id = getattr(self, '_crawl_session_id', str(uuid.uuid4()))
                    
                    # Extract website URL (scheme + netloc)
                    website_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
                    
                    # Use the referring page URL if provided, otherwise use the PDF URL itself
                    visited_url = current_page_url if current_page_url else url
                    
                    # Create PdfDocument
                    pdf_doc = PdfDocument(
                        file_id=file_id,
                        original_file=original_filename,
                        source_url=url,
                        file_path=str(pdf_path),
                        crawl_session_id=crawl_session_id,
                        file_size=len(content),
                        crawl_depth=crawl_depth,
                        status=DocumentStatus.STORED,  # No chunking for direct downloads
                        is_crawled="1",
                        total_pages=0,
                        pages_with_text=0,
                        pages_needing_ocr=0,
                    )
                    
                    # Add PDF to website using nested structure
                    store.add_pdf_to_website(
                        website_url=website_url,
                        crawl_session_id=crawl_session_id,
                        visited_url=visited_url,
                        crawl_depth=crawl_depth,
                        pdf_document=pdf_doc
                    )
                    
                    logger.info(f"Saved PDF to MongoDB 'websites' (nested): {file_id} in {website_url}")
                except Exception as e:
                    logger.warning(f"Failed to save PDF to DocumentStore: {e}", exc_info=True)
                
                return Page(
                    url=url,
                    pdf_path=pdf_path,
                    status_code=response.status,
                )
    
    async def _extract_links_from_html(
        self, html: str, base_url: str, base_domain: str
    ) -> Set[str]:
        """Extract links from HTML content."""
        from bs4 import BeautifulSoup
        
        links = set()
        soup = BeautifulSoup(html, "html.parser")
        
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            full_url = urljoin(base_url, href)
            
            if self._is_same_domain(full_url, base_domain):
                if self._is_valid_page(full_url):
                    links.add(self._normalize_url(full_url))
        
        return links
    
    @staticmethod
    def _get_base_domain(url: str) -> str:
        """Extract base domain from URL."""
        parsed = urlparse(url)
        return parsed.netloc
    
    @staticmethod
    def _normalize_url(url: str) -> str:
        """Normalize URL (remove fragments, trailing slashes)."""
        parsed = urlparse(url)
        # Remove fragment and normalize
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path.rstrip('/')}"
    
    @staticmethod
    def _sanitize_filename(url: str, max_len: int = 100) -> str:
        """Create safe filename from URL."""
        parsed = urlparse(url)
        name = f"{parsed.netloc}{parsed.path}".replace("/", "_").replace(".", "_")
        return name[:max_len]
    
    @staticmethod
    def _is_same_domain(url: str, base_domain: str) -> bool:
        """Check if URL is on same domain."""
        return urlparse(url).netloc == base_domain
    
    @staticmethod
    def _is_valid_page(url: str) -> bool:
        """Check if URL is a valid page (not image, css, js, etc)."""
        invalid_extensions = [
            ".css", ".js", ".png", ".jpg", ".jpeg", ".gif", ".svg",
            ".ico", ".woff", ".woff2", ".ttf", ".eot"
        ]
        path = urlparse(url).path.lower()
        return not any(path.endswith(ext) for ext in invalid_extensions)
    
    @staticmethod
    def _is_pdf(url: str) -> bool:
        """Check if URL is a PDF."""
        return urlparse(url).path.lower().endswith(".pdf")
