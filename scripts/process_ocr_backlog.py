"""
Process OCR Backlog

Processes pages that were marked as needing OCR during orchestrator run.
Runs OCR one page at a time to avoid memory issues.

Usage:
    python scripts/process_ocr_backlog.py
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_logger
from app.crawling.models.document import Page, OCRAction, PageContent, ContentSource
from app.orchestrator.workers.helpers.ocr_processor import process_page_ocr
from app.orchestrator.workers.helpers.text_processor import chunk_page_text
from app.services.document_store import DocumentStore

logger = get_logger(__name__)


async def _store_chunks(
    chunks: list,
    entry: dict,
    doc_store: DocumentStore,
) -> None:
    """
    Store OCR chunks to MongoDB and Qdrant.
    
    Args:
        chunks: List of chunk dictionaries
        entry: Backlog entry with metadata
        doc_store: Document store instance
    """
    from app.config import get_embedding_model, get_qdrant_client
    from qdrant_client.models import PointStruct, Distance, VectorParams
    from datetime import datetime
    import hashlib
    
    url = entry["url"]
    
    # Get or create document
    doc = doc_store.get_by_source_url(url)
    if not doc:
        doc = doc_store.create_document(
            original_file=url,
            source_url=url,
            file_path=entry.get("pdf_path", ""),
            crawl_session_id="ocr_backlog_processing",
            crawl_depth=entry.get("depth", 0),
        )
    
    # Update document with OCR metadata
    doc_store.update_document(
        doc.file_id,
        {
            "is_ocr_required": "1",
            "is_ocr_completed": "1",
            "ocr_completed_at": datetime.utcnow(),
            "is_crawled": "1",
            "vector_count": len(chunks),
        }
    )
    
    # Get embedding model and Qdrant client
    embedding_model, vector_size = get_embedding_model()
    qdrant_client = get_qdrant_client()
    collection_name = "crawled_documents"
    
    # Ensure collection exists
    try:
        qdrant_client.get_collection(collection_name)
    except:
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
    
    # Generate embeddings and store
    points = []
    for i, chunk in enumerate(chunks):
        chunk_text = chunk["text"] if isinstance(chunk, dict) else chunk.text
        embedding = embedding_model.encode(chunk_text).tolist()
        
        chunk_id = hashlib.md5(
            f"{doc.file_id}_ocr_{i}_{chunk_text[:50]}".encode()
        ).hexdigest()
        
        point = PointStruct(
            id=chunk_id,
            vector=embedding,
            payload={
                "file_id": doc.file_id,
                "chunk_index": i,
                "text": chunk_text,
                "word_count": chunk.get("word_count", 0) if isinstance(chunk, dict) else 0,
                "source_url": url,
                "created_at": datetime.utcnow().isoformat(),
                "is_ocr": True,
                "metadata": chunk.get("metadata", {}) if isinstance(chunk, dict) else {},
            }
        )
        points.append(point)
    
    # Batch upsert
    qdrant_client.upsert(
        collection_name=collection_name,
        points=points,
    )
    
    # Mark as vectorized
    doc_store.update_document(
        doc.file_id,
        {
            "is_vectorized": "1",
            "vectorization_completed_at": datetime.utcnow(),
            "status": "VECTORIZED",
        }
    )
    
    logger.info(f"✓ Stored {len(chunks)} chunks to MongoDB + Qdrant: {url}")


async def process_single_page(
    entry: dict,
    doc_store: DocumentStore,
    timeout: int = 120,
) -> bool:
    """
    Process OCR for a single page.
    
    Args:
        entry: Backlog entry with page info
        doc_store: Document store for saving chunks
        timeout: Timeout in seconds (default 120s = 2 minutes)
        
    Returns:
        True if successful, False otherwise
    """
    url = entry["url"]
    pdf_path = entry.get("pdf_path")
    ocr_action = OCRAction(entry["ocr_action"])
    
    if not pdf_path or not Path(pdf_path).exists():
        logger.warning(f"PDF not found for {url}: {pdf_path}")
        return False
    
    logger.info(f"Processing OCR for: {url}")
    
    try:
        # Create Page object
        page = Page(
            url=url,
            depth=entry.get("depth", 0),
            pdf_path=Path(pdf_path),
        )
        
        # Run OCR with timeout
        ocr_text = await asyncio.wait_for(
            process_page_ocr(page, ocr_action),
            timeout=timeout
        )
        
        if not ocr_text:
            logger.warning(f"No OCR text extracted from {url}")
            return False
        
        # Log OCR results
        word_count = len(ocr_text.split())
        logger.info(f"OCR extracted {word_count} words ({len(ocr_text)} chars) from {url}")
        
        # Update page content with OCR text
        page.content = PageContent.from_text(ocr_text, ContentSource.OCR)
        
        # Chunk the OCR'd text with lower threshold for OCR content
        chunks = chunk_page_text(
            page,
            min_words=20,  # Lower threshold for OCR content (was 100)
            max_words=512,
            overlap_words=25,  # Smaller overlap for short content
        )
        
        if not chunks:
            logger.warning(f"No chunks created (text too short: {word_count} words): {url}")
            return False
        
        logger.info(f"✓ Created {len(chunks)} chunks from OCR text: {url}")
        
        # Save chunks to MongoDB and Qdrant
        await _store_chunks(chunks, entry, doc_store)
        
        return True
        
    except asyncio.TimeoutError:
        logger.warning(f"⏱ OCR timeout ({timeout}s) for {url} - skipping")
        return False
        
    except Exception as e:
        logger.error(f"❌ OCR failed for {url}: {e}")
        return False


async def process_ocr_backlog():
    """Process all pages in OCR backlog."""
    backlog_file = Path("outputs/ocr_backlog/pending_ocr.jsonl")
    
    if not backlog_file.exists():
        logger.info("No OCR backlog found")
        return
    
    # Load backlog entries
    entries = []
    with backlog_file.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    
    logger.info(f"Found {len(entries)} pages in OCR backlog")
    
    if not entries:
        return
    
    # Initialize document store
    from app.config import app_config
    doc_store = DocumentStore(
        mongodb_url=app_config.MONGODB_URL,
        database_name=app_config.MONGODB_DATABASE,
    )
    
    # Process each entry
    success_count = 0
    failed_count = 0
    
    for i, entry in enumerate(entries, 1):
        logger.info(f"\n[{i}/{len(entries)}] Processing: {entry['url']}")
        
        success = await process_single_page(entry, doc_store, timeout=120)
        
        if success:
            success_count += 1
        else:
            failed_count += 1
        
        # Small delay between pages to let memory settle
        await asyncio.sleep(2)
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("OCR Backlog Processing Complete")
    logger.info(f"  Success: {success_count}/{len(entries)}")
    logger.info(f"  Failed:  {failed_count}/{len(entries)}")
    logger.info("="*70)
    
    # Archive processed backlog
    archive_file = backlog_file.parent / f"processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    backlog_file.rename(archive_file)
    logger.info(f"Backlog archived to: {archive_file}")


def main():
    """Main entry point."""
    print("\n" + "="*70)
    print("OCR BACKLOG PROCESSOR")
    print("="*70 + "\n")
    
    try:
        asyncio.run(process_ocr_backlog())
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
    except Exception as e:
        logger.error(f"Failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
