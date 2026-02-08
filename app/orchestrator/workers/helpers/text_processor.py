"""
Text processing helpers for page-level operations.

Extracted from pipeline stages to work on individual pages in the worker system.
"""

import re
from typing import Optional, Tuple, List, Dict, Any

from app.crawling.models.document import (
    Page, PageContent, ContentSource, OCRAction, ImageInfo
)
from app.config import get_logger

logger = get_logger(__name__)


# ==================== Text Extraction ====================

def extract_page_text(page: Page, min_chars: int = 100, min_words: int = 20) -> PageContent:
    """
    Extract text content from a single page.
    
    Tries DOM text first, then HTML parsing, with cleaning and validation.
    
    Args:
        page: Page object with HTML/DOM content
        min_chars: Minimum characters for valid content
        min_words: Minimum words for valid content
        
    Returns:
        PageContent with extracted text
    """
    # Try DOM text first (fastest)
    if page.dom_text:
        cleaned = _clean_dom_text(page.dom_text)
        content = PageContent.from_text(cleaned, ContentSource.DOM)
        
        # Add images if available
        if page.scraped_images:
            content.images = [
                ImageInfo(
                    width=img.get("width", 0),
                    height=img.get("height", 0),
                    aspect_ratio=img.get("width", 0) / img.get("height", 1) if img.get("height", 0) > 0 else 0,
                    image_type="unknown",
                    area=img.get("width", 0) * img.get("height", 0)
                )
                for img in page.scraped_images
            ]
        return content
    
    # Try HTML content
    if page.html_content:
        text = _extract_from_html(page.html_content)
        cleaned = _clean_dom_text(text)
        return PageContent.from_text(cleaned, ContentSource.DOM)
    
    # No text available
    return PageContent(
        text="",
        word_count=0,
        char_count=0,
        source=ContentSource.DOM,
    )


def _clean_dom_text(raw_text: str) -> str:
    """
    Clean extracted DOM text.
    
    Removes:
    - Excessive whitespace
    - Navigation noise
    - Cookie banners
    """
    if not raw_text:
        return ""
    
    # Normalize whitespace
    text = re.sub(r"\s+", " ", raw_text)
    
    # Remove common noise patterns
    noise_patterns = [
        r"Accept\s+cookies?",
        r"Cookie\s+policy",
        r"Privacy\s+policy",
        r"Terms\s+of\s+service",
        r"Skip\s+to\s+content",
        r"Toggle\s+navigation",
        r"Loading\.\.\.",
    ]
    
    for pattern in noise_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    
    # Remove very short lines (likely buttons/links)
    lines = text.split("\n")
    cleaned_lines = [
        line.strip() for line in lines
        if len(line.strip()) > 20 or "." in line
    ]
    
    return "\n".join(cleaned_lines).strip()


def _extract_from_html(html: str) -> str:
    """Extract text from HTML using BeautifulSoup."""
    try:
        from bs4 import BeautifulSoup
        
        soup = BeautifulSoup(html, "html.parser")
        
        # Remove script and style elements
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()
        
        # Get text
        text = soup.get_text(separator="\n", strip=True)
        return text
        
    except Exception as e:
        logger.error(f"HTML parsing failed: {e}")
        return ""


# ==================== OCR Decision ====================

def decide_ocr_for_page(
    page: Page,
    min_words: int = 100,
    scanned_max_words: int = 50,
    min_text_bearing_images: int = 3,
    min_text_bearing_ratio: float = 0.5,
) -> Tuple[OCRAction, str]:
    """
    Decide if OCR is needed for a single page.
    
    Returns:
        Tuple of (OCRAction, reason_string)
    """
    content = page.content
    if not content:
        return OCRAction.FULL_PAGE_OCR, "No content extracted"
    
    word_count = content.word_count
    text_bearing_images = len([
        img for img in content.images
        if _is_text_bearing(img, min_text_bearing_images)
    ])
    total_images = len(content.images)
    
    # Calculate ratios
    text_bearing_ratio = (
        text_bearing_images / total_images if total_images > 0 else 0.0
    )
    
    # Decision logic
    
    # Case 1: Very little text - likely scanned document
    if word_count < scanned_max_words:
        return (
            OCRAction.FULL_PAGE_OCR,
            f"Low word count ({word_count} < {scanned_max_words})"
        )
    
    # Case 2: Good text but many text-bearing images
    if word_count >= min_words:
        if (text_bearing_images >= min_text_bearing_images or
            text_bearing_ratio >= min_text_bearing_ratio):
            return (
                OCRAction.OCR_IMAGES_ONLY,
                f"Good text ({word_count} words) + {text_bearing_images} text images"
            )
        else:
            return (
                OCRAction.SKIP_OCR,
                f"Sufficient text ({word_count} words), no significant images"
            )
    
    # Case 3: Moderate text with images - check if images dominate
    if text_bearing_ratio > 0.7:
        return (
            OCRAction.FULL_PAGE_OCR,
            f"Image-heavy page ({text_bearing_ratio:.0%} text-bearing)"
        )
    
    # Case 4: Moderate text, some images
    if text_bearing_images > 0:
        return (
            OCRAction.OCR_IMAGES_ONLY,
            f"Moderate text ({word_count} words) + some images"
        )
    
    # Default: Skip OCR
    return (
        OCRAction.SKIP_OCR,
        f"Moderate text ({word_count} words), no text-bearing images"
    )


def _is_text_bearing(img: ImageInfo, min_text_bearing: int = 3) -> bool:
    """Check if image likely contains text."""
    # Large images are more likely to contain text
    if img.area > 50000:
        return True
    
    # Wide/tall images (tables, charts)
    if img.aspect_ratio > 2.0 or img.aspect_ratio < 0.5:
        return True
    
    return False


# ==================== Chunking ====================

def chunk_page_text(
    page: Page,
    min_words: int = 100,
    max_words: int = 512,
    overlap_words: int = 50,
    filter_language: bool = True,
) -> List[Dict[str, Any]]:
    """
    Chunk page text into high-quality semantic chunks.
    
    Creates overlapping chunks to preserve context and ensure no information is lost.
    Each chunk is self-contained and meaningful.
    
    Args:
        page: Page with extracted content
        min_words: Minimum words per chunk (default: 100 for quality)
        max_words: Maximum words per chunk (default: 512 for embeddings)
        overlap_words: Overlap between chunks (default: 50 to preserve context)
        filter_language: Filter out non-English chunks (default: True)
        
    Returns:
        List of semantic chunks with metadata (English only if filter_language=True)
    """
    if not page.content or not page.content.text:
        return []
    
    text = page.content.text
    doc_id = _get_doc_id_from_page(page)
    
    # Split into sentences
    sentences = _split_into_sentences(text)
    
    # Group into semantic chunks with overlap
    chunk_texts = _group_sentences(sentences, min_words, max_words, overlap_words)
    
    chunks = []
    filtered_count = 0
    
    for i, chunk_text in enumerate(chunk_texts):
        chunk_id = f"{doc_id}_chunk_{i}"
        
        # Filter non-English chunks if enabled
        if filter_language and not _is_english(chunk_text):
            filtered_count += 1
            logger.debug(f"Filtered non-English chunk from {page.url}")
            continue
        
        chunk = {
            "id": chunk_id,
            "text": chunk_text.strip(),
            "doc_id": doc_id,
            "url": page.url,
            "chunk_index": len(chunks),  # Use actual index after filtering
            "word_count": len(chunk_text.split()),
            "char_count": len(chunk_text),
            "source": page.content.source.value,
        }
        chunks.append(chunk)
    
    if filtered_count > 0:
        logger.info(f"Filtered {filtered_count} non-English chunks from {page.url}, kept {len(chunks)} English chunks")
    
    return chunks


def _split_into_sentences(text: str) -> List[str]:
    """Split text into sentences."""
    # Handle common abbreviations
    text = re.sub(r'\b(Dr|Mr|Mrs|Ms|Prof|Inc|Ltd|Jr|Sr)\.',
                  r'\1<PERIOD>', text)
    
    # Split on sentence endings
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Restore abbreviations
    sentences = [s.replace('<PERIOD>', '.') for s in sentences]
    
    return [s.strip() for s in sentences if s.strip()]


def _group_sentences(
    sentences: List[str],
    min_words: int,
    max_words: int,
    overlap: int,
) -> List[str]:
    """Group sentences into chunks."""
    chunks = []
    current_chunk = []
    current_words = 0
    
    for sentence in sentences:
        sentence_words = len(sentence.split())
        
        # Check if adding this sentence exceeds max
        if current_words + sentence_words > max_words and current_words >= min_words:
            # Save current chunk
            chunks.append(" ".join(current_chunk))
            
            # Start new chunk with overlap
            overlap_text = _get_overlap(current_chunk, overlap)
            current_chunk = [overlap_text] if overlap_text else []
            current_words = len(overlap_text.split()) if overlap_text else 0
        
        current_chunk.append(sentence)
        current_words += sentence_words
    
    # Don't forget the last chunk
    if current_chunk and current_words >= min_words // 2:
        chunks.append(" ".join(current_chunk))
    
    return chunks


def _get_overlap(sentences: List[str], target_words: int) -> str:
    """Get overlap text from end of sentences."""
    if not sentences:
        return ""
    
    overlap_sentences = []
    word_count = 0
    
    for sentence in reversed(sentences):
        words = len(sentence.split())
        if word_count + words <= target_words:
            overlap_sentences.insert(0, sentence)
            word_count += words
        else:
            break
    
    return " ".join(overlap_sentences)


def _get_doc_id_from_page(page: Page) -> str:
    """Generate document ID from page URL."""
    from urllib.parse import urlparse
    
    parsed = urlparse(page.url)
    path = parsed.path.replace("/", "_").strip("_")
    return f"{parsed.netloc}_{path}"[:100]  # Limit length


def _is_english(text: str) -> bool:
    """
    Check if text is in English using language detection.
    
    Args:
        text: Text to check
        
    Returns:
        True if English or detection fails (conservative), False otherwise
    """
    try:
        from langdetect import detect, LangDetectException
    except ImportError:
        # If langdetect not installed, assume English (no filtering)
        logger.debug("langdetect not installed, skipping language filtering")
        return True
    
    # Skip very short text (likely technical content, headers, etc.)
    if len(text.strip()) < 30:
        return True
    
    try:
        lang = detect(text)
        return lang == 'en'
    except LangDetectException:
        # If detection fails, assume English (conservative)
        return True
