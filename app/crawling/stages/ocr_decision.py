"""
OCR decision stage - determines if OCR is needed.
"""

from typing import Tuple

from app.crawling.stages.base import PipelineStage
from app.crawling.models.document import (
    Document, Page, PageContent, OCRAction, ImageInfo, ContentSource
)

from app.config import get_logger

logger = get_logger(__name__)


class ImageType:
    """Image classification types."""
    DECORATIVE = "decorative"
    TABLE = "table"
    CHART = "chart"
    SCANNED_TEXT = "scanned_text"
    UNKNOWN = "unknown"


class OCRDecisionStage(PipelineStage):
    """
    OCR decision stage.
    
    Analyzes each page to determine whether OCR is needed:
    - SKIP_OCR: Sufficient text extracted from DOM
    - OCR_IMAGES_ONLY: Text is good, but images may contain text
    - FULL_PAGE_OCR: Page appears to be scanned, needs full OCR
    
    This stage does NOT perform OCR - it just sets the decision.
    """
    
    @property
    def name(self) -> str:
        return "ocr_decision"
    
    async def process(self, document: Document) -> Document:
        """
        Determine OCR action for each page.
        
        Args:
            document: Document with extracted text
            
        Returns:
            Document with ocr_action set for each page
        """
        ocr_config = self.config.ocr if self.config else None
        
        for page in document.pages:
            try:
                action, reason = self._decide_ocr_action(page, ocr_config)
                page.ocr_action = action
                page.ocr_reason = reason
                
                logger.debug(f"{page.url}: {action.value} - {reason}")
                
            except Exception as e:
                logger.error(f"OCR decision failed for {page.url}: {e}")
                page.ocr_action = OCRAction.FULL_PAGE_OCR
                page.ocr_reason = f"Decision failed: {e}"
        
        # Log summary
        actions = [p.ocr_action for p in document.pages if p.ocr_action]
        skip_count = sum(1 for a in actions if a == OCRAction.SKIP_OCR)
        images_count = sum(1 for a in actions if a == OCRAction.OCR_IMAGES_ONLY)
        full_count = sum(1 for a in actions if a == OCRAction.FULL_PAGE_OCR)
        
        logger.info(
            f"OCR decisions: {skip_count} skip, "
            f"{images_count} images-only, {full_count} full-page"
        )
        
        return document
    
    def _decide_ocr_action(self, page: Page, config) -> Tuple[OCRAction, str]:
        """Decide OCR action for a single page."""
        
        content = page.content
        if not content:
            return OCRAction.FULL_PAGE_OCR, "No content extracted"
        
        word_count = content.word_count
        text_bearing_images = len([
            img for img in content.images
            if self._is_text_bearing(img, config)
        ])
        total_images = len(content.images)
        
        # Import conservative thresholds from config
        from app.config import (
            OCR_MIN_WORD_COUNT_SUFFICIENT,
            OCR_SCANNED_PDF_MAX_WORDS,
            OCR_MIN_TEXT_BEARING_IMAGES,
            OCR_MIN_TEXT_BEARING_RATIO,
        )
        
        # Thresholds (use config values for conservative OCR usage)
        min_text_bearing = OCR_MIN_TEXT_BEARING_IMAGES
        min_text_ratio = OCR_MIN_TEXT_BEARING_RATIO
        min_words = OCR_MIN_WORD_COUNT_SUFFICIENT
        scanned_max_words = OCR_SCANNED_PDF_MAX_WORDS
        
        # Calculate ratios
        text_bearing_ratio = text_bearing_images / total_images if total_images > 0 else 0
        significant_text_images = (
            text_bearing_images >= min_text_bearing or
            (total_images > 0 and text_bearing_ratio >= min_text_ratio)
        )
        
        # SCANNED PDF PATTERN: Low text + large images
        is_scanned_pattern = (
            word_count < scanned_max_words and
            total_images > 0 and
            text_bearing_images > 0
        )
        
        if is_scanned_pattern:
            return (
                OCRAction.FULL_PAGE_OCR,
                f"Scanned pattern: {word_count} words, {text_bearing_images} large images"
            )
        
        # Case 1: No text at all
        if word_count == 0:
            return OCRAction.FULL_PAGE_OCR, "No text extracted"
        
        # Case 2: Sufficient text
        if word_count >= min_words:
            if significant_text_images:
                return (
                    OCRAction.OCR_IMAGES_ONLY,
                    f"Text OK ({word_count} words), {text_bearing_images} text images"
                )
            else:
                return (
                    OCRAction.SKIP_OCR,
                    f"Text sufficient ({word_count} words)"
                )
        
        # Case 3: Low text count
        if word_count >= 30:
            # Short document
            if significant_text_images:
                return (
                    OCRAction.OCR_IMAGES_ONLY,
                    f"Short doc ({word_count} words), has text images"
                )
            else:
                return (
                    OCRAction.SKIP_OCR,
                    f"Short doc ({word_count} words), no significant images"
                )
        else:
            # Very low text
            if total_images > 0:
                return (
                    OCRAction.FULL_PAGE_OCR,
                    f"Very low text ({word_count} words) with images"
                )
            else:
                return (
                    OCRAction.SKIP_OCR,
                    f"Sparse text-only page ({word_count} words)"
                )
    
    def _is_text_bearing(self, image: ImageInfo, config) -> bool:
        """Check if image might contain text worth OCRing."""
        from app.config import OCR_MIN_TEXT_BEARING_AREA, OCR_DECORATIVE_MAX_SIZE
        
        min_area = OCR_MIN_TEXT_BEARING_AREA
        decorative_size = OCR_DECORATIVE_MAX_SIZE
        
        # Too small
        if image.width < decorative_size and image.height < decorative_size:
            return False
        
        # Area too small
        if image.area < min_area:
            return False
        
        # Banner/sidebar
        if image.aspect_ratio > 4.0 or image.aspect_ratio < 0.25:
            return False
        
        # Square icons
        if 0.9 <= image.aspect_ratio <= 1.1 and image.width < 400:
            return False
        
        return True
    
    @staticmethod
    def classify_image(width: int, height: int) -> str:
        """Classify image type based on dimensions."""
        aspect_ratio = width / height if height > 0 else 1.0
        area = width * height
        
        # Very small = decorative
        if width < 150 and height < 150:
            return ImageType.DECORATIVE
        
        # Lowered from 300 to 200 based on user feedback
        if width < 200 and height < 200:
            return ImageType.DECORATIVE
        
        # Lowered from 200k to 50k
        if area < 50000:
            return ImageType.DECORATIVE
        
        # Square icons
        if 0.9 <= aspect_ratio <= 1.1 and width < 250:
            return ImageType.DECORATIVE
        
        # Banners
        if aspect_ratio > 5.0 or aspect_ratio < 0.2:
            return ImageType.DECORATIVE
        
        # Table-like
        if 0.5 <= aspect_ratio <= 2.5 and width > 400 and height > 300:
            return ImageType.TABLE
        
        # Chart-like
        if 1.0 <= aspect_ratio <= 2.0 and width > 500:
            return ImageType.CHART
        
        # Large = likely scanned
        if width > 600 and height > 400:
            return ImageType.SCANNED_TEXT
        
        return ImageType.UNKNOWN
