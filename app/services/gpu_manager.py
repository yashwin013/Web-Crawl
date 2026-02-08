"""
GPU Manager for Surya OCR models.

Provides centralized management of Surya detection and recognition predictors
with GPU optimization and caching.
"""

import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Global model cache
_foundation_predictor = None
_det_predictor = None
_rec_predictor = None


def get_surya_predictors() -> Tuple:
    """
    Get Surya detection and recognition predictors.
    
    Uses global singleton pattern to avoid reloading models.
    
    Returns:
        Tuple of (detection_predictor, recognition_predictor)
    """
    global _foundation_predictor, _det_predictor, _rec_predictor
    
    if _det_predictor is None or _rec_predictor is None:
        import torch
        
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing Surya OCR models on {device.upper()}...")
        
        try:
            # Try the new API first (surya >= 0.7.0)
            try:
                from surya.recognition import RecognitionPredictor
                from surya.detection import DetectionPredictor
                from surya.foundation import FoundationPredictor
                
                logger.info("Loading Surya OCR models (new API)...")
                
                # Initialize with explicit device parameter if supported
                try:
                    if _foundation_predictor is None:
                        _foundation_predictor = FoundationPredictor(device=device)
                    _det_predictor = DetectionPredictor(device=device)
                    _rec_predictor = RecognitionPredictor(_foundation_predictor, device=device)
                except TypeError:
                    # If device parameter not supported, try without
                    logger.info("Device parameter not supported, initializing without explicit device...")
                    if _foundation_predictor is None:
                        _foundation_predictor = FoundationPredictor()
                    _det_predictor = DetectionPredictor()
                    _rec_predictor = RecognitionPredictor(_foundation_predictor)
                
                logger.info("Surya OCR models loaded successfully (new API)")
                
            except (ImportError, TypeError) as e:
                # Fall back to old API (surya < 0.7.0)
                logger.info(f"New API failed ({e}), trying legacy API...")
                from surya.recognition import RecognitionPredictor
                from surya.detection import DetectionPredictor
                
                logger.info("Loading Surya OCR models (legacy API)...")
                
                try:
                    _det_predictor = DetectionPredictor(device=device)
                    _rec_predictor = RecognitionPredictor(device=device)
                except TypeError:
                    # If device parameter not supported
                    _det_predictor = DetectionPredictor()
                    _rec_predictor = RecognitionPredictor()
                
                logger.info("Surya OCR models loaded successfully (legacy API)")
            
            # Verify models are properly loaded and move to device if needed
            if hasattr(_det_predictor, 'model') and hasattr(_det_predictor.model, 'to'):
                _det_predictor.model = _det_predictor.model.to(device)
            if hasattr(_rec_predictor, 'model') and hasattr(_rec_predictor.model, 'to'):
                _rec_predictor.model = _rec_predictor.model.to(device)
                
        except ImportError as e:
            logger.error(f"Surya OCR not available: {e}")
            raise ImportError(
                "Surya OCR is required but not installed. "
                "Install with: pip install surya-ocr"
            ) from e
        except Exception as e:
            logger.error(f"Failed to initialize Surya OCR models: {e}", exc_info=True)
            # Try to fallback to CPU
            if device == "cuda":
                logger.warning("GPU initialization failed, falling back to CPU...")
                device = "cpu"
                _foundation_predictor = None
                _det_predictor = None
                _rec_predictor = None
                # Don't recurse, just raise the error
            raise
    
    return _det_predictor, _rec_predictor


def clear_models() -> None:
    """Clear cached models to free GPU memory."""
    global _foundation_predictor, _det_predictor, _rec_predictor
    
    _foundation_predictor = None
    _det_predictor = None
    _rec_predictor = None
    
    import gc
    import torch
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    logger.info("Cleared Surya OCR models from memory")
