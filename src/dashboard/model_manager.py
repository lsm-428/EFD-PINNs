"""Model Manager for EFD3D Dashboard.

This module provides a ModelManager class that handles model loading with
LRU caching to optimize memory usage when switching between different
checkpoint models in the dashboard.
"""

import os
import torch
from typing import Dict, Optional, Any
from pathlib import Path
from functools import lru_cache

from src.models.pinn_two_phase import TwoPhasePINN
from src.dashboard.inference import PINNInferenceEngine


class ModelManager:
    """
    Manages model loading with LRU caching for the dashboard.
    
    Features:
    - Automatic discovery of latest model in outputs/ directory
    - LRU cache with max 5 models to prevent OOM
    - Lazy loading - models only loaded when requested
    """
    
    MAX_CACHED_MODELS = 5
    
    def __init__(self, outputs_dir: Optional[str] = None):
        """
        Initialize ModelManager.
        
        Args:
            outputs_dir: Path to outputs directory containing checkpoints.
                        Defaults to PROJECT_ROOT/outputs
        """
        if outputs_dir is None:
            outputs_dir = str(Path(__file__).resolve().parent.parent / "outputs")
        
        self.outputs_dir = Path(outputs_dir)
        self._cache: Dict[str, PINNInferenceEngine] = {}
        self._access_order: list = []  # LRU tracking
        
        # Discover available models
        self._available_models = self._discover_models()
    
    def _discover_models(self) -> list:
        """
        Recursively discover all .pth/.pt checkpoint files in outputs_dir.
        
        Returns:
            List of (path, mtime) tuples, sorted by modification time (newest first)
        """
        model_files = []
        
        for ext in ("*.pth", "*.pt"):
            for path in self.outputs_dir.rglob(ext):
                try:
                    mtime = path.stat().st_mtime
                    model_files.append((str(path), mtime))
                except OSError:
                    continue
        
        # Sort by modification time, newest first
        model_files.sort(key=lambda x: x[1], reverse=True)
        return model_files
    
    def get_available_models(self) -> list:
        """
        Get list of available models with metadata.
        
        Returns:
            List of dicts with 'path', 'name', 'mtime' keys
        """
        models = []
        for path, mtime in self._available_models:
            p = Path(path)
            models.append({
                "path": path,
                "name": f"{p.parent.name}/{p.name}",
                "mtime": mtime,
                "ctime": mtime,  # alias for compatibility
            })
        return models
    
    def get_latest_model(self) -> Optional[str]:
        """
        Get path to the latest available model.
        
        Returns:
            Path to latest model, or None if no models found
        """
        if not self._available_models:
            return None
        return self._available_models[0][0]
    
    def _evict_oldest(self) -> None:
        """Evict oldest model from cache to make room."""
        if not self._cache:
            return
        
        # LRU: remove least recently accessed
        oldest = self._access_order.pop(0)
        if oldest in self._cache:
            del self._cache[oldest]
    
    def _update_access_order(self, path: str) -> None:
        """Update access order for LRU tracking."""
        if path in self._access_order:
            self._access_order.remove(path)
        self._access_order.append(path)
    
    def load_model(
        self, 
        path: Optional[str] = None, 
        device: str = "auto"
    ) -> PINNInferenceEngine:
        """
        Load a model from checkpoint with LRU caching.
        
        Args:
            path: Path to checkpoint. If None, loads latest model.
            device: Device to load model on ("auto", "cpu", or "cuda")
        
        Returns:
            PINNInferenceEngine instance
            
        Raises:
            FileNotFoundError: If checkpoint not found
            RuntimeError: If model loading fails
        """
        if path is None:
            path = self.get_latest_model()
            if path is None:
                raise FileNotFoundError(
                    f"No model checkpoints found in {self.outputs_dir}"
                )
        
        # Check if already cached
        if path in self._cache:
            self._update_access_order(path)
            return self._cache[path]
        
        # Evict if cache full
        while len(self._cache) >= self.MAX_CACHED_MODELS:
            self._evict_oldest()
        
        # Load model
        try:
            # Use device resolution compatible with dashboard
            if device == "auto":
                actual_device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                actual_device = device
            
            engine = PINNInferenceEngine(path, device=actual_device)
            self._cache[path] = engine
            self._update_access_order(path)
            
            return engine
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {path}: {e}")
    
    def clear_cache(self) -> None:
        """Clear all cached models."""
        self._cache.clear()
        self._access_order.clear()
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dict with 'size', 'max_size', 'models' keys
        """
        return {
            "size": len(self._cache),
            "max_size": self.MAX_CACHED_MODELS,
            "models": list(self._cache.keys()),
            "access_order": self._access_order.copy(),
        }


# Convenience singleton for dashboard use
_default_manager: Optional[ModelManager] = None


def get_default_manager() -> ModelManager:
    """
    Get or create the default ModelManager instance.
    
    Returns:
        ModelManager instance
    """
    global _default_manager
    if _default_manager is None:
        _default_manager = ModelManager()
    return _default_manager


def clear_default_manager() -> None:
    """Clear the default manager instance (for testing)."""
    global _default_manager
    _default_manager = None
