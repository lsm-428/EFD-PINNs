"""DataStore state management for EFD3D dashboard.

This module implements a reactive state management system compatible with
libraries like Panel(param.Parameterized) when available.

For environments without param:
- DataStoreBase provides basic state management without param
- If param is available, DataStore extends param.Parameterized for reactive updates
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional


class DataStoreBase:
    """Base state store with basic functionality.

    Manages shared state including model paths, current simulation parameters,
    and cached inference results.

    Attributes:
        model_paths: List of available model paths
        current_model_path: Currently selected model path
        current_parameters: Current simulation parameters (x, y, z, V_from, V_to, t_since)
        cached_results: Cached inference results
        last_updated: Timestamp of last state update
    """

    def __init__(self) -> None:
        """Initialize DataStore with default state."""
        # Model paths state
        self.model_paths: List[str] = []

        self.current_model_path: Optional[str] = None

        # Simulation parameters (6D Triad format: [x, y, z, V_from, V_to, t_since])
        self.current_parameters: Dict[str, float] = {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "V_from": 0.0,
            "V_to": 0.0,
            "t_since": 0.0,
        }

        # Cached results
        self.cached_results: Dict[str, Any] = {}

        # Timestamp for updates
        self.last_updated: float = time.time()

    def update_model_paths(self, paths: List[str]) -> None:
        """Update available model paths.

        Args:
            paths: List of model checkpoint file paths
        """
        self.model_paths = list(paths)
        self.last_updated = time.time()

    def update_parameters(self, **kwargs: float) -> None:
        """Update simulation parameters.

        Args:
            kwargs: Parameter key-value pairs (x, y, z, V_from, V_to, t_since)
        """
        self.current_parameters.update(kwargs)
        self.last_updated = time.time()

    def cache_results(self, key: str, results: Any) -> None:
        """Cache inference results.

        Args:
            key: Cache key (typically parameter hash)
            results: Inference results to cache
        """
        self.cached_results[key] = results
        self.last_updated = time.time()

    def get_cached(self, key: str) -> Optional[Any]:
        """Get cached results by key.

        Args:
            key: Cache key to retrieve

        Returns:
            Cached results or None if not found
        """
        return self.cached_results.get(key)

    def clear_cache(self) -> None:
        """Clear all cached results."""
        self.cached_results.clear()
        self.last_updated = time.time()

    def reset(self) -> None:
        """Reset all state to defaults."""
        self.model_paths = []
        self.current_model_path = None
        self.current_parameters = {
            "x": 0.0,
            "y": 0.0,
            "z": 0.0,
            "V_from": 0.0,
            "V_to": 0.0,
            "t_since": 0.0,
        }
        self.cached_results = {}
        self.last_updated = time.time()


# Check if param is available for reactive updates
try:
    import param

    _PARAM_AVAILABLE = True

    class DataStore(param.Parameterized, DataStoreBase):
        """Reactive state store for dashboard using param.Parameterized.

        Extends DataStoreBase with reactive parameter updates via param.depends.
        Requires param (Panel) to be installed.

        Attributes:
            model_paths: List of available model paths
            current_model_path: Currently selected model path
            current_parameters: Current simulation parameters
            cached_results: Cached inference results
            last_updated: Timestamp of last state update
        """

        # Override last_updated to be reactive
        last_updated = param.Number(default=0.0, readonly=True)

        # Model paths state
        model_paths = param.List(
            default=[], doc="List of available model checkpoint paths"
        )

        current_model_path = param.String(
            default=None, allow_None=True, doc="Currently selected model path"
        )

        # Simulation parameters (6D Triad format: [x, y, z, V_from, V_to, t_since])
        current_parameters = param.Dict(
            default={
                "x": 0.0,
                "y": 0.0,
                "z": 0.0,
                "V_from": 0.0,
                "V_to": 0.0,
                "t_since": 0.0,
            },
            doc="Current simulation parameters",
        )

        # Cached results
        cached_results = param.Dict(
            default={}, doc="Cached inference results keyed by parameter hash"
        )

        @param.depends("current_model_path", watch=True)
        def _on_model_change(self) -> None:
            """Callback when model path changes."""
            self.last_updated = param.time()

        @param.depends("current_parameters", watch=True)
        def _on_parameters_change(self) -> None:
            """Callback when simulation parameters change."""
            self.last_updated = param.time()

except ImportError:
    _PARAM_AVAILABLE = False
    DataStore = DataStoreBase


__all__ = ["DataStore"]
