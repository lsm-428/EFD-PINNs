"""Dashboard module for EFD3D visualization and interaction.

This module provides dashboard components for interactive exploration
and visualization of EFD3D simulation results.
"""

from src.dashboard.datastore import DataStore
from src.dashboard.model_manager import (
    ModelManager,
    get_default_manager,
    clear_default_manager,
)
from src.dashboard.training_output_analyzer import (
    TrainingOutputScanner,
    TrainingRunInfo,
    TrainingConfigParser,
    LossDataParser,
    MetricsParser,
    ModelLoader,
    ModelInfo,
    TrainingOutputAnalyzer,
)

__all__ = [
    "DataStore",
    "ModelManager",
    "get_default_manager",
    "clear_default_manager",
    "TrainingOutputScanner",
    "TrainingRunInfo",
    "TrainingConfigParser",
    "LossDataParser",
    "MetricsParser",
    "ModelLoader",
    "ModelInfo",
    "TrainingOutputAnalyzer",
]
