"""
EFD-PINNs 源代码包

Physics-Informed Neural Networks for Electrowetting Display Dynamics
"""

from . import models
from . import physics
from . import training
from . import predictors
from . import solvers
from . import visualization
from . import utils

__version__ = "1.0.0"
__all__ = [
    "models",
    "physics", 
    "training",
    "predictors",
    "solvers",
    "visualization",
    "utils",
]
