"""
物理约束模块

包含物理约束层、数据生成器和接触角计算
"""

from .constraints import PhysicsConstraints
from .data_generator import ElectrowettingPhysicsGenerator
from .contact_angle import ContactAngleLoss

__all__ = [
    "PhysicsConstraints",
    "ElectrowettingPhysicsGenerator",
    "ContactAngleLoss",
]
