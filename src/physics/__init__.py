"""
物理约束模块

包含物理约束层

注意：ElectrowettingPhysicsGenerator 已废弃，
现在使用 src.models.pinn_two_phase.DataGenerator
"""

from .constraints import PhysicsConstraints

__all__ = [
    "PhysicsConstraints",
]
