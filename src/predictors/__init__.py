"""
预测器模块

包含：
- HybridPredictor: Stage 1 接触角预测器
- PINNAperturePredictor: Stage 3 PINN 开口率预测器
"""

from .hybrid_predictor import HybridPredictor

# 延迟导入 PINN 预测器（避免循环依赖）
def get_pinn_aperture_predictor():
    """获取 PINN 开口率预测器"""
    from .pinn_aperture import PINNAperturePredictor
    return PINNAperturePredictor

__all__ = [
    "HybridPredictor",
    "get_pinn_aperture_predictor",
]
