"""
预测器模块

包含：
- HybridPredictor: Stage 1 接触角预测器
- PINNAperturePredictor: Stage 3 PINN 开口率预测器

注意: LSTM-Hybrid-PINN 预测器已移至 experimental/lstm_pinn/
"""

from .hybrid_predictor import HybridPredictor


def get_pinn_aperture_predictor():
    """获取 PINN 开口率预测器"""
    from .pinn_aperture import PINNAperturePredictor

    return PINNAperturePredictor


__all__ = [
    "HybridPredictor",
    "get_pinn_aperture_predictor",
]
