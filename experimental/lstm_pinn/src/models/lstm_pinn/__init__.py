"""
LSTM-MLP-PINN 模块

用于电润湿像素的两相流动态模拟，支持任意电压序列和多步跳变。

架构:
- VoltageEncoder: LSTM 编码器，处理电压时间序列
- PhiDecoder: MLP 解码器，将空间坐标和隐状态映射到 φ 值
- LSTMPINNModel: 主模型，组合编码器和解码器
"""

from .encoder import VoltageEncoder
from .decoder import PhiDecoder, VelocityDecoder
from .model import LSTMPINNModel
from .config import (
    LSTMPINNConfig,
    load_lstm_pinn_config,
    LSTMConfig,
    PhiDecoderConfig,
    VelocityDecoderConfig,
    SequenceConfig,
    TrainingConfig,
    MaterialsConfig,
    DynamicsConfig,
    GeometryConfig,
)
from .physics import ElectrowettingPhysics
from .data_generator import SequenceDataGenerator
from .physics_loss import LSTMPINNPhysicsLoss, SimplifiedPhysicsLoss
from .trainer import LSTMPINNTrainer, train_lstm_pinn
from .response_time import ResponseTimeCalculator, compute_response_time
from .visualization import LSTMPINNVisualizer
from .hybrid_model import LSTMHybridPINN

__all__ = [
    "VoltageEncoder",
    "PhiDecoder",
    "VelocityDecoder",
    "LSTMPINNModel",
    "LSTMPINNConfig",
    "load_lstm_pinn_config",
    "LSTMConfig",
    "PhiDecoderConfig",
    "VelocityDecoderConfig",
    "SequenceConfig",
    "TrainingConfig",
    "MaterialsConfig",
    "DynamicsConfig",
    "GeometryConfig",
    "ElectrowettingPhysics",
    "SequenceDataGenerator",
    "LSTMPINNPhysicsLoss",
    "SimplifiedPhysicsLoss",
    "LSTMPINNTrainer",
    "train_lstm_pinn",
    "ResponseTimeCalculator",
    "compute_response_time",
    "LSTMPINNVisualizer",
    "LSTMHybridPINN",
]
