"""
LSTM-PINN 配置加载器

加载和验证 LSTM-PINN 配置文件
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class LSTMConfig:
    """LSTM 编码器配置"""
    input_dim: int = 2
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    bidirectional: bool = False


@dataclass
class PhiDecoderConfig:
    """φ 解码器配置"""
    spatial_dim: int = 3
    hidden_layers: list = field(default_factory=lambda: [128, 64, 32])
    activation: str = "tanh"
    use_skip_connections: bool = False


@dataclass
class VelocityDecoderConfig:
    """速度解码器配置"""
    enabled: bool = False
    hidden_layers: list = field(default_factory=lambda: [64, 32])
    activation: str = "tanh"


@dataclass
class SequenceConfig:
    """序列配置"""
    length: int = 50
    dt: float = 0.001
    t_max: float = 0.05


@dataclass
class TrainingConfig:
    """训练配置"""
    epochs: int = 30000
    batch_size: int = 1024
    optimizer: str = "adam"
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    lr_scheduler: str = "cosine"
    warmup_epochs: int = 500
    min_lr: float = 1e-6
    early_stopping_patience: int = 2000
    gradient_clip: float = 1.0
    mixed_precision: bool = True


@dataclass
class MaterialsConfig:
    """材料参数配置"""
    theta0: float = 120.0
    theta_30V: float = 67.5
    theta_wall: float = 71.0
    epsilon_r: float = 12.0
    epsilon_hydrophobic: float = 1.9
    gamma: float = 0.015
    dielectric_thickness: float = 4e-7
    hydrophobic_thickness: float = 4e-7
    V_threshold: float = 3.0
    aperture_max: float = 0.85


@dataclass
class DynamicsConfig:
    """动力学参数配置"""
    tau: float = 0.005
    tau_recovery: float = 0.0075
    zeta: float = 0.8


@dataclass
class GeometryConfig:
    """几何参数配置"""
    Lx: float = 0.000174
    Ly: float = 0.000174
    Lz: float = 2e-5
    pixel_size: float = 0.000174
    ink_thickness: float = 3e-6
    wall_height: float = 3.5e-6


class LSTMPINNConfig:
    """
    LSTM-PINN 完整配置
    
    从 JSON 配置文件加载并验证配置
    """
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        初始化配置
        
        Args:
            config_dict: 配置字典，如果为 None 则使用默认值
        """
        if config_dict is None:
            config_dict = {}
        
        self._raw_config = config_dict
        
        # 解析各部分配置
        self.lstm = self._parse_lstm_config(config_dict.get("lstm_model", {}).get("lstm", {}))
        self.phi_decoder = self._parse_phi_decoder_config(
            config_dict.get("lstm_model", {}).get("phi_decoder", {})
        )
        self.velocity_decoder = self._parse_velocity_decoder_config(
            config_dict.get("lstm_model", {}).get("velocity_decoder", {})
        )
        self.sequence = self._parse_sequence_config(config_dict.get("sequence", {}))
        self.training = self._parse_training_config(config_dict.get("training", {}))
        self.materials = self._parse_materials_config(config_dict.get("materials", {}))
        self.dynamics = self._parse_dynamics_config(config_dict.get("dynamics", {}))
        self.geometry = self._parse_geometry_config(config_dict.get("geometry", {}))
        
        # 其他配置
        self.data = config_dict.get("data", {})
        self.physics_loss = config_dict.get("physics_loss", {})
        self.normalization = config_dict.get("normalization", {})
        self.output = config_dict.get("output", {})
        self.seed = config_dict.get("seed", 42)
        self.deterministic = config_dict.get("deterministic", True)
    
    def _parse_lstm_config(self, config: Dict) -> LSTMConfig:
        return LSTMConfig(
            input_dim=config.get("input_dim", 2),
            hidden_dim=config.get("hidden_dim", 128),
            num_layers=config.get("num_layers", 2),
            dropout=config.get("dropout", 0.1),
            bidirectional=config.get("bidirectional", False)
        )
    
    def _parse_phi_decoder_config(self, config: Dict) -> PhiDecoderConfig:
        return PhiDecoderConfig(
            spatial_dim=config.get("spatial_dim", 3),
            hidden_layers=config.get("hidden_layers", [128, 64, 32]),
            activation=config.get("activation", "tanh"),
            use_skip_connections=config.get("use_skip_connections", False)
        )
    
    def _parse_velocity_decoder_config(self, config: Dict) -> VelocityDecoderConfig:
        return VelocityDecoderConfig(
            enabled=config.get("enabled", False),
            hidden_layers=config.get("hidden_layers", [64, 32]),
            activation=config.get("activation", "tanh")
        )
    
    def _parse_sequence_config(self, config: Dict) -> SequenceConfig:
        return SequenceConfig(
            length=config.get("length", 50),
            dt=config.get("dt", 0.001),
            t_max=config.get("t_max", 0.05)
        )
    
    def _parse_training_config(self, config: Dict) -> TrainingConfig:
        return TrainingConfig(
            epochs=config.get("epochs", 30000),
            batch_size=config.get("batch_size", 1024),
            optimizer=config.get("optimizer", "adam"),
            learning_rate=config.get("learning_rate", 0.001),
            weight_decay=config.get("weight_decay", 1e-5),
            lr_scheduler=config.get("lr_scheduler", "cosine"),
            warmup_epochs=config.get("warmup_epochs", 500),
            min_lr=config.get("min_lr", 1e-6),
            early_stopping_patience=config.get("early_stopping_patience", 2000),
            gradient_clip=config.get("gradient_clip", 1.0),
            mixed_precision=config.get("mixed_precision", True)
        )
    
    def _parse_materials_config(self, config: Dict) -> MaterialsConfig:
        return MaterialsConfig(
            theta0=config.get("theta0", 120.0),
            theta_30V=config.get("theta_30V", 67.5),
            theta_wall=config.get("theta_wall", 71.0),
            epsilon_r=config.get("epsilon_r", 12.0),
            epsilon_hydrophobic=config.get("epsilon_hydrophobic", 1.9),
            gamma=config.get("gamma", 0.015),
            dielectric_thickness=config.get("dielectric_thickness", 4e-7),
            hydrophobic_thickness=config.get("hydrophobic_thickness", 4e-7),
            V_threshold=config.get("V_threshold", 3.0),
            aperture_max=config.get("aperture_max", 0.85)
        )
    
    def _parse_dynamics_config(self, config: Dict) -> DynamicsConfig:
        return DynamicsConfig(
            tau=config.get("tau", 0.005),
            tau_recovery=config.get("tau_recovery", 0.0075),
            zeta=config.get("zeta", 0.8)
        )
    
    def _parse_geometry_config(self, config: Dict) -> GeometryConfig:
        return GeometryConfig(
            Lx=config.get("Lx", 0.000174),
            Ly=config.get("Ly", 0.000174),
            Lz=config.get("Lz", 2e-5),
            pixel_size=config.get("pixel_size", 0.000174),
            ink_thickness=config.get("ink_thickness", 3e-6),
            wall_height=config.get("wall_height", 3.5e-6)
        )
    
    def to_model_config(self) -> Dict[str, Any]:
        """
        转换为模型初始化所需的配置字典
        
        Returns:
            模型配置字典
        """
        return {
            "lstm": {
                "input_dim": self.lstm.input_dim,
                "hidden_dim": self.lstm.hidden_dim,
                "num_layers": self.lstm.num_layers,
                "dropout": self.lstm.dropout,
                "bidirectional": self.lstm.bidirectional
            },
            "phi_decoder": {
                "spatial_dim": self.phi_decoder.spatial_dim,
                "hidden_layers": self.phi_decoder.hidden_layers,
                "activation": self.phi_decoder.activation,
                "use_skip_connections": self.phi_decoder.use_skip_connections
            },
            "velocity_decoder": {
                "enabled": self.velocity_decoder.enabled,
                "hidden_layers": self.velocity_decoder.hidden_layers,
                "activation": self.velocity_decoder.activation
            }
        }
    
    def get_raw_config(self) -> Dict[str, Any]:
        """返回原始配置字典"""
        return self._raw_config
    
    @classmethod
    def from_json(cls, config_path: str) -> "LSTMPINNConfig":
        """
        从 JSON 文件加载配置
        
        Args:
            config_path: 配置文件路径
        
        Returns:
            LSTMPINNConfig 实例
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        return cls(config_dict)
    
    def save_json(self, config_path: str):
        """
        保存配置到 JSON 文件
        
        Args:
            config_path: 保存路径
        """
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self._raw_config, f, indent=2, ensure_ascii=False)


def load_lstm_pinn_config(config_path: str) -> LSTMPINNConfig:
    """
    加载 LSTM-PINN 配置的便捷函数
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        LSTMPINNConfig 实例
    """
    return LSTMPINNConfig.from_json(config_path)
