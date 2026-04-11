"""
LSTM-MLP-PINN 主模型

组合 LSTM 编码器和 MLP 解码器，用于电润湿像素的两相流动态模拟
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Any

from .encoder import VoltageEncoder
from .decoder import PhiDecoder, VelocityDecoder


class LSTMPINNModel(nn.Module):
    """
    LSTM-PINN 主模型
    
    输入:
        - spatial_coords: (batch, 3) - (x, y, z) 空间坐标
        - voltage_sequence: (batch, seq_len, 1) - 电压时间序列
        - time_sequence: (batch, seq_len, 1) - 对应的时间序列
    
    输出:
        - phi: (batch, 1) - φ 值 [0, 1]
        - velocity: (batch, 3) - 速度场 (u, v, w)，可选
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化 LSTM-PINN 模型
        
        Args:
            config: 配置字典，包含以下键:
                - lstm: LSTM 编码器配置
                - phi_decoder: φ 解码器配置
                - velocity_decoder: 速度解码器配置（可选）
        """
        super().__init__()
        
        self.config = config
        
        # 获取 LSTM 配置
        lstm_config = config.get("lstm", {})
        self.lstm_encoder = VoltageEncoder(
            input_dim=lstm_config.get("input_dim", 2),
            hidden_dim=lstm_config.get("hidden_dim", 128),
            num_layers=lstm_config.get("num_layers", 2),
            dropout=lstm_config.get("dropout", 0.1),
            bidirectional=lstm_config.get("bidirectional", False)
        )
        
        # 获取 φ 解码器配置
        phi_config = config.get("phi_decoder", {})
        self.phi_decoder = PhiDecoder(
            spatial_dim=phi_config.get("spatial_dim", 3),
            hidden_dim=self.lstm_encoder.get_output_dim(),
            hidden_layers=phi_config.get("hidden_layers", [128, 64, 32]),
            activation=phi_config.get("activation", "tanh"),
            use_skip_connections=phi_config.get("use_skip_connections", False)
        )
        
        # 获取速度解码器配置（可选）
        velocity_config = config.get("velocity_decoder", {})
        if velocity_config.get("enabled", False):
            self.velocity_decoder = VelocityDecoder(
                spatial_dim=phi_config.get("spatial_dim", 3),
                hidden_dim=self.lstm_encoder.get_output_dim(),
                hidden_layers=velocity_config.get("hidden_layers", [64, 32]),
                activation=velocity_config.get("activation", "tanh")
            )
        else:
            self.velocity_decoder = None
    
    def forward(
        self,
        spatial_coords: torch.Tensor,
        voltage_sequence: torch.Tensor,
        time_sequence: Optional[torch.Tensor] = None,
        return_velocity: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            spatial_coords: (batch, 3) 空间坐标 (x, y, z)
            voltage_sequence: (batch, seq_len, 1) 电压序列
            time_sequence: (batch, seq_len, 1) 时间序列，可选
            return_velocity: 是否返回速度场
        
        Returns:
            输出字典:
                - phi: (batch, 1) φ 值
                - velocity: (batch, 3) 速度场（如果 return_velocity=True）
                - hidden: (batch, hidden_dim) LSTM 隐状态
        """
        # 构建 LSTM 输入
        if time_sequence is not None:
            # 计算时间步长 dt
            dt = torch.zeros_like(time_sequence)
            dt[:, 1:, :] = time_sequence[:, 1:, :] - time_sequence[:, :-1, :]
            lstm_input = torch.cat([voltage_sequence, dt], dim=-1)
        else:
            # 如果没有时间序列，使用固定 dt
            batch_size, seq_len, _ = voltage_sequence.shape
            dt = torch.ones(batch_size, seq_len, 1, device=voltage_sequence.device)
            lstm_input = torch.cat([voltage_sequence, dt], dim=-1)
        
        # LSTM 编码
        hidden, all_hidden = self.lstm_encoder(lstm_input)
        
        # φ 解码
        phi = self.phi_decoder(spatial_coords, hidden)
        
        # 构建输出
        output = {
            "phi": phi,
            "hidden": hidden
        }
        
        # 速度解码（可选）
        if return_velocity and self.velocity_decoder is not None:
            velocity = self.velocity_decoder(spatial_coords, hidden)
            output["velocity"] = velocity
        
        return output
    
    def predict_phi(
        self,
        spatial_coords: torch.Tensor,
        voltage_sequence: torch.Tensor,
        time_sequence: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        预测 φ 值（简化接口）
        
        Args:
            spatial_coords: (batch, 3) 空间坐标
            voltage_sequence: (batch, seq_len, 1) 电压序列
            time_sequence: (batch, seq_len, 1) 时间序列
        
        Returns:
            phi: (batch, 1) φ 值
        """
        output = self.forward(spatial_coords, voltage_sequence, time_sequence)
        return output["phi"]
    
    def get_hidden_state(
        self,
        voltage_sequence: torch.Tensor,
        time_sequence: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        获取 LSTM 隐状态（用于分析）
        
        Args:
            voltage_sequence: (batch, seq_len, 1) 电压序列
            time_sequence: (batch, seq_len, 1) 时间序列
        
        Returns:
            hidden: (batch, hidden_dim) LSTM 隐状态
        """
        # 构建 LSTM 输入
        if time_sequence is not None:
            dt = torch.zeros_like(time_sequence)
            dt[:, 1:, :] = time_sequence[:, 1:, :] - time_sequence[:, :-1, :]
            lstm_input = torch.cat([voltage_sequence, dt], dim=-1)
        else:
            batch_size, seq_len, _ = voltage_sequence.shape
            dt = torch.ones(batch_size, seq_len, 1, device=voltage_sequence.device)
            lstm_input = torch.cat([voltage_sequence, dt], dim=-1)
        
        hidden, _ = self.lstm_encoder(lstm_input)
        return hidden
    
    def save_checkpoint(self, path: str, optimizer: Optional[Any] = None, epoch: int = 0, **kwargs):
        """
        保存模型 checkpoint
        
        Args:
            path: 保存路径
            optimizer: 优化器（可选）
            epoch: 当前 epoch
            **kwargs: 其他需要保存的信息
        """
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "config": self.config,
            "epoch": epoch,
            "model_type": "lstm_pinn"
        }
        
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        
        checkpoint.update(kwargs)
        
        torch.save(checkpoint, path)
    
    @classmethod
    def load_checkpoint(cls, path: str, device: str = "cpu") -> Tuple["LSTMPINNModel", Dict]:
        """
        加载模型 checkpoint
        
        Args:
            path: checkpoint 路径
            device: 目标设备
        
        Returns:
            model: 加载的模型
            checkpoint: checkpoint 字典
        """
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        
        # 验证模型类型
        if checkpoint.get("model_type") != "lstm_pinn":
            raise RuntimeError(
                f"Checkpoint model type mismatch: expected 'lstm_pinn', "
                f"got '{checkpoint.get('model_type')}'"
            )
        
        # 创建模型
        model = cls(checkpoint["config"])
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        
        return model, checkpoint
