"""
LSTM-PINN 混合模型

基于已训练的 TwoPhasePINN，添加 LSTM 编码器处理电压跳变序列。

核心设计：
- LSTM 输入：(V_from, V_to, t_since) 三元组序列
- 每个三元组表示一次电压跳变及其持续时间
- LSTM 编码完整电压历史，输出替代原来的 (V, V_prev)

例如 0→20→30→20→0 的序列:
  Step 1: (0, 20, 0.010)    # 0→20V，持续10ms
  Step 2: (20, 30, 0.005)   # 20→30V，持续5ms
  Step 3: (30, 20, 0.008)   # 30→20V，持续8ms
  Step 4: (20, 0, 0.003)    # 20→0V，持续3ms（当前）

策略：
1. 加载预训练的 TwoPhasePINN 权重
2. 用 LSTM 编码器替换 (V, V_prev) 输入
3. 支持冻结预训练权重，只训练 LSTM
4. 支持端到端微调
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, Any
import logging

from .encoder import VoltageEncoder

logger = logging.getLogger("LSTM-PINN-Hybrid")


class LSTMHybridPINN(nn.Module):
    """
    LSTM 混合 PINN 模型

    基于预训练的 TwoPhasePINN，用 LSTM 编码电压跳变序列替换 (V, V_prev)

    输入:
        - spatial_coords: (batch, 3) 空间坐标 (x, y, z)
        - t: (batch, 1) 当前时间（序列结束后的相对时间）
        - voltage_seq: (batch, seq_len, 3) 电压跳变序列
          * [:, :, 0]: V_from - 跳变前电压 (归一化)
          * [:, :, 1]: V_to - 跳变后电压 (归一化)
          * [:, :, 2]: t_since - 该步持续时间 (归一化)

    输出:
        - (batch, 5) - (u, v, w, p, phi)

    物理意义:
        LSTM 学习电压历史的累积效应：
        - 升压路径依赖：0→30 vs 0→20→30 可能有不同响应
        - 降压滞后：从高电压降下来的恢复过程
        - 多次升降压的累积效应
    """

    def __init__(
        self,
        pretrained_pinn: Optional[nn.Module] = None,
        config: Optional[Dict] = None,
        freeze_pinn: bool = True,
    ):
        """
        初始化混合模型

        Args:
            pretrained_pinn: 预训练的 TwoPhasePINN 模型
            config: 配置字典
            freeze_pinn: 是否冻结预训练权重
        """
        super().__init__()

        config = config or {}
        lstm_config = config.get("lstm", {})

        # LSTM 编码器 - 输入维度为 3: (V_from, V_to, t_since)
        self.lstm_encoder = VoltageEncoder(
            input_dim=lstm_config.get("input_dim", 3),  # 三元组输入
            hidden_dim=lstm_config.get("hidden_dim", 128),
            num_layers=lstm_config.get("num_layers", 2),
            dropout=lstm_config.get("dropout", 0.0),
        )

        # 隐状态投影层：输出 (V_eff, V_prev_eff)
        # 语义：V_eff = 当前等效电压，V_prev_eff = 历史等效电压
        # 这样预训练的 phi_net 可以直接理解这两个值
        hidden_dim = lstm_config.get("hidden_dim", 128)
        self.v_eff_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Sigmoid(),  # 输出 [0, 1]，对应 V/30
        )
        self.v_prev_eff_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Sigmoid(),  # 输出 [0, 1]，对应 V_prev/30
        )

        # 预训练的 PINN 网络
        if pretrained_pinn is not None:
            self.phi_net = pretrained_pinn.phi_net
            self.vel_net = pretrained_pinn.vel_net

            # 复制几何参数
            self.Lx = pretrained_pinn.Lx
            self.Ly = pretrained_pinn.Ly
            self.Lz = pretrained_pinn.Lz
            self.t_max = pretrained_pinn.t_max

            if freeze_pinn:
                logger.info("冻结预训练 PINN 权重，只训练 LSTM 编码器")
                for param in self.phi_net.parameters():
                    param.requires_grad = False
                for param in self.vel_net.parameters():
                    param.requires_grad = False
        else:
            # 如果没有预训练模型，创建新的网络
            model_cfg = config.get("model", {})
            hidden_phi = model_cfg.get("hidden_phi", [64, 64, 64, 32])
            hidden_vel = model_cfg.get("hidden_vel", [64, 64, 32])

            self.phi_net = self._build_network(6, 1, hidden_phi)
            self.vel_net = self._build_network(7, 4, hidden_vel)

            # 默认几何参数
            self.Lx = 174e-6
            self.Ly = 174e-6
            self.Lz = 20e-6
            self.t_max = 0.05

        self.freeze_pinn = freeze_pinn

    def _build_network(
        self, input_dim: int, output_dim: int, hidden_layers: list
    ) -> nn.Sequential:
        """构建全连接网络"""
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.Tanh())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)

    def forward(
        self, spatial_coords: torch.Tensor, t: torch.Tensor, voltage_seq: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            spatial_coords: (batch, 3) 空间坐标 (x, y, z)
            t: (batch, 1) 当前时间（t_since）
            voltage_seq: (batch, seq_len, 3) 电压跳变序列
                - [:, :, 0]: V_from (归一化到 [0, 1])
                - [:, :, 1]: V_to (归一化到 [0, 1])
                - [:, :, 2]: t_since (归一化到 [0, 1])

        Returns:
            (batch, 5) - (u, v, w, p, phi)
        """
        # LSTM 编码电压历史
        hidden, _ = self.lstm_encoder(voltage_seq)  # (batch, hidden_dim)

        # 输出等效电压 (V_eff, V_prev_eff)
        # 语义：V_eff = V_to（当前电压），V_prev_eff = V_from（跳变前电压）
        V_eff = self.v_eff_head(hidden)  # (batch, 1) 当前等效电压 (V_to)
        V_prev_eff = self.v_prev_eff_head(hidden)  # (batch, 1) 历史等效电压 (V_from)

        # 归一化空间坐标
        x_norm = spatial_coords[:, 0:1] / self.Lx
        y_norm = spatial_coords[:, 1:2] / self.Ly
        z_norm = spatial_coords[:, 2:3] / self.Lz
        t_norm = t / self.t_max

        # Phi 网络输入：(x, y, z, V_from, V_to, t_since) - 三元组格式
        # V_prev_eff 对应 V_from，V_eff 对应 V_to
        phi_input = torch.cat(
            [
                x_norm,
                y_norm,
                z_norm,
                V_prev_eff,
                V_eff,
                t_norm,  # (V_from, V_to, t_since)
            ],
            dim=-1,
        )

        phi_raw = self.phi_net(phi_input)
        phi = torch.sigmoid(phi_raw)

        # 速度网络输入：(x, y, z, V_from, V_to, t_since, phi)
        vel_input = torch.cat(
            [x_norm, y_norm, z_norm, V_prev_eff, V_eff, t_norm, phi], dim=-1
        )

        vel_out = self.vel_net(vel_input)
        u, v, w, p = vel_out[:, 0:1], vel_out[:, 1:2], vel_out[:, 2:3], vel_out[:, 3:4]

        return torch.cat([u, v, w, p, phi], dim=-1)

    def predict_phi(
        self, spatial_coords: torch.Tensor, t: torch.Tensor, voltage_seq: torch.Tensor
    ) -> torch.Tensor:
        """预测 φ 值"""
        output = self.forward(spatial_coords, t, voltage_seq)
        return output[:, 4:5]

    def get_effective_voltage(
        self, voltage_seq: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取 LSTM 编码的等效电压

        用于辅助损失：单步场景下 V_eff ≈ V_to, V_prev_eff ≈ V_from

        Returns:
            (V_eff, V_prev_eff): 各 (batch, 1)
        """
        hidden, _ = self.lstm_encoder(voltage_seq)
        V_eff = self.v_eff_head(hidden)
        V_prev_eff = self.v_prev_eff_head(hidden)
        return V_eff, V_prev_eff

    def unfreeze_pinn(self):
        """解冻 PINN 权重，进行端到端微调"""
        logger.info("解冻 PINN 权重，开始端到端微调")
        for param in self.phi_net.parameters():
            param.requires_grad = True
        for param in self.vel_net.parameters():
            param.requires_grad = True
        self.freeze_pinn = False

    def unfreeze_phi_only(self):
        """只解冻 phi_net 权重，保持 vel_net 冻结（渐进式解冻 Stage 2）"""
        logger.info("解冻 phi_net 权重，vel_net 保持冻结")
        for param in self.phi_net.parameters():
            param.requires_grad = True
        for param in self.vel_net.parameters():
            param.requires_grad = False

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        config: Optional[Dict] = None,
        freeze_pinn: bool = True,
        device: str = "cpu",
    ) -> "LSTMHybridPINN":
        """
        从预训练的 TwoPhasePINN checkpoint 创建混合模型
        """
        from src.models.pinn_two_phase import TwoPhasePINN

        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )

        pinn_config = checkpoint.get("config", {})
        pretrained_pinn = TwoPhasePINN(pinn_config)
        pretrained_pinn.load_state_dict(checkpoint["model_state_dict"])
        pretrained_pinn.to(device)

        logger.info(f"从 {checkpoint_path} 加载预训练 TwoPhasePINN")

        model = cls(
            pretrained_pinn=pretrained_pinn, config=config, freeze_pinn=freeze_pinn
        )
        model.to(device)

        return model

    def save_checkpoint(
        self,
        path: str,
        optimizer: Optional[Any] = None,
        epoch: int = 0,
        config: Optional[Dict] = None,
        **kwargs,
    ):
        """保存 checkpoint"""
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "model_type": "lstm_hybrid_pinn",
            "epoch": epoch,
            "freeze_pinn": self.freeze_pinn,
        }

        if config is not None:
            checkpoint["config"] = config

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        checkpoint.update(kwargs)
        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(
        cls, path: str, device: str = "cpu"
    ) -> Tuple["LSTMHybridPINN", Dict]:
        """加载 checkpoint"""
        checkpoint = torch.load(path, map_location=device, weights_only=True)

        if checkpoint.get("model_type") != "lstm_hybrid_pinn":
            raise RuntimeError("Model type mismatch: expected 'lstm_hybrid_pinn'")

        model = cls(
            pretrained_pinn=None, freeze_pinn=checkpoint.get("freeze_pinn", False)
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

        return model, checkpoint
