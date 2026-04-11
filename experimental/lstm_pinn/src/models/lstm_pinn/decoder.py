"""
φ 场解码器

将空间坐标和 LSTM 隐状态映射到 φ 值
"""

import torch
import torch.nn as nn
from typing import List, Optional


class PhiDecoder(nn.Module):
    """
    φ 场解码器
    
    输入:
        - spatial: (batch, 3) - (x, y, z) 归一化坐标
        - hidden: (batch, hidden_dim) - LSTM 隐状态
    
    输出:
        - phi: (batch, 1) - φ 值 [0, 1]
    """
    
    def __init__(
        self,
        spatial_dim: int = 3,
        hidden_dim: int = 128,
        hidden_layers: Optional[List[int]] = None,
        activation: str = "tanh",
        use_skip_connections: bool = False
    ):
        """
        初始化 φ 解码器
        
        Args:
            spatial_dim: 空间坐标维度，默认 3 (x, y, z)
            hidden_dim: LSTM 隐状态维度
            hidden_layers: MLP 隐藏层维度列表，默认 [128, 64, 32]
            activation: 激活函数类型 ("tanh", "relu", "gelu")
            use_skip_connections: 是否使用跳跃连接
        """
        super().__init__()
        
        self.spatial_dim = spatial_dim
        self.hidden_dim = hidden_dim
        self.use_skip_connections = use_skip_connections
        
        if hidden_layers is None:
            hidden_layers = [128, 64, 32]
        
        # 输入维度 = 空间坐标 + LSTM 隐状态
        input_dim = spatial_dim + hidden_dim
        
        # 选择激活函数
        if activation == "tanh":
            act_fn = nn.Tanh
        elif activation == "relu":
            act_fn = nn.ReLU
        elif activation == "gelu":
            act_fn = nn.GELU
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # 构建 MLP 层
        layers = []
        prev_dim = input_dim
        
        for i, h_dim in enumerate(hidden_layers):
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(act_fn())
            prev_dim = h_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
        
        # 跳跃连接投影（如果需要）
        if use_skip_connections and len(hidden_layers) > 0:
            self.skip_projection = nn.Linear(input_dim, hidden_layers[-1])
        else:
            self.skip_projection = None
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """Xavier 初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        spatial: torch.Tensor,
        hidden: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            spatial: (batch, spatial_dim) 空间坐标
            hidden: (batch, hidden_dim) LSTM 隐状态
        
        Returns:
            phi: (batch, 1) φ 值 [0, 1]
        """
        # 拼接空间坐标和 LSTM 隐状态
        x = torch.cat([spatial, hidden], dim=-1)
        
        # MLP 前向传播
        phi_raw = self.mlp(x)
        
        # 使用 sigmoid 确保输出在 [0, 1] 范围内
        phi = torch.sigmoid(phi_raw)
        
        return phi


class VelocityDecoder(nn.Module):
    """
    速度场解码器（可选）
    
    输入:
        - spatial: (batch, 3) - (x, y, z) 归一化坐标
        - hidden: (batch, hidden_dim) - LSTM 隐状态
    
    输出:
        - velocity: (batch, 3) - 速度场 (u, v, w)
    """
    
    def __init__(
        self,
        spatial_dim: int = 3,
        hidden_dim: int = 128,
        hidden_layers: Optional[List[int]] = None,
        activation: str = "tanh"
    ):
        """
        初始化速度解码器
        
        Args:
            spatial_dim: 空间坐标维度
            hidden_dim: LSTM 隐状态维度
            hidden_layers: MLP 隐藏层维度列表
            activation: 激活函数类型
        """
        super().__init__()
        
        if hidden_layers is None:
            hidden_layers = [64, 32]
        
        input_dim = spatial_dim + hidden_dim
        
        # 选择激活函数
        if activation == "tanh":
            act_fn = nn.Tanh
        elif activation == "relu":
            act_fn = nn.ReLU
        elif activation == "gelu":
            act_fn = nn.GELU
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # 构建 MLP 层
        layers = []
        prev_dim = input_dim
        
        for h_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(act_fn())
            prev_dim = h_dim
        
        # 输出层 - 3 个速度分量
        layers.append(nn.Linear(prev_dim, 3))
        
        self.mlp = nn.Sequential(*layers)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """Xavier 初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        spatial: torch.Tensor,
        hidden: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            spatial: (batch, spatial_dim) 空间坐标
            hidden: (batch, hidden_dim) LSTM 隐状态
        
        Returns:
            velocity: (batch, 3) 速度场 (u, v, w)
        """
        x = torch.cat([spatial, hidden], dim=-1)
        velocity = self.mlp(x)
        return velocity
