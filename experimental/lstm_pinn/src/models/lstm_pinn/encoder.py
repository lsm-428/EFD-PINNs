"""
LSTM 电压编码器

处理电压跳变序列，捕获时序依赖（升压/降压滞后）

输入格式：(V_from, V_to, t_since) 三元组序列
- V_from: 跳变前电压 (归一化)
- V_to: 跳变后电压 (归一化)  
- t_since: 该步持续时间 (归一化)

例如 0→20→30→20→0 的序列:
  Step 1: (0/30, 20/30, 10ms/50ms)  # 0→20V，持续10ms
  Step 2: (20/30, 30/30, 5ms/50ms)  # 20→30V，持续5ms
  Step 3: (30/30, 20/30, 8ms/50ms)  # 30→20V，持续8ms
  Step 4: (20/30, 0/30, 3ms/50ms)   # 20→0V，持续3ms
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class VoltageEncoder(nn.Module):
    """
    LSTM 电压编码器
    
    输入:
        - sequence: (batch, seq_len, 3) - (V_from, V_to, t_since) 序列
          * V_from: 跳变前电压 (归一化到 [0, 1])
          * V_to: 跳变后电压 (归一化到 [0, 1])
          * t_since: 该步持续时间 (归一化到 [0, 1])
    
    输出:
        - hidden: (batch, hidden_dim) - 最终隐状态
        - all_hidden: (batch, seq_len, hidden_dim) - 所有时刻的隐状态
    
    物理意义:
        LSTM 学习电压历史的累积效应，例如：
        - 多次升降压后的滞后效应
        - 不同路径到达同一电压的差异（0→30 vs 0→20→30）
    """
    
    def __init__(
        self,
        input_dim: int = 3,  # (V_from, V_to, t_since)
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False
    ):
        """
        初始化 LSTM 编码器
        
        Args:
            input_dim: 输入特征维度，默认 2 (V, dt)
            hidden_dim: LSTM 隐状态维度
            num_layers: LSTM 层数
            dropout: Dropout 比例（仅在 num_layers > 1 时生效）
            bidirectional: 是否使用双向 LSTM
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # LSTM 层
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # 如果是双向 LSTM，需要投影到 hidden_dim
        if bidirectional:
            self.projection = nn.Linear(hidden_dim * 2, hidden_dim)
        else:
            self.projection = None
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(
        self,
        sequence: torch.Tensor,
        initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            sequence: (batch, seq_len, input_dim) 输入序列
            initial_state: 可选的初始隐状态 (h_0, c_0)
        
        Returns:
            hidden: (batch, hidden_dim) 最终隐状态
            all_hidden: (batch, seq_len, hidden_dim) 所有时刻的隐状态
        """
        batch_size = sequence.size(0)
        
        # 初始化隐状态
        if initial_state is None:
            h_0 = torch.zeros(
                self.num_layers * self.num_directions,
                batch_size,
                self.hidden_dim,
                device=sequence.device,
                dtype=sequence.dtype
            )
            c_0 = torch.zeros_like(h_0)
            initial_state = (h_0, c_0)
        
        # LSTM 前向传播
        output, (h_n, c_n) = self.lstm(sequence, initial_state)
        
        # 处理双向 LSTM 的输出
        if self.bidirectional:
            # output: (batch, seq_len, hidden_dim * 2)
            all_hidden = self.projection(output)
            # h_n: (num_layers * 2, batch, hidden_dim)
            # 取最后一层的前向和后向隐状态
            h_forward = h_n[-2]  # 最后一层前向
            h_backward = h_n[-1]  # 最后一层后向
            hidden = self.projection(torch.cat([h_forward, h_backward], dim=-1))
        else:
            all_hidden = output
            # 取最后一层的隐状态
            hidden = h_n[-1]
        
        # 层归一化
        hidden = self.layer_norm(hidden)
        
        return hidden, all_hidden
    
    def get_output_dim(self) -> int:
        """返回输出维度"""
        return self.hidden_dim
