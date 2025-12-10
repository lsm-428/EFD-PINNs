#!/usr/bin/env python3
"""
OptimizedEWPINN - 增强型神经网络架构
====================================

用于接触角预测的优化 PINN 模型。

作者: EFD-PINNs Team
"""

import math
import torch
import torch.nn as nn


class SimpleAttention(nn.Module):
    """简单注意力机制"""
    
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        scale = math.sqrt(x.size(-1))
        attention = self.softmax(q @ k.transpose(-2, -1) / scale)
        
        return (attention @ v) + x


class OptimizedEWPINN(nn.Module):
    """
    增强型 PINN 神经网络
    
    特点：
    - 可选批量归一化
    - 可选残差连接
    - 可选注意力机制
    - 多种激活函数支持
    """
    
    def __init__(self, input_dim, hidden_dims, output_dim, activation='relu', config=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.config = config or {}
        self.use_batch_norm = self.config.get('use_batch_norm', True)
        self.use_residual = self.config.get('use_residual', True)
        self.use_attention = self.config.get('use_attention', False)
        
        layers = []
        prev_dim = input_dim
        
        for i, h_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, h_dim))
            
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(h_dim))
            
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.1))
            else:
                layers.append(nn.ReLU())
            
            if self.use_attention and i == len(hidden_dims) // 2:
                layers.append(SimpleAttention(h_dim))
            
            prev_dim = h_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.main_layers = nn.Sequential(*layers)
        
        if self.use_residual and input_dim == output_dim:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = None
        
        self.apply(self._initialize_weights)
    
    def _initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        out = self.main_layers(x)
        if self.residual_layer is not None:
            out = out + self.residual_layer(x)
        return out
