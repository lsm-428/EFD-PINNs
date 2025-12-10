#!/usr/bin/env python3
"""
训练组件库
==========

包含数据标准化、损失稳定、数据增强等核心训练组件。

作者: EFD-PINNs Team
"""

import numpy as np
import torch
import torch.nn as nn

try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
except ImportError:
    StandardScaler = MinMaxScaler = RobustScaler = None


class DataNormalizer:
    """增强版数据标准化器"""
    
    def __init__(self, method: str = "standard", config: dict = None):
        self.method = method
        self.config = config or {}
        self.scaler = None
        self.mean = None
        self.std = None
        self.min_val = None
        self.max_val = None
        self.output_normalizer = None  # 用于输出反归一化
        
        if method == "standard" and StandardScaler:
            self.scaler = StandardScaler()
        elif method == "minmax" and MinMaxScaler:
            self.scaler = MinMaxScaler()
        elif method == "robust" and RobustScaler:
            self.scaler = RobustScaler()

    def fit(self, X: np.ndarray):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.min_val = np.min(X, axis=0)
        self.max_val = np.max(X, axis=0)
        if self.scaler:
            self.scaler.fit(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.scaler:
            return self.scaler.transform(X)
        elif self.mean is not None and self.std is not None:
            return (X - self.mean) / (self.std + 1e-8)
        return X

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        if self.scaler:
            return self.scaler.inverse_transform(X)
        elif self.mean is not None and self.std is not None:
            return X * (self.std + 1e-8) + self.mean
        return X

    def state_dict(self):
        return {
            "method": self.method,
            "mean": self.mean.tolist() if self.mean is not None else None,
            "std": self.std.tolist() if self.std is not None else None,
            "min_val": self.min_val.tolist() if self.min_val is not None else None,
            "max_val": self.max_val.tolist() if self.max_val is not None else None,
        }

    def load_state_dict(self, state):
        self.method = state.get("method", "standard")
        self.mean = np.array(state["mean"]) if state.get("mean") else None
        self.std = np.array(state["std"]) if state.get("std") else None
        self.min_val = np.array(state["min_val"]) if state.get("min_val") else None
        self.max_val = np.array(state["max_val"]) if state.get("max_val") else None


class LossStabilizer:
    """高级损失稳定器，支持多种权重策略"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.loss_type = self.config.get('loss_type', 'mse')
        self.epsilon = self.config.get('epsilon', 1e-8)
        self.weight_strategy = self.config.get('weight_strategy', 'fixed')
        self.history_size = self.config.get('history_size', 100)
        self.loss_history = []
        self.early_stopping_patience = self.config.get('early_stopping_patience', 20)
        self.early_stopping_min_delta = self.config.get('early_stopping_min_delta', 1e-5)
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.base_physics_weight = self.config.get('base_physics_weight', 1.0)
        self.max_physics_weight = self.config.get('max_physics_weight', 10.0)
    
    def safe_mse_loss(self, pred, target):
        return torch.mean(torch.clamp((pred - target) ** 2, max=1e8))
    
    def compute_loss(self, pred, target, physics_loss=None, physics_weight=0.0):
        base_loss = self.safe_mse_loss(pred, target)
        if physics_loss is not None:
            total_loss = base_loss + physics_weight * physics_loss
        else:
            total_loss = base_loss
        self.loss_history.append(total_loss.item())
        if len(self.loss_history) > self.history_size:
            self.loss_history.pop(0)
        return total_loss
    
    def check_early_stopping(self):
        if not self.loss_history:
            return False
        current_loss = self.loss_history[-1]
        if current_loss < self.best_loss - self.early_stopping_min_delta:
            self.best_loss = current_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.early_stopping_patience
    
    def get_dynamic_physics_weight(self, epoch=0):
        if self.weight_strategy == 'fixed':
            return self.base_physics_weight
        elif self.weight_strategy == 'adaptive' and len(self.loss_history) >= 10:
            recent_avg = np.mean(self.loss_history[-10:])
            earlier_avg = np.mean(self.loss_history[:10])
            if earlier_avg > 0:
                improvement = (earlier_avg - recent_avg) / earlier_avg
                if improvement < 0.01:
                    return min(self.max_physics_weight, self.base_physics_weight * 1.5)
        return self.base_physics_weight


class EnhancedDataAugmenter:
    """增强型数据增强器"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.noise_level = self.config.get('noise_level', 0.01)
        self.enable_noise = self.config.get('enable_noise_augmentation', True)
        self.enable_scaling = self.config.get('enable_scaling', True)
        self.scaling_range = self.config.get('scaling_range', [0.95, 1.05])
    
    def augment(self, x, y=None):
        if self.enable_noise and self.noise_level > 0:
            noise = torch.randn_like(x) * self.noise_level
            x = x + noise
        
        if self.enable_scaling:
            scale = torch.rand(1, device=x.device) * (self.scaling_range[1] - self.scaling_range[0]) + self.scaling_range[0]
            x = x * scale
        
        return x, y
    
    def __call__(self, x, y=None):
        return self.augment(x, y)


class DynamicWeightIntegration:
    """动态权重调整管理器"""
    
    def __init__(self, strategy='adaptive', initial_weight=1.0, config=None):
        self.strategy = strategy
        self.initial_weight = initial_weight
        self.current_weight = initial_weight
        self.config = config or {}
        self.loss_history = {'data': [], 'physics': []}
    
    def update(self, data_loss, physics_loss, epoch=None, total_epochs=None):
        self.loss_history['data'].append(data_loss)
        self.loss_history['physics'].append(physics_loss)
        
        if self.strategy == 'adaptive' and total_epochs and epoch is not None:
            progress = epoch / total_epochs
            weight_factor = 1.0 + 9.0 * progress
            self.current_weight = self.initial_weight * weight_factor
    
    def get_weight(self):
        return self.current_weight
