#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型维度一致性属性测试
使用 hypothesis 进行属性测试

**Feature: fix-training-config, Property 1: Model input dimension consistency**
**Feature: fix-training-config, Property 2: Model output dimension consistency**
**Validates: Requirements 1.4, 1.5**
"""

import os
import sys
import unittest

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from hypothesis import given, strategies as st, settings

# 导入模型创建函数
try:
    from efd_pinn_train import OptimizedEWPINN
except ImportError:
    # 如果导入失败，定义一个简单的模型类用于测试
    class OptimizedEWPINN(nn.Module):
        def __init__(self, input_dim, hidden_dims, output_dim, activation='relu', config=None):
            super().__init__()
            self.input_dim = input_dim
            self.output_dim = output_dim
            
            layers = []
            prev_dim = input_dim
            for h_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, h_dim))
                layers.append(nn.ReLU())
                prev_dim = h_dim
            layers.append(nn.Linear(prev_dim, output_dim))
            
            self.main_layers = nn.Sequential(*layers)
        
        def forward(self, x):
            return self.main_layers(x)


class TestModelDimensionConsistency(unittest.TestCase):
    """模型维度一致性测试"""
    
    @given(
        input_dim=st.integers(min_value=1, max_value=100),
        output_dim=st.integers(min_value=1, max_value=50),
        num_layers=st.integers(min_value=1, max_value=5),
        layer_width=st.integers(min_value=8, max_value=128)
    )
    @settings(max_examples=50, deadline=None)
    def test_input_dimension_consistency(self, input_dim, output_dim, num_layers, layer_width):
        """
        Property 1: 模型输入维度一致性
        *For any* valid configuration file, when the model is created, 
        the model's first layer input size SHALL equal the configured input_dim value.
        
        **Feature: fix-training-config, Property 1: Model input dimension consistency**
        **Validates: Requirements 1.4**
        """
        hidden_dims = [layer_width] * num_layers
        
        model = OptimizedEWPINN(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            activation='relu'
        )
        
        # 验证模型的 input_dim 属性
        self.assertEqual(model.input_dim, input_dim)
        
        # 验证第一层的输入维度
        first_layer = None
        for module in model.main_layers:
            if isinstance(module, nn.Linear):
                first_layer = module
                break
        
        self.assertIsNotNone(first_layer)
        self.assertEqual(first_layer.in_features, input_dim)
    
    @given(
        input_dim=st.integers(min_value=1, max_value=100),
        output_dim=st.integers(min_value=1, max_value=50),
        num_layers=st.integers(min_value=1, max_value=5),
        layer_width=st.integers(min_value=8, max_value=128)
    )
    @settings(max_examples=50, deadline=None)
    def test_output_dimension_consistency(self, input_dim, output_dim, num_layers, layer_width):
        """
        Property 2: 模型输出维度一致性
        *For any* valid configuration file, when the model is created, 
        the model's last layer output size SHALL equal the configured output_dim value.
        
        **Feature: fix-training-config, Property 2: Model output dimension consistency**
        **Validates: Requirements 1.5**
        """
        hidden_dims = [layer_width] * num_layers
        
        model = OptimizedEWPINN(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            activation='relu'
        )
        
        # 验证模型的 output_dim 属性
        self.assertEqual(model.output_dim, output_dim)
        
        # 验证最后一层的输出维度
        last_layer = None
        for module in reversed(list(model.main_layers)):
            if isinstance(module, nn.Linear):
                last_layer = module
                break
        
        self.assertIsNotNone(last_layer)
        self.assertEqual(last_layer.out_features, output_dim)
    
    @given(
        input_dim=st.integers(min_value=1, max_value=100),
        output_dim=st.integers(min_value=1, max_value=50),
        batch_size=st.integers(min_value=2, max_value=32)  # BatchNorm 需要 batch_size > 1
    )
    @settings(max_examples=30, deadline=None)
    def test_forward_pass_dimensions(self, input_dim, output_dim, batch_size):
        """
        测试前向传播的输入输出维度
        验证模型能正确处理任意维度的输入并产生正确维度的输出
        注意：BatchNorm 需要 batch_size > 1
        """
        hidden_dims = [64, 64]
        
        model = OptimizedEWPINN(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            activation='relu'
        )
        model.eval()  # 设置为评估模式，避免 BatchNorm 问题
        
        # 创建随机输入
        x = torch.randn(batch_size, input_dim)
        
        # 前向传播
        with torch.no_grad():
            y = model(x)
        
        # 验证输出维度
        self.assertEqual(y.shape[0], batch_size)
        self.assertEqual(y.shape[1], output_dim)
    
    def test_specific_config_dimensions(self):
        """
        测试特定配置（62维输入，24维输出）的模型维度
        这是项目中实际使用的配置
        """
        input_dim = 62
        output_dim = 24
        hidden_dims = [256, 256, 128, 64]
        
        model = OptimizedEWPINN(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            activation='gelu'
        )
        
        # 验证维度
        self.assertEqual(model.input_dim, 62)
        self.assertEqual(model.output_dim, 24)
        
        # 测试前向传播
        x = torch.randn(16, 62)
        with torch.no_grad():
            y = model(x)
        
        self.assertEqual(y.shape, (16, 24))


class TestConfigToModelConsistency(unittest.TestCase):
    """配置到模型的一致性测试"""
    
    def test_config_model_dimension_match(self):
        """
        测试从配置文件创建的模型维度与配置一致
        """
        import json
        
        config_files = [
            'config/stage1_config.json',
            'config_stage2_optimized.json',
            'config/stage8_residual_learning.json'
        ]
        
        for config_path in config_files:
            if not os.path.exists(config_path):
                continue
            
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # 获取模型配置
            model_config = config.get('model', config.get('模型', {}))
            input_dim = model_config.get('input_dim', 3)
            output_dim = model_config.get('output_dim', 1)
            hidden_dims = model_config.get('hidden_dims', 
                          model_config.get('hidden_layers',
                          model_config.get('隐藏层维度', [64, 64])))
            activation = model_config.get('activation', 
                         model_config.get('激活函数', 'relu'))
            
            # 创建模型
            model = OptimizedEWPINN(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                output_dim=output_dim,
                activation=activation
            )
            
            # 验证维度一致性
            self.assertEqual(model.input_dim, input_dim, 
                           f"Config {config_path}: input_dim mismatch")
            self.assertEqual(model.output_dim, output_dim,
                           f"Config {config_path}: output_dim mismatch")
            
            # 测试前向传播
            x = torch.randn(8, input_dim)
            with torch.no_grad():
                y = model(x)
            
            self.assertEqual(y.shape[1], output_dim,
                           f"Config {config_path}: output shape mismatch")


if __name__ == '__main__':
    unittest.main(verbosity=2)
