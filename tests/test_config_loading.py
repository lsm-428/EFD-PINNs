#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置加载单元测试
测试中英文键名解析和默认值回退机制

**Feature: fix-training-config, Property 3: Bilingual key support**
**Validates: Requirements 2.2, 2.3**
"""

import json
import os
import sys
import unittest

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ConfigParser:
    """统一配置解析器，支持中英文键名"""
    
    # 键名映射表
    KEY_MAPPING = {
        'model': ['model', '模型', '模型架构'],
        'input_dim': ['input_dim', '输入维度'],
        'output_dim': ['output_dim', '输出维度'],
        'hidden_layers': ['hidden_layers', '隐藏层维度', 'hidden_dims'],
        'activation': ['activation', '激活函数'],
        'use_batch_norm': ['use_batch_norm', '批量归一化'],
        'training': ['training', '训练流程'],
        'data': ['data', '数据'],
        'optimizer': ['optimizer', '优化器'],
    }
    
    # 默认值
    DEFAULTS = {
        'input_dim': 3,
        'output_dim': 1,
        'hidden_layers': [64, 64],
        'activation': 'relu',
        'use_batch_norm': False,
    }
    
    @classmethod
    def get_value(cls, config: dict, key: str, default=None):
        """从配置中获取值，支持多种键名"""
        if key in cls.KEY_MAPPING:
            for alt_key in cls.KEY_MAPPING[key]:
                if alt_key in config:
                    return config[alt_key]
        
        # 直接查找
        if key in config:
            return config[key]
        
        # 返回默认值
        if default is not None:
            return default
        return cls.DEFAULTS.get(key)
    
    @classmethod
    def get_model_config(cls, config: dict) -> dict:
        """获取标准化的模型配置"""
        # 首先尝试获取模型配置节
        model_config = None
        for key in cls.KEY_MAPPING['model']:
            if key in config:
                model_config = config[key]
                break
        
        if model_config is None:
            model_config = {}
        
        # 提取标准化的模型参数
        return {
            'input_dim': cls.get_value(model_config, 'input_dim', cls.DEFAULTS['input_dim']),
            'output_dim': cls.get_value(model_config, 'output_dim', cls.DEFAULTS['output_dim']),
            'hidden_layers': cls.get_value(model_config, 'hidden_layers', cls.DEFAULTS['hidden_layers']),
            'activation': cls.get_value(model_config, 'activation', cls.DEFAULTS['activation']),
            'use_batch_norm': cls.get_value(model_config, 'use_batch_norm', cls.DEFAULTS['use_batch_norm']),
        }
    
    @classmethod
    def validate_config(cls, config: dict) -> list:
        """验证配置完整性，返回警告列表"""
        warnings = []
        
        # 检查模型配置
        model_config = cls.get_model_config(config)
        
        # 检查必需的维度参数
        if model_config['input_dim'] == cls.DEFAULTS['input_dim']:
            # 检查是否真的缺失
            found = False
            for key in cls.KEY_MAPPING['model']:
                if key in config:
                    mc = config[key]
                    for dim_key in cls.KEY_MAPPING['input_dim']:
                        if dim_key in mc:
                            found = True
                            break
            if not found:
                warnings.append("Missing input_dim in model config, using default: 3")
        
        if model_config['output_dim'] == cls.DEFAULTS['output_dim']:
            found = False
            for key in cls.KEY_MAPPING['model']:
                if key in config:
                    mc = config[key]
                    for dim_key in cls.KEY_MAPPING['output_dim']:
                        if dim_key in mc:
                            found = True
                            break
            if not found:
                warnings.append("Missing output_dim in model config, using default: 1")
        
        return warnings


class TestConfigParser(unittest.TestCase):
    """配置解析器测试"""
    
    def test_english_keys(self):
        """测试英文键名解析"""
        config = {
            "model": {
                "input_dim": 62,
                "output_dim": 24,
                "hidden_layers": [128, 128, 128],
                "activation": "gelu",
                "use_batch_norm": True
            }
        }
        
        model_config = ConfigParser.get_model_config(config)
        
        self.assertEqual(model_config['input_dim'], 62)
        self.assertEqual(model_config['output_dim'], 24)
        self.assertEqual(model_config['hidden_layers'], [128, 128, 128])
        self.assertEqual(model_config['activation'], 'gelu')
        self.assertTrue(model_config['use_batch_norm'])
    
    def test_chinese_keys(self):
        """测试中文键名解析"""
        config = {
            "模型": {
                "input_dim": 62,
                "output_dim": 24,
                "隐藏层维度": [256, 256, 128, 64],
                "激活函数": "relu",
                "批量归一化": False
            }
        }
        
        model_config = ConfigParser.get_model_config(config)
        
        self.assertEqual(model_config['input_dim'], 62)
        self.assertEqual(model_config['output_dim'], 24)
        self.assertEqual(model_config['hidden_layers'], [256, 256, 128, 64])
        self.assertEqual(model_config['activation'], 'relu')
        self.assertFalse(model_config['use_batch_norm'])
    
    def test_mixed_keys(self):
        """测试混合键名解析"""
        config = {
            "model": {
                "input_dim": 62,
                "output_dim": 24,
                "hidden_dims": [64, 64],  # 另一种英文键名
            },
            "模型": {
                "激活函数": "tanh",
                "批量归一化": True
            }
        }
        
        # 英文键优先
        model_config = ConfigParser.get_model_config(config)
        
        self.assertEqual(model_config['input_dim'], 62)
        self.assertEqual(model_config['output_dim'], 24)
        self.assertEqual(model_config['hidden_layers'], [64, 64])
    
    def test_default_values(self):
        """测试默认值回退"""
        config = {}  # 空配置
        
        model_config = ConfigParser.get_model_config(config)
        
        self.assertEqual(model_config['input_dim'], 3)  # 默认值
        self.assertEqual(model_config['output_dim'], 1)  # 默认值
        self.assertEqual(model_config['hidden_layers'], [64, 64])  # 默认值
        self.assertEqual(model_config['activation'], 'relu')  # 默认值
        self.assertFalse(model_config['use_batch_norm'])  # 默认值
    
    def test_validation_missing_dims(self):
        """测试缺失维度参数时的验证警告"""
        config = {
            "model": {
                "activation": "gelu"
                # 缺少 input_dim 和 output_dim
            }
        }
        
        warnings = ConfigParser.validate_config(config)
        
        self.assertTrue(len(warnings) > 0)
        self.assertTrue(any('input_dim' in w for w in warnings))
        self.assertTrue(any('output_dim' in w for w in warnings))
    
    def test_validation_complete_config(self):
        """测试完整配置时无警告"""
        config = {
            "model": {
                "input_dim": 62,
                "output_dim": 24,
                "hidden_layers": [128, 128]
            }
        }
        
        warnings = ConfigParser.validate_config(config)
        
        self.assertEqual(len(warnings), 0)
    
    def test_bilingual_equivalence(self):
        """
        Property 3: 中英文键名解析结果相同
        **Feature: fix-training-config, Property 3: Bilingual key support**
        **Validates: Requirements 2.2, 2.3**
        """
        # 英文配置
        config_en = {
            "model": {
                "input_dim": 62,
                "output_dim": 24,
                "hidden_layers": [256, 256, 128, 64],
                "activation": "gelu",
                "use_batch_norm": True
            }
        }
        
        # 中文配置（相同值）
        config_cn = {
            "模型": {
                "input_dim": 62,
                "output_dim": 24,
                "隐藏层维度": [256, 256, 128, 64],
                "激活函数": "gelu",
                "批量归一化": True
            }
        }
        
        model_en = ConfigParser.get_model_config(config_en)
        model_cn = ConfigParser.get_model_config(config_cn)
        
        # 验证两种配置解析结果相同
        self.assertEqual(model_en['input_dim'], model_cn['input_dim'])
        self.assertEqual(model_en['output_dim'], model_cn['output_dim'])
        self.assertEqual(model_en['hidden_layers'], model_cn['hidden_layers'])
        self.assertEqual(model_en['activation'], model_cn['activation'])
        self.assertEqual(model_en['use_batch_norm'], model_cn['use_batch_norm'])


class TestConfigFiles(unittest.TestCase):
    """测试实际配置文件"""
    
    def test_stage1_config(self):
        """测试 stage1_config.json"""
        config_path = 'config/stage1_config.json'
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            model_config = ConfigParser.get_model_config(config)
            
            self.assertEqual(model_config['input_dim'], 62)
            self.assertEqual(model_config['output_dim'], 24)
    
    def test_stage2_optimized_config(self):
        """测试 config_stage2_optimized.json"""
        config_path = 'config_stage2_optimized.json'
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            model_config = ConfigParser.get_model_config(config)
            
            self.assertEqual(model_config['input_dim'], 62)
            self.assertEqual(model_config['output_dim'], 24)
    
    def test_stage8_config(self):
        """测试 stage8_residual_learning.json"""
        config_path = 'config/stage8_residual_learning.json'
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            model_config = ConfigParser.get_model_config(config)
            
            self.assertEqual(model_config['input_dim'], 62)
            self.assertEqual(model_config['output_dim'], 24)


if __name__ == '__main__':
    unittest.main(verbosity=2)
