"""
EWPINN 配置管理模块
处理训练配置、超参数设置和配置文件管理
"""

import json
import os
from typing import Dict, Any, Optional
import torch
import logging

logger = logging.getLogger('EWPINN_Config')

class ConfigManager:
    """配置管理器 - 处理所有训练配置"""
    
    def __init__(self):
        self.config = self._get_default_config()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认训练配置"""
        return {
            # 基础配置
            'stage': 3,
            'input_dim': 62,
            'output_dim': 24,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            
            # 数据配置
            'num_train_samples': 1000,
            'num_val_samples': 200,
            'num_test_samples': 100,
            'batch_size': 32,
            'num_workers': 4,
            
            # 训练配置
            'early_stopping': True,
            'patience': 10,
            'save_best_only': True,
            'monitor_metric': 'val_loss',
            'min_delta': 1e-6,
            
            # 多阶段训练配置
            'multi_stage_config': {
                1: {
                    'epochs': 50,
                    'learning_rate': 1e-3,
                    'optimizer': 'AdamW',
                    'weight_decay': 1e-5,
                    'scheduler': 'cosine',
                    'warmup_epochs': 5,
                    'description': '预训练阶段'
                },
                2: {
                    'epochs': 100,
                    'learning_rate': 5e-4,
                    'optimizer': 'AdamW',
                    'weight_decay': 1e-5,
                    'scheduler': 'cosine',
                    'description': '物理强化训练'
                },
                3: {
                    'epochs': 150,
                    'learning_rate': 1e-4,
                    'optimizer': 'AdamW',
                    'weight_decay': 1e-6,
                    'scheduler': 'cosine',
                    'description': '精细调优阶段'
                }
            },
            
            # 损失权重
            'loss_weights': {
                'physics': 1.0,
                'boundary': 10.0,
                'initial': 10.0,
                'data': 1.0
            },
            
            # 保存配置
            'checkpoint_dir': 'checkpoints',
            'log_dir': 'logs',
            'save_frequency': 10,
            'save_optimizer': True,
            
            # 验证配置
            'validation_frequency': 5,
            'physics_validation_frequency': 10,
            
            # 梯度裁剪
            'gradient_clip_value': 1.0,
            'gradient_clip_norm': 1.0,
            
            # 随机种子
            'seed': 42,
            'deterministic': True
        }
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """从文件加载配置"""
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                
                # 合并配置，优先使用加载的配置
                self.config.update(loaded_config)
                logger.info(f"配置已从 {config_path} 加载")
                return self.config
            else:
                logger.warning(f"配置文件 {config_path} 不存在，使用默认配置")
                return self.config
                
        except Exception as e:
            logger.error(f"加载配置文件失败: {str(e)}")
            return self.config
    
    def save_config(self, config_path: str) -> None:
        """保存配置到文件"""
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
            logger.info(f"配置已保存到 {config_path}")
        except Exception as e:
            logger.error(f"保存配置文件失败: {str(e)}")
    
    def get_config(self) -> Dict[str, Any]:
        """获取当前配置"""
        return self.config.copy()
    
    def update_config(self, config_updates: Dict[str, Any]) -> None:
        """更新配置"""
        try:
            self.config.update(config_updates)
            logger.info(f"配置已更新: {list(config_updates.keys())}")
        except Exception as e:
            logger.error(f"更新配置失败: {str(e)}")
    
    def get_stage_config(self, stage: int) -> Dict[str, Any]:
        """获取特定阶段的配置"""
        if stage in self.config['multi_stage_config']:
            return self.config['multi_stage_config'][stage]
        else:
            logger.warning(f"阶段 {stage} 配置不存在，返回默认配置")
            return {
                'epochs': 50,
                'learning_rate': 1e-3,
                'optimizer': 'AdamW',
                'weight_decay': 1e-5,
                'scheduler': 'cosine',
                'description': f'阶段 {stage}'
            }
    
    def create_quick_config(self) -> Dict[str, Any]:
        """创建快速训练配置"""
        quick_config = self.config.copy()
        quick_config.update({
            'num_train_samples': 200,
            'num_val_samples': 50,
            'num_test_samples': 20,
            'batch_size': 16,
            'num_workers': 0,
            'early_stopping': True,
            'patience': 5,
            'multi_stage_config': {
                1: {
                    'epochs': 10,
                    'learning_rate': 1e-3,
                    'optimizer': 'AdamW',
                    'weight_decay': 1e-5,
                    'scheduler': 'cosine',
                    'description': '快速预训练'
                },
                2: {
                    'epochs': 15,
                    'learning_rate': 5e-4,
                    'optimizer': 'AdamW',
                    'weight_decay': 1e-5,
                    'scheduler': 'cosine',
                    'description': '快速物理强化'
                },
                3: {
                    'epochs': 20,
                    'learning_rate': 1e-4,
                    'optimizer': 'AdamW',
                    'weight_decay': 1e-6,
                    'scheduler': 'cosine',
                    'description': '快速精细调优'
                }
            }
        })
        return quick_config
    
    def create_stable_config(self) -> Dict[str, Any]:
        """创建稳定训练配置（用于修复损失过高问题）"""
        stable_config = self.config.copy()
        stable_config.update({
            'num_train_samples': 50,
            'num_val_samples': 20,
            'num_test_samples': 10,
            'batch_size': 10,
            'num_workers': 0,
            'early_stopping': True,
            'patience': 3,
            'multi_stage_config': {
                1: {
                    'epochs': 50,
                    'learning_rate': 1e-4,
                    'optimizer': 'AdamW',
                    'weight_decay': 1e-5,
                    'scheduler': 'cosine',
                    'warmup_epochs': 5,
                    'description': '稳定预训练'
                },
                2: {
                    'epochs': 100,
                    'learning_rate': 5e-5,
                    'optimizer': 'AdamW',
                    'weight_decay': 1e-5,
                    'scheduler': 'cosine',
                    'description': '稳定物理强化'
                },
                3: {
                    'epochs': 50,
                    'learning_rate': 1e-5,
                    'optimizer': 'AdamW',
                    'weight_decay': 1e-6,
                    'scheduler': 'cosine',
                    'description': '稳定精细调优'
                }
            },
            'loss_weights': {
                'physics': 0.1,
                'boundary': 1.0,
                'initial': 1.0,
                'data': 0.1
            },
            'gradient_clip_value': 0.5,
            'gradient_clip_norm': 0.5
        })
        return stable_config
    
    def validate_config(self) -> bool:
        """验证配置有效性"""
        try:
            # 检查必要字段
            required_fields = ['input_dim', 'output_dim', 'stage']
            for field in required_fields:
                if field not in self.config:
                    logger.error(f"缺少必要配置字段: {field}")
                    return False
            
            # 检查数值范围
            if self.config['batch_size'] <= 0:
                logger.error("batch_size 必须大于0")
                return False
            
            if self.config['learning_rate'] <= 0 or self.config['learning_rate'] > 1:
                logger.error("learning_rate 必须在 (0, 1] 范围内")
                return False
            
            # 检查设备
            if self.config['device'] not in ['cpu', 'cuda']:
                logger.error("device 必须是 'cpu' 或 'cuda'")
                return False
            
            logger.info("配置验证通过")
            return True
            
        except Exception as e:
            logger.error(f"配置验证失败: {str(e)}")
            return False