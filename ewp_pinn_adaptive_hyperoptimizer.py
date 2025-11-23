import torch
import numpy as np
import json
import os
from datetime import datetime
import logging
from typing import Dict, List, Any, Optional, Tuple

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('EWPINN_HyperOptimizer')

class AdaptiveHyperparameterOptimizer:
    """
    自适应超参数优化器 - 为EWPINN模型提供动态超参数调整功能
    """
    
    def __init__(self, config_path: Optional[str] = None, device: Optional[str] = None):
        """
        初始化自适应超参数优化器
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.optimization_history = []
        self.iteration = 0
        
        # 默认配置
        self.config = {
            'learning_rate': {
                'initial': 1e-3,
                'min': 1e-6,
                'max': 1e-2,
                'patience': 5,
                'factor': 0.5,
                'cooldown': 3
            },
            'batch_size': {
                'initial': 32,
                'min': 8,
                'max': 128,
                'scale_factor': 2,
                'patience': 8
            },
            'physics_constraint': {
                'initial_weight': 0.1,
                'max_weight': 0.8,
                'growth_rate': 0.05
            },
            'regularization': {
                'weight_decay': {
                    'initial': 1e-4,
                    'min': 1e-6,
                    'max': 1e-3,
                    'patience': 10
                },
                'dropout': {
                    'initial': 0.1
                }
            },
            'early_stopping': {
                'patience': 15,
                'min_delta': 1e-4
            }
        }
        
        # 加载配置文件
        if config_path and os.path.exists(config_path):
            self._load_config(config_path)
        
        # 初始化当前超参数
        self.current_hyperparams = {
            'learning_rate': self.config['learning_rate']['initial'],
            'batch_size': self.config['batch_size']['initial'],
            'physics_weight': self.config['physics_constraint']['initial_weight'],
            'weight_decay': self.config['regularization']['weight_decay']['initial'],
            'dropout_rate': self.config['regularization']['dropout']['initial']
        }
        
        # 初始化状态跟踪器
        self.state = {
            'lr_patience': 0,
            'batch_size_patience': 0,
            'reg_patience': 0,
            'cooldown': 0,
            'best_val_loss': float('inf'),
            'best_epoch': 0,
            'loss_trend': [],
            'lr_history': [],
            'physics_weight_history': []
        }
        
        logger.info(f"✅ 自适应超参数优化器初始化完成")
    
    def _load_config(self, config_path: str):
        """
        从配置文件加载优化器设置
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
                if '自适应超参数' in user_config:
                    adaptive_config = user_config['自适应超参数']
                    for key, value in adaptive_config.items():
                        if key in self.config and isinstance(value, dict):
                            self.config[key].update(value)
            logger.info(f"✅ 成功加载超参数优化器配置")
        except Exception as e:
            logger.error(f"❌ 加载配置文件失败: {str(e)}")
    
    def get_hyperparams(self) -> Dict[str, float]:
        """
        获取当前超参数
        """
        return self.current_hyperparams.copy()
    
    def update_optimizer_lr(self, optimizer: torch.optim.Optimizer):
        """
        更新优化器的学习率
        """
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.current_hyperparams['learning_rate']
    
    def adaptive_update(self, metrics: Dict[str, float], epoch: int = 0) -> bool:
        """
        基于性能指标自适应更新超参数
        """
        self.iteration += 1
        should_stop = False
        
        # 记录指标
        current_val_loss = metrics.get('val_loss', float('inf'))
        self.state['loss_trend'].append(current_val_loss)
        self.state['lr_history'].append(self.current_hyperparams['learning_rate'])
        self.state['physics_weight_history'].append(self.current_hyperparams['physics_weight'])
        
        # 更新最佳指标
        if current_val_loss < self.state['best_val_loss'] - self.config['early_stopping']['min_delta']:
            self.state['best_val_loss'] = current_val_loss
            self.state['best_epoch'] = epoch
            self.state['lr_patience'] = 0
            self.state['batch_size_patience'] = 0
            self.state['reg_patience'] = 0
        
        # 检查冷却期
        if self.state['cooldown'] > 0:
            self.state['cooldown'] -= 1
            return should_stop
        
        # 自适应调整超参数
        if epoch >= 10:  # 预热后开始调整
            # 学习率调整
            self._adjust_learning_rate(current_val_loss)
            
            # 物理约束权重调整
            self._adjust_physics_weight(metrics)
            
            # 批次大小调整
            self._adjust_batch_size(metrics, epoch)
            
            # 正则化调整
            self._adjust_regularization(metrics)
        
        # 记录历史
        self.optimization_history.append({
            'epoch': epoch,
            'metrics': metrics.copy(),
            'hyperparams': self.current_hyperparams.copy()
        })
        
        # 检查早停
        if len(self.state['loss_trend']) > self.config['early_stopping']['patience']:
            recent_losses = self.state['loss_trend'][-self.config['early_stopping']['patience']:]
            if min(recent_losses) >= self.state['best_val_loss'] - self.config['early_stopping']['min_delta']:
                should_stop = True
        
        return should_stop
    
    def _adjust_learning_rate(self, val_loss: float):
        """
        调整学习率
        """
        # 如果验证损失没有改善，降低学习率
        if len(self.state['loss_trend']) >= 2:
            if self.state['loss_trend'][-1] >= self.state['loss_trend'][-2] - self.config['early_stopping']['min_delta']:
                self.state['lr_patience'] += 1
                
                if self.state['lr_patience'] >= self.config['learning_rate']['patience']:
                    new_lr = self.current_hyperparams['learning_rate'] * self.config['learning_rate']['factor']
                    new_lr = max(new_lr, self.config['learning_rate']['min'])
                    
                    if new_lr < self.current_hyperparams['learning_rate']:
                        self.current_hyperparams['learning_rate'] = new_lr
                        self.state['lr_patience'] = 0
                        self.state['cooldown'] = self.config['learning_rate']['cooldown']
                        logger.info(f"📉 学习率衰减至: {new_lr}")
    
    def _adjust_physics_weight(self, metrics: Dict[str, float]):
        """
        调整物理约束权重
        """
        data_loss = metrics.get('data_loss', 0.0)
        physics_loss = metrics.get('physics_loss', 0.0)
        
        if data_loss > 0 and physics_loss > 0:
            loss_ratio = data_loss / physics_loss
            
            # 动态调整物理权重
            if loss_ratio > 2.0 and self.current_hyperparams['physics_weight'] < self.config['physics_constraint']['max_weight']:
                new_weight = min(
                    self.current_hyperparams['physics_weight'] + self.config['physics_constraint']['growth_rate'],
                    self.config['physics_constraint']['max_weight']
                )
                self.current_hyperparams['physics_weight'] = new_weight
            elif loss_ratio < 0.5:
                new_weight = max(
                    self.current_hyperparams['physics_weight'] - self.config['physics_constraint']['growth_rate'],
                    0.0
                )
                self.current_hyperparams['physics_weight'] = new_weight
    
    def _adjust_batch_size(self, metrics: Dict[str, float], epoch: int):
        """
        调整批次大小
        """
        if len(self.state['loss_trend']) >= 10:
            recent_losses = self.state['loss_trend'][-10:]
            loss_decrease_rate = (recent_losses[0] - recent_losses[-1]) / recent_losses[0]
            
            if loss_decrease_rate < 0.05:  # 损失下降缓慢
                self.state['batch_size_patience'] += 1
                
                if self.state['batch_size_patience'] >= self.config['batch_size']['patience']:
                    new_batch_size = max(
                        self.current_hyperparams['batch_size'] // self.config['batch_size']['scale_factor'],
                        self.config['batch_size']['min']
                    )
                    
                    if new_batch_size < self.current_hyperparams['batch_size']:
                        self.current_hyperparams['batch_size'] = new_batch_size
                        self.state['batch_size_patience'] = 0
    
    def _adjust_regularization(self, metrics: Dict[str, float]):
        """
        调整正则化强度
        """
        train_loss = metrics.get('train_loss', 0.0)
        val_loss = metrics.get('val_loss', 0.0)
        
        if train_loss > 0 and val_loss > 0:
            overfitting_ratio = val_loss / train_loss
            
            if overfitting_ratio > 1.5:  # 可能过拟合
                if self.state['reg_patience'] >= self.config['regularization']['weight_decay']['patience']:
                    new_weight_decay = min(
                        self.current_hyperparams['weight_decay'] * 1.5,
                        self.config['regularization']['weight_decay']['max']
                    )
                    self.current_hyperparams['weight_decay'] = new_weight_decay
            elif overfitting_ratio < 1.1:
                new_weight_decay = max(
                    self.current_hyperparams['weight_decay'] * 0.8,
                    self.config['regularization']['weight_decay']['min']
                )
                self.current_hyperparams['weight_decay'] = new_weight_decay
                self.state['reg_patience'] = 0
    
    def save_history(self, save_path: str):
        """
        保存优化历史
        """
        try:
            history_data = {
                'optimization_history': self.optimization_history,
                'final_hyperparams': self.current_hyperparams,
                'timestamp': datetime.now().isoformat()
            }
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"❌ 保存优化历史失败: {str(e)}")
    
    def export_recommended_config(self, save_path: str):
        """
        导出推荐的配置文件
        """
        try:
            # 获取最佳超参数
            if not self.optimization_history:
                best_hyperparams = self.current_hyperparams
            else:
                best_entry = min(self.optimization_history, key=lambda x: x['metrics'].get('val_loss', float('inf')))
                best_hyperparams = best_entry['hyperparams']
            
            recommended_config = {
                '模型配置': {
                    '输入维度': 62,
                    '输出维度': 24,
                    '网络架构': {
                        '隐藏层': [128, 64, 32],
                        'Dropout': best_hyperparams['dropout_rate']
                    }
                },
                '训练配置': {
                    '批次大小': best_hyperparams['batch_size'],
                    '学习率': best_hyperparams['learning_rate'],
                    '正则化': {
                        '权重衰减': best_hyperparams['weight_decay']
                    }
                },
                '物理约束': {
                    '权重': best_hyperparams['physics_weight']
                },
                '导出时间': datetime.now().isoformat()
            }
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(recommended_config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"❌ 导出推荐配置失败: {str(e)}")

# 集成适配器
def integrate_adaptive_optimizer(config_path: str = None):
    """
    集成自适应超参数优化器
    """
    optimizer = AdaptiveHyperparameterOptimizer(config_path=config_path)
    logger.info("✅ 自适应超参数优化器已集成")
    return optimizer