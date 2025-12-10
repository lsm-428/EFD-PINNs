"""
训练稳定性管理器

提供学习率预热、NaN/Inf 检测恢复、梯度裁剪等功能
"""

import logging
from typing import Dict, Any, Optional
import torch
import torch.nn as nn

logger = logging.getLogger("EWP-TrainingStabilizer")


class TrainingStabilizer:
    """
    训练稳定性管理器
    
    功能:
    - 学习率预热
    - NaN/Inf 检测和恢复
    - 梯度裁剪
    
    Args:
        model: 神经网络模型
        config: 训练配置字典
    """
    
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        self.model = model
        self.grad_clip = config.get("gradient_clip", 1.0)
        self.warmup_epochs = config.get("warmup_epochs", 1000)
        self.base_lr = config.get("learning_rate", 5e-4)
        self.last_valid_state: Optional[Dict[str, torch.Tensor]] = None
        self.nan_recovery_count = 0
    
    def get_warmup_lr(self, epoch: int) -> float:
        """
        获取预热阶段学习率
        
        在前 warmup_epochs 个 epoch 内，学习率从 0 线性增加到 base_lr。
        
        Args:
            epoch: 当前 epoch
            
        Returns:
            当前学习率
        """
        if epoch < self.warmup_epochs:
            return self.base_lr * (epoch + 1) / self.warmup_epochs
        return self.base_lr
    
    def save_valid_state(self) -> None:
        """保存当前有效的模型状态"""
        self.last_valid_state = {
            k: v.clone().detach()
            for k, v in self.model.state_dict().items()
        }
    
    def restore_on_nan(
        self,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer
    ) -> bool:
        """
        检测 NaN/Inf 并恢复模型状态
        
        如果损失为 NaN 或 Inf，恢复上一个有效状态并减半学习率。
        
        Args:
            loss: 当前损失
            optimizer: 优化器
            
        Returns:
            是否进行了恢复
        """
        if torch.isnan(loss) or torch.isinf(loss):
            if self.last_valid_state is not None:
                self.model.load_state_dict(self.last_valid_state)
                
                # 减半学习率
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5
                
                self.nan_recovery_count += 1
                logger.warning(
                    f"NaN/Inf detected! Restored model state and halved learning rate. "
                    f"Recovery count: {self.nan_recovery_count}"
                )
                return True
            else:
                logger.error("NaN/Inf detected but no valid state to restore!")
        return False
    
    def clip_gradients(self) -> float:
        """
        梯度裁剪
        
        Returns:
            裁剪前的梯度范数
        """
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.grad_clip
        )
        return grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
    
    def update_lr(self, optimizer: torch.optim.Optimizer, epoch: int) -> None:
        """
        更新学习率（预热阶段）
        
        Args:
            optimizer: 优化器
            epoch: 当前 epoch
        """
        if epoch < self.warmup_epochs:
            lr = self.get_warmup_lr(epoch)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
