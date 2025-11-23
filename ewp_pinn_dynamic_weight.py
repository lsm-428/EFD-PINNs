"""
动态物理损失权重调整模块

该模块实现了基于训练进度和性能指标的物理损失权重动态调整策略，
旨在更好地平衡物理约束和数据拟合，提高模型训练效果。

主要功能:
1. 基于训练阶段的权重调整
2. 基于损失比例的自适应权重调整
3. 基于验证性能的反馈调整
4. 权重变化平滑处理，避免剧烈波动
"""

import torch
import numpy as np
import logging
from typing import Dict, Optional, Tuple, List, Union

logger = logging.getLogger('DynamicPhysicsWeight')

class DynamicPhysicsWeightScheduler:
    """
    动态物理损失权重调度器
    
    根据训练进度和性能指标动态调整物理损失权重，实现物理约束和数据拟合的平衡
    """
    
    def __init__(self, 
                 initial_weight: float = 0.1,
                 min_weight: float = 0.01,
                 max_weight: float = 5.0,
                 adjustment_strategy: str = "adaptive",
                 smoothing_factor: float = 0.9,
                 adjustment_interval: int = 100,
                 target_loss_ratio: float = 1.0,
                 patience: int = 500,
                 verbose: bool = True):
        """
        初始化动态物理权重调度器
        
        参数:
            initial_weight: 初始物理损失权重
            min_weight: 最小物理损失权重
            max_weight: 最大物理损失权重
            adjustment_strategy: 调整策略 ("stage", "adaptive", "performance", "combined")
            smoothing_factor: 权重变化平滑因子 (0-1), 越接近1变化越平滑
            adjustment_interval: 调整间隔 (步数)
            target_loss_ratio: 目标数据损失与物理损失比例
            patience: 性能不提升时的等待步数
            verbose: 是否打印调整信息
        """
        # 兼容错误传入的字典/结构体
        try:
            if isinstance(initial_weight, dict):
                iw = initial_weight.get('initial_weight', initial_weight.get('value', 0.1))
                self.initial_weight = float(iw)
            else:
                self.initial_weight = float(initial_weight)
        except Exception:
            self.initial_weight = 0.1
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.adjustment_strategy = adjustment_strategy
        self.smoothing_factor = smoothing_factor
        self.adjustment_interval = adjustment_interval
        self.target_loss_ratio = target_loss_ratio
        self.patience = patience
        self.verbose = verbose
        
        # 当前权重
        self.current_weight = self.initial_weight
        
        # 训练状态跟踪
        self.step_count = 0
        self.last_adjustment_step = 0
        self.best_val_loss = float('inf')
        self.steps_without_improvement = 0
        
        # 损失历史记录
        self.loss_history = {
            'data_loss': [],
            'physics_loss': [],
            'val_loss': [],
            'weight_history': []
        }
        
        # 阶段性权重设置
        self.stage_weights = {
            1: 0.05,   # 初始阶段，主要关注数据拟合
            2: 0.1,    # 中期阶段，引入物理约束
            3: 0.5,    # 后期阶段，加强物理约束
            4: 1.0     # 最后阶段，平衡物理约束和数据拟合
        }
        
        logger.info(f"动态物理权重调度器初始化: 初始权重={initial_weight}, 策略={adjustment_strategy}")
    
    def get_current_weight(self) -> float:
        """获取当前物理损失权重"""
        return self.current_weight
    
    def update(self, 
               data_loss: float, 
               physics_loss: float, 
               val_loss: Optional[float] = None,
               epoch: Optional[int] = None,
               stage: Optional[int] = None) -> float:
        """
        更新物理损失权重
        
        参数:
            data_loss: 当前数据损失
            physics_loss: 当前物理损失
            val_loss: 当前验证损失 (可选)
            epoch: 当前训练轮次 (可选)
            stage: 当前训练阶段 (可选)
            
        返回:
            更新后的物理损失权重
        """
        # 更新步数
        self.step_count += 1
        
        # 记录损失历史
        self.loss_history['data_loss'].append(data_loss)
        self.loss_history['physics_loss'].append(physics_loss)
        if val_loss is not None:
            self.loss_history['val_loss'].append(val_loss)
        self.loss_history['weight_history'].append(self.current_weight)
        
        # 检查是否需要调整权重
        if self.step_count - self.last_adjustment_step >= self.adjustment_interval:
            new_weight = self._compute_new_weight(data_loss, physics_loss, val_loss, epoch, stage)
            
            # 应用平滑处理
            smoothed_weight = (self.smoothing_factor * self.current_weight + 
                              (1 - self.smoothing_factor) * new_weight)
            
            # 确保权重在合理范围内
            smoothed_weight = max(self.min_weight, min(self.max_weight, smoothed_weight))
            
            # 记录权重变化
            weight_change = abs(smoothed_weight - self.current_weight)
            if weight_change > 1e-6 and self.verbose:
                logger.info(f"步骤 {self.step_count}: 物理权重 {self.current_weight:.6f} -> {smoothed_weight:.6f} "
                          f"(变化: {weight_change:.6f}, 数据损失: {data_loss:.6f}, 物理损失: {physics_loss:.6f})")
            
            # 更新当前权重
            self.current_weight = smoothed_weight
            self.last_adjustment_step = self.step_count
            
            # 更新最佳验证损失和等待步数
            if val_loss is not None:
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.steps_without_improvement = 0
                else:
                    self.steps_without_improvement += 1
        
        return self.current_weight
    
    def _compute_new_weight(self, 
                          data_loss: float, 
                          physics_loss: float, 
                          val_loss: Optional[float] = None,
                          epoch: Optional[int] = None,
                          stage: Optional[int] = None) -> float:
        """计算新的物理损失权重"""
        if self.adjustment_strategy == "stage" and stage is not None:
            return self._stage_based_adjustment(stage)
        elif self.adjustment_strategy == "adaptive":
            return self._adaptive_adjustment(data_loss, physics_loss)
        elif self.adjustment_strategy == "performance" and val_loss is not None:
            return self._performance_based_adjustment(val_loss)
        elif self.adjustment_strategy == "combined":
            # 组合多种策略
            stage_weight = self._stage_based_adjustment(stage) if stage is not None else self.current_weight
            adaptive_weight = self._adaptive_adjustment(data_loss, physics_loss)
            performance_weight = self._performance_based_adjustment(val_loss) if val_loss is not None else self.current_weight
            
            # 加权平均
            return (0.4 * stage_weight + 0.4 * adaptive_weight + 0.2 * performance_weight)
        else:
            # 默认使用自适应调整
            return self._adaptive_adjustment(data_loss, physics_loss)
    
    def _stage_based_adjustment(self, stage: int) -> float:
        """基于训练阶段的权重调整"""
        if stage in self.stage_weights:
            return self.stage_weights[stage]
        else:
            # 如果阶段超出预定义范围，使用最后一个阶段的权重
            return self.stage_weights[max(self.stage_weights.keys())]
    
    def _adaptive_adjustment(self, data_loss: float, physics_loss: float) -> float:
        """基于损失比例的自适应权重调整"""
        # 避免除零
        if physics_loss < 1e-8:
            physics_loss = 1e-8
        
        # 计算当前损失比例
        current_ratio = data_loss / physics_loss
        
        # 根据比例调整权重
        if current_ratio > self.target_loss_ratio * 2:
            # 数据损失远大于物理损失，增加物理权重
            adjustment_factor = 1.2
        elif current_ratio > self.target_loss_ratio:
            # 数据损失略大于物理损失，略微增加物理权重
            adjustment_factor = 1.1
        elif current_ratio < self.target_loss_ratio / 2:
            # 物理损失远大于数据损失，减少物理权重
            adjustment_factor = 0.8
        elif current_ratio < self.target_loss_ratio:
            # 物理损失略大于数据损失，略微减少物理权重
            adjustment_factor = 0.9
        else:
            # 比例接近目标，不调整
            adjustment_factor = 1.0
        
        # 应用调整因子
        new_weight = self.current_weight * adjustment_factor
        
        return new_weight
    
    def _performance_based_adjustment(self, val_loss: float) -> float:
        """基于验证性能的权重调整"""
        # 如果验证性能提升，保持当前权重
        if val_loss < self.best_val_loss:
            return self.current_weight
        
        # 如果性能长时间没有提升，尝试调整权重
        if self.steps_without_improvement > self.patience:
            # 根据当前权重决定调整方向
            if self.current_weight > (self.min_weight + self.max_weight) / 2:
                # 当前权重较高，尝试降低权重
                return self.current_weight * 0.9
            else:
                # 当前权重较低，尝试增加权重
                return self.current_weight * 1.1
        
        # 默认保持当前权重
        return self.current_weight
    
    def get_loss_history(self) -> Dict[str, List[float]]:
        """获取损失历史记录"""
        return self.loss_history
    
    def reset(self):
        """重置调度器状态"""
        self.current_weight = self.initial_weight
        self.step_count = 0
        self.last_adjustment_step = 0
        self.best_val_loss = float('inf')
        self.steps_without_improvement = 0
        self.loss_history = {
            'data_loss': [],
            'physics_loss': [],
            'val_loss': [],
            'weight_history': []
        }
        logger.info("动态物理权重调度器已重置")


class PhysicsWeightIntegration:
    """
    物理权重集成类，用于将动态权重调整集成到现有训练流程中
    """
    
    def __init__(self, 
                 weight_scheduler: DynamicPhysicsWeightScheduler,
                 integration_method: str = "multiplicative"):
        """
        初始化物理权重集成
        
        参数:
            weight_scheduler: 动态权重调度器
            integration_method: 集成方法 ("multiplicative", "additive", "replacement")
        """
        self.weight_scheduler = weight_scheduler
        self.integration_method = integration_method
        
    def apply_dynamic_weight(self, 
                           base_physics_loss, 
                           data_loss,
                           val_loss: Optional[torch.Tensor] = None,
                           epoch: Optional[int] = None,
                           stage: Optional[int] = None) -> torch.Tensor:
        """
        应用动态权重到物理损失
        
        参数:
            base_physics_loss: 基础物理损失
            data_loss: 数据损失
            val_loss: 验证损失 (可选)
            epoch: 当前训练轮次 (可选)
            stage: 当前训练阶段 (可选)
            
        返回:
            应用动态权重后的物理损失
        """
        # 获取当前动态权重
        # 统一为张量
        if not isinstance(base_physics_loss, torch.Tensor):
            base_physics_loss = torch.tensor(float(base_physics_loss))
        if not isinstance(data_loss, torch.Tensor):
            data_loss = torch.tensor(float(data_loss))
        if val_loss is not None and not isinstance(val_loss, torch.Tensor):
            val_loss = torch.tensor(float(val_loss))

        current_weight = self.weight_scheduler.update(
            data_loss.item(),
            base_physics_loss.item(),
            val_loss.item() if val_loss is not None else None,
            epoch, stage
        )
        
        weight_tensor = torch.tensor(current_weight, 
                                    device=base_physics_loss.device, 
                                    dtype=base_physics_loss.dtype)
        
        # 根据集成方法应用权重
        if self.integration_method == "multiplicative":
            return base_physics_loss * weight_tensor
        elif self.integration_method == "additive":
            return base_physics_loss + weight_tensor
        elif self.integration_method == "replacement":
            # 使用权重作为新的损失值
            return weight_tensor * torch.mean(base_physics_loss)
        else:
            # 默认使用乘法
            return base_physics_loss * weight_tensor
    
    def get_current_weight(self) -> float:
        """获取当前权重值"""
        return self.weight_scheduler.get_current_weight()
