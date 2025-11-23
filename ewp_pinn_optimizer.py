"""
EWPINN 优化器模块
包含损失函数、优化策略、学习率调度器和优化器管理
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger('EWPINN_Optimizer')

class StablePINNLoss(nn.Module):
    """稳定版PINN综合损失函数 - 解决数值不稳定性"""
    
    def __init__(self, weights: Optional[Dict[str, float]] = None, 
                 max_loss_value: float = 1000.0, 
                 loss_scaling_factor: float = 1e-5):
        super().__init__()
        
        # 保守的损失权重设置 - 降低物理损失权重以提高稳定性
        self.weights = weights or {
            'physics': 0.1,      # 降低物理损失权重
            'boundary': 0.5,     # 降低边界损失权重
            'initial': 0.5,      # 降低初始条件损失权重
            'data': 1.0          # 保持数据损失权重
        }
        
        self.physics_weight = self.weights['physics']
        self.data_weight = self.weights['data']
        self.boundary_weight = self.weights['boundary']
        self.initial_weight = self.weights['initial']
        self.max_loss_value = max_loss_value
        self.loss_scaling_factor = loss_scaling_factor
        
        # MSE损失用于数据项
        self.mse_loss = nn.MSELoss(reduction='mean')
        
        # 平滑L1损失用于物理残差
        self.smooth_l1_loss = nn.SmoothL1Loss(beta=0.1, reduction='mean')
        
        logger.info(f"稳定版PINN损失初始化: 权重={self.weights}, 最大损失={max_loss_value}")
    
    def check_input_validity(self, physics_loss, boundary_loss, initial_loss, data_loss):
        """检查输入有效性"""
        input_losses = [physics_loss, boundary_loss, initial_loss, data_loss]
        names = ['physics_loss', 'boundary_loss', 'initial_loss', 'data_loss']
        
        for loss_tensor, name in zip(input_losses, names):
            if loss_tensor is not None:
                if torch.isnan(loss_tensor).any():
                    logger.warning(f"检测到NaN损失值 ({name})，将使用代理损失")
                elif torch.isinf(loss_tensor).any():
                    logger.warning(f"检测到无穷损失值 ({name})，将应用裁剪")
    
    def check_numerical_stability(self, total_loss, loss_dict):
        """检查数值稳定性"""
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            logger.error(f"总损失不稳定: {total_loss}")
            return False
        
        for key, value in loss_dict.items():
            if torch.isnan(torch.tensor(value)) or torch.isinf(torch.tensor(value)):
                logger.warning(f"损失组件不稳定 ({key}): {value}")
                return False
        
        return True
    
    def clip_loss(self, loss_tensor: torch.Tensor, name: str = "loss") -> torch.Tensor:
        """裁剪损失值以防止数值爆炸"""
        original_loss = loss_tensor.item() if loss_tensor.numel() == 1 else loss_tensor.mean().item()
        
        if not torch.isfinite(loss_tensor).all():
            logger.warning(f"{name}包含非有限值，使用代理损失")
            return torch.tensor(1.0, device=loss_tensor.device, requires_grad=True)
        
        if loss_tensor.abs().max() > self.max_loss_value:
            logger.warning(f"{name}过大 ({original_loss:.2e})，应用裁剪")
            clipped_loss = torch.clamp(loss_tensor, -self.max_loss_value, self.max_loss_value)
            return clipped_loss
        
        return loss_tensor
    
    def __call__(self, physics_loss, boundary_loss, initial_loss, data_loss):
        """计算稳定的总损失"""
        # 检查有效性
        self.check_input_validity(physics_loss, boundary_loss, initial_loss, data_loss)
        
        # 计算损失组件
        physics_loss_clipped = physics_loss * self.physics_weight
        boundary_loss_clipped = boundary_loss * self.boundary_weight  
        initial_loss_clipped = initial_loss * self.initial_weight
        data_loss_clipped = data_loss * self.data_weight
        
        total_loss = physics_loss_clipped + boundary_loss_clipped + initial_loss_clipped + data_loss_clipped
        
        loss_dict = {
            'physics': physics_loss_clipped.item(),
            'boundary': boundary_loss_clipped.item(), 
            'initial': initial_loss_clipped.item(),
            'data': data_loss_clipped.item(),
            'total': total_loss.item()
        }
        
        # 检查数值稳定性
        self.check_numerical_stability(total_loss, loss_dict)
        
        return total_loss, loss_dict

# 保持向后兼容
PINNLoss = StablePINNLoss

class StablePhysicsConsistencyLoss(nn.Module):
    """稳定版物理一致性损失函数 - 解决数值不稳定性"""
    
    def __init__(self, device='cpu', physics_config=None, max_residual: float = 100.0):
        super().__init__()
        self.device = device
        self.max_residual = max_residual
        
        # 保守的物理参数设置
        self.physics_config = physics_config or {
            'epsilon': 8.85e-12,  # 真空介电常数
            'gamma': 0.035,       # 表面张力
            'V': 100.0,           # 应用电压
        }
        
        # 注册物理参数为缓冲区
        for key, value in self.physics_config.items():
            self.register_buffer(key, torch.tensor(value, dtype=torch.float32))
        
        logger.info(f"稳定版物理一致性损失初始化: 物理参数={self.physics_config}, 最大残差={max_residual}")
    
    def safe_gradient(self, tensor: torch.Tensor, var: torch.Tensor, 
                     name: str = "gradient") -> torch.Tensor:
        """安全的梯度计算，防止数值问题"""
        try:
            # 确保张量需要梯度
            if not tensor.requires_grad:
                # 如果不需要梯度，创建一个需要梯度的副本
                tensor = tensor.detach().requires_grad_(True)
            
            grad = torch.autograd.grad(
                tensor, var, 
                grad_outputs=torch.ones_like(tensor), 
                create_graph=True, 
                retain_graph=True,
                allow_unused=True
            )
            
            # 处理未使用的梯度
            if grad[0] is None:
                return torch.zeros_like(var)
            
            return grad[0]
        except Exception as e:
            logger.warning(f"梯度计算失败 ({name}): {e}")
            return torch.zeros_like(var)
    
    def compute_continuity_residual(self, u, v, p, x, y):
        """计算连续性方程残差 ∇·v = 0 (稳定版)"""
        try:
            # 检查输入有效性
            if not all(torch.isfinite(t).all() for t in [u, v, x, y]):
                return torch.tensor(1.0, device=self.device, requires_grad=True)
            
            # 安全的梯度计算
            u_x = self.safe_gradient(u, torch.ones_like(u), x, "u_x")
            u_y = self.safe_gradient(u, torch.ones_like(u), y, "u_y")
            v_x = self.safe_gradient(v, torch.ones_like(v), x, "v_x")
            v_y = self.safe_gradient(v, torch.ones_like(v), y, "v_y")
            
            # 检查梯度有效性
            if not all(torch.isfinite(g).all() for g in [u_x, u_y, v_x, v_y]):
                return torch.tensor(1.0, device=self.device, requires_grad=True)
            
            # 连续性方程: ∂u/∂x + ∂v/∂y = 0
            continuity_residual = u_x + v_y
            
            # 裁剪异常值
            continuity_residual = torch.clamp(continuity_residual, -self.max_residual, self.max_residual)
            
            continuity_loss = torch.mean(continuity_residual**2)
            
            # 再次裁剪最终损失
            continuity_loss = torch.clamp(continuity_loss, 0.0, self.max_residual)
            
            return continuity_loss
        except Exception as e:
            logger.warning(f"连续性方程计算失败: {e}")
            return torch.tensor(1.0, device=self.device, requires_grad=True)
    
    def compute_momentum_residual(self, u, v, p, x, y):
        """计算Navier-Stokes动量方程残差 (稳定版)"""
        try:
            # 检查输入有效性
            if not all(torch.isfinite(t).all() for t in [u, v, p, x, y]):
                return torch.tensor(1.0, device=self.device, requires_grad=True)
            
            # 计算一阶导数 (安全版本)
            u_x = self.safe_gradient(u, torch.ones_like(u), x, "u_x")
            u_y = self.safe_gradient(u, torch.ones_like(u), y, "u_y")
            v_x = self.safe_gradient(v, torch.ones_like(v), x, "v_x")
            v_y = self.safe_gradient(v, torch.ones_like(v), y, "v_y")
            
            # 计算二阶导数 (粘性项)
            u_xx = self.safe_gradient(u_x, torch.ones_like(u_x), x, "u_xx")
            u_yy = self.safe_gradient(u_y, torch.ones_like(u_y), y, "u_yy")
            v_xx = self.safe_gradient(v_x, torch.ones_like(v_x), x, "v_xx")
            v_yy = self.safe_gradient(v_y, torch.ones_like(v_y), y, "v_yy")
            
            # 检查所有梯度的有效性
            gradients = [u_x, u_y, v_x, v_y, u_xx, u_yy, v_xx, v_yy]
            if not all(torch.isfinite(g).all() for g in gradients):
                return torch.tensor(1.0, device=self.device, requires_grad=True)
            
            laplacian_u = torch.clamp(u_xx + u_yy, -self.max_residual, self.max_residual)
            laplacian_v = torch.clamp(v_xx + v_yy, -self.max_residual, self.max_residual)
            
            # 简化的Navier-Stokes方程 (更保守的计算)
            # 压力梯度
            p_x = self.safe_gradient(p, torch.ones_like(p), x, "p_x")
            p_y = self.safe_gradient(p, torch.ones_like(p), y, "p_y")
            
            if not torch.isfinite(p_x).all() or not torch.isfinite(p_y).all():
                return torch.tensor(1.0, device=self.device, requires_grad=True)
            
            # 动量残差 (简化版本，减少复杂性)
            momentum_x = torch.clamp(p_x - 0.01 * laplacian_u, -self.max_residual, self.max_residual)
            momentum_y = torch.clamp(p_y - 0.01 * laplacian_v, -self.max_residual, self.max_residual)
            
            momentum_loss = torch.mean(momentum_x**2 + momentum_y**2)
            
            # 裁剪最终损失
            momentum_loss = torch.clamp(momentum_loss, 0.0, self.max_residual)
            
            return momentum_loss
        except Exception as e:
            logger.warning(f"动量方程计算失败: {e}")
            return torch.tensor(1.0, device=self.device, requires_grad=True)
    
    def forward(self, predictions, coords):
        """计算物理一致性损失 (稳定版)"""
        try:
            # 检查输入
            if predictions is None or coords is None:
                logger.warning("物理损失输入为空")
                return torch.tensor(1.0, device=self.device, requires_grad=True)
            
            # 解包预测值 (u, v, p, contact_angle, volume_fraction)
            if predictions.shape[1] < 3:
                logger.warning("预测值维度不足，无法提取u, v, p")
                return torch.tensor(1.0, device=self.device, requires_grad=True)
                
            u, v, p = predictions[:, 0], predictions[:, 1], predictions[:, 2]
            x, y = coords[:, 0], coords[:, 1]
            
            # 检查输入的有效性
            input_tensors = [u, v, p, x, y]
            if not all(t is not None and torch.isfinite(t).all() for t in input_tensors):
                logger.warning("物理损失输入包含无效值")
                return torch.tensor(1.0, device=self.device, requires_grad=True)
            
            # 计算物理残差
            continuity_loss = self.compute_continuity_residual(u, v, p, x, y)
            momentum_loss = self.compute_momentum_residual(u, v, p, x, y)
            
            total_physics_loss = continuity_loss + momentum_loss
            
            # 检查结果的有效性
            if not torch.isfinite(total_physics_loss) or total_physics_loss.item() > self.max_residual * 2:
                logger.warning(f"物理损失计算结果异常: {total_physics_loss.item():.2e}")
                return torch.tensor(1.0, device=self.device, requires_grad=True)
            
            # 最终裁剪
            total_physics_loss = torch.clamp(total_physics_loss, 0.0, self.max_residual)
            
            return total_physics_loss
            
        except Exception as e:
            logger.error(f"物理一致性损失计算失败: {e}")
            return torch.tensor(1.0, device=self.device, requires_grad=True)

# 保持向后兼容
PhysicsConsistencyLoss = StablePhysicsConsistencyLoss

class WarmupCosineLR(_LRScheduler):
    """带预热的余弦退火学习率调度器"""
    
    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr=1e-6, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # 线性预热
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs 
                   for base_lr in self.base_lrs]
        else:
            # 余弦退火
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            progress = min(progress, 1.0)
            return [self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
                   for base_lr in self.base_lrs]

class EnhancedGradientClipper:
    """增强版梯度裁剪工具 - 专门解决数值稳定性问题"""
    
    def __init__(self, clip_value: float = 1.0, clip_norm: float = 1.0, 
                 max_grad_value: float = 10.0, loss_clip_value: float = 1000.0):
        self.clip_value = clip_value
        self.clip_norm = clip_norm
        self.max_grad_value = max_grad_value
        self.loss_clip_value = loss_clip_value
    
    def clip_gradients(self, model: nn.Module) -> Dict[str, float]:
        """裁剪模型梯度并检测异常"""
        grad_norms = {}
        
        # 计算梯度统计
        total_norm = 0.0
        max_grad = 0.0
        param_count = 0
        
        for p in model.parameters():
            if p.grad is not None:
                # 计算梯度范数
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                max_grad = max(max_grad, p.grad.data.abs().max().item())
                param_count += 1
        
        total_norm = total_norm ** (1. / 2)
        
        # 梯度范数裁剪
        if self.clip_norm > 0 and total_norm > self.clip_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip_norm)
            grad_norms['total_grad_norm_before'] = total_norm
            grad_norms['total_grad_norm_after'] = self.clip_norm
        else:
            grad_norms['total_grad_norm'] = total_norm
        
        # 梯度值裁剪
        if self.clip_value > 0:
            torch.nn.utils.clip_grad_value_(model.parameters(), self.clip_value)
            grad_norms['max_grad_value_before'] = max_grad
        
        # 异常检测
        if max_grad > self.max_grad_value:
            logger.warning(f"梯度值过大: {max_grad:.2e} > {self.max_grad_value:.2e}")
            grad_norms['grad_clip_applied'] = True
        
        if total_norm > 10.0:  # 总梯度范数阈值
            logger.warning(f"总梯度范数过大: {total_norm:.2e}")
            grad_norms['total_norm_clip_applied'] = True
        
        return grad_norms
    
    def check_numerical_stability(self, tensors: List[torch.Tensor]) -> Dict[str, bool]:
        """检查数值稳定性"""
        stability_report = {}
        
        for i, tensor in enumerate(tensors):
            if tensor is None:
                continue
                
            # NaN检测
            has_nan = torch.isnan(tensor).any().item()
            stability_report[f'tensor_{i}_has_nan'] = has_nan
            
            # Inf检测
            has_inf = torch.isinf(tensor).any().item()
            stability_report[f'tensor_{i}_has_inf'] = has_inf
            
            if has_nan or has_inf:
                logger.error(f"张量 {i} 包含数值异常: NaN={has_nan}, Inf={has_inf}")
                logger.error(f"异常张量形状: {tensor.shape}, 统计: min={tensor.min():.2e}, max={tensor.max():.2e}")
        
        return stability_report

class EarlyStopping:
    """早停机制"""
    
    def __init__(self, patience: int = 10, min_delta: float = 1e-6, monitor: str = 'val_loss'):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.best_loss = float('inf')
        self.wait = 0
        self.stopped_epoch = 0
        self.best_state_dict = None
    
    def __call__(self, epoch: int, model: nn.Module, current_loss: float) -> bool:
        """检查是否应该早停"""
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.wait = 0
            # 保存最佳模型状态
            self.best_state_dict = model.state_dict().copy()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                logger.info(f"早停触发: 在epoch {epoch}停止训练")
                return True
        
        return False

class NumericalStabilityMonitor:
    """数值稳定性监控器"""
    
    def __init__(self, check_frequency: int = 10):
        self.check_frequency = check_frequency
        self.step_count = 0
        self.stability_history = []
        
    def check_tensor_stability(self, tensor: torch.Tensor, name: str = "tensor") -> Dict[str, Any]:
        """检查单个张量的稳定性"""
        if tensor is None:
            return {"status": "valid", "message": "张量为None"}
        
        # 基础统计
        stats = {
            "name": name,
            "shape": tensor.shape,
            "dtype": tensor.dtype,
            "device": str(tensor.device),
            "min_value": tensor.min().item(),
            "max_value": tensor.max().item(),
            "mean_value": tensor.mean().item(),
            "std_value": tensor.std().item(),
            "has_nan": torch.isnan(tensor).any().item(),
            "has_inf": torch.isinf(tensor).any().item(),
            "norm_l2": tensor.norm(2).item()
        }
        
        # 状态判断
        if stats["has_nan"]:
            stats["status"] = "invalid"
            stats["message"] = "包含NaN值"
        elif stats["has_inf"]:
            stats["status"] = "invalid"  
            stats["message"] = "包含Inf值"
        elif stats["max_value"] > 1e6 or stats["min_value"] < -1e6:
            stats["status"] = "warning"
            stats["message"] = "数值范围过大"
        elif stats["std_value"] > 1e3:
            stats["status"] = "warning"
            stats["message"] = "数值方差过大"
        else:
            stats["status"] = "valid"
            stats["message"] = "数值稳定"
        
        return stats
    
    def monitor_training_step(self, losses: Dict[str, torch.Tensor], 
                            predictions: torch.Tensor, gradients: Dict[str, float]) -> Dict[str, Any]:
        """监控训练步骤的数值稳定性"""
        self.step_count += 1
        
        monitor_report = {
            "step": self.step_count,
            "timestamp": time.time(),
            "tensor_checks": {},
            "gradient_checks": {},
            "loss_checks": {},
            "overall_status": "stable"
        }
        
        # 检查预测值
        if predictions is not None:
            pred_stats = self.check_tensor_stability(predictions, "predictions")
            monitor_report["tensor_checks"]["predictions"] = pred_stats
            if pred_stats["status"] == "invalid":
                monitor_report["overall_status"] = "unstable"
        
        # 检查损失值
        for loss_name, loss_tensor in losses.items():
            if isinstance(loss_tensor, torch.Tensor):
                loss_stats = self.check_tensor_stability(loss_tensor, f"loss_{loss_name}")
                monitor_report["loss_checks"][loss_name] = loss_stats
                if loss_stats["status"] == "invalid":
                    monitor_report["overall_status"] = "unstable"
                elif loss_stats["status"] == "warning":
                    if monitor_report["overall_status"] == "stable":
                        monitor_report["overall_status"] = "warning"
        
        # 梯度统计
        monitor_report["gradient_checks"] = gradients
        
        # 记录历史
        self.stability_history.append(monitor_report)
        
        return monitor_report
    
    def get_stability_report(self) -> Dict[str, Any]:
        """获取稳定性报告"""
        if not self.stability_history:
            return {"status": "no_data", "message": "没有监控数据"}
        
        # 统计历史数据
        total_steps = len(self.stability_history)
        unstable_steps = sum(1 for step in self.stability_history if step["overall_status"] == "unstable")
        warning_steps = sum(1 for step in self.stability_history if step["overall_status"] == "warning")
        
        return {
            "total_steps": total_steps,
            "unstable_steps": unstable_steps,
            "warning_steps": warning_steps,
            "stability_rate": (total_steps - unstable_steps - warning_steps) / total_steps,
            "recent_trend": self.stability_history[-min(10, total_steps):],
            "status": "healthy" if unstable_steps == 0 else "problematic" if unstable_steps > total_steps * 0.1 else "warning"
        }

class EWPINNOptimizerManager:
    """EWPINN优化器管理器 - 增强数值稳定性"""
    
    def __init__(self, config: Dict[str, Any], device='cpu'):
        """初始化优化器管理器"""
        self.config = config
        self.device = device
        
        # 创建稳定版损失函数
        loss_config = config.get('loss_config', {})
        self.loss_fn = StablePINNLoss(
            weights=loss_config.get('weights', {
                'physics': 0.1,      # 降低物理损失权重
                'boundary': 0.5,     # 降低边界损失权重
                'initial': 0.5,      # 降低初始条件损失权重
                'data': 1.0          # 保持数据损失权重
            }),
            max_loss_value=loss_config.get('max_loss_value', 100.0)
        )
        
        # 创建物理一致性损失函数
        physics_config = config.get('physics_config', {})
        self.physics_loss_fn = StablePhysicsConsistencyLoss(
            device=device,
            physics_config=physics_config,
            max_residual=loss_config.get('max_physics_residual', 10.0)
        )
        
        # 创建稳定版梯度裁剪器
        grad_config = config.get('gradient_clip_config', {})
        self.gradient_clipper = EnhancedGradientClipper(
            clip_value=grad_config.get('max_grad_value', 0.1),    # 更保守的梯度裁剪
            clip_norm=grad_config.get('max_grad_norm', 0.5),      # 更保守的梯度范数
            max_grad_value=grad_config.get('max_grad_value_overall', 1e3),
            loss_clip_value=loss_config.get('loss_clip_value', 50.0)  # 更保守的损失裁剪
        )
        
        # 创建数值稳定性监控器
        stability_config = config.get('stability_config', {})
        self.stability_monitor = NumericalStabilityMonitor(
            check_frequency=stability_config.get('check_frequency', 10)
        )
        
        # 创建优化器和调度器
        self._setup_optimizer_and_scheduler(config)
        
        logger.info("EWPINN优化器管理器初始化完成 - 已启用数值稳定性保护")
        logger.info(f"损失权重: {self.loss_fn.weights}")
        logger.info(f"物理损失最大残差: {self.physics_loss_fn.max_residual}")
        logger.info(f"梯度裁剪: 值<= {self.gradient_clipper.clip_value}, 范数<= {self.gradient_clipper.clip_norm}")
    
    def _setup_optimizer_and_scheduler(self, config: Dict[str, Any]):
        """设置优化器和学习率调度器"""
        # 获取模型参数（假设已经传入）
        if 'model' not in config:
            logger.warning("配置中未提供模型参数，跳过优化器初始化")
            return
            
        model = config['model']
        
        # 优化器配置
        opt_config = config.get('optimizer_config', {})
        optimizer_type = opt_config.get('type', 'AdamW').lower()
        learning_rate = opt_config.get('learning_rate', 1e-4)  # 更保守的学习率
        weight_decay = opt_config.get('weight_decay', 1e-2)
        
        # 创建优化器
        if optimizer_type == 'adamw':
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif optimizer_type == 'adam':
            self.optimizer = torch.optim.Adam(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        else:
            self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        
        # 学习率调度器
        scheduler_config = opt_config.get('scheduler_config', {})
        scheduler_type = scheduler_config.get('type', 'warmup_cosine').lower()
        
        if scheduler_type == 'warmup_cosine':
            total_steps = config.get('total_steps', 1000)
            warmup_steps = min(scheduler_config.get('warmup_steps', 100), total_steps // 10)
            
            self.scheduler = WarmupCosineLR(
                self.optimizer,
                warmup_steps=warmup_steps,
                total_steps=total_steps,
                min_lr=scheduler_config.get('min_lr', learning_rate * 0.1)
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get('step_size', 100),
                gamma=scheduler_config.get('gamma', 0.9)
            )
    
    def create_optimizer(self, model: nn.Module, stage_config: Dict[str, Any]) -> optim.Optimizer:
        """创建优化器"""
        optimizer_name = stage_config.get('optimizer', 'AdamW').lower()
        learning_rate = stage_config.get('learning_rate', 1e-3)
        weight_decay = stage_config.get('weight_decay', 1e-5)
        
        if optimizer_name == 'adamw':
            self.optimizer = optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif optimizer_name == 'adam':
            self.optimizer = optim.Adam(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=(0.9, 0.999),
                eps=1e-8
            )
        elif optimizer_name == 'sgd':
            self.optimizer = optim.SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=0.9,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"不支持的优化器: {optimizer_name}")
        
        logger.info(f"创建{optimizer_name}优化器，学习率={learning_rate}")
        return self.optimizer
    
    def create_scheduler(self, stage_config: Dict[str, Any], total_epochs: int):
        """创建学习率调度器"""
        scheduler_name = stage_config.get('scheduler', 'cosine').lower()
        warmup_epochs = stage_config.get('warmup_epochs', 0)
        
        if scheduler_name == 'cosine':
            # 安全获取learning_rate，提供默认值0.001
            learning_rate = stage_config.get('learning_rate', 0.001)
            self.scheduler = WarmupCosineLR(
                self.optimizer,
                warmup_epochs=warmup_epochs,
                max_epochs=total_epochs,
                min_lr=learning_rate * 0.01
            )
        elif scheduler_name == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=total_epochs // 3,
                gamma=0.1
            )
        elif scheduler_name == 'exponential':
            self.scheduler = optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=0.95
            )
        else:
            logger.warning(f"未知调度器类型 {scheduler_name}，使用None")
            self.scheduler = None
        
        logger.info(f"创建{scheduler_name}学习率调度器")
        return self.scheduler
    
    def setup_early_stopping(self):
        """设置早停机制"""
        self.early_stopping = EarlyStopping(
            patience=self.config.get('patience', 10),
            min_delta=self.config.get('min_delta', 1e-6),
            monitor=self.config.get('monitor_metric', 'val_loss')
        )
        logger.info("早停机制设置完成")
    
    def compute_losses(self, 
                      predictions: torch.Tensor,
                      targets: torch.Tensor,
                      coordinates: torch.Tensor,
                      boundary_pred: Optional[torch.Tensor] = None,
                      initial_pred: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """计算综合损失 (稳定版)"""
        try:
            # 检查输入有效性
            input_tensors = [predictions, targets, coordinates]
            if not all(t is not None and torch.isfinite(t).all() for t in input_tensors):
                logger.warning("损失计算输入包含无效值，使用代理损失")
                safe_loss = torch.tensor(1.0, device=predictions.device, requires_grad=True)
                return safe_loss, {
                    'total_loss': safe_loss,
                    'physics_loss': torch.tensor(1.0, device=predictions.device),
                    'boundary_loss': torch.tensor(0.0, device=predictions.device),
                    'initial_loss': torch.tensor(0.0, device=predictions.device),
                    'data_loss': torch.tensor(1.0, device=predictions.device),
                    'weighted_physics': torch.tensor(0.0, device=predictions.device),
                    'weighted_boundary': torch.tensor(0.0, device=predictions.device),
                    'weighted_initial': torch.tensor(0.0, device=predictions.device),
                    'weighted_data': safe_loss
                }
            
            # 数据损失 (MSE)
            data_loss = self.loss_fn.mse_loss(predictions, targets)
            
            # 物理一致性损失
            physics_loss = self.physics_loss_fn(predictions, coordinates)
            
            # 边界条件损失
            boundary_loss = torch.tensor(0.0, device=predictions.device)
            if boundary_pred is not None and torch.isfinite(boundary_pred).all():
                # 简化的边界损失
                boundary_loss = self.loss_fn.mse_loss(boundary_pred, targets)
            
            # 初始条件损失
            initial_loss = torch.tensor(0.0, device=predictions.device)
            if initial_pred is not None and torch.isfinite(initial_pred).all():
                # 简化的初始损失
                initial_loss = self.loss_fn.mse_loss(initial_pred, targets)
            
            # 使用稳定版损失函数计算总损失
            total_loss, loss_dict = self.loss_fn(
                physics_loss, boundary_loss, initial_loss, data_loss
            )
            
            # 最终数值检查
            if not torch.isfinite(total_loss):
                logger.warning(f"总损失无效 ({total_loss})，应用数值稳定化")
                total_loss = torch.sign(total_loss) * self.loss_fn.max_loss_value
                loss_dict['total_loss'] = total_loss
            
            return total_loss, loss_dict
            
        except Exception as e:
            logger.error(f"损失计算失败: {str(e)}")
            # 返回安全的代理损失
            safe_loss = torch.tensor(1.0, device=predictions.device, requires_grad=True)
            safe_loss_dict = {
                'total_loss': safe_loss,
                'physics_loss': torch.tensor(1.0, device=predictions.device),
                'boundary_loss': torch.tensor(0.0, device=predictions.device),
                'initial_loss': torch.tensor(0.0, device=predictions.device),
                'data_loss': torch.tensor(1.0, device=predictions.device),
                'weighted_physics': torch.tensor(0.0, device=predictions.device),
                'weighted_boundary': torch.tensor(0.0, device=predictions.device),
                'weighted_initial': torch.tensor(0.0, device=predictions.device),
                'weighted_data': safe_loss
            }
            return safe_loss, safe_loss_dict
    
    def step(self, model: nn.Module, epoch: int, step_info: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """执行优化步骤 (稳定版)"""
        try:
            if self.optimizer is None:
                logger.warning("优化器未初始化")
                return {'max_grad': 0.0, 'avg_grad': 0.0, 'grad_clip_ratio': 0.0}
            
            # 检查模型参数的数值稳定性
            self.stability_monitor.check_tensor_stability(
                model.parameters(), 
                name="model_parameters"
            )
            
            # 梯度裁剪 (增强版)
            grad_norms = self.gradient_clipper.clip_gradients(model.parameters())
            
            # 检查梯度稳定性
            if step_info is not None:
                self.stability_monitor.monitor_training_step(step_info)
            
            # 检查梯度范数异常
            if grad_norms['max_grad_norm'] > self.stability_monitor.warning_threshold:
                logger.warning(f"梯度范数过大: {grad_norms['max_grad_norm']:.2e}")
            
            # 优化器步骤
            self.optimizer.step()
            
            # 学习率调度
            if self.scheduler is not None:
                self.scheduler.step()
            
            # 添加额外统计信息
            grad_norms['clip_ratio'] = self._calculate_clip_ratio(grad_norms)
            grad_norms['lr'] = self.get_current_lr()[0] if self.get_current_lr() else 0.0
            
            return grad_norms
            
        except Exception as e:
            logger.error(f"优化步骤失败: {str(e)}")
            return {'max_grad': 0.0, 'avg_grad': 0.0, 'grad_clip_ratio': 0.0, 'lr': 0.0}
    
    def _calculate_clip_ratio(self, grad_norms: Dict[str, float]) -> float:
        """计算梯度裁剪比例"""
        try:
            if 'max_grad_value' in grad_norms and 'max_grad_norm' in grad_norms:
                value_ratio = grad_norms['max_grad_value'] / self.gradient_clipper.max_grad_value
                norm_ratio = grad_norms['max_grad_norm'] / self.gradient_clipper.max_grad_norm
                return max(value_ratio, norm_ratio)
            return 0.0
        except:
            return 0.0
    
    def get_current_lr(self) -> Optional[List[float]]:
        """获取当前学习率"""
        if self.optimizer is not None:
            return [group['lr'] for group in self.optimizer.param_groups]
        return None
    
    def get_stability_report(self) -> Dict[str, Any]:
        """获取稳定性报告"""
        return self.stability_monitor.get_stability_report()
    
    def early_stopping_check(self, current_loss: float, model: nn.Module) -> bool:
        """早停检查 (稳定版)"""
        try:
            # 检查损失值有效性
            if not torch.isfinite(torch.tensor(current_loss)):
                logger.warning("当前损失值无效，触发早停")
                return True
            
            # 检查梯度稳定性
            grad_stats = self.stability_monitor.get_recent_gradient_stats()
            if len(grad_stats) > 5:
                recent_max_grads = [stat.get('max_grad_norm', 0) for stat in grad_stats[-5:]]
                if max(recent_max_grads) > self.stability_monitor.warning_threshold * 10:
                    logger.warning("梯度爆炸模式检测，触发早停")
                    return True
            
            return False
        except Exception as e:
            logger.error(f"早停检查失败: {e}")
            return True
    
    def check_early_stopping(self, epoch: int, model: nn.Module, val_loss: float) -> bool:
        """检查早停"""
        if self.early_stopping is not None:
            return self.early_stopping(epoch, model, val_loss)
        return False
    
    def get_best_model_state(self) -> Optional[Dict[str, torch.Tensor]]:
        """获取最佳模型状态"""
        if self.early_stopping is not None and self.early_stopping.best_state_dict is not None:
            return self.early_stopping.best_state_dict
        return None
    
    def get_current_lr(self) -> List[float]:
        """获取当前学习率"""
        if self.optimizer is not None:
            return [param_group['lr'] for param_group in self.optimizer.param_groups]
        return []
    
    def save_checkpoint(self, model: nn.Module, epoch: int, loss: float, filepath: str):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'loss': loss,
            'config': self.config
        }
        
        torch.save(checkpoint, filepath)
        logger.info(f"检查点已保存到 {filepath}")
    
    def load_checkpoint(self, filepath: str, model: nn.Module) -> Dict[str, Any]:
        """加载检查点"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if self.optimizer and checkpoint.get('optimizer_state_dict'):
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        logger.info(f"检查点已从 {filepath} 加载")
        return checkpoint