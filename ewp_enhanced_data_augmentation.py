#!/usr/bin/env python3
"""
增强版EWPINN数据增强模块

此模块提供高级数据增强技术，用于提升EWPINN模型的泛化能力和物理一致性
特性包括：
- 多类型数据增强策略
- 物理约束感知的数据变换
- 自适应增强强度调整
- 批量增强支持
- 增强效果分析工具
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('EnhancedDataAugmentation')

class EnhancedDataAugmenter:
    """
    增强版数据增强器类
    提供多种数据增强策略，支持物理约束感知的数据变换
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化数据增强器
        
        Args:
            config: 增强配置字典，包含各种增强策略的参数
        """
        # 默认配置
        self.default_config = {
            'enabled': True,
            'base_intensity': 0.1,  # 基础增强强度
            'strategies': {
                'random_scaling': {
                    'enabled': True,
                    'intensity': 0.1,
                    'min_scale': 0.9,
                    'max_scale': 1.1
                },
                'nonlinear_transformation': {
                    'enabled': True,
                    'intensity': 0.1,
                    'transform_prob': 0.7,
                    'max_transform_dims': 5
                },
                'feature_shuffling': {
                    'enabled': True,
                    'shuffle_prob': 0.3,
                    'shuffle_group_size': 4
                },
                'random_noise': {
                    'enabled': True,
                    'noise_level': 0.05,
                    'noise_type': 'gaussian'
                },
                'elastic_deformation': {
                    'enabled': True,
                    'alpha': 0.5,
                    'sigma': 0.05
                },
                'physics_informed_distortion': {
                    'enabled': True,
                    'intensity': 0.08,
                    'distortion_type': 'local'
                },
                'frequency_domain': {
                    'enabled': True,
                    'intensity': 0.05
                },
                # 新增增强策略
                'time_shift': {
                    'enabled': True,
                    'shift_prob': 0.4,
                    'max_shift_fraction': 0.1
                },
                'feature_importance_aware': {
                    'enabled': True,
                    'importance_guided': True,
                    'base_intensity': 0.05
                },
                'dynamic_batch_augmentation': {
                    'enabled': True,
                    'adaptive_variation': True,
                    'max_variation_factor': 0.2
                },
                # 新增增强策略
                'random_rotation': {
                    'enabled': True,
                    'intensity': 0.1,
                    'rotation_prob': 0.6
                },
                'mixup_augmentation': {
                    'enabled': True,
                    'alpha': 0.4,
                    'mix_prob': 0.5
                },
                'random_crop_pad': {
                    'enabled': True,
                    'crop_prob': 0.4,
                    'max_crop_ratio': 0.1
                }
            },
            'adaptive': {
                'enabled': True,
                'max_intensity_factor': 2.0,  # 增加最大强度
                'min_intensity_factor': 0.3,  # 降低最小强度
                'learning_rate': 0.02,        # 增加学习率
                'window_size': 5,            # 使用更大的窗口
                'overfitting_threshold': 0.05  # 过拟合检测阈值
            },
            'batch_level': {
                'enabled': True,
                'variation_intensity': 0.1,
                'group_augmentation': True    # 批次分组增强
            },
            'validation': {
                'apply_weak_augmentation': True,  # 验证时应用弱增强
                'validation_intensity_factor': 0.3  # 验证时的强度因子
            }
        }
        
        # 合并配置
        self.config = self.default_config.copy()
        if config:
            self._merge_config(config)
        
        # 初始化自适应参数
        self.current_intensity_factor = 1.0
        self.performance_history = []
        
        logger.info("增强版数据增强器已初始化")
    
    def _merge_config(self, new_config: Dict):
        """
        递归合并配置字典
        """
        for key, value in new_config.items():
            if key in self.config and isinstance(self.config[key], dict) and isinstance(value, dict):
                self._merge_config_recursive(self.config[key], value)
            else:
                self.config[key] = value
    
    def _merge_config_recursive(self, base: Dict, update: Dict):
        """
        递归合并嵌套配置
        """
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config_recursive(base[key], value)
            else:
                base[key] = value
    
    def apply_augmentation(self, data: torch.Tensor, 
                          strategy_weights: Optional[Dict[str, float]] = None,
                          metadata: Optional[Dict] = None,
                          is_validation: bool = False) -> torch.Tensor:
        """
        应用数据增强
        
        Args:
            data: 输入数据张量 (batch_size, feature_dim)
            strategy_weights: 各策略的权重字典
            metadata: 可选的元数据，用于物理约束感知增强
            is_validation: 是否为验证模式（应用弱增强）
            
        Returns:
            增强后的数据张量
        """
        if not self.config['enabled']:
            return data
        
        augmented = data.clone()
        
        # 应用批量级别的增强变化
        if self.config['batch_level']['enabled']:
            batch_size = augmented.size(0)
            
            # 批次分组增强
            if self.config['batch_level']['group_augmentation'] and batch_size > 4:
                # 将批次分成小组，每组应用相似但略有不同的增强
                group_size = max(2, batch_size // 5)  # 最多分成5组
                num_groups = (batch_size + group_size - 1) // group_size
                
                for group_idx in range(num_groups):
                    start_idx = group_idx * group_size
                    end_idx = min(start_idx + group_size, batch_size)
                    group_size_actual = end_idx - start_idx
                    
                    if group_size_actual > 0:
                        # 为每个组生成一个基本增强因子
                        group_factor = 1.0 + (torch.randn(1) * self.config['batch_level']['variation_intensity']).item()
                        
                        # 在组内添加小的变化
                        for i in range(start_idx, end_idx):
                            individual_factor = group_factor * (1.0 + (torch.randn(1) * 0.05).item())
                            current_intensity = self.current_intensity_factor * individual_factor
                            augmented[i] = self._apply_instance_augmentation(
                                augmented[i], strategy_weights, metadata, current_intensity, is_validation
                            )
            elif batch_size > 1:
                # 常规批量增强
                batch_factors = 1.0 + (torch.randn(batch_size) * 
                                     self.config['batch_level']['variation_intensity'])
                # 应用批量级别因子
                for i in range(batch_size):
                    current_intensity = self.current_intensity_factor * batch_factors[i]
                    augmented[i] = self._apply_instance_augmentation(
                        augmented[i], strategy_weights, metadata, current_intensity, is_validation
                    )
            else:
                augmented = self._apply_instance_augmentation(
                    augmented, strategy_weights, metadata, self.current_intensity_factor, is_validation
                )
        else:
            # 对每个样本应用相同强度的增强
            augmented = self._apply_instance_augmentation(
                augmented, strategy_weights, metadata, self.current_intensity_factor, is_validation
            )
        
        # 在所有其他增强后应用Mixup（如果启用）
        if self.config['strategies']['mixup_augmentation']['enabled'] and not is_validation:
            batch_size = augmented.size(0)
            if batch_size > 1 and np.random.random() < self.config['strategies']['mixup_augmentation']['mix_prob']:
                alpha = self.config['strategies']['mixup_augmentation']['alpha']
                augmented, _ = self._mixup_augmentation(augmented, alpha)
        
        return augmented
    
    def _apply_instance_augmentation(self, instance: torch.Tensor, 
                                    strategy_weights: Optional[Dict[str, float]],
                                    metadata: Optional[Dict],
                                    intensity_factor: float,
                                    is_validation: bool = False) -> torch.Tensor:
        """
        对单个样本应用增强
        
        Args:
            instance: 输入数据张量
            strategy_weights: 各策略的权重字典
            metadata: 可选的元数据
            intensity_factor: 强度因子
            is_validation: 是否为验证模式（应用弱增强）
        """
        augmented = instance.clone()
        
        # 验证模式下调整强度
        if is_validation and self.config['validation']['apply_weak_augmentation']:
            current_intensity_factor = intensity_factor * self.config['validation']['validation_intensity_factor']
        else:
            current_intensity_factor = intensity_factor
        
        # 应用随机缩放
        if self.config['strategies']['random_scaling']['enabled']:
            weight = strategy_weights.get('random_scaling', 1.0) if strategy_weights else 1.0
            intensity = self.config['strategies']['random_scaling']['intensity'] * \
                       current_intensity_factor * weight
            augmented = self._random_scaling(augmented, intensity)
        
        # 应用非线性变换
        if self.config['strategies']['nonlinear_transformation']['enabled']:
            weight = strategy_weights.get('nonlinear_transformation', 1.0) if strategy_weights else 1.0
            intensity = self.config['strategies']['nonlinear_transformation']['intensity'] * \
                       current_intensity_factor * weight
            augmented = self._nonlinear_transformation(augmented, intensity)
        
        # 应用特征洗牌
        if self.config['strategies']['feature_shuffling']['enabled']:
            if np.random.random() < self.config['strategies']['feature_shuffling']['shuffle_prob']:
                augmented = self._feature_shuffling(augmented)
        
        # 应用随机噪声
        if self.config['strategies']['random_noise']['enabled']:
            weight = strategy_weights.get('random_noise', 1.0) if strategy_weights else 1.0
            intensity = self.config['strategies']['random_noise']['noise_level'] * \
                       current_intensity_factor * weight
            augmented = self._add_noise(augmented, intensity)
        
        # 应用弹性变形
        if self.config['strategies']['elastic_deformation']['enabled']:
            weight = strategy_weights.get('elastic_deformation', 1.0) if strategy_weights else 1.0
            alpha = self.config['strategies']['elastic_deformation']['alpha'] * current_intensity_factor * weight
            sigma = self.config['strategies']['elastic_deformation']['sigma']
            augmented = self._elastic_deformation(augmented, alpha, sigma)
        
        # 应用物理感知扭曲
        if self.config['strategies']['physics_informed_distortion']['enabled']:
            weight = strategy_weights.get('physics_informed_distortion', 1.0) if strategy_weights else 1.0
            intensity = self.config['strategies']['physics_informed_distortion']['intensity'] * \
                       current_intensity_factor * weight
            augmented = self._physics_informed_distortion(augmented, intensity, metadata)
        
        # 应用频域增强
        if self.config['strategies']['frequency_domain']['enabled']:
            weight = strategy_weights.get('frequency_domain', 1.0) if strategy_weights else 1.0
            intensity = self.config['strategies']['frequency_domain']['intensity'] * \
                       current_intensity_factor * weight
            augmented = self._frequency_domain_augmentation(augmented, intensity)
        
        # 应用时间位移增强
        if self.config['strategies']['time_shift']['enabled'] and not is_validation:
            if np.random.random() < self.config['strategies']['time_shift']['shift_prob']:
                max_shift = int(instance.size(0) * self.config['strategies']['time_shift']['max_shift_fraction'])
                augmented = self._time_shift(augmented, max_shift)
        
        # 应用特征重要性感知增强
        if self.config['strategies']['feature_importance_aware']['enabled']:
            weight = strategy_weights.get('feature_importance_aware', 1.0) if strategy_weights else 1.0
            base_intensity = self.config['strategies']['feature_importance_aware']['base_intensity'] * \
                           current_intensity_factor * weight
            augmented = self._feature_importance_aware_augmentation(augmented, base_intensity, metadata)
        
        # 应用旋转增强
        if self.config['strategies']['random_rotation']['enabled'] and not is_validation:
            if np.random.random() < self.config['strategies']['random_rotation']['rotation_prob']:
                weight = strategy_weights.get('random_rotation', 1.0) if strategy_weights else 1.0
                intensity_rot = self.config['strategies']['random_rotation']['intensity'] * \
                           current_intensity_factor * weight
                augmented = self._random_rotation(augmented, intensity_rot)
        
        # 应用随机裁剪/填充增强
        if self.config['strategies']['random_crop_pad']['enabled'] and not is_validation:
            if np.random.random() < self.config['strategies']['random_crop_pad']['crop_prob']:
                max_crop_ratio = self.config['strategies']['random_crop_pad']['max_crop_ratio']
                augmented = self._random_crop_pad(augmented, max_crop_ratio)
        
        return augmented
    
    def _random_rotation(self, data: torch.Tensor, intensity: float) -> torch.Tensor:
        """
        对多维数据应用旋转增强，模拟不同角度的物理观测
        """
        if data.size(0) < 2:
            return data
        
        augmented = data.clone()
        
        # 只对数据的前几个维度应用旋转
        num_rotated = min(4, data.size(0))  # 最多旋转4个维度
        
        # 计算旋转角度
        angle = (torch.rand(1).item() - 0.5) * 2 * np.pi * intensity  # -π*intensity 到 π*intensity
        
        # 创建旋转矩阵（2D旋转）
        rotation_matrix = torch.eye(num_rotated, device=data.device)
        if num_rotated >= 2:
            rotation_matrix[0, 0] = torch.cos(torch.tensor(angle))
            rotation_matrix[0, 1] = -torch.sin(torch.tensor(angle))
            rotation_matrix[1, 0] = torch.sin(torch.tensor(angle))
            rotation_matrix[1, 1] = torch.cos(torch.tensor(angle))
        
        # 应用旋转
        augmented[:num_rotated] = rotation_matrix @ augmented[:num_rotated]
        
        return augmented
    
    def _mixup_augmentation(self, data_batch: torch.Tensor, alpha: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        应用Mixup增强，混合批次中的样本
        """
        batch_size = data_batch.size(0)
        device = data_batch.device
        
        # 生成混合权重
        lam = torch.distributions.beta.Beta(alpha, alpha).sample((batch_size, 1)).to(device)
        
        # 打乱批次顺序
        indices = torch.randperm(batch_size).to(device)
        
        # 混合数据
        mixed_data = lam * data_batch + (1 - lam) * data_batch[indices]
        
        return mixed_data, lam
    
    def _random_crop_pad(self, data: torch.Tensor, max_crop_ratio: float) -> torch.Tensor:
        """
        应用随机裁剪和填充增强，模拟数据的部分丢失和恢复
        """
        feature_dim = data.size(0)
        if feature_dim <= 3:  # 维度太少不适合裁剪
            return data
        
        # 计算裁剪长度
        max_crop_len = max(1, int(feature_dim * max_crop_ratio))
        crop_len = torch.randint(1, max_crop_len + 1, (1,)).item()
        
        # 随机选择裁剪起始位置
        start_pos = torch.randint(0, feature_dim - crop_len + 1, (1,)).item()
        
        # 创建掩码
        mask = torch.ones_like(data)
        mask[start_pos:start_pos + crop_len] = 0
        
        # 裁剪后的数据
        cropped = data * mask
        
        # 对裁剪区域进行填充
        if start_pos > 0 and start_pos + crop_len < feature_dim:
            left_mean = data[start_pos - 1].mean() if start_pos > 0 else 0
            right_mean = data[start_pos + crop_len].mean() if start_pos + crop_len < feature_dim else 0
            fill_value = (left_mean + right_mean) / 2
        else:
            fill_value = data.mean()
        
        # 添加一些随机噪声到填充值
        fill_value = fill_value * (1 + (torch.randn(1).item() - 0.5) * 0.2)
        
        # 填充裁剪区域
        cropped[start_pos:start_pos + crop_len] = fill_value
        
        return cropped
    
    def _random_scaling(self, data: torch.Tensor, intensity: float) -> torch.Tensor:
        """
        应用随机缩放增强
        """
        min_scale = max(0.5, 1.0 - intensity)
        max_scale = min(2.0, 1.0 + intensity)
        
        # 对每个维度应用不同的缩放因子
        scale_factors = torch.rand(data.size()) * (max_scale - min_scale) + min_scale
        return data * scale_factors
    
    def _nonlinear_transformation(self, data: torch.Tensor, intensity: float) -> torch.Tensor:
        """
        应用多种非线性变换
        """
        augmented = data.clone()
        
        # 随机选择变换类型
        transform_type = np.random.choice(['sine', 'tanh', 'log', 'exp'])
        
        # 选择要变换的维度
        num_transformed = min(
            np.random.randint(1, min(8, data.size(0) + 1)),  # 最多变换8个维度或所有维度
            int(data.size(0) * 0.3)  # 最多变换30%的维度
        )
        
        transform_indices = torch.randperm(data.size(0))[:num_transformed]
        
        for idx in transform_indices:
            if transform_type == 'sine':
                augmented[idx] = augmented[idx] + torch.sin(augmented[idx]) * intensity
            elif transform_type == 'tanh':
                augmented[idx] = augmented[idx] + torch.tanh(augmented[idx]) * intensity
            elif transform_type == 'log':
                # 安全地应用对数变换
                abs_val = torch.abs(augmented[idx]) + 1e-8
                sign = torch.sign(augmented[idx])
                augmented[idx] = augmented[idx] + sign * torch.log(abs_val) * intensity * 0.1
            elif transform_type == 'exp':
                # 限制指数变换范围以避免数值问题
                clamped = torch.clamp(augmented[idx], min=-1.0, max=1.0)
                augmented[idx] = augmented[idx] + torch.exp(clamped) * intensity * 0.1
        
        return augmented
    
    def _feature_shuffling(self, data: torch.Tensor) -> torch.Tensor:
        """
        对特征进行分组洗牌，保持物理相关性
        """
        augmented = data.clone()
        feature_dim = data.size(0)
        group_size = self.config['strategies']['feature_shuffling']['shuffle_group_size']
        
        # 计算可分组的数量
        num_groups = feature_dim // group_size
        remaining = feature_dim % group_size
        
        # 对每个组内的特征进行洗牌
        for i in range(num_groups):
            start_idx = i * group_size
            end_idx = start_idx + group_size
            
            # 在组内洗牌
            shuffle_idx = torch.randperm(group_size)
            augmented[start_idx:end_idx] = augmented[start_idx:end_idx][shuffle_idx]
        
        # 处理剩余特征
        if remaining > 1:
            start_idx = num_groups * group_size
            shuffle_idx = torch.randperm(remaining)
            augmented[start_idx:] = augmented[start_idx:][shuffle_idx]
        
        return augmented
    
    def _add_noise(self, data: torch.Tensor, noise_level: float) -> torch.Tensor:
        """
        添加各种类型的噪声
        """
        noise_type = self.config['strategies']['random_noise']['noise_type']
        
        if noise_type == 'gaussian':
            noise = torch.randn_like(data) * noise_level
        elif noise_type == 'uniform':
            noise = (torch.rand_like(data) - 0.5) * 2 * noise_level
        elif noise_type == 'laplace':
            noise = torch.distributions.laplace.Laplace(0, noise_level).sample(data.size()).to(data.device)
        else:
            noise = torch.randn_like(data) * noise_level
        
        # 添加稀疏噪声：只有部分维度添加噪声
        mask = torch.rand_like(data) < 0.7  # 70%的维度添加噪声
        noise = noise * mask
        
        return data + noise
    
    def _elastic_deformation(self, data: torch.Tensor, alpha: float, sigma: float) -> torch.Tensor:
        """
        应用弹性变形，模拟物理系统中的微小变形
        """
        # 生成随机位移场
        displacement = torch.randn_like(data) * alpha
        
        # 使用高斯平滑位移场
        # 对于1D数据，我们使用简单的卷积近似
        kernel_size = max(3, int(6 * sigma))
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # 创建高斯核
        kernel = torch.exp(-torch.arange(-kernel_size//2, kernel_size//2 + 1)**2 / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        
        # 应用卷积（在CPU上执行以避免复杂的1D卷积实现）
        displacement_np = displacement.cpu().numpy()
        kernel_np = kernel.cpu().numpy()
        
        # 使用numpy的卷积函数
        smoothed_displacement = np.convolve(displacement_np, kernel_np, mode='same')
        smoothed_displacement = torch.tensor(smoothed_displacement, device=data.device, dtype=data.dtype)
        
        # 应用变形
        return data + smoothed_displacement
    
    def _physics_informed_distortion(self, data: torch.Tensor, intensity: float, 
                                    metadata: Optional[Dict] = None) -> torch.Tensor:
        """
        应用物理约束感知的扭曲，保持关键物理关系
        """
        augmented = data.clone()
        feature_dim = data.size(0)
        
        # 根据不同位置应用不同的扭曲强度
        # 假设前几个维度是关键的物理参数
        if self.config['strategies']['physics_informed_distortion']['distortion_type'] == 'local':
            # 为每个位置创建不同的扭曲权重
            position_weights = torch.ones(feature_dim, device=data.device)
            
            # 假设前10个特征是更关键的物理参数，应用较小的扭曲
            if feature_dim > 10:
                position_weights[:10] = 0.3  # 减少关键物理参数的扭曲
                
            # 应用位置感知的扭曲
            distortion = torch.randn_like(data) * intensity * position_weights
        else:
            # 全局扭曲，但避开某些关键维度
            distortion = torch.randn_like(data) * intensity
            if feature_dim > 5:
                # 保护前5个维度（假设是最重要的物理参数）
                distortion[:5] *= 0.2
        
        # 如果有元数据，可以进一步调整扭曲策略
        if metadata and 'sensitive_indices' in metadata:
            sensitive_indices = metadata['sensitive_indices']
            for idx in sensitive_indices:
                if idx < feature_dim:
                    distortion[idx] *= 0.1  # 进一步减少敏感维度的扭曲
        
        return augmented + distortion
    
    def _frequency_domain_augmentation(self, data: torch.Tensor, intensity: float) -> torch.Tensor:
        """
        在频域中应用增强，捕获数据的全局特征
        """
        # 将数据转换到频域
        # 对于1D数据，我们使用简单的方法模拟频域增强
        # 对于更复杂的实现，可以使用FFT
        
        # 计算差分（近似高频分量）
        if data.size(0) > 1:
            diff = data[1:] - data[:-1]
            high_freq = torch.cat([diff, diff[-1:]], dim=0)
            
            # 修改高频分量 - 使用更智能的频率调整
            # 低频部分（前30%）变化较小，高频部分（后70%）变化较大
            freq_weights = torch.ones_like(high_freq)
            split_point = int(data.size(0) * 0.3)
            
            # 低频部分权重较小
            freq_weights[:split_point] = 0.3
            # 高频部分权重较大
            freq_weights[split_point:] = 1.5
            
            # 应用频率感知的修改
            high_freq_modified = high_freq * (1.0 + torch.randn_like(high_freq) * intensity * freq_weights)
            
            # 重建信号
            augmented = torch.zeros_like(data)
            augmented[0] = data[0]
            for i in range(1, data.size(0)):
                augmented[i] = augmented[i-1] + high_freq_modified[i-1]
            
            return augmented
        else:
            return data
    
    def _time_shift(self, data: torch.Tensor, max_shift: int) -> torch.Tensor:
        """
        应用时间位移增强，模拟时间序列的不同起始点
        
        Args:
            data: 输入数据张量
            max_shift: 最大位移量
            
        Returns:
            位移后的数据
        """
        if max_shift <= 0 or data.size(0) <= 1:
            return data
        
        # 随机选择位移方向和大小
        shift_direction = 1 if torch.rand(1).item() > 0.5 else -1
        shift_amount = torch.randint(1, max_shift + 1, (1,)).item()
        total_shift = shift_direction * shift_amount
        
        # 执行循环位移
        if total_shift > 0:
            # 向右位移
            shifted = torch.cat([data[-total_shift:], data[:-total_shift]])
        else:
            # 向左位移
            shifted = torch.cat([data[-total_shift:], data[:-total_shift]])
        
        return shifted
    
    def _feature_importance_aware_augmentation(self, data: torch.Tensor, 
                                             base_intensity: float, 
                                             metadata: Optional[Dict] = None) -> torch.Tensor:
        """
        基于特征重要性的增强，对不同重要性的特征应用不同强度的增强
        
        Args:
            data: 输入数据张量
            base_intensity: 基础增强强度
            metadata: 包含特征重要性信息的元数据
            
        Returns:
            增强后的数据
        """
        if not self.config['strategies']['feature_importance_aware']['importance_guided']:
            return data
        
        # 创建特征重要性权重
        feature_dim = data.size(0)
        importance_weights = torch.ones(feature_dim, device=data.device)
        
        # 如果有元数据中的重要性信息，使用它
        if metadata and 'feature_importance' in metadata:
            importance = torch.tensor(metadata['feature_importance'], device=data.device)
            # 归一化重要性权重
            importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)
            # 重要性高的特征增强强度低，反之亦然
            importance_weights = 1.0 - importance * 0.7  # 最高减少70%的增强
        else:
            # 默认假设：前20%的特征更重要，应用较小的增强
            num_important = max(1, int(feature_dim * 0.2))
            importance_weights[:num_important] = 0.4  # 重要特征只应用40%的增强强度
            
            # 中间60%的特征应用中等增强
            mid_start = num_important
            mid_end = int(feature_dim * 0.8)
            importance_weights[mid_start:mid_end] = 0.8
            
            # 最后20%的特征应用较强增强
            importance_weights[mid_end:] = 1.2
        
        # 应用特征重要性感知的随机噪声
        noise = torch.randn_like(data) * base_intensity * importance_weights
        
        return data + noise
    
    def update_adaptive_parameters(self, performance_metrics: Dict[str, float]):
        """
        根据模型性能自适应调整增强强度和策略
        
        Args:
            performance_metrics: 包含模型性能指标的字典，如损失、准确率等
        """
        if not self.config['adaptive']['enabled']:
            return
        
        # 存储性能历史
        self.performance_history.append(performance_metrics)
        
        # 获取配置的窗口大小
        window_size = self.config['adaptive'].get('window_size', 5)
        
        # 只保留最近的性能记录
        if len(self.performance_history) > window_size * 2:
            self.performance_history.pop(0)
        
        # 至少需要窗口大小个性能记录才能进行自适应调整
        if len(self.performance_history) < window_size:
            return
        
        # 提取最近的损失值
        recent_losses = [metrics.get('loss', float('inf')) for metrics in self.performance_history[-window_size:]]
        recent_train_losses = [metrics.get('train_loss', float('inf')) for metrics in self.performance_history[-window_size:]]
        
        # 计算训练损失和验证损失的比率（过拟合检测）
        if recent_train_losses[-1] > 0 and recent_losses[-1] > 0:
            loss_ratio = recent_losses[-1] / recent_train_losses[-1]
            
            # 如果训练损失远低于验证损失，可能存在过拟合
            if loss_ratio > 1.5:  # 验证损失是训练损失的1.5倍以上
                self.current_intensity_factor = min(
                    self.current_intensity_factor * (1.0 + self.config['adaptive']['learning_rate'] * 1.5),
                    self.config['adaptive']['max_intensity_factor']
                )
                logger.info(f"检测到严重过拟合，大幅增加增强强度: {self.current_intensity_factor:.3f} (损失比: {loss_ratio:.3f})")
                return
        
        # 计算损失变化趋势（使用最后几个样本）
        if len(recent_losses) >= 3:
            # 计算平均损失变化率
            loss_diffs = [recent_losses[i] - recent_losses[i-1] for i in range(1, len(recent_losses))]
            avg_loss_diff = np.mean(loss_diffs[-2:])
            
            # 计算损失的方差（稳定性指标）
            loss_variance = np.var(recent_losses)
            
            # 检查是否过拟合（验证损失上升且趋势明显）
            threshold = self.config['adaptive'].get('overfitting_threshold', 0.05)
            if avg_loss_diff > threshold:
                # 严重过拟合情况，快速增加增强强度
                adjustment_factor = 1.0 + self.config['adaptive']['learning_rate'] * 2
                self.current_intensity_factor = min(
                    self.current_intensity_factor * adjustment_factor,
                    self.config['adaptive']['max_intensity_factor']
                )
                logger.info(f"检测到过拟合趋势，增加增强强度: {self.current_intensity_factor:.3f} (平均变化: {avg_loss_diff:.6f})")
            # 如果损失下降稳定，适当减少增强强度以加快收敛
            elif avg_loss_diff < -threshold and loss_variance < threshold:
                self.current_intensity_factor = max(
                    self.current_intensity_factor * (1.0 - self.config['adaptive']['learning_rate']),
                    self.config['adaptive']['min_intensity_factor']
                )
                logger.info(f"训练稳定下降，微调减少增强强度: {self.current_intensity_factor:.3f} (稳定性: {loss_variance:.6f})")
            # 如果损失波动较大，略微增加增强强度以提高稳定性
            elif loss_variance > threshold * 2:
                self.current_intensity_factor = min(
                    self.current_intensity_factor * (1.0 + self.config['adaptive']['learning_rate'] * 0.5),
                    self.config['adaptive']['max_intensity_factor']
                )
                logger.info(f"检测到损失波动，小幅增加增强强度: {self.current_intensity_factor:.3f} (波动: {loss_variance:.6f})")
        
        # 动态调整批量级别增强的变化强度
        if self.config['strategies']['dynamic_batch_augmentation']['enabled'] and \
           self.config['strategies']['dynamic_batch_augmentation']['adaptive_variation']:
            # 根据训练稳定性调整批量变化强度
            if len(recent_losses) >= window_size:
                train_stability = 1.0 / (np.var(recent_train_losses) + 1e-8)
                self.config['batch_level']['variation_intensity'] = min(
                    0.1 + 0.1 * (1.0 - min(train_stability, 1.0)),
                    self.config['strategies']['dynamic_batch_augmentation']['max_variation_factor']
                )
    
    def get_current_config(self) -> Dict:
        """
        获取当前的增强配置
        
        Returns:
            当前配置字典
        """
        return {
            **self.config,
            'current_intensity_factor': self.current_intensity_factor,
            'performance_history_length': len(self.performance_history)
        }
    
    def visualize_augmentation_effect(self, original_data: torch.Tensor, 
                                     augmented_data: torch.Tensor, 
                                     save_path: Optional[str] = None):
        """
        可视化增强效果
        
        Args:
            original_data: 原始数据
            augmented_data: 增强后的数据
            save_path: 保存可视化结果的路径
        """
        try:
            import matplotlib.pyplot as plt
            
            # 确保数据在CPU上并转换为numpy
            original = original_data.cpu().numpy()
            augmented = augmented_data.cpu().numpy()
            
            # 创建可视化
            plt.figure(figsize=(12, 6))
            
            # 绘制原始数据和增强数据
            plt.subplot(2, 1, 1)
            plt.plot(original, 'b-', label='原始数据')
            plt.plot(augmented, 'r-', alpha=0.7, label='增强数据')
            plt.title('原始数据 vs 增强数据')
            plt.legend()
            
            # 绘制差异
            plt.subplot(2, 1, 2)
            plt.plot(augmented - original, 'g-')
            plt.title('增强差异')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                logger.info(f"增强效果可视化已保存到: {save_path}")
            else:
                plt.show()
                
        except ImportError:
            logger.warning("matplotlib未安装，无法可视化增强效果")


def create_augmentation_pipeline(config: Optional[Dict] = None) -> EnhancedDataAugmenter:
    """
    创建数据增强流水线
    
    Args:
        config: 增强配置
        
    Returns:
        增强器实例
    """
    return EnhancedDataAugmenter(config)


def apply_multi_strategy_augmentation(
    data: torch.Tensor,
    strategies: List[str],
    intensities: Optional[List[float]] = None,
    **kwargs
) -> torch.Tensor:
    """
    应用多种增强策略
    
    Args:
        data: 输入数据
        strategies: 要应用的策略列表
        intensities: 各策略的强度列表
        **kwargs: 额外参数
        
    Returns:
        增强后的数据
    """
    # 创建配置
    config = {
        'strategies': {}
    }
    
    # 启用指定的策略
    for i, strategy in enumerate(strategies):
        config['strategies'][strategy] = {
            'enabled': True,
            'intensity': intensities[i] if intensities and i < len(intensities) else 0.1
        }
    
    # 禁用其他策略
    all_strategies = ['random_scaling', 'nonlinear_transformation', 'feature_shuffling',
                    'random_noise', 'elastic_deformation', 'physics_informed_distortion',
                    'frequency_domain']
    
    for strategy in all_strategies:
        if strategy not in config['strategies']:
            config['strategies'][strategy] = {'enabled': False}
    
    # 创建增强器并应用
    augmenter = EnhancedDataAugmenter(config)
    return augmenter.apply_augmentation(data, **kwargs)


def generate_augmentation_report(augmenter: EnhancedDataAugmenter, 
                                original_sample: torch.Tensor,
                                num_reports: int = 5) -> Dict:
    """
    生成增强效果报告
    
    Args:
        augmenter: 增强器实例
        original_sample: 原始样本
        num_reports: 生成的报告数量
        
    Returns:
        包含增强统计信息的字典
    """
    reports = []
    
    for i in range(num_reports):
        augmented = augmenter.apply_augmentation(original_sample)
        
        # 计算统计信息
        stats = {
            'original_mean': original_sample.mean().item(),
            'original_std': original_sample.std().item(),
            'augmented_mean': augmented.mean().item(),
            'augmented_std': augmented.std().item(),
            'mean_absolute_diff': torch.abs(augmented - original_sample).mean().item(),
            'max_absolute_diff': torch.abs(augmented - original_sample).max().item(),
            'correlation': torch.corrcoef(torch.stack([original_sample, augmented]))[0, 1].item()
        }
        
        reports.append(stats)
    
    # 计算平均统计信息
    avg_report = {
        'avg_mean_absolute_diff': np.mean([r['mean_absolute_diff'] for r in reports]),
        'avg_max_absolute_diff': np.mean([r['max_absolute_diff'] for r in reports]),
        'avg_correlation': np.mean([r['correlation'] for r in reports]),
        'current_intensity_factor': augmenter.current_intensity_factor,
        'config_summary': {
            'enabled_strategies': [k for k, v in augmenter.config['strategies'].items() 
                                if v.get('enabled', False)]
        }
    }
    
    logger.info(f"增强效果报告: 平均MAE={avg_report['avg_mean_absolute_diff']:.4f}, "
                f"平均相关性={avg_report['avg_correlation']:.4f}")
    
    return avg_report


# 使用示例
if __name__ == "__main__":
    # 创建示例数据
    example_data = torch.randn(62)  # 假设输入维度为62
    
    # 创建增强器
    augmenter = EnhancedDataAugmenter()
    
    # 应用增强
    augmented_data = augmenter.apply_augmentation(example_data)
    
    # 生成报告
    report = generate_augmentation_report(augmenter, example_data, num_reports=10)
    print(f"增强报告: {report}")
    
    # 测试自适应参数更新
    augmenter.update_adaptive_parameters({'loss': 0.1})
    augmenter.update_adaptive_parameters({'loss': 0.12})
    augmenter.update_adaptive_parameters({'loss': 0.15})  # 模拟过拟合
    
    # 再次应用增强（应该使用更强的增强）
    augmented_data2 = augmenter.apply_augmentation(example_data)
    
    print("\n增强器配置:")
    import json
    print(json.dumps(augmenter.get_current_config(), indent=2, ensure_ascii=False))
    
    print("\n增强版数据增强模块测试完成！")