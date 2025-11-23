#!/usr/bin/env python3
"""
EWPINN优化训练脚本 - 解决损失过高问题
包含数据标准化、损失稳定化、渐进式训练
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import json
import copy
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import sklearn.exceptions
sklearn.exceptions.EfficiencyWarning = FutureWarning  # 忽略sklearn警告
from ewp_pinn_input_layer import EWPINNInputLayer
from ewp_pinn_output_layer import EWPINNOutputLayer
from ewp_data_interface import create_dataset, create_dataloader
from ewp_data_interface import validate_units
from ewp_pinn_physics import PINNConstraintLayer, PhysicsEnhancedLoss
from ewp_pinn_adaptive_hyperoptimizer import AdaptiveHyperparameterOptimizer
from ewp_pinn_performance_monitor import ModelPerformanceMonitor
from ewp_pinn_regularization import AdvancedRegularizer, GradientNoiseRegularizer, apply_regularization_to_model
from ewp_pinn_optimized_architecture import EfficientEWPINN, create_optimized_model, get_model_optimization_suggestions

class OptimizedEWPINN(nn.Module):
    """
    优化版EWPINN模型 - 增强型神经网络架构，支持配置文件加载
    特性：批量标准化、改进的初始化、灵活的架构配置
    """
    def __init__(self, input_dim=62, output_dim=24, hidden_layers=None, dropout_rate=0.1,
                 activation='ReLU', batch_norm=True, config_path=None, device='cpu'):
        super(OptimizedEWPINN, self).__init__()
        
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        
        # 模型配置信息，用于版本控制和兼容性检查
        self.model_info = {
            'version': '1.0.0',
            'input_dim': input_dim,
            'output_dim': output_dim,
            'hidden_layers': hidden_layers if hidden_layers else [128, 64, 32],
            'dropout_rate': dropout_rate,
            'activation': activation,
            'batch_norm': batch_norm,
            'architecture': 'EWPINN',
            'created_at': datetime.now().isoformat()
        }
        
        # 从配置文件加载参数（如果提供）
        if config_path and os.path.exists(config_path):
            self._load_config(config_path)
        
        # 默认隐藏层配置
        if hidden_layers is None:
            hidden_layers = [128, 64, 32]
        self.hidden_layers = hidden_layers
        
        # 更新model_info中的hidden_layers
        self.model_info['hidden_layers'] = hidden_layers
        self.model_info['activation'] = activation
        
        # 选择激活函数
        activation_map = {
            'ReLU': nn.ReLU,
            'LeakyReLU': nn.LeakyReLU,
            'GELU': nn.GELU,
            'SiLU': nn.SiLU
        }
        activation_fn = activation_map.get(activation, nn.ReLU)
        
        # 构建网络
        layers = []
        prev_dim = input_dim
        
        # 构建隐藏层
        for i, hidden_dim in enumerate(hidden_layers):
            # 线性层
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # 批量标准化（如果启用）
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            # 激活函数
            layers.append(activation_fn())
            
            # Dropout（除了最后一层）
            if i < len(hidden_layers) - 1 and dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers).to(device)
        
        # 初始化权重
        self._initialize_weights()
        
        # 打印模型信息
        self._print_model_info(activation)
    
    def _load_config(self, config_path):
        """从JSON配置文件加载模型参数"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                
            if '模型配置' in config:
                model_config = config['模型配置']
                if '输入维度' in model_config:
                    self.input_dim = model_config['输入维度']
                if '输出维度' in model_config:
                    self.output_dim = model_config['输出维度']
                if '网络架构' in model_config:
                    net_config = model_config['网络架构']
                    if '隐藏层' in net_config:
                        self.hidden_layers = net_config['隐藏层']
                    if 'Dropout' in net_config:
                        self.dropout_rate = net_config['Dropout']
                print(f"✅ 成功加载配置文件: {config_path}")
        except Exception as e:
            print(f"⚠️  加载配置文件失败: {str(e)}")
            print("   将使用默认配置")
    
    def _initialize_weights(self):
        """使用He初始化方法改进权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 使用He初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)
    
    def _print_model_info(self, activation):
        """打印模型架构信息"""
        print(f"🚀 优化EWPINN模型已初始化 - 设备: {self.device}")
        print(f"   输入维度: {self.input_dim}, 输出维度: {self.output_dim}")
        print(f"   激活函数: {activation}")
        print(f"   批量标准化: {'启用' if self.batch_norm else '禁用'}")
        print(f"   Dropout率: {self.dropout_rate}")
        
        # 打印网络结构
        structure_str = f"{self.input_dim}"
        for dim in self.hidden_layers:
            structure_str += f" -> {dim}"
        structure_str += f" -> {self.output_dim}"
        print(f"   网络结构: {structure_str}")
        
        # 计算参数数量
        param_count = sum(p.numel() for p in self.parameters())
        print(f"   参数数量: {param_count:,}")
    
    def forward(self, x):
        """前向传播"""
        return self.model(x)
    
    def get_model_summary(self):
        """返回模型摘要信息"""
        summary = {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'hidden_layers': self.hidden_layers,
            'dropout_rate': self.dropout_rate,
            'batch_norm': self.batch_norm,
            'parameter_count': sum(p.numel() for p in self.parameters()),
            'device': str(self.device)
        }
        return summary

class LossStabilizer:
    """
    高级损失稳定器 - 增强数值稳定性和灵活性
    特性：多种损失函数、自适应稳定化、配置文件支持
    """
    def __init__(self, epsilon=1e-10, loss_type='mse', safe_clamp=True,
                 config_path=None, patience=5, reduction_factor=0.5):
        self.epsilon = epsilon
        self.loss_type = loss_type
        self.safe_clamp = safe_clamp
        self.patience = patience
        self.reduction_factor = reduction_factor
        self.current_loss = float('inf')
        self.loss_history = []
        self.best_loss = float('inf')
        self.patience_counter = 0
        
        # 从配置文件加载参数（如果提供）
        if config_path and os.path.exists(config_path):
            self._load_config(config_path)
        
        print(f"📊 高级损失稳定器已初始化")
        print(f"   损失类型: {loss_type}")
        print(f"   数值安全参数: {epsilon}")
        print(f"   安全裁剪: {'启用' if safe_clamp else '禁用'}")
        print(f"   自适应稳定化: {'启用' if patience > 0 else '禁用'} (耐心: {patience})")
    
    def _load_config(self, config_path):
        """从JSON配置文件加载损失稳定器参数"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                
            if '物理约束' in config:
                constraint_config = config['物理约束']
                if '数值稳定性参数' in constraint_config:
                    self.epsilon = constraint_config['数值稳定性参数']
                if '损失类型' in constraint_config:
                    self.loss_type = constraint_config['损失类型']
                if '安全裁剪' in constraint_config:
                    self.safe_clamp = constraint_config['安全裁剪']
                if '自适应稳定化' in constraint_config:
                    adapt_config = constraint_config['自适应稳定化']
                    if '耐心' in adapt_config:
                        self.patience = adapt_config['耐心']
                    if '减少因子' in adapt_config:
                        self.reduction_factor = adapt_config['减少因子']
            print(f"✅ 成功加载损失稳定器配置")
        except Exception as e:
            print(f"⚠️  加载损失稳定器配置失败: {str(e)}")
    
    def safe_mse_loss(self, pred, target, max_loss_value=1e6):
        """安全的MSE损失计算"""
        # 1. 数据预裁剪 - 避免极端值
        if self.safe_clamp:
            pred_clipped = torch.clamp(pred, -1000, 1000)
            target_clipped = torch.clamp(target, -1000, 1000)
        else:
            pred_clipped, target_clipped = pred, target
        
        # 2. 计算MSE
        mse = nn.functional.mse_loss(pred_clipped, target_clipped)
        
        # 3. 立即裁剪损失值
        safe_mse = torch.clamp(mse, 0, max_loss_value)
        
        # 4. 对数变换稳定（可选）
        if safe_mse > 1.0:
            stable_loss = torch.log(1 + safe_mse)
        else:
            stable_loss = safe_mse
            
        return stable_loss
    
    def relative_loss(self, pred, target, epsilon=None):
        """相对损失 - 对数据量级不敏感"""
        # 使用实例或参数epsilon
        eps = epsilon if epsilon is not None else self.epsilon
        
        # 计算相对误差
        denominator = torch.abs(target) + eps
        relative_error = torch.abs(pred - target) / denominator
        
        # 返回平均相对误差
        return torch.mean(relative_error)
    
    def huber_loss(self, pred, target, delta=1.0):
        """Huber损失，结合MSE和MAE的优点"""
        if self.safe_clamp:
            pred = torch.clamp(pred, -1000, 1000)
            target = torch.clamp(target, -1000, 1000)
        
        # 计算绝对误差
        abs_error = torch.abs(pred - target)
        
        # 对小误差使用平方，大误差使用线性
        quadratic = torch.clamp(abs_error, max=delta)
        linear = abs_error - quadratic
        
        loss = 0.5 * quadratic.pow(2) + delta * linear
        
        return torch.mean(loss)
    
    def combined_loss(self, pred, target, mse_weight=0.5, relative_weight=0.5):
        """组合损失函数，平衡MSE和相对误差"""
        mse = self.safe_mse_loss(pred, target)
        relative = self.relative_loss(pred, target)
        
        # 归一化损失值
        total_weight = mse_weight + relative_weight
        if total_weight > 0:
            mse_weight /= total_weight
            relative_weight /= total_weight
        
        return mse_weight * mse + relative_weight * relative
    
    def compute_loss(self, pred, target):
        """根据配置计算损失"""
        loss_mapping = {
            'mse': self.safe_mse_loss,
            'relative': self.relative_loss,
            'huber': self.huber_loss,
            'combined': self.combined_loss
        }
        
        loss_fn = loss_mapping.get(self.loss_type, self.safe_mse_loss)
        loss = loss_fn(pred, target)
        
        # 记录损失历史
        self.loss_history.append(loss.item())
        self.current_loss = loss.item()
        
        return loss
    
    def adaptive_stabilization(self, current_loss):
        """自适应稳定化机制，检测并处理训练不稳定情况"""
        if self.patience <= 0:
            return False, 1.0  # 不启用自适应稳定化
        
        # 更新最佳损失
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.patience_counter = 0
            return False, 1.0
        
        # 增加耐心计数器
        self.patience_counter += 1
        
        # 如果超过耐心阈值，触发稳定化
        if self.patience_counter >= self.patience:
            self.patience_counter = 0
            scale_factor = self.reduction_factor
            print(f"⚠️  检测到训练不稳定，应用稳定化因子: {scale_factor}")
            return True, scale_factor
        
        return False, 1.0
    
    def should_stop_early(self, threshold=1e-6):
        """早停机制，检测训练是否收敛"""
        if len(self.loss_history) < 10:
            return False
        
        # 检查最近10次损失的变化
        recent_losses = self.loss_history[-10:]
        loss_std = np.std(recent_losses)
        
        return loss_std < threshold

class DataNormalizer:
    """
    高级数据标准化器 - 增强的数据预处理和标准化功能
    特性：多种标准化方法、异常值处理、配置文件支持
    """
    def __init__(self, feature_method='standard', label_method='minmax',
                 handle_outliers=True, outlier_threshold=3.0, config_path=None):
        self.feature_method = feature_method
        self.label_method = label_method
        self.handle_outliers = handle_outliers
        self.outlier_threshold = outlier_threshold
        self.is_fitted = False
        
        # 初始化标准化器
        self.input_scaler = self._create_scaler(feature_method)
        self.output_scaler = self._create_scaler(label_method)
        
        # 从配置文件加载参数（如果提供）
        if config_path and os.path.exists(config_path):
            self._load_config(config_path)
        
        print(f"🔄 高级数据标准化器已初始化")
        print(f"   特征标准化方法: {feature_method}")
        print(f"   标签标准化方法: {label_method}")
        print(f"   异常值处理: {'启用' if handle_outliers else '禁用'} (阈值: {outlier_threshold})")
    
    def _create_scaler(self, method):
        """根据方法创建标准化器"""
        if method == 'minmax':
            return MinMaxScaler(feature_range=(0, 1))
        elif method == 'robust':
            from sklearn.preprocessing import RobustScaler
            return RobustScaler()
        elif method == 'power':
            from sklearn.preprocessing import PowerTransformer
            return PowerTransformer(method='yeo-johnson')
        else:  # standard
            return StandardScaler()
    
    def _load_config(self, config_path):
        """从JSON配置文件加载数据标准化器参数"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                
            if '数据处理' in config:
                data_config = config['数据处理']
                if '特征标准化方法' in data_config:
                    self.feature_method = data_config['特征标准化方法']
                if '标签标准化方法' in data_config:
                    self.label_method = data_config['标签标准化方法']
                if '异常值处理' in data_config:
                    outlier_config = data_config['异常值处理']
                    if '启用' in outlier_config:
                        self.handle_outliers = outlier_config['启用']
                    if '阈值' in outlier_config:
                        self.outlier_threshold = outlier_config['阈值']
                
                # 重新创建标准化器以应用新方法
                self.input_scaler = self._create_scaler(self.feature_method)
                self.output_scaler = self._create_scaler(self.label_method)
                
            print(f"✅ 成功加载数据标准化器配置")
        except Exception as e:
            print(f"⚠️  加载数据标准化器配置失败: {str(e)}")
    
    def _handle_outliers(self, data, threshold=None):
        """处理异常值"""
        if not self.handle_outliers:
            return data
        
        threshold = threshold if threshold is not None else self.outlier_threshold
        
        # 转换为numpy数组
        data_np = data.cpu().numpy() if torch.is_tensor(data) else data
        
        # 使用IQR方法检测异常值
        q1 = np.percentile(data_np, 25, axis=0)
        q3 = np.percentile(data_np, 75, axis=0)
        iqr = q3 - q1
        
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        
        # 裁剪异常值
        data_clipped = np.clip(data_np, lower_bound, upper_bound)
        
        # 计算异常值数量
        outliers = np.sum((data_np < lower_bound) | (data_np > upper_bound))
        total_values = data_np.size
        
        if outliers > 0:
            outlier_percent = (outliers / total_values) * 100
            print(f"⚠️  检测并处理了 {outliers} 个异常值 ({outlier_percent:.2f}%)")
        
        return data_clipped
    
    def fit(self, features, labels):
        """拟合一化器"""
        # 转换为numpy数组
        features_np = features.cpu().numpy() if torch.is_tensor(features) else features
        labels_np = labels.cpu().numpy() if torch.is_tensor(labels) else labels
        
        # 处理异常值
        if self.handle_outliers:
            features_np = self._handle_outliers(features_np)
            labels_np = self._handle_outliers(labels_np)
        
        # 标准化输入特征
        self.input_scaler.fit(features_np)
        
        # 归一化输出标签
        self.output_scaler.fit(labels_np)
        
        self.is_fitted = True
        print(f"✅ 数据标准化器拟合完成")
        
        # 打印特征统计信息
        if hasattr(self.input_scaler, 'mean_'):
            print(f"   特征均值范围: {self.input_scaler.mean_.min():.4f} ~ {self.input_scaler.mean_.max():.4f}")
        
        # 打印标签统计信息
        if hasattr(self.output_scaler, 'mean_'):
            print(f"   标签均值范围: {self.output_scaler.mean_.min():.4f} ~ {self.output_scaler.mean_.max():.4f}")
    
    def transform_features(self, features):
        """标准化特征"""
        if not self.is_fitted:
            raise ValueError("标准化器未拟合")
        
        is_tensor = torch.is_tensor(features)
        if is_tensor:
            device = features.device
            features_np = features.cpu().numpy()
            if self.handle_outliers:
                features_np = self._handle_outliers(features_np)
            features_normalized = self.input_scaler.transform(features_np)
            return torch.tensor(features_normalized, dtype=torch.float32, device=device)
        else:
            data_np = features
            if self.handle_outliers:
                data_np = self._handle_outliers(data_np)
            features_normalized = self.input_scaler.transform(data_np)
            return features_normalized
    
    def transform_labels(self, labels):
        """标准化标签"""
        if not self.is_fitted:
            raise ValueError("标准化器未拟合")
        
        is_tensor = torch.is_tensor(labels)
        if is_tensor:
            device = labels.device
            labels_np = labels.cpu().numpy()
            if self.handle_outliers:
                labels_np = self._handle_outliers(labels_np)
            labels_normalized = self.output_scaler.transform(labels_np)
            labels_tensor = torch.tensor(labels_normalized, dtype=torch.float32, device=device)
            labels_tensor = torch.clamp(labels_tensor, 0.0, 1.0)
            return labels_tensor
        else:
            data_np = labels
            if self.handle_outliers:
                data_np = self._handle_outliers(data_np)
            labels_normalized = self.output_scaler.transform(data_np)
            import numpy as np
            labels_normalized = np.clip(labels_normalized, 0.0, 1.0)
            return labels_normalized
    
    def inverse_transform_labels(self, labels_normalized):
        """反归一化标签 - 增强版本，确保数值稳定性"""
        if not self.is_fitted:
            raise ValueError("标准化器未拟合")
        
        # 处理PyTorch张量
        if torch.is_tensor(labels_normalized):
            device = labels_normalized.device
            labels_np = labels_normalized.cpu().numpy()
        else:
            labels_np = labels_normalized.copy()  # 创建副本以避免修改原始数据
        
        # 关键改进：在逆标准化前将数据裁剪到合理范围
        # 特别是对于minmax标准化，确保数据在[0,1]范围内
        if isinstance(self.output_scaler, MinMaxScaler):
            labels_np = np.clip(labels_np, 0.0, 1.0)
        else:
            # 对于其他标准化方法，使用更宽松的裁剪范围
            # 找出数据的四分位数以确定合理范围
            q1 = np.percentile(labels_np, 25, axis=0)
            q3 = np.percentile(labels_np, 75, axis=0)
            iqr = q3 - q1
            # 使用更宽松的范围，避免过度裁剪
            lower_bound = q1 - 5 * iqr
            upper_bound = q3 + 5 * iqr
            # 处理可能的零IQR情况
            if np.any(iqr == 0):
                # 使用标准差来确定范围
                std = np.std(labels_np, axis=0)
                mean = np.mean(labels_np, axis=0)
                # 更新零IQR维度的边界
                for i in range(len(iqr)):
                    if iqr[i] == 0:
                        lower_bound[i] = mean[i] - 5 * std[i] if std[i] > 0 else mean[i] - 1.0
                        upper_bound[i] = mean[i] + 5 * std[i] if std[i] > 0 else mean[i] + 1.0
            
            # 裁剪到计算的范围内
            labels_np = np.clip(labels_np, lower_bound, upper_bound)
        
        # 添加数值稳定性检查
        # 检查并替换NaN和无穷大值
        labels_np = np.nan_to_num(labels_np)
        
        # 进行逆标准化
        try:
            labels_original = self.output_scaler.inverse_transform(labels_np)
            
            # 再次检查逆标准化后的数值稳定性
            labels_original = np.nan_to_num(labels_original)
            
            # 进一步防止极端值
            # 计算逆标准化后数据的合理范围
            if labels_original.size > 0:  # 确保数组不为空
                # 使用稳健的统计量确定范围
                q1_inv = np.percentile(labels_original, 25, axis=0)
                q3_inv = np.percentile(labels_original, 75, axis=0)
                iqr_inv = q3_inv - q1_inv
                
                # 使用更严格的范围来防止异常大的值
                lower_bound_inv = q1_inv - 3 * iqr_inv
                upper_bound_inv = q3_inv + 3 * iqr_inv
                
                # 处理零IQR情况
                if np.any(iqr_inv == 0):
                    std_inv = np.std(labels_original, axis=0)
                    mean_inv = np.mean(labels_original, axis=0)
                    for i in range(len(iqr_inv)):
                        if iqr_inv[i] == 0:
                            lower_bound_inv[i] = mean_inv[i] - 3 * std_inv[i] if std_inv[i] > 0 else mean_inv[i] - 10.0
                            upper_bound_inv[i] = mean_inv[i] + 3 * std_inv[i] if std_inv[i] > 0 else mean_inv[i] + 10.0
                
                # 最终裁剪，确保值不会过大
                labels_original = np.clip(labels_original, lower_bound_inv, upper_bound_inv)
            
            # 如果输入是张量，转换回张量
            if torch.is_tensor(labels_normalized):
                return torch.tensor(labels_original, dtype=torch.float32, device=device)
            else:
                return labels_original
        except Exception as e:
            print(f"⚠️  逆标准化过程中出错: {str(e)}")
            # 出错时返回一个合理的默认值（原始范围的中位数附近）
            if hasattr(self.output_scaler, 'data_min_'):
                default_value = (self.output_scaler.data_min_ + self.output_scaler.data_max_) / 2
                return np.full_like(labels_np, default_value) if isinstance(labels_normalized, np.ndarray) else \
                       torch.full_like(labels_normalized, default_value, device=device)
            else:
                # 返回零矩阵作为后备
                return np.zeros_like(labels_np) if isinstance(labels_normalized, np.ndarray) else \
                       torch.zeros_like(labels_normalized, device=device)
    
    def get_scaler_info(self):
        """获取标准化器信息"""
        info = {
            'feature_method': self.feature_method,
            'label_method': self.label_method,
            'handle_outliers': self.handle_outliers,
            'is_fitted': self.is_fitted
        }
        
        # 添加特征标准化器参数
        if hasattr(self.input_scaler, 'mean_'):
            info['feature_mean'] = self.input_scaler.mean_.tolist()
        if hasattr(self.input_scaler, 'scale_'):
            info['feature_scale'] = self.input_scaler.scale_.tolist()
        
        # 添加标签标准化器参数
        if hasattr(self.output_scaler, 'mean_'):
            info['label_mean'] = self.output_scaler.mean_.tolist()
        if hasattr(self.output_scaler, 'scale_'):
            info['label_scale'] = self.output_scaler.scale_.tolist()
        
        return info

def generate_realistic_data(model, num_samples=200, config_path=None, seed=None, data_augmentation=True):
    """
    高级数据生成器 - 基于物理约束的数据生成和增强
    特性：配置文件支持、数据增强、质量控制、参数优化
    """
    # 设置随机种子以确保可重复性
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    print(f"🔄 高级数据生成器启动 - {num_samples}个样本")
    print(f"   数据增强: {'启用' if data_augmentation else '禁用'}")
    
    # 默认配置
    data_config = {
        'implementation_stage': 3,
        'parameter_ranges': {
            'frequency': {'min': 0.1, 'max': 10.0},
            'power': {'min': 0.01, 'max': 100.0},
            'dimension': {'min': 1e-6, 'max': 1e-3},
            'size': {'min': 1e-6, 'max': 1e-3},
            'default': {'min': None, 'max': None}
        },
        'augmentation_level': 0.1,
        'noise_level': 0.05,
        'correlation_strength': 0.7,
        'validation_ratio': 0.1
    }
    
    # 从配置文件加载参数
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                if '数据处理' in config and '数据生成' in config['数据处理']:
                    gen_config = config['数据处理']['数据生成']
                    for key, value in gen_config.items():
                        if key in data_config:
                            if key == 'parameter_ranges' and isinstance(value, dict):
                                # 合并参数范围配置
                                for param, ranges in value.items():
                                    if param not in data_config['parameter_ranges']:
                                        data_config['parameter_ranges'][param] = ranges
                                    else:
                                        data_config['parameter_ranges'][param].update(ranges)
                            else:
                                data_config[key] = value
                    print(f"✅ 成功加载数据生成配置")
        except Exception as e:
            print(f"⚠️  加载数据生成配置失败: {str(e)}")
    
    # 初始化输入输出层
    device = model.device
    input_layer = EWPINNInputLayer(device=device)
    output_layer = EWPINNOutputLayer(device=device)
    
    # 设置实现阶段
    stage = data_config['implementation_stage']
    input_layer.set_implementation_stage(stage)
    output_layer.set_implementation_stage(stage)
    print(f"   实现阶段: {stage}")
    
    features_list = []
    labels_list = []
    success_count = 0
    
    # 生成数据
    start_time = time.time()
    for i in range(num_samples):
        try:
            # 创建输入字典
            input_dict = input_layer.generate_example_input()
            
            # 对输入参数进行合理化调整
            input_dict = normalize_input_parameters(input_dict, data_config['parameter_ranges'])
            
            # 转换为输入向量
            input_vector = input_layer.create_input_vector(input_dict)
            
            # 确保是torch张量
            if not isinstance(input_vector, torch.Tensor):
                input_vector = torch.tensor(input_vector, dtype=torch.float32, device=device)
            
            # 应用数据增强
            if data_augmentation and np.random.random() < 0.7:  # 70%概率应用增强
                input_vector = apply_data_augmentation(input_vector, 
                                                      level=data_config['augmentation_level'])
            
            # 添加轻微噪声
            if data_config['noise_level'] > 0:
                noise = torch.randn_like(input_vector) * data_config['noise_level']
                input_vector = input_vector + noise
            
            features_list.append(input_vector)
            
            # 生成对应的输出标签
            random_output = output_layer.generate_random_output(batch_size=1)
            if isinstance(random_output, torch.Tensor):
                label_vector = random_output[0]
            else:
                # 确保转换为torch张量
                label_vector = torch.tensor(random_output[0], dtype=torch.float32, device=device)
            
            # 添加输出噪声
            if data_config['noise_level'] > 0:
                output_noise = torch.randn_like(label_vector) * (data_config['noise_level'] * 0.5)
                label_vector = label_vector + output_noise
            
            labels_list.append(label_vector)
            success_count += 1
            
            # 进度显示
            if (i + 1) % 100 == 0 or i == num_samples - 1:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"   进度: {i+1}/{num_samples} ({rate:.1f}样本/秒)")
                
        except Exception as e:
            if (i + 1) % 50 == 0:
                print(f"⚠️  样本 {i} 生成失败: {str(e)}")
            # 使用零向量作为备选
            zero_vector = torch.zeros(24, dtype=torch.float32, device=device)
            zero_feature = torch.zeros(62, dtype=torch.float32, device=device)
            features_list.append(zero_feature)
            labels_list.append(zero_vector)
    
    # 转换为张量
    features = torch.stack(features_list)
    labels = torch.stack(labels_list)
    
    # 计算数据统计信息
    gen_time = time.time() - start_time
    success_rate = (success_count / num_samples) * 100 if num_samples > 0 else 0
    
    print(f"✅ 数据生成完成")
    print(f"   样本数量: {features.shape[0]}")
    print(f"   成功生成率: {success_rate:.1f}%")
    print(f"   生成时间: {gen_time:.2f}秒 ({num_samples/gen_time:.1f}样本/秒)")
    print(f"   输入形状: {features.shape}, 输出形状: {labels.shape}")
    
    # 分离训练集和验证集
    val_size = int(num_samples * data_config['validation_ratio'])
    if val_size > 0:
        indices = torch.randperm(num_samples)
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]
        
        X_train, y_train = features[train_indices], labels[train_indices]
        X_val, y_val = features[val_indices], labels[val_indices]
        
        print(f"   训练集: {X_train.shape[0]}样本, 验证集: {X_val.shape[0]}样本")
        
        return X_train, y_train, X_val, y_val
    else:
        return features, labels

def normalize_input_parameters(input_dict, parameter_ranges):
    """
    标准化输入参数，确保在合理范围内
    """
    for key, value in input_dict.items():
        if isinstance(value, (int, float)):
            # 查找参数范围
            param_range = None
            for param_key, ranges in parameter_ranges.items():
                if param_key.lower() in key.lower():
                    param_range = ranges
                    break
            
            # 如果找到范围，应用限制
            if param_range:
                min_val = param_range.get('min')
                max_val = param_range.get('max')
                
                if min_val is not None and value < min_val:
                    input_dict[key] = min_val
                if max_val is not None and value > max_val:
                    input_dict[key] = max_val
    
    return input_dict

def apply_data_augmentation(input_vector, level=0.1):
    """
    应用数据增强到输入向量
    包括随机缩放、轻微旋转和非线性变换
    """
    augmented = input_vector.clone()
    
    # 随机缩放 - 对不同维度应用不同的缩放因子
    scale_factors = 1.0 + (torch.randn_like(augmented) * level)
    augmented = augmented * scale_factors
    
    # 选择部分维度进行非线性变换
    num_transformed = min(5, augmented.size(0))
    transform_indices = torch.randperm(augmented.size(0))[:num_transformed]
    
    for idx in transform_indices:
        # 应用正弦变换作为非线性增强
        augmented[idx] = augmented[idx] + torch.sin(augmented[idx]) * level
    
    return augmented

def progressive_training(config_path='model_config.json', resume_training=False, resume_checkpoint=None, mixed_precision=True, model_init_seed=None, use_adaptive_hyperopt=False, enable_performance_monitor=True, enable_advanced_regularization=True, use_efficient_architecture=True, model_compression_factor=1.0):
    """
    高级渐进式训练策略 - 集成PINN物理约束和集成学习支持
    特性：配置文件支持、多种优化器、高级学习率调度、早停机制、混合精度训练、物理约束、集成学习支持、自适应超参数优化、模型性能监控与诊断
    
    参数:
    - config_path: 配置文件路径
    - resume_training: 是否从检查点恢复训练
    - resume_checkpoint: 恢复训练的检查点路径
    - mixed_precision: 是否启用混合精度训练
    - model_init_seed: 模型初始化种子，用于创建具有不同初始权重的模型（支持集成学习）
    - use_adaptive_hyperopt: 是否启用自适应超参数优化
    - enable_performance_monitor: 是否启用模型性能监控与诊断工具
    """
    print("🚀 EWPINN高级优化训练系统启动 - 集成物理约束")
    print("=" * 60)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 使用设备: {device}")
    print(f"⚡ 混合精度训练: {'启用' if mixed_precision and torch.cuda.is_available() else '禁用'}")
    print(f"🔬 PINN物理约束: 已启用")
    print(f"🎯 自适应超参数优化: {'启用' if use_adaptive_hyperopt else '禁用'}")
    print(f"📊 模型性能监控: {'启用' if enable_performance_monitor else '禁用'}")

    # 抑制物理模块日志输出（仅显示错误）
    import logging
    physics_logger = logging.getLogger('EWPINN_Physics')
    physics_logger.setLevel(logging.ERROR)
    physics_logger.propagate = True
    
    # 创建时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 模型初始化种子（用于集成学习，创建不同的初始权重）
    print(f"🎲 模型初始化种子: {'随机' if model_init_seed is None else model_init_seed} (用于集成学习)")
    
    # 默认配置
    default_config = {
        '模型': {
            '输入维度': 62,
            '输出维度': 24,
            '隐藏层': [128, 64, 32],
            '激活函数': 'ReLU',
            '批标准化': True,
            'Dropout率': 0.1
        },
        '训练': {
            '渐进式训练': [
                {
                    '名称': '预热阶段',
                    '轮次': 10,
                    '学习率': 1e-4,
                    '批次大小': 16,
                    '权重衰减': 1e-5,
                    '优化器': 'AdamW',
                    '调度策略': 'CosineAnnealing',
                    '调度参数': {'T_max': 10},
                    '描述': '小学习率预热，激活函数适应',
                    '物理约束权重': 0.05
                },
                {
                    '名称': '主训练阶段',
                    '轮次': 20,
                    '学习率': 5e-4,
                    '批次大小': 32,
                    '权重衰减': 1e-5,
                    '优化器': 'AdamW',
                    '调度策略': 'CosineAnnealing',
                    '调度参数': {'T_max': 20},
                    '描述': '主要训练阶段，平衡收敛速度与稳定性',
                    '物理约束权重': 0.1
                },
                {
                    '名称': '精细调优',
                    '轮次': 10,
                    '学习率': 1e-4,
                    '批次大小': 32,
                    '权重衰减': 1e-6,
                    '优化器': 'AdamW',
                    '调度策略': 'CosineAnnealing',
                    '调度参数': {'T_max': 10},
                    '描述': '精细调优，提高精度',
                    '物理约束权重': 0.2
                }
            ],
            '早停配置': {
                '启用': True,
                '耐心值': 5,
                '最小改进': 5e-4,
                '恢复最佳模型': True
            },
            '梯度裁剪': 1.0,
            '梯度累积步数': 1
        },
        '数据': {
            '样本数量': 300,
            '数据增强': True,
            '训练比例': 0.8,
            '验证比例': 0.1,
            '测试比例': 0.1
        },
        '物理约束': {
            '启用': True,
            '初始权重': 0.1,
            '权重衰减': 0.99,
            '物理点数量': 500,
            '残差权重': {
                '连续性': 1.0,
                '动量_u': 0.1,
                '动量_v': 0.1,
                '动量_w': 0.1
            },
            '自适应权重': True
        }
    }
    
    # 加载配置文件
    config = default_config.copy()
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
                # 深度合并配置
                if '模型' in user_config:
                    config['模型'].update(user_config['模型'])
                if '训练' in user_config:
                    if '渐进式训练' in user_config['训练']:
                        config['训练']['渐进式训练'] = user_config['训练']['渐进式训练']
                    if '早停配置' in user_config['训练']:
                        config['训练']['早停配置'].update(user_config['训练']['早停配置'])
                    for key, value in user_config['训练'].items():
                        if key not in ['渐进式训练', '早停配置']:
                            config['训练'][key] = value
                if '数据' in user_config:
                    config['数据'].update(user_config['数据'])
            print(f"✅ 成功加载配置文件: {config_path}")
        except Exception as e:
            print(f"⚠️  加载配置文件失败，使用默认配置: {str(e)}")
    else:
        print(f"ℹ️  配置文件不存在或未指定，使用默认配置")
    
    # 初始化模型
    # 如果设置了模型初始化种子，使用它来创建不同的初始权重分布
    if model_init_seed is not None:
        # 临时设置随机种子用于模型初始化
        original_state = torch.get_rng_state()
        torch.manual_seed(model_init_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(model_init_seed)
    
    # 根据参数选择模型架构
    if use_efficient_architecture:
        print(f"🔧 使用高效EWPINN架构，压缩因子: {model_compression_factor}")
        model = EfficientEWPINN(
            input_dim=config['模型']['输入维度'],
            output_dim=config['模型']['输出维度'],
            hidden_layers=config['模型']['隐藏层'],
            dropout_rate=config['模型']['Dropout率'],
            activation=config['模型']['激活函数'],
            batch_norm=config['模型']['批标准化'],
            device=device,
            compression_factor=model_compression_factor,
            use_residual=True,
            use_attention=True,
            gradient_checkpointing=False
        )
        
        # 获取模型优化建议
        optimization_suggestions = get_model_optimization_suggestions(model)
        for suggestion in optimization_suggestions:
            print(f"💡 优化建议: {suggestion}")
    else:
        model = OptimizedEWPINN(
            input_dim=config['模型']['输入维度'],
            output_dim=config['模型']['输出维度'],
            hidden_layers=config['模型']['隐藏层'],
            dropout_rate=config['模型']['Dropout率'],
            activation=config['模型']['激活函数'],
            batch_norm=config['模型']['批标准化'],
            device=device
        )
    
    # 恢复原始随机种子状态
    if model_init_seed is not None:
        torch.set_rng_state(original_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(original_state)
    
    # 初始化自适应超参数优化器
    hyperoptimizer = None
    if use_adaptive_hyperopt:
        hyperoptimizer = AdaptiveHyperparameterOptimizer(
            config=config,
            device=device,
            patience=5,
            reduction_factor=0.5,
            verbose=True
        )
    
    # 初始化模型性能监控器
    performance_monitor = None
    if enable_performance_monitor:
        perf_dir = os.path.join(os.getcwd(), 'performance_reports')
        try:
            os.makedirs(perf_dir, exist_ok=True)
        except Exception:
            pass
        performance_monitor = ModelPerformanceMonitor(
            device=device,
            save_dir=perf_dir
        )
    
    # 初始化高级正则化器
    regularizer = None
    gradient_noise_reg = None
    if enable_advanced_regularization:
        # 从配置中获取正则化参数
        reg_config = config.get('正则化配置', {})
        regularizer = AdvancedRegularizer(
            config_path=None,  # 可以从配置文件加载
            l1_lambda=reg_config.get('L1正则化系数', 0.0),
            l2_lambda=reg_config.get('L2正则化系数', 0.001),
            dropout_rate=reg_config.get('Dropout率', config['模型'].get('Dropout率', 0.1)),
            use_weight_clipping=reg_config.get('使用权重裁剪', False),
            weight_clip_value=reg_config.get('权重裁剪阈值', 1.0),
            use_spectral_norm=reg_config.get('使用谱归一化', False),
            enable_early_stopping=reg_config.get('启用早停', True),
            patience=reg_config.get('早停耐心值', 10),
            device=device
        )
        
        # 初始化梯度噪声正则化器
        apply_gradient_noise = reg_config.get('应用梯度噪声', False)
        if apply_gradient_noise:
            gradient_noise_reg = GradientNoiseRegularizer(
                eta=reg_config.get('梯度噪声系数', 0.01),
                gamma=reg_config.get('梯度噪声衰减率', 0.55)
            )
        
        # 应用正则化到模型
        apply_dropconnect = reg_config.get('应用DropConnect', False)
        if apply_dropconnect:
            model = apply_regularization_to_model(
                model,
                regularizer,
                apply_dropconnect=True,
                dropconnect_rate=reg_config.get('DropConnect率', 0.2)
            )
        
        print(f"✅ 高级正则化已启用")
    
    # 初始化物理约束层（如果启用）
    pinn_layer = None
    physics_enabled = config.get('物理约束', {}).get('启用', True)
    if physics_enabled:
        # 创建物理约束层
        residual_weights = config['物理约束'].get('残差权重', {})
        pinn_layer = PINNConstraintLayer(
            residual_weights=residual_weights,
        )
        pinn_layer.adaptive_weights = config['物理约束'].get('自适应权重', True)
        pinn_layer = pinn_layer.to(device)
        print(f"✅ 物理约束层已初始化: 自适应权重={pinn_layer.adaptive_weights}")
    
    # 生成数据（使用改进后的版本）
    num_samples = config['数据']['样本数量']
    data_augmentation = config['数据']['数据增强']
    
    # 如果数据生成函数支持训练/验证/测试集分离，直接获取
    if hasattr(generate_realistic_data, '__code__') and 'validation_ratio' in generate_realistic_data.__code__.co_varnames:
        X_train_raw, y_train_raw, X_val_raw, y_val_raw = generate_realistic_data(
            model, 
            num_samples=num_samples, 
            config_path=config_path,
            seed=42,  # 设置固定种子确保可重复性
            data_augmentation=data_augmentation
        )
        # 初始化标准化器
        normalizer = DataNormalizer(
            feature_method='robust',  # 使用robust方法更适合处理异常值
            label_method='minmax',
            handle_outliers=True,
            outlier_threshold=2.5  # 降低阈值以处理更多异常值
        )
        # 对所有数据进行标准化
        # 仅在训练集上拟合标准化器
        combined_features = torch.cat([X_train_raw, X_val_raw])
        combined_labels = torch.cat([y_train_raw, y_val_raw])
        normalizer.fit(combined_features, combined_labels)
        
        # 标准化各个数据集
        X_train = normalizer.transform_features(X_train_raw)
        y_train = normalizer.transform_labels(y_train_raw)
        X_val = normalizer.transform_features(X_val_raw)
        y_val = normalizer.transform_labels(y_val_raw)
        
        # 手动创建测试集
        test_size = int(len(X_train) * (config['数据']['测试比例'] / config['数据']['训练比例']))
        X_test, y_test = X_train[:test_size], y_train[:test_size]
        X_train, y_train = X_train[test_size:], y_train[test_size:]
    else:
        # 处理数据生成逻辑，支持返回2个或4个值
        data_result = generate_realistic_data(model, num_samples=num_samples)
        if len(data_result) == 4:  # 返回的是X_train, y_train, X_val, y_val
            features, labels = data_result[0], data_result[1]  # 只使用训练集作为原始特征和标签
        else:  # 返回的是features, labels
            features, labels = data_result
        
        # 数据标准化
        normalizer = DataNormalizer()
        normalizer.fit(features, labels)
        
        # 标准化数据
        features_normalized = normalizer.transform_features(features)
        labels_normalized = normalizer.transform_labels(labels)
        
        # 分割数据集
        train_size = int(config['数据']['训练比例'] * len(features_normalized))
        val_size = int(config['数据']['验证比例'] * len(features_normalized))
        
        X_train = features_normalized[:train_size]
        y_train = labels_normalized[:train_size]
        X_val = features_normalized[train_size:train_size+val_size]
        y_val = labels_normalized[train_size:train_size+val_size]
        X_test = features_normalized[train_size+val_size:]
        y_test = labels_normalized[train_size+val_size:]

    # 统一数据接口：创建测试数据集
    device = next(model.parameters()).device
    test_dataset = create_dataset(X_test, y_test, input_layer=None, stage=None, device=device)
    
    # 生成物理点（如果启用物理约束）
    X_phys = None
    if physics_enabled:
        num_phys_samples = config['物理约束'].get('物理点数量', 500)
        print(f"🔬 生成物理约束点: {num_phys_samples}个")
        # 使用数据生成器生成物理点，但不需要对应的标签
        phys_input_layer = EWPINNInputLayer(device=device)
        phys_input_layer.set_implementation_stage(config['数据'].get('implementation_stage', 3))
        
        X_phys_list = []
        for _ in range(num_phys_samples):
            input_dict = phys_input_layer.generate_example_input()
            input_vector = phys_input_layer.create_input_vector(input_dict)
            if not isinstance(input_vector, torch.Tensor):
                input_vector = torch.tensor(input_vector, dtype=torch.float32, device=device)
            else:
                input_vector = input_vector.to(device)
            X_phys_list.append(input_vector)
        
        X_phys = torch.stack(X_phys_list).to(device)
        print(f"✅ 物理约束点生成完成: {X_phys.shape[0]}个样本")
    
    # 确保有normalizer实例
    if 'normalizer' not in locals():
        normalizer = DataNormalizer()
        normalizer.fit(X_train, y_train)
    
    print(f"📊 标准化后数据范围:")
    print(f"   输入: [{X_train.min():.3f}, {X_train.max():.3f}]")
    print(f"   输出: [{y_train.min():.3f}, {y_train.max():.3f}]")
    print(f"📈 数据集划分: 训练{len(X_train)}, 验证{len(X_val)}, 测试{len(X_test)}")
    
    # 创建保存目录
    if resume_training and resume_checkpoint:
        # 从检查点路径推断保存目录
        save_dir = os.path.dirname(resume_checkpoint)
        print(f"🔄 从检查点恢复训练: {resume_checkpoint}")
    else:
        save_dir = f"checkpoints_optimized_{timestamp}"
        os.makedirs(save_dir, exist_ok=True)
        print(f"📁 创建保存目录: {save_dir}")
    
    # 训练历史
    train_history = []
    val_history = []
    best_val_loss = float('inf')
    best_model_state = None
    no_improve_count = 0
    
    # 早停配置
    early_stopping = config['训练']['早停配置']['启用']
    # 增加耐心值，给物理约束PINN模型更多收敛时间
    patience = config['训练']['早停配置'].get('耐心值', 30)
    min_improvement = config['训练']['早停配置'].get('最小改进', 1e-6)
    restore_best = config['训练']['早停配置'].get('恢复最佳模型', True)
    
    # 梯度裁剪和累积
    gradient_clip = config['训练']['梯度裁剪']
    gradient_accumulation_steps = config['训练']['梯度累积步数']
    
    # 准备混合精度训练
    scaler = torch.cuda.amp.GradScaler() if mixed_precision and torch.cuda.is_available() else None
    
    # 初始化LossStabilizer
    loss_stabilizer = LossStabilizer(config_path=config_path)
    
    # 恢复训练状态（如果启用）
    start_stage = 0
    if resume_training and resume_checkpoint and os.path.exists(resume_checkpoint):
        try:
            checkpoint = torch.load(resume_checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            train_history = checkpoint.get('train_history', [])
            val_history = checkpoint.get('val_history', [])
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            start_stage = checkpoint.get('last_stage', 0)
            print(f"✅ 成功恢复训练状态，从阶段 {start_stage + 1} 继续")
        except Exception as e:
            print(f"⚠️  恢复训练状态失败: {str(e)}")
    
    # 渐进式训练阶段
    training_stages = config['训练']['渐进式训练']
    
    # 开始渐进式训练
    for stage_idx, stage_config in enumerate(training_stages):
        # 如果是恢复训练，跳过已完成的阶段
        if stage_idx < start_stage:
            continue
            
        print(f"\n🎯 {stage_config['名称']} (阶段 {stage_idx + 1}/{len(training_stages)})")
        print(f"   {stage_config['描述']}")
        print(f"   优化器: {stage_config['优化器']}, 调度策略: {stage_config['调度策略']}")
        
        # 创建优化器 - 支持多种优化器
        optimizer = create_optimizer(model, stage_config)
        
        # 创建学习率调度器 - 支持多种调度策略
        scheduler = create_lr_scheduler(optimizer, stage_config)

        # 使用统一数据接口创建本阶段的数据加载器
        train_loader = create_dataloader(
            X_train, y_train, batch_size=stage_config.get('批次大小', 16), shuffle=True,
            device=device, num_workers=0, drop_last=True, pin_memory=False
        )
        val_loader = create_dataloader(
            X_val, y_val, batch_size=stage_config.get('批次大小', 16), shuffle=False,
            device=device, num_workers=0, drop_last=False, pin_memory=False
        )
        
        # 训练循环
        stage_start_time = time.time()
        stage_train_losses = []
        stage_val_losses = []
        
        for epoch in range(stage_config['轮次']):
            # 训练阶段
            model.train()
            train_loss = 0.0
            train_data_loss = 0.0
            train_physics_loss = 0.0
            train_mae = 0.0
            num_train_batches = 0
            batch_start_time = time.time()
            
            # 使用统一的DataLoader进行批处理
            num_train_batches = len(train_loader)
            optimizer.zero_grad()
            for batch_idx, (batch_features, batch_labels) in enumerate(train_loader):
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                stage_physics_weight = stage_config.get('物理约束权重', config['物理约束'].get('初始权重', 0.1))
                if physics_enabled and X_phys is not None:
                    phys_indices = torch.randperm(len(X_phys))[:batch_features.size(0)]
                    X_phys_batch = X_phys[phys_indices].to(device)
                with torch.cuda.amp.autocast(enabled=scaler is not None):
                    predictions = model(batch_features)
                    if physics_enabled and X_phys is not None:
                        data_loss = loss_stabilizer.compute_loss(predictions, batch_labels)
                        phys_outputs = model(X_phys_batch)
                        physics_loss_val, _ = pinn_layer.compute_physics_loss(X_phys_batch, phys_outputs)
                        total_loss = data_loss + stage_physics_weight * physics_loss_val
                    else:
                        data_loss = loss_stabilizer.compute_loss(predictions, batch_labels)
                        total_loss = data_loss
                        physics_loss_val = torch.tensor(0.0, device=device)
                    if regularizer is not None:
                        total_loss = total_loss + regularizer.compute_regularization_loss(model)
                    mae = torch.mean(torch.abs(predictions - batch_labels))
                combined_loss = total_loss + mae * 0.1
                if scaler is not None:
                    scaler.scale(combined_loss / gradient_accumulation_steps).backward()
                else:
                    (combined_loss / gradient_accumulation_steps).backward()
                if ((batch_idx + 1) % gradient_accumulation_steps == 0) or (batch_idx == num_train_batches - 1):
                    if gradient_clip > 0:
                        if scaler is not None:
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
                train_loss += total_loss.item() * batch_features.size(0)
                train_data_loss += data_loss.item() * batch_features.size(0)
                physics_loss_value = physics_loss_val.item() if hasattr(physics_loss_val, 'item') else physics_loss_val
                if physics_enabled and X_phys is not None:
                    train_physics_loss += physics_loss_value * stage_physics_weight * batch_features.size(0)
                train_mae += mae.item() * batch_features.size(0)
            
            # 更新性能监控器（如果启用）
            if enable_performance_monitor and performance_monitor is not None:
                avg_train_loss = train_loss / len(X_train)
                avg_train_mae = train_mae / len(X_train)
                performance_monitor.log_training_metrics(
                    epoch=epoch,
                    train_loss=avg_train_loss,
                    val_loss=0.0,  # 临时值，后续可替换为实际验证损失
                    train_mae=avg_train_mae,
                    data_loss=train_data_loss / len(X_train) if physics_enabled else avg_train_loss,
                    physics_loss=train_physics_loss / len(X_train) if physics_enabled else 0.0,
                    learning_rate=scheduler.get_last_lr()[0] if scheduler is not None else stage_config['学习率']
                )
            
            # 验证阶段
            model.eval()
            val_loss = 0.0
            val_data_loss = 0.0
            val_physics_loss = 0.0
            val_mae = 0.0
            num_val_batches = 0
            
            with torch.no_grad():
                for batch_features, batch_labels in val_loader:
                    batch_features = batch_features.to(device)
                    batch_labels = batch_labels.to(device)
                    
                    # 验证时也可以计算物理约束损失
                    if physics_enabled and X_phys is not None:
                        phys_val_indices = torch.randperm(len(X_phys))[:len(batch_features)]
                        X_phys_val_batch = X_phys[phys_val_indices].to(device)
                    
                    with torch.cuda.amp.autocast(enabled=scaler is not None):
                        predictions = model(batch_features)
                        
                        # 集成物理约束的损失计算
                        if physics_enabled and X_phys is not None:
                            # 计算数据损失
                            data_val_loss = loss_stabilizer.compute_loss(predictions, batch_labels)
                            
                            # 计算物理约束损失（在物理点上）
                            phys_val_outputs = model(X_phys_val_batch)
                            physics_val_loss, _ = pinn_layer.compute_physics_loss(
                                X_phys_val_batch, phys_val_outputs
                            )
                            
                            # 组合损失
                            total_val_loss = data_val_loss + stage_physics_weight * physics_val_loss
                        else:
                            # 仅数据损失
                            total_val_loss = loss_stabilizer.compute_loss(predictions, batch_labels)
                            data_val_loss = total_val_loss
                            physics_val_loss = torch.tensor(0.0, device=device)
                        
                        val_batch_mae = torch.mean(torch.abs(predictions - batch_labels))
                    
                    val_loss += total_val_loss.item() * len(batch_features)
                    if physics_enabled and X_phys is not None:
                          val_data_loss += data_val_loss.item() * len(batch_features)
                          # 处理physics_val_loss可能是float类型的情况
                          physics_val_loss_value = physics_val_loss.item() if hasattr(physics_val_loss, 'item') else physics_val_loss
                          val_physics_loss += physics_val_loss_value * stage_physics_weight * len(batch_features)
                    # 处理val_batch_mae可能是float类型的情况
                    val_batch_mae_value = val_batch_mae.item() if hasattr(val_batch_mae, 'item') else val_batch_mae
                    val_mae += val_batch_mae_value * len(batch_features)
                    num_val_batches += 1
            
            # 计算平均损失（考虑批次大小不同）
            avg_train_loss = train_loss / len(X_train)
            avg_train_mae = train_mae / len(X_train)
            avg_val_loss = val_loss / len(X_val)
            avg_val_mae = val_mae / len(X_val)
            
            # 计算平均物理损失（如果启用）
            avg_train_data_loss = 0
            avg_train_physics_loss = 0
            avg_val_data_loss = 0
            avg_val_physics_loss = 0
            if physics_enabled and X_phys is not None:
                avg_train_data_loss = train_data_loss / len(X_train)
                avg_train_physics_loss = train_physics_loss / len(X_train)
                avg_val_data_loss = val_data_loss / len(X_val)
                avg_val_physics_loss = val_physics_loss / len(X_val)
            
            stage_train_losses.append(avg_train_loss)
            stage_val_losses.append(avg_val_loss)
            
            # 更新性能监控器的验证指标（如果启用）
            if enable_performance_monitor and performance_monitor is not None:
                performance_monitor.log_training_metrics(
                    epoch=epoch,
                    train_loss=0.0,  # 临时值，主要记录验证指标
                    val_loss=avg_val_loss,
                    val_mae=avg_val_mae,
                    data_loss=avg_val_data_loss if physics_enabled else avg_val_loss,
                    physics_loss=avg_val_physics_loss if physics_enabled else 0.0
                )
                
                # 定期生成性能诊断报告和可视化
                if (epoch + 1) % 10 == 0 or epoch == stage_config['轮次'] - 1:
                    # 确保metrics_history中有足够的数据点且数组长度匹配
                    if len(performance_monitor.metrics_history.get('epoch', [])) > 0:
                        # 检查关键数组长度是否匹配
                        has_consistent_lengths = True
                        epoch_len = len(performance_monitor.metrics_history['epoch'])
                        for key in ['train_mae', 'val_mae']:
                            if key in performance_monitor.metrics_history and len(performance_monitor.metrics_history[key]) != epoch_len:
                                has_consistent_lengths = False
                                break
                        
                        if has_consistent_lengths:
                            performance_monitor.export_diagnostics()
                    performance_monitor.generate_performance_report()
            
            # 自适应超参数优化器更新
            if hyperoptimizer is not None and (epoch + 1) % 5 == 0:  # 每5个epoch调整一次超参数
                # 收集当前训练状态信息
                train_state = {
                    'epoch': epoch,
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'train_mae': avg_train_mae,
                    'val_mae': avg_val_mae,
                    'learning_rate': current_lr,
                    'physics_weight': stage_physics_weight,
                    'batch_size': stage_config['批次大小']
                }
                
                # 调用超参数优化器调整参数
                updated_params = hyperoptimizer.adjust_hyperparameters(
                    train_state=train_state,
                    model=model,
                    optimizer=optimizer,
                    stage_config=stage_config
                )
                
                # 更新阶段配置中的参数
                if 'learning_rate' in updated_params:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = updated_params['learning_rate']
                
                if 'batch_size' in updated_params:
                    stage_config['批次大小'] = updated_params['batch_size']
                
                if 'physics_weight' in updated_params:
                    stage_physics_weight = updated_params['physics_weight']
                
                if 'dropout_rate' in updated_params and hasattr(model, 'dropout'):
                    model.dropout.p = updated_params['dropout_rate']
            
            # 早停逻辑
            if avg_val_loss < best_val_loss - min_improvement:
                best_val_loss = avg_val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                no_improve_count = 0
                # 保存最佳模型
                best_checkpoint = {
                    'model_state_dict': best_model_state,
                    'normalizer': normalizer,
                    'train_history': train_history + stage_train_losses,
                    'val_history': val_history + stage_val_losses,
                    'best_val_loss': best_val_loss,
                    'best_epoch': len(train_history) + epoch,
                    'last_stage': stage_idx
                }
                torch.save(best_checkpoint, f"{save_dir}/best_model.pth")
            else:
                no_improve_count += 1
            
            # 学习率调度
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()
            
            # 计算批次处理时间
            epoch_time = time.time() - batch_start_time
            batches_per_sec = num_train_batches / epoch_time if epoch_time > 0 else 0
            
            # 打印进度
            current_lr = optimizer.param_groups[0]['lr']
            if epoch % 10 == 0 or epoch == stage_config['轮次'] - 1:
                if physics_enabled and X_phys is not None:
                    print(f"   Epoch {epoch:3d}/{stage_config['轮次']} | "
                          f"Train: {avg_train_loss:.6f} (数据: {avg_train_data_loss:.6f}, 物理: {avg_train_physics_loss:.6f}) | "
                          f"Val: {avg_val_loss:.6f} (数据: {avg_val_data_loss:.6f}, 物理: {avg_val_physics_loss:.6f}) | "
                          f"LR: {current_lr:.2e}, 物理权重: {stage_physics_weight:.3f} | "
                          f"速度: {batches_per_sec:.1f}批/秒")
                else:
                    print(f"   Epoch {epoch:3d}/{stage_config['轮次']} | "
                          f"Train: {avg_train_loss:.6f} (MAE: {avg_train_mae:.6f}) | "
                          f"Val: {avg_val_loss:.6f} (MAE: {avg_val_mae:.6f}) | "
                          f"LR: {current_lr:.2e} | "
                          f"速度: {batches_per_sec:.1f}批/秒")
            
            # 早停检查
            if early_stopping and no_improve_count >= patience:
                print(f"⚠️  早停触发: 验证损失 {patience} 轮未改善")
                break
        
        # 保存阶段结果
        stage_info = {
            'stage': stage_config['名称'],
            'train_losses': stage_train_losses,
            'val_losses': stage_val_losses,
            'final_train_loss': stage_train_losses[-1],
            'final_val_loss': stage_val_losses[-1],
            'duration': time.time() - stage_start_time
        }
        
        train_history.extend(stage_train_losses)
        val_history.extend(stage_val_losses)
        
        # 保存阶段检查点
        checkpoint = {
            'stage': stage_config['名称'],
            'epoch': stage_config['轮次'],
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': stage_train_losses[-1],
            'val_loss': stage_val_losses[-1],
            'normalizer': normalizer,
            'config': stage_config,
            'train_history': train_history,
            'val_history': val_history,
            'last_stage': stage_idx,
            '物理约束启用': physics_enabled,
            '物理约束权重': stage_physics_weight
        }
        torch.save(checkpoint, f"{save_dir}/stage_{stage_idx+1}_{stage_config['名称'].replace(' ', '_')}.pth")
        print(f"✅ 阶段 {stage_idx+1} 检查点已保存: stage_{stage_idx+1}_{stage_config['名称'].replace(' ', '_')}.pth")
        try:
            from scripts.generate_constraint_report import compute_constraint_stats
            import json, os
            rep = compute_constraint_stats(model, X_train, y_train, X_phys, device)
            out_dir = os.path.join(save_dir, 'consistency_data')
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, f'constraint_diagnostics_stage_{stage_idx+1}.json'), 'w', encoding='utf-8') as f:
                json.dump(rep, f, indent=2, ensure_ascii=False)
            try:
                from scripts.visualize_constraint_report import plot_residual_stats, plot_weight_series
                plot_residual_stats(rep, out_dir)
                plot_weight_series(rep, out_dir)
            except Exception:
                pass
        except Exception:
            pass
        
        print(f"✅ {stage_config['名称']} 完成: 训练损失={stage_train_losses[-1]:.6f}, 验证损失={stage_val_losses[-1]:.6f}")
        print(f"   阶段用时: {time.time() - stage_start_time:.2f}秒")
    
    # 恢复最佳模型（如果启用）
    if restore_best and best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"🔄 恢复最佳模型，验证损失: {best_val_loss:.6f}")
    
    # 最终测试
    print(f"\n🧪 最终测试...")
    model.eval()
    test_loss = 0.0
    test_data_loss = 0.0
    test_physics_loss = 0.0
    test_mae = 0.0
    
    with torch.no_grad():
        for batch_features, batch_labels in torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False):
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            
            # 测试时也可以计算物理约束损失
            if physics_enabled and X_phys is not None:
                phys_test_indices = torch.randperm(len(X_phys))[:len(batch_features)]
                X_phys_test_batch = X_phys[phys_test_indices].to(device)
            
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                predictions = model(batch_features)
                
                # 集成物理约束的损失计算
                if physics_enabled and X_phys is not None:
                    # 获取最后一个阶段的物理权重
                    last_stage_physics_weight = training_stages[-1].get('物理约束权重', 
                                                                      config['物理约束'].get('初始权重', 0.1))
                    
                    # 计算数据损失
                    data_test_loss = loss_stabilizer.compute_loss(predictions, batch_labels)
                    
                    # 计算物理约束损失（在物理点上）
                    phys_test_outputs = model(X_phys_test_batch)
                    physics_test_loss, _ = pinn_layer.compute_physics_loss(
                        X_phys_test_batch, phys_test_outputs
                    )
                    
                    # 组合损失
                    total_test_loss = data_test_loss + last_stage_physics_weight * physics_test_loss
                else:
                    # 仅数据损失
                    total_test_loss = loss_stabilizer.compute_loss(predictions, batch_labels)
                    data_test_loss = total_test_loss
                    physics_test_loss = torch.tensor(0.0, device=device)
                    
                test_batch_mae = torch.mean(torch.abs(predictions - batch_labels))
            
            test_loss += total_test_loss.item() * len(batch_features)
            if physics_enabled and X_phys is not None:
                    test_data_loss += data_test_loss.item() * len(batch_features)
                    last_stage_physics_weight = training_stages[-1].get('物理约束权重', 
                                                                       config['物理约束'].get('初始权重', 0.1))
                    # 检查physics_test_loss是否为张量，如果是则调用item()，否则直接使用
                    if hasattr(physics_test_loss, 'item'):
                        test_physics_loss += physics_test_loss.item() * last_stage_physics_weight * len(batch_features)
                    else:
                        test_physics_loss += physics_test_loss * last_stage_physics_weight * len(batch_features)
            test_mae += test_batch_mae.item() * len(batch_features)
    
    # 计算平均测试指标
    avg_test_loss = test_loss / len(X_test)
    avg_test_mae = test_mae / len(X_test)
    
    # 计算平均物理损失（如果启用）
    avg_test_data_loss = 0
    avg_test_physics_loss = 0
    if physics_enabled and X_phys is not None:
        avg_test_data_loss = test_data_loss / len(X_test)
        avg_test_physics_loss = test_physics_loss / len(X_test)
    
    print(f"📊 最终测试结果:")
    if physics_enabled and X_phys is not None:
        print(f"   测试损失: {avg_test_loss:.6f} (数据: {avg_test_data_loss:.6f}, 物理: {avg_test_physics_loss:.6f})")
    else:
        print(f"   测试损失: {avg_test_loss:.6f}")
    print(f"   测试MAE: {avg_test_mae:.6f}")
    
    # 保存最终模型
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'normalizer': normalizer,
        'train_history': train_history,
        'val_history': val_history,
        'best_val_loss': best_val_loss,
        'final_train_loss': train_history[-1],
        'final_val_loss': val_history[-1],
        'test_loss': avg_test_loss,
        'test_mae': avg_test_mae,
        'test_data_loss': avg_test_data_loss,
        'test_physics_loss': avg_test_physics_loss,
        'training_stages': training_stages,
        'config': config,
        '物理约束启用': physics_enabled,
        '物理约束权重系列': [stage.get('物理约束权重', config['物理约束'].get('初始权重', 0.1)) for stage in training_stages],
        '模型版本': model.model_info['version'],
        '模型初始化种子': model_init_seed,
        '训练完成时间': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # 添加超参数优化历史记录
    if 'hyperoptimizer' in locals() and hyperoptimizer is not None:
        final_checkpoint['hyperparameter_optimization_history'] = hyperoptimizer.get_optimization_history()
        final_checkpoint['best_hyperparameters'] = hyperoptimizer.get_best_hyperparameters()
    
    # 保存最终检查点
    final_checkpoint_path = os.path.join(save_dir, "final_optimized_model.pth")
    torch.save(final_checkpoint, final_checkpoint_path)
    print(f"✅ 最终模型检查点已保存: {final_checkpoint_path}")
    try:
        from scripts.generate_constraint_report import compute_constraint_stats
        import json, os
        rep = compute_constraint_stats(model, X_train, y_train, X_phys, device)
        out_dir = os.path.join(save_dir, 'consistency_data')
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f'constraint_diagnostics_final.json'), 'w', encoding='utf-8') as f:
            json.dump(rep, f, indent=2, ensure_ascii=False)
        try:
            from scripts.visualize_constraint_report import plot_residual_stats, plot_weight_series
            plot_residual_stats(rep, out_dir)
            plot_weight_series(rep, out_dir)
        except Exception:
            pass
    except Exception:
        pass
    
    # 生成并保存最终性能报告（如果启用了性能监控）
    if enable_performance_monitor and performance_monitor is not None:
        # 直接将测试指标添加到metrics_history字典中
        performance_monitor.metrics_history['test_loss'] = [avg_test_loss]
        performance_monitor.metrics_history['test_mae'] = [avg_test_mae]
        if physics_enabled:
            performance_monitor.metrics_history['test_data_loss'] = [avg_test_data_loss]
            performance_monitor.metrics_history['test_physics_loss'] = [avg_test_physics_loss]
        # 追加 CSV 指标日志
        try:
            import csv
            csv_path = os.path.join(save_dir, 'metrics_summary.csv')
            headers = ['metric', 'value']
            rows = [
                ['test_loss', avg_test_loss],
                ['test_mae', avg_test_mae],
                ['test_data_loss', avg_test_data_loss],
                ['test_physics_loss', avg_test_physics_loss]
            ]
            with open(csv_path, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(headers)
                w.writerows(rows)
        except Exception:
            pass
        # 使用实际存在的方法生成报告和诊断
        performance_monitor.generate_performance_report()
        performance_monitor.export_diagnostics()
        
        print(f"📊 模型性能监控报告已生成并保存")
    
    # 使用save_model函数保存模型
    metadata = {
        'training_stages': training_stages,
        'physics_enabled': physics_enabled,
        'best_val_loss': best_val_loss,
        'test_metrics': {
            'loss': avg_test_loss,
            'mae': avg_test_mae,
            'data_loss': avg_test_data_loss,
            'physics_loss': avg_test_physics_loss
        },
        'model_init_seed': model_init_seed,
        'training_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    try:
        unit_meta = validate_units(X_train.cpu().numpy())
        metadata['unit_checks'] = unit_meta.get('unit_checks', [])
    except Exception:
        metadata['unit_checks'] = []
    
    # 添加超参数优化信息到metadata
    if 'hyperoptimizer' in locals() and hyperoptimizer is not None:
        metadata['hyperparameter_optimization'] = {
            'enabled': True,
            'best_hyperparameters': hyperoptimizer.get_best_hyperparameters(),
            'optimization_rounds': len(hyperoptimizer.get_optimization_history())
        }
    else:
        metadata['hyperparameter_optimization'] = {'enabled': False}
    
    # 添加额外的训练信息
    metadata.update({
        '总轮次': sum(stage['轮次'] for stage in training_stages),
        '最佳验证损失': best_val_loss,
        '最终测试损失': avg_test_loss,
        '最终测试MAE': avg_test_mae,
        '时间戳': datetime.now().strftime("%Y%m%d_%H%M%S"),
        '模型版本': model.model_info['version'],
        '物理约束启用': physics_enabled,
        'PINN集成': physics_enabled,
        '训练阶段数': len(training_stages),
        '模型初始化种子': model_init_seed
    })
    
    # 使用save_model函数保存模型
    save_model(
        model=model,
        normalizer=normalizer,
        save_path=os.path.join(save_dir, "optimized_model_with_physics.pth"),
        config=config,
        metadata=metadata,
        export_onnx=config.get('导出ONNX', False),
        onnx_path=os.path.join(save_dir, "optimized_model_with_physics.onnx")
    )
    
    # 生成并保存训练曲线图
    plot_training_curves(train_history, val_history, save_dir)
    
    print(f"\n🏁 训练完成！")
    print(f"📁 所有结果保存在: {save_dir}")
    print(f"📊 最佳验证损失: {best_val_loss:.6f}")
    print(f"📈 物理约束集成: {'✅ 已启用' if physics_enabled else '❌ 未启用'}")
    if model_init_seed is not None:
        print(f"🎲 模型初始化种子: {model_init_seed} (适用于集成学习)")
    if physics_enabled:
        print(f"⚖️  物理约束权重系列: {[stage.get('物理约束权重', config['物理约束'].get('初始权重', 0.1)) for stage in training_stages]}")
    
    return model, normalizer, config
    torch.save(final_checkpoint, f"{save_dir}/final_optimized_model.pth")
    
    # 绘制训练曲线
    plot_training_curves(train_history, val_history, save_dir)
    
    # 计算总体训练统计信息
    total_epochs = len(train_history)
    total_duration = sum(stage_info.get('duration', 0) for stage_info in locals().get('stage_info', []))
    loss_improvement = (train_history[0] - train_history[-1]) / train_history[0] * 100 if train_history else 0
    
    print(f"\n🎉 优化训练完成!")
    print(f"💾 模型保存至: {save_dir}")
    print(f"📈 总训练轮次: {total_epochs}")
    print(f"⏱️  总训练时间: {total_duration:.2f}秒")
    print(f"📉 训练损失改善: {train_history[0]:.6f} → {train_history[-1]:.6f} ({loss_improvement:.1f}%)")
    print(f"📉 验证损失改善: {val_history[0]:.6f} → {val_history[-1]:.6f}")
    print(f"🔍 最佳验证损失: {best_val_loss:.6f}")
    
    return model, normalizer, final_checkpoint

def create_optimizer(model, stage_config):
    """
    创建优化器，支持多种优化器类型
    """
    optimizer_type = stage_config.get('优化器', 'AdamW').lower()
    lr = stage_config.get('学习率', 1e-4)
    weight_decay = stage_config.get('权重衰减', 1e-5)
    
    if optimizer_type == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'sgd':
        momentum = stage_config.get('动量', 0.9)
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_type == 'radam':
        # 尝试导入RAdam，如果不可用则回退到AdamW
        try:
            from torch_optimizer import RAdam
            return RAdam(model.parameters(), lr=lr, weight_decay=weight_decay)
        except ImportError:
            print("⚠️  RAdam不可用，使用AdamW代替")
            return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        # 默认使用AdamW
        print(f"⚠️  未知优化器类型: {optimizer_type}，使用AdamW代替")
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

def create_lr_scheduler(optimizer, stage_config):
    """
    创建学习率调度器，支持多种调度策略
    """
    scheduler_type = stage_config.get('调度策略', 'CosineAnnealing').lower()
    epochs = stage_config.get('轮次', 100)
    base_lr = stage_config.get('学习率', 1e-4)
    
    if scheduler_type == 'cosineannealing':
        T_max = stage_config.get('调度参数', {}).get('T_max', epochs)
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    elif scheduler_type == 'reducelronplateau':
        patience = stage_config.get('调度参数', {}).get('patience', 5)
        factor = stage_config.get('调度参数', {}).get('factor', 0.5)
        min_lr = stage_config.get('调度参数', {}).get('min_lr', base_lr * 0.01)
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=patience, factor=factor, min_lr=min_lr
        )
    elif scheduler_type == 'onecycle':
        max_lr = stage_config.get('调度参数', {}).get('max_lr', base_lr * 10)
        return optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_lr, total_steps=epochs
        )
    elif scheduler_type == 'lineardecay':
        # 线性衰减到基础学习率的10%
        final_lr = base_lr * 0.1
        return optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=final_lr/base_lr, total_iters=epochs
        )
    else:
        # 默认使用CosineAnnealing
        print(f"⚠️  未知调度策略: {scheduler_type}，使用CosineAnnealing代替")
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

def plot_training_curves(train_losses, val_losses, save_dir):
    """绘制训练曲线"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失', alpha=0.8)
    plt.plot(val_losses, label='验证损失', alpha=0.8)
    plt.xlabel('训练轮次')
    plt.ylabel('损失值')
    plt.title('训练过程 - 损失曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    # 只显示后50%的数据，更清楚地看到收敛过程
    mid_point = len(train_losses) // 2
    plt.plot(range(mid_point, len(train_losses)), train_losses[mid_point:], label='训练损失', alpha=0.8)
    plt.plot(range(mid_point, len(val_losses)), val_losses[mid_point:], label='验证损失', alpha=0.8)
    plt.xlabel('训练轮次')
    plt.ylabel('损失值')
    plt.title('训练过程 - 后期收敛')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 训练曲线已保存: {save_dir}/training_curves.png")

def quantize_model(model, quantized_path, calibration_dataset=None, dynamic_quantization=True):
    """
    模型量化功能，减小模型体积和加速推理
    
    参数:
    - model: 原始模型
    - quantized_path: 量化后保存路径
    - calibration_dataset: 校准数据集（动态量化不需要）
    - dynamic_quantization: 是否使用动态量化
    """
    try:
        print(f"⚡ 开始模型量化...")
        
        # 确保模型处于评估模式
        model.eval()
        
        if dynamic_quantization:
            # 动态量化（更简单，适用范围更广）
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear, nn.LSTM, nn.GRU},
                dtype=torch.qint8
            )
            print("✅ 动态量化完成")
        else:
            # 静态量化（需要校准，精度更高）
            if calibration_dataset is None:
                raise ValueError("静态量化需要提供校准数据集")
            
            # 准备量化配置
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            torch.quantization.prepare(model, inplace=True)
            
            # 校准
            print("🔍 正在进行模型校准...")
            with torch.no_grad():
                for data in calibration_dataset:
                    if isinstance(data, tuple):
                        model(data[0].to(model.device))
                    else:
                        model(data.to(model.device))
            
            # 执行量化
            quantized_model = torch.quantization.convert(model, inplace=False)
            print("✅ 静态量化完成")
        
        # 保存量化模型
        torch.save(quantized_model.state_dict(), quantized_path)
        
        # 计算模型大小减小比例
        original_size = os.path.getsize(quantized_path.replace('.quantized.pth', '.pth')) / (1024 * 1024)
        quantized_size = os.path.getsize(quantized_path) / (1024 * 1024)
        size_reduction = (1 - quantized_size / original_size) * 100
        
        print(f"✅ 量化模型已保存至: {quantized_path}")
        print(f"📉 模型大小: {original_size:.2f}MB → {quantized_size:.2f}MB (-{size_reduction:.1f}%)")
        
        return quantized_model
    except Exception as e:
        print(f"❌ 模型量化失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def compare_model_performance(model1, model2, test_data, test_labels, device='cpu'):
    """
    比较两个模型的性能（原始模型和量化模型）
    """
    try:
        print("⚖️  比较模型性能...")
        
        # 准备数据
        test_data = test_data.to(device)
        test_labels = test_labels.to(device)
        
        # 性能评估函数
        def evaluate_model(model):
            model.eval()
            with torch.no_grad():
                start_time = time.time()
                predictions = model(test_data)
                inference_time = time.time() - start_time
                
                # 计算损失和精度
                mse_loss = nn.MSELoss()(predictions, test_labels).item()
                mae_loss = nn.L1Loss()(predictions, test_labels).item()
                
                return {
                    'inference_time': inference_time,
                    'mse_loss': mse_loss,
                    'mae_loss': mae_loss
                }
        
        # 评估两个模型
        perf1 = evaluate_model(model1)
        perf2 = evaluate_model(model2)
        
        # 打印比较结果
        print("📊 模型性能比较:")
        print(f"  原始模型 - 推理时间: {perf1['inference_time']:.4f}秒, MSE: {perf1['mse_loss']:.6f}, MAE: {perf1['mae_loss']:.6f}")
        print(f"  优化模型 - 推理时间: {perf2['inference_time']:.4f}秒, MSE: {perf2['mse_loss']:.6f}, MAE: {perf2['mae_loss']:.6f}")
        
        # 计算改进比例
        speedup = perf1['inference_time'] / perf2['inference_time'] if perf2['inference_time'] > 0 else float('inf')
        mse_diff = (perf1['mse_loss'] - perf2['mse_loss']) / perf1['mse_loss'] * 100 if perf1['mse_loss'] > 0 else 0
        mae_diff = (perf1['mae_loss'] - perf2['mae_loss']) / perf1['mae_loss'] * 100 if perf1['mae_loss'] > 0 else 0
        
        print(f"📈 性能改进:")
        print(f"  推理速度提升: {speedup:.2f}x")
        print(f"  MSE变化: {mse_diff:+.1f}%")
        print(f"  MAE变化: {mae_diff:+.1f}%")
        
        return perf1, perf2
    except Exception as e:
        print(f"❌ 模型性能比较失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

def create_model_summary(model, input_size=(1, 62), device='cpu'):
    """
    创建模型结构摘要
    """
    try:
        # 准备输入
        input_tensor = torch.randn(*input_size).to(device)
        
        # 打印模型结构
        print("📋 模型结构摘要:")
        print(model)
        
        # 计算参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"📊 参数量统计:")
        print(f"  总参数量: {total_params:,}")
        print(f"  可训练参数量: {trainable_params:,}")
        
        # 前向传播以显示输出形状
        with torch.no_grad():
            output = model(input_tensor)
        
        print(f"📏 输入形状: {input_tensor.shape}")
        print(f"📏 输出形状: {output.shape}")
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'input_shape': input_tensor.shape,
            'output_shape': output.shape
        }
    except Exception as e:
        print(f"❌ 创建模型摘要失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# 增强的模型保存功能
def save_model(model, normalizer, save_path, config=None, metadata=None, export_onnx=False, onnx_path=None):
    """
    增强的模型保存功能
    
    参数:
    - model: 要保存的模型
    - normalizer: 数据标准化器
    - save_path: 保存路径
    - config: 训练配置
    - metadata: 额外元数据
    - export_onnx: 是否导出为ONNX格式
    - onnx_path: ONNX文件保存路径
    """
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 构建保存字典
        save_dict = {
            'model_state_dict': model.state_dict(),
            'model_info': getattr(model, 'model_info', {'version': 'unknown'}),
            'normalizer': normalizer,
            'save_time': datetime.now().isoformat(),
            'torch_version': torch.__version__
        }
        
        # 添加配置和元数据
        if config is not None:
            save_dict['config'] = config
        if metadata is not None:
            save_dict['metadata'] = metadata
        
        # 保存模型
        torch.save(save_dict, save_path)
        print(f"✅ 模型已保存至: {save_path}")
        
        # 导出为ONNX格式
        if export_onnx and HAS_ONNX:
            try:
                if onnx_path is None:
                    onnx_path = save_path.replace('.pth', '.onnx')
                
                # 准备示例输入
                dummy_input = torch.randn(1, model.model_info['input_dim']).to(model.device)
                
                # 导出为ONNX
                torch.onnx.export(
                    model,
                    dummy_input,
                    onnx_path,
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={'input': {0: 'batch_size'},
                                 'output': {0: 'batch_size'}},
                    verbose=False
                )
                
                # 验证ONNX模型
                onnx_model = onnx.load(onnx_path)
                onnx.checker.check_model(onnx_model)
                print(f"✅ 模型已导出为ONNX格式至: {onnx_path}")
                
            except Exception as e:
                print(f"⚠️  ONNX导出失败: {str(e)}")
        
        return True
    except Exception as e:
        print(f"❌ 保存模型失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

# 增强的模型加载功能
def load_model(load_path, device='cpu'):
    """
    增强的模型加载功能，包含版本检查和兼容性验证
    
    参数:
    - load_path: 加载路径
    - device: 设备
    
    返回:
    - model: 加载的模型
    - normalizer: 加载的标准化器
    - metadata: 元数据字典
    """
    try:
        print(f"🔄 正在加载模型: {load_path}")
        
        # 检查文件是否存在
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"模型文件不存在: {load_path}")
        
        # 加载模型
        checkpoint = torch.load(load_path, map_location=device)
        
        # 提取模型信息
        model_info = checkpoint.get('model_info', {})
        print(f"📋 模型信息: 版本={model_info.get('version', 'unknown')}, "
              f"架构={model_info.get('architecture', 'unknown')}")
        
        # 兼容性检查
        check_model_compatibility(model_info)
        
        # 重建模型
        model = OptimizedEWPINN(
            input_dim=model_info.get('input_dim', 62),
            output_dim=model_info.get('output_dim', 24),
            hidden_layers=model_info.get('hidden_layers', [128, 64, 32]),
            dropout_rate=model_info.get('dropout_rate', 0.1),
            activation=model_info.get('activation', 'ReLU'),
            batch_norm=model_info.get('batch_norm', True),
            device=device
        )
        
        # 加载模型权重
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # 提取标准化器
        normalizer = checkpoint.get('normalizer', None)
        if normalizer is None:
            print("⚠️  模型中未找到标准化器")
        
        # 提取额外信息
        metadata = {
            'save_time': checkpoint.get('save_time', 'unknown'),
            'torch_version': checkpoint.get('torch_version', 'unknown'),
            'config': checkpoint.get('config', None),
            'metadata': checkpoint.get('metadata', None)
        }
        
        print(f"✅ 模型加载成功")
        return model, normalizer, metadata
        
    except Exception as e:
        print(f"❌ 加载模型失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None

def check_model_compatibility(model_info):
    """
    检查模型兼容性
    """
    current_version = '1.0.0'
    model_version = model_info.get('version', 'unknown')
    
    # 简单版本兼容性检查
    if model_version != current_version:
        warnings.warn(
            f"模型版本不匹配: 加载的模型版本={model_version}, 当前版本={current_version}"
        )
    
    # 架构检查
    architecture = model_info.get('architecture', 'unknown')
    if architecture != 'EWPINN':
        raise ValueError(f"不支持的模型架构: {architecture}")
    
    print(f"✅ 模型兼容性检查通过")

def load_and_test_model(load_path, test_data=None, test_labels=None, device='cpu'):
    """
    加载并测试保存的模型
    
    参数:
    - load_path: 模型加载路径
    - test_data: 测试数据（可选）
    - test_labels: 测试标签（可选）
    - device: 设备
    """
    # 加载模型
    model, normalizer, metadata = load_model(load_path, device=device)
    
    if model is None:
        return None
    
    # 创建模型摘要
    create_model_summary(model, input_size=(1, model.model_info['input_dim']), device=device)
    
    # 如果提供了测试数据，则进行测试
    if test_data is not None and test_labels is not None:
        print("\n🧪 进行模型测试...")
        
        # 确保数据格式正确
        if isinstance(test_data, np.ndarray):
            test_data = torch.tensor(test_data, dtype=torch.float32).to(device)
        if isinstance(test_labels, np.ndarray):
            test_labels = torch.tensor(test_labels, dtype=torch.float32).to(device)
        
        # 如果有标准化器，使用它
        if normalizer is not None:
            test_data = normalizer.transform_features(test_data)
            test_labels = normalizer.transform_labels(test_labels)
        
        # 测试模型
        model.eval()
        with torch.no_grad():
            predictions = model(test_data)
            
            # 计算指标
            mse_loss = nn.MSELoss()(predictions, test_labels).item()
            mae_loss = nn.L1Loss()(predictions, test_labels).item()
            
            print(f"📊 测试结果:")
            print(f"  MSE损失: {mse_loss:.6f}")
            print(f"  MAE损失: {mae_loss:.6f}")
    
    return model, normalizer, metadata

def parse_arguments():
    """
    解析命令行参数
    """
    import argparse
    parser = argparse.ArgumentParser(description='EWPINN模型训练与测试脚本')
    
    # 模式选择
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'infer'],
                        help='运行模式: train(训练), test(测试), infer(推理)')
    
    # 配置文件
    parser.add_argument('--config', type=str, default='model_config.json',
                        help='配置文件路径')
    
    # 训练相关参数
    parser.add_argument('--resume', action='store_true',
                        help='从检查点恢复训练')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='恢复训练的检查点路径')
    parser.add_argument('--mixed-precision', action='store_true', default=True,
                        help='启用混合精度训练')
    parser.add_argument('--model-seed', type=int, default=None,
                        help='模型初始化种子，用于集成学习时创建不同初始权重的模型')
    parser.add_argument('--efficient-architecture', action='store_true', default=True,
                        help='使用高效EWPINN架构（包含残差连接和注意力机制）')
    parser.add_argument('--model-compression', type=float, default=1.0,
                        help='模型压缩因子，小于1.0将减少网络参数数量（默认为1.0，不压缩）'),
    
    # 测试/推理相关参数
    parser.add_argument('--model-path', type=str, default='models/best_model.pth',
                        help='测试/推理使用的模型路径')
    parser.add_argument('--export-onnx', action='store_true',
                        help='导出模型为ONNX格式')
    
    # 数据相关参数
    parser.add_argument('--num-samples', type=int, default=200,
                        help='生成的样本数量')
    parser.add_argument('--data-augmentation', action='store_true', default=True,
                        help='启用数据增强')
    
    # 输出相关参数
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='输出目录')
    parser.add_argument('--device', type=str, default=None,
                        help='运行设备 (cpu, cuda, cuda:0 等)')
    
    return parser.parse_args()

def run_test_mode(args):
    """
    运行测试模式
    """
    print(f"\n📊 进入测试模式")
    
    # 确定设备
    device = args.device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  使用设备: {device}")
    
    # 加载模型
    model, normalizer, metadata = load_model(args.model_path, device=device)
    if model is None:
        print(f"❌ 无法加载模型: {args.model_path}")
        return False
    
    # 生成测试数据
    print(f"🔄 生成测试数据 ({args.num_samples} 样本)")
    X_test, y_test = generate_realistic_data(
        model, 
        num_samples=args.num_samples,
        config_path=args.config,
        data_augmentation=args.data_augmentation
    )
    
    # 数据标准化
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
    
    if normalizer is not None:
        X_test = normalizer.transform_features(X_test)
        y_test = normalizer.transform_labels(y_test)
    
    # 测试模型
    print(f"🧪 测试模型性能...")
    test_results = load_and_test_model(
        args.model_path, 
        test_data=X_test,
        test_labels=y_test,
        device=device
    )
    
    # 导出ONNX（如果需要）
    if args.export_onnx and HAS_ONNX:
        onnx_path = args.model_path.replace('.pth', '.onnx')
        success = save_model(
            model, 
            normalizer, 
            args.model_path,
            export_onnx=True,
            onnx_path=onnx_path
        )
        if success:
            print(f"✅ 模型已导出为ONNX格式")
    
    return True

def run_infer_mode(args):
    """
    运行推理模式
    """
    print(f"\n🤖 进入推理模式")
    
    # 确定设备
    device = args.device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  使用设备: {device}")
    
    # 加载模型
    model, normalizer, metadata = load_model(args.model_path, device=device)
    if model is None:
        print(f"❌ 无法加载模型: {args.model_path}")
        return False
    
    print(f"🔍 模型信息:")
    for key, value in model.model_info.items():
        print(f"  - {key}: {value}")
    
    # 示例推理
    print(f"\n🧪 执行示例推理...")
    # 创建随机输入（在实际应用中应替换为真实输入）
    sample_input = torch.randn(1, model.input_dim).to(device)
    
    if normalizer is not None:
        sample_input = normalizer.transform_features(sample_input)
    
    model.eval()
    with torch.no_grad():
        prediction = model(sample_input)
    
    if normalizer is not None:
        prediction = normalizer.inverse_transform_labels(prediction)
    
    print(f"📊 推理结果:")
    print(f"  输入特征维度: {sample_input.shape}")
    print(f"  预测输出维度: {prediction.shape}")
    print(f"  预测示例值: {prediction[0, :5].cpu().numpy()}")  # 显示前5个输出
    
    return True

def main():
    """
    主函数 - 完整的训练与测试脚本入口
    """
    try:
        # 打印欢迎信息
        print("\n🚀 EWPINN 模型训练与测试系统")
        print("=========================================")
        print(f"🕒 启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📱 PyTorch 版本: {torch.__version__}")
        print(f"📊 CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU 名称: {torch.cuda.get_device_name(0)}")
        print(f"📦 ONNX 支持: {HAS_ONNX}")
        print("=========================================")
        
        # 解析命令行参数
        args = parse_arguments()
        print(f"\n⚙️  运行配置:")
        print(f"   模式: {args.mode}")
        print(f"   配置文件: {args.config}")
        print(f"   输出目录: {args.output_dir}")
        
        # 确保输出目录存在
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 根据模式执行不同功能
        if args.mode == 'train':
            print(f"\n📈 进入训练模式")
            print(f"   恢复训练: {args.resume}")
            if args.resume and args.checkpoint:
                print(f"   检查点路径: {args.checkpoint}")
            print(f"   混合精度训练: {args.mixed_precision}")
            print(f"   模型初始化种子: {args.model_seed if args.model_seed is not None else '随机'}")
            print(f"   使用高效架构: {args.efficient_architecture}")
            print(f"   模型压缩因子: {args.model_compression}")
            
            # 执行训练
            model, normalizer, metadata = progressive_training(
                config_path=args.config,
                resume_training=args.resume,
                resume_checkpoint=args.checkpoint,
                mixed_precision=args.mixed_precision,
                model_init_seed=args.model_seed,
                use_efficient_architecture=args.efficient_architecture,
                model_compression_factor=args.model_compression
            )
            
            print(f"\n✅ 训练成功完成！")
            
            # 保存最终模型
            final_model_path = os.path.join(args.output_dir, 'final_model.pth')
            save_model(
                model, 
                normalizer, 
                final_model_path,
                config=args.config,
                metadata={
                    'training_completed': datetime.now().isoformat(),
                    'samples_generated': args.num_samples,
                    'mixed_precision': args.mixed_precision
                },
                export_onnx=args.export_onnx
            )
            
        elif args.mode == 'test':
            run_test_mode(args)
            
        elif args.mode == 'infer':
            run_infer_mode(args)
            
        print(f"\n✅ 任务完成！")
        print(f"📁 所有输出已保存至: {args.output_dir}")
        
    except Exception as e:
        print(f"\n❌ 执行失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return -1
    
    return 0

if __name__ == '__main__':
    # 全局变量检查
    HAS_ONNX = False
    try:
        import onnx
        import onnxruntime
        HAS_ONNX = True
        print(f"✅ ONNX支持已启用")
    except ImportError:
        print(f"⚠️ ONNX支持未启用，将跳过ONNX导出功能")
    
    # 运行主函数
    exit_code = main()
    
    # 根据退出码决定是否显示成功信息
    if exit_code == 0:
        print(f"\n🎉 程序成功执行完成！")
        print(f"\n📚 使用说明:")
        print(f"  训练模型: python ewp_pinn_optimized_train.py --mode train --config your_config.json")
        print(f"  训练指定种子的模型: python ewp_pinn_optimized_train.py --mode train --model-seed 42")
        print(f"  使用高效架构: python ewp_pinn_optimized_train.py --mode train --efficient-architecture")
        print(f"  使用模型压缩: python ewp_pinn_optimized_train.py --mode train --model-compression 0.8")
        print(f"  测试模型: python ewp_pinn_optimized_train.py --mode test --model-path your_model.pth")
        print(f"  模型推理: python ewp_pinn_optimized_train.py --mode infer --model-path your_model.pth")
        print(f"  启用ONNX导出: python ewp_pinn_optimized_train.py --mode train --export-onnx")
    else:
        print(f"❌ 程序执行失败，退出码: {exit_code}")
