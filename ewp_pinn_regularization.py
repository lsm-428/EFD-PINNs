import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import math
from typing import Dict, List, Optional, Union, Tuple


class AdvancedRegularizer:
    """
    高级正则化器，提供多种正则化技术以提升模型泛化能力
    
    主要功能：
    1. L1/L2正则化及弹性网络正则化
    2. Dropout优化和DropConnect
    3. 权重约束（如权重裁剪、谱归一化）
    4. 早停策略
    5. 混合正则化策略
    """
    
    def __init__(self, 
                 config_path: Optional[str] = None,
                 l1_lambda: float = 0.0,
                 l2_lambda: float = 0.001,
                 weight_decay: float = 0.0,
                 use_dropout: bool = True,
                 dropout_rate: float = 0.1,
                 use_weight_clipping: bool = False,
                 weight_clip_value: float = 1.0,
                 use_spectral_norm: bool = False,
                 use_batch_norm: bool = True,
                 enable_early_stopping: bool = True,
                 patience: int = 10,
                 min_improvement: float = 1e-5,
                 device: str = 'cpu',
                 l1_strength: Optional[float] = None,
                 spectral_norm: Optional[bool] = None):
        """
        初始化高级正则化器
        
        参数：
        - config_path: 配置文件路径，可从中加载正则化参数
        - l1_lambda: L1正则化系数
        - l2_lambda: L2正则化系数
        - weight_decay: 权重衰减系数
        - use_dropout: 是否使用Dropout
        - dropout_rate: Dropout概率
        - use_weight_clipping: 是否使用权重裁剪
        - weight_clip_value: 权重裁剪阈值
        - use_spectral_norm: 是否使用谱归一化
        - use_batch_norm: 是否使用批归一化
        - enable_early_stopping: 是否启用早停
        - patience: 早停耐心值
        - min_improvement: 最小改进阈值
        - device: 计算设备
        """
        self.device = device
        self.config = self._load_config(config_path) if config_path else {}
        
        # 从配置或参数中获取正则化设置
        self.l1_lambda = self.config.get('L1正则化系数', l1_lambda)
        self.l2_lambda = self.config.get('L2正则化系数', l2_lambda)
        self.weight_decay = self.config.get('权重衰减', weight_decay)
        self.use_dropout = self.config.get('使用Dropout', use_dropout)
        self.dropout_rate = self.config.get('Dropout率', dropout_rate)
        self.use_weight_clipping = self.config.get('使用权重裁剪', use_weight_clipping)
        self.weight_clip_value = self.config.get('权重裁剪阈值', weight_clip_value)
        self.use_spectral_norm = self.config.get('使用谱归一化', use_spectral_norm)
        self.use_batch_norm = self.config.get('使用批归一化', use_batch_norm)

        if l1_strength is not None:
            self.l1_lambda = l1_strength
        if spectral_norm is not None:
            self.use_spectral_norm = spectral_norm
        
        # 早停设置
        self.enable_early_stopping = self.config.get('启用早停', enable_early_stopping)
        self.patience = self.config.get('早停耐心值', patience)
        self.min_improvement = self.config.get('最小改进阈值', min_improvement)
        
        # 早停状态变量
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.should_stop = False
        
        # 保存正则化历史记录
        self.regularization_history = []
        
        # 初始化Dropout层
        self.dropout_layer = nn.Dropout(self.dropout_rate)
        
        if self.enable_early_stopping:
            print(f"✅ 早停机制已启用: 耐心值={self.patience}, 最小改进={self.min_improvement}")
        
        print(f"📊 正则化配置: L1={self.l1_lambda}, L2={self.l2_lambda}, Dropout={self.dropout_rate if self.use_dropout else '禁用'}")
    
    def _load_config(self, config_path: str) -> Dict:
        """
        从配置文件加载正则化参数
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                full_config = json.load(f)
                return full_config.get('正则化配置', {})
        except Exception as e:
            print(f"⚠️  加载正则化配置失败: {str(e)}")
            return {}
    
    def compute_regularization_loss(self, model: nn.Module) -> torch.Tensor:
        """
        计算模型的正则化损失
        
        参数：
        - model: PyTorch模型
        
        返回：
        - 正则化损失张量
        """
        regularization_loss = torch.tensor(0.0, device=self.device)
        
        # 计算L1正则化损失
        if self.l1_lambda > 0:
            l1_loss = sum(torch.norm(param, 1) for param in model.parameters() if param.requires_grad)
            regularization_loss += self.l1_lambda * l1_loss
        
        # 计算L2正则化损失
        if self.l2_lambda > 0:
            l2_loss = sum(torch.norm(param, 2) for param in model.parameters() if param.requires_grad)
            regularization_loss += self.l2_lambda * l2_loss
        
        # 记录正则化损失
        self.regularization_history.append({
            'l1_loss': (self.l1_lambda * sum(torch.norm(param, 1).item() for param in model.parameters() if param.requires_grad)) if self.l1_lambda > 0 else 0.0,
            'l2_loss': (self.l2_lambda * sum(torch.norm(param, 2).item() for param in model.parameters() if param.requires_grad)) if self.l2_lambda > 0 else 0.0,
            'total_reg_loss': regularization_loss.item()
        })
        
        return regularization_loss
    
    def apply_dropout(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        应用Dropout正则化
        
        参数：
        - x: 输入张量
        - training: 是否处于训练模式
        
        返回：
        - Dropout后的张量
        """
        if self.use_dropout and training:
            return self.dropout_layer(x)
        return x
    
    def apply_weight_clipping(self, model: nn.Module) -> None:
        """
        应用权重裁剪
        
        参数：
        - model: PyTorch模型
        """
        if self.use_weight_clipping:
            with torch.no_grad():
                for param in model.parameters():
                    if param.requires_grad:
                        param.clamp_(-self.weight_clip_value, self.weight_clip_value)
    
    def apply_spectral_normalization(self, model: nn.Module) -> nn.Module:
        """
        应用谱归一化到模型的线性层
        
        参数：
        - model: PyTorch模型
        
        返回：
        - 应用谱归一化后的模型
        """
        if self.use_spectral_norm:
            # 递归遍历模型的所有子模块
            for name, module in list(model.named_children()):
                if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                    # 应用谱归一化
                    setattr(model, name, nn.utils.spectral_norm(module))
                else:
                    # 递归应用到子模块
                    self.apply_spectral_normalization(module)
        return model
    
    def check_early_stopping(self, val_loss: float) -> bool:
        """
        检查是否应该早停
        
        参数：
        - val_loss: 当前验证损失
        
        返回：
        - 是否应该停止训练
        """
        if not self.enable_early_stopping:
            return False
        
        # 如果验证损失有足够的改进
        if val_loss < self.best_val_loss - self.min_improvement:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            self.should_stop = False
        else:
            # 没有足够改进，增加计数器
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                self.should_stop = True
                print(f"📉 触发早停: {self.patience_counter}轮未改进")
        
        return self.should_stop
    
    def reset_early_stopping(self) -> None:
        """
        重置早停状态
        """
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.should_stop = False
    
    def get_regularization_info(self) -> Dict:
        """
        获取正则化配置信息
        
        返回：
        - 正则化配置字典
        """
        return {
            'l1_lambda': self.l1_lambda,
            'l2_lambda': self.l2_lambda,
            'weight_decay': self.weight_decay,
            'use_dropout': self.use_dropout,
            'dropout_rate': self.dropout_rate,
            'use_weight_clipping': self.use_weight_clipping,
            'weight_clip_value': self.weight_clip_value,
            'use_spectral_norm': self.use_spectral_norm,
            'use_batch_norm': self.use_batch_norm,
            'enable_early_stopping': self.enable_early_stopping,
            'patience': self.patience,
            'min_improvement': self.min_improvement
        }
    
    def get_regularization_history(self) -> List[Dict]:
        """
        获取正则化历史记录
        
        返回：
        - 正则化历史记录列表
        """
        return self.regularization_history


class DropConnectLayer(nn.Module):
    """
    DropConnect层实现 - 对网络权重进行随机失活而非激活值
    """
    
    def __init__(self, module: nn.Module, drop_rate: float = 0.1, active: bool = True):
        """
        初始化DropConnect层
        
        参数：
        - module: 要应用DropConnect的模块（通常是nn.Linear）
        - drop_rate: DropConnect概率
        - active: 是否激活DropConnect
        """
        super(DropConnectLayer, self).__init__()
        self.module = module
        self.drop_rate = drop_rate
        self.active = active
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数：
        - x: 输入张量
        
        返回：
        - 输出张量
        """
        if not self.training or not self.active or self.drop_rate == 0:
            return self.module(x)
        
        # 生成掩码，保留概率为1-drop_rate
        with torch.no_grad():
            # 为权重创建掩码
            mask = torch.bernoulli(
                torch.ones_like(self.module.weight) * (1 - self.drop_rate)
            ) / (1 - self.drop_rate)  # 缩放以保持期望输出不变
            
            # 对权重应用掩码
            masked_weight = self.module.weight * mask
            
            # 如果存在偏置，也创建掩码并应用
            if self.module.bias is not None:
                bias_mask = torch.bernoulli(
                    torch.ones_like(self.module.bias) * (1 - self.drop_rate)
                ) / (1 - self.drop_rate)
                masked_bias = self.module.bias * bias_mask
                return F.linear(x, masked_weight, masked_bias)
            else:
                return F.linear(x, masked_weight)


class GradientNoiseRegularizer:
    """
    梯度噪声正则化器 - 在梯度中添加噪声以提高泛化能力
    """
    
    def __init__(self, 
                 eta: float = 0.01, 
                 gamma: float = 0.55, 
                 enabled: bool = True,
                 noise_stddev: Optional[float] = None,
                 noise_decay: Optional[float] = None,
                 noise_annealing: Optional[bool] = None,
                 device: Optional[str] = None):
        """
        初始化梯度噪声正则化器
        
        参数：
        - eta: 噪声强度参数
        - gamma: 噪声衰减率参数
        - enabled: 是否启用
        """
        self.eta = eta if noise_stddev is None else noise_stddev
        self.gamma = gamma if noise_decay is None else noise_decay
        self.enabled = enabled if noise_annealing is None else noise_annealing
        self.device = device
        self.iteration = 0
    
    def add_gradient_noise(self, parameters: List[torch.Tensor]) -> None:
        """
        向参数梯度添加噪声
        
        参数：
        - parameters: 要添加噪声的参数列表
        """
        if not self.enabled or self.eta == 0:
            return
        
        # 更新迭代计数
        self.iteration += 1
        
        # 计算噪声标准差
        sigma = self.eta / ((1 + self.iteration) ** self.gamma)
        
        # 向每个参数的梯度添加噪声
        for param in parameters:
            if param.grad is not None:
                noise = torch.randn_like(param.grad) * sigma
                param.grad.add_(noise)
    
    def reset(self) -> None:
        """
        重置迭代计数
        """
        self.iteration = 0


class EnsembleRegularization:
    """
    集成正则化 - 通过模型集成提升泛化能力
    """
    
    def __init__(self, num_models: int = 3, device: str = 'cpu'):
        """
        初始化集成正则化器
        
        参数：
        - num_models: 集成模型数量
        - device: 计算设备
        """
        self.num_models = num_models
        self.device = device
        self.models = []
    
    def create_ensemble(self, model_class: type, **model_kwargs) -> List[nn.Module]:
        """
        创建模型集成
        
        参数：
        - model_class: 模型类
        - model_kwargs: 模型初始化参数
        
        返回：
        - 模型列表
        """
        self.models = []
        for i in range(self.num_models):
            # 为每个模型设置不同的随机种子以增加多样性
            torch.manual_seed(torch.initial_seed() + i)
            model = model_class(**model_kwargs).to(self.device)
            self.models.append(model)
        
        return self.models
    
    def ensemble_predict(self, x: torch.Tensor, aggregation: str = 'mean') -> torch.Tensor:
        """
        集成预测
        
        参数：
        - x: 输入数据
        - aggregation: 聚合方法 ('mean', 'median', 'vote')
        
        返回：
        - 集成预测结果
        """
        predictions = []
        
        # 获取每个模型的预测
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)
        
        # 聚合预测
        predictions_tensor = torch.stack(predictions)
        
        if aggregation == 'mean':
            return torch.mean(predictions_tensor, dim=0)
        elif aggregation == 'median':
            return torch.median(predictions_tensor, dim=0)[0]
        elif aggregation == 'vote':
            # 对于分类任务的投票机制（需要根据具体任务调整）
            return torch.mode(predictions_tensor, dim=0)[0]
        else:
            raise ValueError(f"不支持的聚合方法: {aggregation}")
    
    def save_ensemble(self, save_dir: str) -> None:
        """
        保存集成模型
        
        参数：
        - save_dir: 保存目录
        """
        import os
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        for i, model in enumerate(self.models):
            save_path = os.path.join(save_dir, f"ensemble_model_{i}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"✅ 集成模型 {i} 已保存至: {save_path}")


class VariationalDropout(nn.Module):
    """
    变分Dropout实现 - 对同一特征的所有时间步使用相同的dropout掩码
    适用于RNN/LSTM等循环神经网络和具有层次结构的神经网络
    """
    
    def __init__(self, drop_rate: float = 0.1, batch_first: bool = True):
        """
        初始化变分Dropout
        
        参数：
        - drop_rate: Dropout概率
        - batch_first: 输入是否为batch_first格式
        """
        super(VariationalDropout, self).__init__()
        self.drop_rate = drop_rate
        self.batch_first = batch_first
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数：
        - x: 输入张量，形状为 (batch_size, seq_len, features) 或 (seq_len, batch_size, features)
        
        返回：
        - Dropout后的张量
        """
        if not self.training or self.drop_rate == 0:
            return x
        
        # 确定输入维度顺序
        if self.batch_first:
            batch_size, seq_len, features = x.size()
            # 创建形状为 (batch_size, 1, features) 的掩码，对每个样本的所有序列使用相同的掩码
            mask = torch.bernoulli(
                torch.ones(batch_size, 1, features, device=x.device) * (1 - self.drop_rate)
            ) / (1 - self.drop_rate)
        else:
            seq_len, batch_size, features = x.size()
            # 创建形状为 (1, batch_size, features) 的掩码
            mask = torch.bernoulli(
                torch.ones(1, batch_size, features, device=x.device) * (1 - self.drop_rate)
            ) / (1 - self.drop_rate)
        
        # 应用掩码
        return x * mask


def apply_regularization_to_model(model: nn.Module, 
                                 regularizer: AdvancedRegularizer,
                                 apply_dropconnect: bool = False,
                                 dropconnect_rate: float = 0.2) -> nn.Module:
    """
    将正则化技术应用到模型中
    
    参数：
    - model: 要应用正则化的模型
    - regularizer: 正则化器实例
    - apply_dropconnect: 是否应用DropConnect
    - dropconnect_rate: DropConnect概率
    
    返回：
    - 应用正则化后的模型
    """
    # 应用谱归一化（如果启用）
    if regularizer.use_spectral_norm:
        model = regularizer.apply_spectral_normalization(model)
    
    # 应用DropConnect（如果启用）
    if apply_dropconnect:
        # 递归遍历模型并替换线性层为DropConnect层
        for name, module in list(model.named_children()):
            if isinstance(module, nn.Linear) and name != 'output_layer':  # 保留输出层不变
                setattr(model, name, DropConnectLayer(module, drop_rate=dropconnect_rate))
            else:
                # 递归应用到子模块
                apply_regularization_to_model(module, regularizer, apply_dropconnect, dropconnect_rate)
    
    return model


def compute_model_complexity(model: nn.Module, input_size: Tuple[int, ...] = (1, 62)) -> Dict[str, float]:
    """
    计算模型复杂度指标
    
    参数：
    - model: PyTorch模型
    - input_size: 输入张量大小
    
    返回：
    - 模型复杂度指标字典
    """
    import torch.nn.utils.prune as prune
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 计算FLOPs（浮点运算次数）
    # 使用钩子来跟踪计算
    class FLOPCounter:
        def __init__(self):
            self.flops = 0
        
        def hook_fn(self, module, input, output):
            # 只计算Conv和Linear层
            if isinstance(module, nn.Conv2d):
                # 对于卷积层: FLOPs = (input_channels * kernel_h * kernel_w * output_channels * out_h * out_w) / groups
                batch_size = input[0].size(0)
                out_h = output.size(2)
                out_w = output.size(3)
                kernel_h, kernel_w = module.kernel_size
                in_channels = module.in_channels
                out_channels = module.out_channels
                groups = module.groups
                
                flops = batch_size * out_h * out_w * in_channels * out_channels * kernel_h * kernel_w / groups
                self.flops += flops
                
            elif isinstance(module, nn.Linear):
                # 对于线性层: FLOPs = 2 * batch_size * in_features * out_features
                batch_size = input[0].size(0)
                flops = 2 * batch_size * module.in_features * module.out_features
                self.flops += flops
    
    counter = FLOPCounter()
    hooks = []
    
    # 注册钩子
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(counter.hook_fn))
    
    # 前向传播以计算FLOPs
    device = next(model.parameters()).device
    input_tensor = torch.randn(input_size, device=device)
    with torch.no_grad():
        model(input_tensor)
    
    # 移除钩子
    for hook in hooks:
        hook.remove()
    
    return {
        'total_params': total_params,
        'total_params_million': total_params / 1e6,
        'flops': counter.flops,
        'flops_million': counter.flops / 1e6,
        'flops_billion': counter.flops / 1e9
    }