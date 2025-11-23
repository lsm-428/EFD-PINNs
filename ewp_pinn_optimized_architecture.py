import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json
from datetime import datetime
import numpy as np

class EfficientResidualLayer(nn.Module):
    """高效残差层 - 包含批归一化和激活函数的残差块"""
    def __init__(self, in_features, out_features, activation_fn=F.relu, dropout_rate=0.1):
        super(EfficientResidualLayer, self).__init__()
        
        self.linear1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.linear2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation_fn = activation_fn
        
        # 快捷连接 - 如果维度不匹配需要线性变换
        self.shortcut = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        # He初始化
        nn.init.kaiming_normal_(self.linear1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.linear2.weight, mode='fan_in', nonlinearity='relu')
        if hasattr(self.shortcut, 'weight'):
            nn.init.kaiming_normal_(self.shortcut.weight, mode='fan_in', nonlinearity='relu')
        
        # 偏置初始化为小值
        if self.linear1.bias is not None:
            nn.init.constant_(self.linear1.bias, 0.01)
        if self.linear2.bias is not None:
            nn.init.constant_(self.linear2.bias, 0.01)
        if hasattr(self.shortcut, 'bias') and self.shortcut.bias is not None:
            nn.init.constant_(self.shortcut.bias, 0.01)
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.linear1(x)
        out = self.bn1(out)
        out = self.activation_fn(out)
        out = self.dropout(out)
        
        out = self.linear2(out)
        out = self.bn2(out)
        
        # 残差连接
        out += residual
        out = self.activation_fn(out)
        
        return out

class AttentionMechanism(nn.Module):
    """简单的通道注意力机制，用于增强重要特征"""
    def __init__(self, feature_dim):
        super(AttentionMechanism, self).__init__()
        self.fc1 = nn.Linear(feature_dim, feature_dim // 4)
        self.fc2 = nn.Linear(feature_dim // 4, feature_dim)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # 全局平均池化 - 修复维度处理
        if x.dim() == 1:
            # 一维输入 [features]
            attention = x.mean().unsqueeze(0).unsqueeze(0)  # 转为 [1, 1]
        elif x.dim() == 2:
            # 二维输入 [batch, features]
            attention = x.mean(dim=0, keepdim=True)  # [1, features]
        else:
            # 多维输入，默认在最后一个维度平均
            attention = x.mean(dim=-1, keepdim=True)
        
        # 注意力门控 - 确保维度匹配
        attention = self.fc1(attention)
        attention = F.relu(attention)
        attention = self.fc2(attention)
        attention = self.sigmoid(attention)
        
        # 应用注意力权重 - 确保广播正确
        if x.dim() == 1 and attention.dim() == 2:
            attention = attention.squeeze(0)  # 调整为一维
        
        return x * attention

class EfficientEWPINN(nn.Module):
    """高效EWPINN模型 - 增强型神经网络架构，支持残差连接和注意力机制
    特性：残差网络、注意力机制、动态网络结构、梯度累积支持、量化友好设计"""
    
    def __init__(self, input_dim=62, output_dim=24, hidden_layers=None, dropout_rate=0.1, 
                 activation='ReLU', batch_norm=True, use_residual=True, use_attention=True,
                 config_path=None, device='cpu', compression_factor=1.0, gradient_checkpointing=False):
        super(EfficientEWPINN, self).__init__()
        
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.use_residual = use_residual
        self.use_attention = use_attention
        self.compression_factor = compression_factor  # 网络压缩因子
        self.gradient_checkpointing = gradient_checkpointing
        
        # 模型配置信息
        self.model_info = {
            'version': '2.0.0',  # 升级版本号
            'input_dim': input_dim,
            'output_dim': output_dim,
            'hidden_layers': hidden_layers if hidden_layers else [128, 64, 32],
            'dropout_rate': dropout_rate,
            'activation': activation,
            'batch_norm': batch_norm,
            'use_residual': use_residual,
            'use_attention': use_attention,
            'compression_factor': compression_factor,
            'architecture': 'EfficientEWPINN',
            'created_at': datetime.now().isoformat()
        }
        
        # 从配置文件加载参数（如果提供）
        if config_path and os.path.exists(config_path):
            self._load_config(config_path)
        
        # 默认隐藏层配置
        if hidden_layers is None:
            hidden_layers = [128, 64, 32]
        
        # 应用压缩因子
        if compression_factor != 1.0:
            hidden_layers = [max(8, int(dim * compression_factor)) for dim in hidden_layers]
        
        self.hidden_layers = hidden_layers
        self.model_info['hidden_layers'] = hidden_layers
        self.model_info['activation'] = activation
        
        # 选择激活函数
        activation_map = {
            'ReLU': F.relu,
            'LeakyReLU': F.leaky_relu,
            'GELU': F.gelu,
            'SiLU': F.silu
        }
        self.activation_fn = activation_map.get(activation, F.relu)
        
        # 构建网络
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        
        # 构建隐藏层
        for i, hidden_dim in enumerate(hidden_layers):
            if use_residual and i > 0:  # 第一个层通常不使用残差
                # 使用残差块
                layer = EfficientResidualLayer(prev_dim, hidden_dim, self.activation_fn, dropout_rate)
            else:
                # 标准层
                layer_components = [nn.Linear(prev_dim, hidden_dim)]
                if batch_norm:
                    layer_components.append(nn.BatchNorm1d(hidden_dim))
                layer_components.append(nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity())
                layer = nn.Sequential(*layer_components)
            
            self.layers.append(layer)
            prev_dim = hidden_dim
        
        # 注意力机制层（如果启用）
        self.attention = AttentionMechanism(prev_dim) if use_attention else nn.Identity()
        self.attention_mechanisms = self.attention
        
        # 输出层 - 使用较小的权重初始化，有助于稳定训练
        self.output_layer = nn.Linear(prev_dim, output_dim)
        
        # 初始化权重
        self._initialize_weights()
        
        # 如果启用梯度检查点，应用到相应的层
        if gradient_checkpointing:
            self._apply_gradient_checkpointing()
        
        # 将模型移动到指定设备
        self.to(self.device)

        self.residual_layers = [layer for layer in self.layers if isinstance(layer, EfficientResidualLayer)]
        
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
                    if '残差连接' in net_config:
                        self.use_residual = net_config['残差连接']
                    if '注意力机制' in net_config:
                        self.use_attention = net_config['注意力机制']
                    if '压缩因子' in net_config:
                        self.compression_factor = net_config['压缩因子']
                print(f"✅ 成功加载配置文件: {config_path}")
        except Exception as e:
            print(f"⚠️  加载配置文件失败: {str(e)}")
            print("   将使用默认配置")
    
    def _initialize_weights(self):
        """高级权重初始化方法"""
        # 输出层使用较小的权重初始化
        nn.init.normal_(self.output_layer.weight, mean=0.0, std=0.01)
        if self.output_layer.bias is not None:
            nn.init.constant_(self.output_layer.bias, 0.0)
    
    def _apply_gradient_checkpointing(self):
        """应用梯度检查点以减少内存使用"""
        for layer in self.layers:
            if hasattr(layer, 'gradient_checkpointing_enable'):
                layer.gradient_checkpointing_enable()
    
    def _print_model_info(self, activation):
        """打印模型架构信息"""
        print(f"🚀 高效EWPINN模型已初始化 - 设备: {self.device}")
        print(f"   输入维度: {self.input_dim}, 输出维度: {self.output_dim}")
        print(f"   激活函数: {activation}")
        print(f"   批量标准化: {'启用' if self.batch_norm else '禁用'}")
        print(f"   Dropout率: {self.dropout_rate}")
        print(f"   残差连接: {'启用' if self.use_residual else '禁用'}")
        print(f"   注意力机制: {'启用' if self.use_attention else '禁用'}")
        print(f"   网络压缩因子: {self.compression_factor}")
        
        # 打印网络结构
        structure_str = f"{self.input_dim}"
        for dim in self.hidden_layers:
            structure_str += f" -> {dim}"
        structure_str += f" -> {self.output_dim}"
        print(f"   网络结构: {structure_str}")
        
        # 计算参数数量
        param_count = sum(p.numel() for p in self.parameters())
        print(f"   参数数量: {param_count:,}")
        
        # 计算理论FLOPs（简化计算）
        flops = 0
        prev_dim = self.input_dim
        for dim in self.hidden_layers:
            flops += 2 * prev_dim * dim  # 乘加操作
            prev_dim = dim
        flops += 2 * prev_dim * self.output_dim
        print(f"   理论FLOPs: {flops:,}")
    
    def forward(self, x):
        """高效前向传播"""
        for layer in self.layers:
            # 对于残差层，直接调用
            if isinstance(layer, EfficientResidualLayer):
                x = layer(x)
            else:
                # 对于标准层，先经过层，再应用激活函数
                x = layer(x)
                x = self.activation_fn(x)
        
        # 应用注意力机制（如果启用）
        x = self.attention(x)
        
        # 输出层
        x = self.output_layer(x)
        
        return x
    
    def get_model_summary(self):
        """返回模型摘要信息"""
        summary = {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'hidden_layers': self.hidden_layers,
            'dropout_rate': self.dropout_rate,
            'batch_norm': self.batch_norm,
            'use_residual': self.use_residual,
            'use_attention': self.use_attention,
            'compression_factor': self.compression_factor,
            'parameter_count': sum(p.numel() for p in self.parameters()),
            'device': str(self.device),
            'architecture': 'EfficientEWPINN'
        }
        return summary
    
    def enable_quantization(self):
        """准备模型进行量化"""
        # 将模型设置为评估模式
        self.eval()
        
        # 确保批归一化层使用移动统计
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                m.track_running_stats = True
        
        print("✅ 模型已准备好进行量化")
    
    def prune_model(self, pruning_ratio=0.2):
        """对模型进行剪枝以减少参数数量"""
        if pruning_ratio <= 0:
            return
        
        # 对线性层应用L1范数剪枝
        parameters_to_prune = []
        for module in self.modules():
            if isinstance(module, nn.Linear) and module.out_features > 10:  # 避免对输出层和小层剪枝
                parameters_to_prune.append((module, 'weight'))
        
        if parameters_to_prune:
            # 应用全局剪枝
            from torch.nn.utils import prune
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=pruning_ratio,
            )
            
            # 使剪枝永久化
            for module, name in parameters_to_prune:
                prune.remove(module, name)
            
            print(f"✅ 模型剪枝完成，剪枝比例: {pruning_ratio*100:.1f}%")
            print(f"   剪枝后参数数量: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")

def create_optimized_model(config_path=None, device='cpu', **kwargs):
    """工厂函数：创建优化的EWPINN模型"""
    return EfficientEWPINN(config_path=config_path, device=device, **kwargs)

def get_model_optimization_suggestions(model):
    """分析模型并提供优化建议"""
    suggestions = []
    param_count = sum(p.numel() for p in model.parameters())
    
    # 基于参数数量的压缩建议
    if param_count > 100000:
        suggestions.append(f"参数数量较大 ({param_count:,})，建议使用压缩因子 0.75-0.5 减少计算量")
    
    # 检查是否使用了残差连接
    if hasattr(model, 'use_residual') and not model.use_residual:
        suggestions.append("建议启用残差连接以提高深度网络的训练稳定性")
    
    # 检查是否使用了注意力机制
    if hasattr(model, 'use_attention') and not model.use_attention and param_count > 50000:
        suggestions.append("对于较大的模型，建议启用注意力机制以提高特征利用率")
    
    return suggestions

def benchmark_model_performance(model, input_tensor, iterations=100, warmup=10):
    """基准测试模型性能"""
    model.eval()
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    
    # 预热
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)
    
    # 测量推理时间
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(iterations):
            _ = model(input_tensor)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.time()
    
    avg_time = (end_time - start_time) / iterations * 1000  # 毫秒
    throughput = iterations / (end_time - start_time)  # 样本/秒
    
    return {
        'average_inference_time_ms': avg_time,
        'throughput_samples_per_second': throughput,
        'device': str(device),
        'iterations': iterations
    }

# 导入必要的库
import time