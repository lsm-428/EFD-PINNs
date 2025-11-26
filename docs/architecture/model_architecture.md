# 模型架构详解

## 整体架构概览

EFD3D采用分层模块化设计，核心架构包含以下组件：

```python
class OptimizedEWPINN(nn.Module):
    """优化的电润湿PINN模型架构"""
    
    def __init__(self, input_dim=62, output_dim=24, hidden_layers=[256, 512, 256]):
        super().__init__()
        
        # 输入预处理层
        self.input_layer = EnhancedInputLayer(input_dim, hidden_layers[0])
        
        # 核心特征提取层
        self.feature_layers = nn.ModuleList([
            ResidualBlock(hidden_layers[i], hidden_layers[i+1]) 
            for i in range(len(hidden_layers)-1)
        ])
        
        # 注意力机制层
        self.attention_layer = MultiHeadAttention(hidden_layers[-1])
        
        # 物理约束层
        self.physics_layer = PINNConstraintLayer(hidden_layers[-1])
        
        # 输出层
        self.output_layer = AdaptiveOutputLayer(hidden_layers[-1], output_dim)
        
        # 辅助输出层（用于诊断和监控）
        self.auxiliary_outputs = AuxiliaryOutputLayer(hidden_layers[-1])
    
    def forward(self, x):
        """前向传播流程"""
        # 输入预处理
        x = self.input_layer(x)
        
        # 特征提取
        for layer in self.feature_layers:
            x = layer(x)
        
        # 注意力机制
        x_attended = self.attention_layer(x)
        
        # 物理约束计算
        physics_outputs = self.physics_layer(x_attended)
        
        # 主输出
        main_output = self.output_layer(x_attended)
        
        # 辅助输出
        auxiliary_outputs = self.auxiliary_outputs(x_attended)
        
        return {
            'main_predictions': main_output,
            'physics_constraints': physics_outputs,
            'auxiliary_outputs': auxiliary_outputs,
            'attention_weights': x_attended.attention_weights
        }
```

## 核心模块详解

### 1. 增强输入层（EnhancedInputLayer）
**功能**：输入数据预处理和特征工程

```python
class EnhancedInputLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        # 基础线性变换
        self.linear = nn.Linear(input_dim, output_dim)
        
        # 特征归一化
        self.batch_norm = nn.BatchNorm1d(output_dim)
        
        # 激活函数
        self.activation = nn.SiLU()  # Swish激活函数
        
        # 特征交互层
        self.feature_interaction = FeatureInteractionLayer(input_dim)
    
    def forward(self, x):
        # 特征交互
        x_interacted = self.feature_interaction(x)
        
        # 线性变换
        x = self.linear(x_interacted)
        
        # 归一化
        x = self.batch_norm(x)
        
        # 激活
        x = self.activation(x)
        
        return x

class FeatureInteractionLayer(nn.Module):
    """特征交互层：捕捉输入特征间的非线性关系"""
    
    def __init__(self, input_dim):
        super().__init__()
        # 实现特征交叉和交互
        self.cross_terms = nn.Linear(input_dim * 2, input_dim)
    
    def forward(self, x):
        # 生成特征交叉项
        batch_size = x.size(0)
        cross_features = []
        
        for i in range(x.size(1)):
            for j in range(i+1, x.size(1)):
                cross_term = x[:, i] * x[:, j]
                cross_features.append(cross_term.unsqueeze(1))
        
        if cross_features:
            cross_tensor = torch.cat(cross_features, dim=1)
            # 选择重要的交叉特征
            selected_cross = self.select_important_cross_terms(cross_tensor)
            x = torch.cat([x, selected_cross], dim=1)
        
        return x
```

### 2. 残差块（ResidualBlock）
**功能**：深层特征提取，防止梯度消失

```python
class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.1):
        super().__init__()
        
        # 主路径
        self.linear1 = nn.Linear(input_dim, output_dim)
        self.bn1 = nn.BatchNorm1d(output_dim)
        self.activation1 = nn.SiLU()
        
        self.linear2 = nn.Linear(output_dim, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
        self.activation2 = nn.SiLU()
        
        # 残差连接
        if input_dim != output_dim:
            self.residual_connection = nn.Linear(input_dim, output_dim)
        else:
            self.residual_connection = nn.Identity()
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        residual = self.residual_connection(x)
        
        # 主路径
        out = self.linear1(x)
        out = self.bn1(out)
        out = self.activation1(out)
        out = self.dropout(out)
        
        out = self.linear2(out)
        out = self.bn2(out)
        
        # 残差连接
        out += residual
        out = self.activation2(out)
        
        return out
```

### 3. 多头注意力机制（MultiHeadAttention）
**功能**：捕捉特征间的长程依赖关系

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim必须能被num_heads整除"
        
        # 查询、键、值投影
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # 输出投影
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # 缩放因子
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()
        
        # 投影到查询、键、值
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # 转置以便矩阵乘法
        q = q.transpose(1, 2)  # [batch, heads, seq_len, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # 应用注意力权重
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        # 输出投影
        output = self.out_proj(attn_output)
        
        # 保存注意力权重用于可视化
        self.attention_weights = attn_weights.detach()
        
        return output
```

### 4. 物理约束层（PINNConstraintLayer）
**功能**：计算物理约束残差

```python
class PINNConstraintLayer(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        
        # 物理约束计算网络
        self.constraint_networks = nn.ModuleDict({
            'navier_stokes': NavierStokesConstraint(feature_dim),
            'young_lippmann': YoungLippmannConstraint(feature_dim),
            'mass_conservation': MassConservationConstraint(feature_dim),
            'energy_conservation': EnergyConservationConstraint(feature_dim),
            'charge_conservation': ChargeConservationConstraint(feature_dim)
        })
        
        # 约束权重学习
        self.constraint_weights = nn.Parameter(torch.ones(len(self.constraint_networks)))
    
    def forward(self, features):
        constraints = {}
        
        for name, constraint_net in self.constraint_networks.items():
            constraint_value = constraint_net(features)
            constraints[name] = constraint_value
        
        # 应用学习到的权重
        weighted_constraints = {}
        for i, (name, constraint) in enumerate(constraints.items()):
            weighted_constraints[name] = constraint * self.constraint_weights[i]
        
        return weighted_constraints
```

### 5. 自适应输出层（AdaptiveOutputLayer）
**功能**：多尺度输出预测

```python
class AdaptiveOutputLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        
        # 多尺度输出
        self.output_scales = [1, 2, 4]  # 不同尺度的输出
        
        self.scale_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim // scale),
                nn.BatchNorm1d(input_dim // scale),
                nn.SiLU(),
                nn.Linear(input_dim // scale, output_dim)
            ) for scale in self.output_scales
        ])
        
        # 输出融合
        self.fusion_layer = nn.Linear(len(self.output_scales) * output_dim, output_dim)
    
    def forward(self, x):
        scale_outputs = []
        
        for layer in self.scale_layers:
            output = layer(x)
            scale_outputs.append(output)
        
        # 融合多尺度输出
        fused_output = torch.cat(scale_outputs, dim=1)
        final_output = self.fusion_layer(fused_output)
        
        return final_output
```

## 架构优化技术

### 1. 梯度检查点（Gradient Checkpointing）
```python
def forward_with_checkpointing(self, x):
    """使用梯度检查点减少内存使用"""
    
    # 将网络分段，设置检查点
    segments = [
        (self.input_layer, self.feature_layers[:2]),
        (self.feature_layers[2:4],),
        (self.feature_layers[4:], self.attention_layer),
        (self.physics_layer, self.output_layer)
    ]
    
    # 使用检查点的前向传播
    for i, segment in enumerate(segments):
        if i < len(segments) - 1:
            x = checkpoint.checkpoint(self._forward_segment, x, segment)
        else:
            x = self._forward_segment(x, segment)
    
    return x

def _forward_segment(self, x, layers):
    """单个网络段的前向传播"""
    for layer in layers:
        if isinstance(layer, (list, tuple)):
            for sublayer in layer:
                x = sublayer(x)
        else:
            x = layer(x)
    return x
```

### 2. 模型量化（Model Quantization）
```python
class QuantizedEWPINN(nn.Module):
    """量化版本的模型，减少内存占用和推理时间"""
    
    def __init__(self, original_model):
        super().__init__()
        
        # 准备模型量化
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        
        # 量化模型组件
        self.quantized_layers = self._quantize_layers(original_model)
    
    def _quantize_layers(self, model):
        """量化模型层"""
        quantized_model = torch.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8
        )
        return quantized_model
    
    def forward(self, x):
        x = self.quant(x)  # 量化输入
        x = self.quantized_layers(x)
        x = self.dequant(x)  # 反量化输出
        return x
```

### 3. 知识蒸馏（Knowledge Distillation）
```python
class DistilledEWPINN(nn.Module):
    """通过知识蒸馏得到的轻量级模型"""
    
    def __init__(self, teacher_model, student_config):
        super().__init__()
        
        self.teacher_model = teacher_model
        self.student_model = self._build_student_model(student_config)
        
        # 蒸馏温度
        self.temperature = 3.0
    
    def forward(self, x):
        # 教师模型预测（用于蒸馏）
        with torch.no_grad():
            teacher_outputs = self.teacher_model(x)
        
        # 学生模型预测
        student_outputs = self.student_model(x)
        
        return {
            'student_predictions': student_outputs,
            'teacher_predictions': teacher_outputs
        }
    
    def compute_distillation_loss(self, student_outputs, teacher_outputs, targets):
        """计算知识蒸馏损失"""
        # 学生损失
        student_loss = F.mse_loss(student_outputs, targets)
        
        # 蒸馏损失（KL散度）
        distillation_loss = F.kl_div(
            F.log_softmax(student_outputs / self.temperature, dim=1),
            F.softmax(teacher_outputs / self.temperature, dim=1),
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # 组合损失
        alpha = 0.7  # 蒸馏损失权重
        total_loss = alpha * distillation_loss + (1 - alpha) * student_loss
        
        return total_loss
```

## 架构选择指南

### 根据任务复杂度选择架构

| 任务复杂度 | 推荐架构 | 隐藏层配置 | 特殊组件 |
|-----------|----------|------------|----------|
| 简单任务 | 基础PINN | [128, 256, 128] | 基础约束层 |
| 中等任务 | 标准PINN | [256, 512, 256] | 残差块+注意力 |
| 复杂任务 | 增强PINN | [512, 1024, 512, 256] | 全组件+蒸馏 |

### 性能与资源权衡

```python
def select_architecture_based_on_resources(available_memory, task_complexity):
    """根据可用资源和任务复杂度选择架构"""
    
    if available_memory < 4:  # < 4GB
        return {
            'hidden_layers': [64, 128, 64],
            'use_attention': False,
            'use_residual': True,
            'quantization': True
        }
    elif available_memory < 8:  # < 8GB
        return {
            'hidden_layers': [128, 256, 128],
            'use_attention': True,
            'use_residual': True,
            'quantization': False
        }
    else:  # >= 8GB
        return {
            'hidden_layers': [256, 512, 256],
            'use_attention': True,
            'use_residual': True,
            'quantization': False
        }
```

这个详细的模型架构文档为开发者提供了完整的架构设计、优化技术和选择指南。