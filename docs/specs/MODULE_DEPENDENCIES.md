# 模块依赖关系图

## 📦 核心模块

```
┌─────────────────────────────────────────────────────────────┐
│                    efd_pinns_train.py                       │
│                      (主训练脚本)                            │
│  - 数据生成                                                  │
│  - 训练循环                                                  │
│  - 检查点保存                                                │
└────────────┬────────────────────────────────────────────────┘
             │
             ├──────────────┬──────────────┬──────────────┐
             ↓              ↓              ↓              ↓
    ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐
    │ ewp_pinn_  │  │ ewp_pinn_  │  │ ewp_pinn_  │  │ experiment_│
    │ model.py   │  │ physics.py │  │ optimizer  │  │ management │
    │            │  │            │  │ .py        │  │            │
    │ EWPINN模型 │  │ 物理约束层  │  │ 优化器管理  │  │ 实验管理   │
    └────────────┘  └────────────┘  └────────────┘  └────────────┘
         │               │
         ↓               ↓
    ┌────────────┐  ┌────────────┐
    │ ewp_pinn_  │  │ ewp_pinn_  │
    │ input_     │  │ dynamic_   │
    │ layer.py   │  │ weight.py  │
    │            │  │            │
    │ 输入特征   │  │ 动态权重   │
    └────────────┘  └────────────┘
```

## 🔍 详细依赖树

```
efd_pinns_train.py
│
├─► ewp_pinn_model.py
│   ├─► ewp_pinn_input_layer.py
│   │   └─► 定义62维输入特征
│   │
│   └─► ewp_pinn_output_layer.py
│       └─► 定义24维输出解析
│
├─► ewp_pinn_physics.py
│   ├─► 计算Young-Lippmann残差
│   ├─► 计算Navier-Stokes残差
│   ├─► 计算体积守恒残差
│   └─► 计算界面稳定性残差
│
├─► ewp_pinn_optimizer.py
│   ├─► Adam/AdamW优化器
│   └─► 学习率调度器
│
├─► ewp_pinn_dynamic_weight.py
│   └─► 自适应物理权重调整
│
├─► ewp_pinn_training_tracker.py
│   └─► 训练过程记录
│
└─► experiment_management/
    ├─► ExperimentManager
    └─► ConfigVersionManager
```

## 🎯 数据流向

```
配置文件 (JSON)
    ↓
efd_pinns_train.py::main()
    ↓
generate_training_data()
    ├─► 生成62维输入 X
    │   └─► [x, y, z, t, V, ...]
    │
    └─► 计算24维输出 Y
        └─► [θ, u, v, w, p, ...]
    ↓
DataLoader (批处理)
    ↓
训练循环
    ├─► model(X_batch)
    │   └─► EWPINN.forward()
    │       ├─► encoding_layer
    │       ├─► branch1/2/3
    │       ├─► multihead_att
    │       ├─► fusion_layer
    │       └─► output_layer
    │           └─► predictions (24维)
    │
    ├─► 计算数据损失
    │   └─► MSE(predictions, Y_batch)
    │
    ├─► 计算物理损失
    │   └─► PINNConstraintLayer(X_physics, predictions)
    │       ├─► young_lippmann_residual
    │       ├─► navier_stokes_residual
    │       ├─► volume_conservation_residual
    │       └─► interface_stability_residual
    │
    ├─► 总损失
    │   └─► total_loss = data_loss + α × physics_loss
    │
    └─► 反向传播
        └─► optimizer.step()
    ↓
保存检查点
    └─► checkpoint_epoch_N.pth
```

## 🔄 分析流程

```
checkpoint.pth
    ↓
analyze_young_lippmann.py
    ├─► 加载模型
    │   └─► EWPINN.load_state_dict()
    │
    ├─► 生成测试数据
    │   ├─► V_test = [0, 1, ..., 80]
    │   └─► X_test = generate_features(V_test)
    │
    ├─► 模型预测
    │   └─► θ_pred = model(X_test)[:, 0]
    │
    ├─► 理论计算
    │   └─► θ_theory = Young_Lippmann(V_test)
    │
    ├─► 线性度分析
    │   ├─► cos(θ_pred) vs V²
    │   └─► 计算 R²
    │
    └─► 输出结果
        ├─► 图表: young_lippmann_analysis.png
        └─► 报告: YOUNG_LIPPMANN_ANALYSIS.md
```

## 📊 模型内部结构

```
EWPINN (ewp_pinn_model.py)
│
├─► encoding_layer (ResNet编码)
│   ├─► ResidualBlock(62 → 256)
│   ├─► ResidualBlock(256 → 256)
│   ├─► ResidualBlock(256 → 256)
│   ├─► ResidualBlock(256 → 256)
│   ├─► ResidualBlock(256 → 256)
│   ├─► ResidualBlock(256 → 128)
│   └─► ResidualBlock(128 → 128)
│
├─► 多分支处理
│   ├─► branch1: ResidualBlock(128 → 128)
│   ├─► branch2: ResidualBlock(128 → 128)
│   └─► branch3: ResidualBlock(128 → 128)
│       └─► 融合: branch1 + branch2 + branch3
│
├─► multihead_att (多头注意力)
│   ├─► query: Linear(128 → 128)
│   ├─► key: Linear(128 → 128)
│   ├─► value: Linear(128 → 128)
│   └─► attention = softmax(Q·K^T/√d) · V
│
├─► fusion_layer (特征融合)
│   ├─► ResidualBlock(128 → 128)
│   ├─► ResidualBlock(128 → 128)
│   ├─► ResidualBlock(128 → 128)
│   └─► ResidualBlock(128 → 128)
│
└─► 输出层
    ├─► output_layer: Linear(128 → 24)
    ├─► auxiliary_output_layer: Linear(128 → 16)
    ├─► volume_fraction_layer: Linear(128 → 1)
    ├─► interface_curvature_layer: Linear(128 → 1)
    └─► ink_potential_layer: Linear(128 → 1)
```

## 🧮 物理约束层结构

```
PINNConstraintLayer (ewp_pinn_physics.py)
│
├─► Young-Lippmann约束 (权重=5.0)
│   └─► residual = cos(θ_pred) - [cos(θ₀) + (εε₀V²)/(2γd)]
│
├─► Navier-Stokes约束 (权重=0.01)
│   ├─► 连续性: ∂u/∂x + ∂v/∂y + ∂w/∂z = 0
│   ├─► 动量u: ρ(∂u/∂t + u·∇u) = -∂p/∂x + μ∇²u
│   ├─► 动量v: ρ(∂v/∂t + v·∇v) = -∂p/∂y + μ∇²v
│   └─► 动量w: ρ(∂w/∂t + w·∇w) = -∂p/∂z + μ∇²w
│
├─► 体积守恒约束 (权重=1.0)
│   └─► residual = |V_total - V_initial|
│
├─► 界面稳定性约束 (权重=0.5)
│   └─► residual = |∇²κ|
│
└─► 其他约束 (权重=0.01-0.5)
    ├─► 接触线动力学
    ├─► 介电充电
    ├─► 热力学
    └─► ...

总物理损失 = Σ (权重ᵢ × 残差ᵢ²)
```

## 🎛️ 配置文件结构

```
config_stage1_physics_validation.json
│
├─► model (模型配置)
│   ├─► input_dim: 62
│   ├─► output_dim: 24
│   ├─► hidden_dims: [256, 256, 128, 64]
│   ├─► activation: "gelu"
│   ├─► use_batch_norm: true
│   └─► dropout: 0.05
│
├─► training (训练配置)
│   ├─► epochs: 200
│   ├─► batch_size: 32
│   ├─► optimizer: "adam"
│   ├─► learning_rate: 0.001
│   ├─► lr_scheduler: "step"
│   └─► gradient_clipping: 0.5
│
├─► physics (物理配置)
│   ├─► physics_weight: 0.1
│   ├─► adaptive_physics_weight: true
│   ├─► num_physics_points: 100
│   └─► residual_weights:
│       ├─► young_lippmann: 5.0
│       ├─► contact_angle_constraint: 1.0
│       ├─► volume_conservation: 1.0
│       └─► ...
│
└─► data (数据配置)
    ├─► num_samples: 500
    ├─► train_ratio: 0.6
    ├─► val_ratio: 0.2
    └─► normalization: "standard"
```

## 🔗 关键接口

### 1. 模型接口
```python
# 输入
X: torch.Tensor  # shape: (batch, 62)

# 输出
outputs: dict = {
    'main_predictions': torch.Tensor,      # (batch, 24)
    'auxiliary_predictions': torch.Tensor, # (batch, 16)
    'volume_fraction': torch.Tensor,       # (batch, 1)
    'interface_curvature': torch.Tensor,   # (batch, 1)
    'ink_potential': torch.Tensor,         # (batch, 1)
    'features': torch.Tensor               # (batch, 128)
}
```

### 2. 物理约束接口
```python
# 输入
x: torch.Tensor           # 物理点坐标 (batch, 62)
predictions: torch.Tensor # 模型预测 (batch, 24)

# 输出
physics_loss: torch.Tensor  # 标量
residuals: dict = {
    'young_lippmann': float,
    'continuity': float,
    'momentum_u': float,
    ...
}
```

### 3. 数据生成接口
```python
# 输入
num_samples: int
config: dict

# 输出
X_train: torch.Tensor  # (n_train, 62)
Y_train: torch.Tensor  # (n_train, 24)
X_val: torch.Tensor    # (n_val, 62)
Y_val: torch.Tensor    # (n_val, 24)
X_test: torch.Tensor   # (n_test, 62)
Y_test: torch.Tensor   # (n_test, 24)
physics_points: torch.Tensor  # (n_physics, 62)
normalizer: DataNormalizer
```

## 📝 使用示例

### 完整训练流程
```python
# 1. 加载配置
config = load_config('config_stage1_physics_validation.json')

# 2. 生成数据
X_train, Y_train, ... = generate_training_data(config, num_samples=500)

# 3. 创建模型
model = EWPINN(input_dim=62, output_dim=24)

# 4. 创建物理约束
physics = PINNConstraintLayer(config)

# 5. 训练循环
for epoch in range(200):
    # 前向传播
    predictions = model(X_batch)
    
    # 计算损失
    data_loss = MSE(predictions, Y_batch)
    physics_loss = physics(X_physics, predictions)
    total_loss = data_loss + 0.1 * physics_loss
    
    # 反向传播
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

# 6. 保存模型
torch.save(model.state_dict(), 'checkpoint.pth')
```

### 分析流程
```python
# 1. 加载模型
model = EWPINN(input_dim=62, output_dim=24)
model.load_state_dict(torch.load('checkpoint.pth'))

# 2. 生成测试数据
V_test = np.linspace(0, 80, 100)
X_test = generate_test_features(V_test)

# 3. 预测
predictions = model(X_test)
theta_pred = predictions[:, 0]

# 4. 分析
R2 = compute_linearity(theta_pred, V_test)
print(f'R² = {R2:.4f}')
```

---

**最后更新**: 2025-11-28  
**用途**: 理解项目模块间的依赖和数据流向
