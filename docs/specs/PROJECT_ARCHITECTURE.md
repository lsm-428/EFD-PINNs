# EFD-PINNs 项目架构与逻辑关系

## 🎯 项目目标
使用物理信息神经网络(PINN)模拟电润湿显示(Electrowetting Display)的多物理场行为

## 📊 核心流程图

```
配置文件 → 数据生成 → 模型训练 → 结果分析
   ↓          ↓          ↓          ↓
config.json  62维输入   EWPINN    Young-Lippmann
             24维输出   模型      R²分析
```

---

## 🏗️ 架构层次

### 第1层：配置与接口
```
config_stage1_physics_validation.json
    ↓
定义训练参数、模型结构、物理权重
```

### 第2层：数据层
```
efd_pinns_train.py::generate_training_data()
    ↓
生成62维输入特征 (x,y,z,t,V,...)
    ↓
使用Young-Lippmann方程计算24维输出 (θ,u,v,p,...)
    ↓
返回训练/验证/测试数据集
```

### 第3层：模型层
```
ewp_pinn_model.py::EWPINN
    ├── encoding_layer (ResNet编码)
    ├── branch1/2/3 (多分支处理)
    ├── multihead_att (注意力融合)
    ├── fusion_layer (特征融合)
    └── output_layer (输出预测)
```

### 第4层：物理约束层
```
ewp_pinn_physics.py::PINNConstraintLayer
    ├── Young-Lippmann残差
    ├── Navier-Stokes残差
    ├── 体积守恒残差
    └── 界面稳定性残差
```

### 第5层：训练与优化
```
efd_pinns_train.py::train_loop
    ├── 数据损失 (MSE)
    ├── 物理损失 (残差)
    ├── 总损失 = 数据损失 + α×物理损失
    └── 反向传播更新参数
```

### 第6层：分析与验证
```
analyze_young_lippmann.py
    ├── 加载训练好的模型
    ├── 生成测试数据
    ├── 计算cos(θ) vs V²的线性度
    └── 输出R²评估结果
```

---

## 🔄 详细数据流

### 1. 训练阶段

```
[配置加载]
config.json → 读取参数
    ↓
[数据生成]
generate_training_data()
    输入: num_samples=500
    处理:
        1. 生成62维随机特征 X
        2. 提取电压 V = X[:, 5]
        3. 计算接触角 θ = f_YL(V)  # Young-Lippmann
        4. 计算流场 u,v,w,p = f_flow(θ)
        5. 组装24维输出 Y = [θ, u, v, w, p, ...]
    输出: X_train, Y_train, X_val, Y_val, X_test, Y_test
    ↓
[模型初始化]
EWPINN(input_dim=62, output_dim=24)
    结构:
        Input(62) → Encoding(256) → Branches(128×3) 
        → Attention(128) → Fusion(128) → Output(24)
    ↓
[物理约束初始化]
PINNConstraintLayer(config)
    加载残差权重:
        young_lippmann: 5.0
        contact_angle: 1.0
        volume_conservation: 1.0
        ...
    ↓
[训练循环] for epoch in range(200):
    
    前向传播:
        predictions = model(X_batch)  # (batch, 24)
    
    计算损失:
        data_loss = MSE(predictions, Y_batch)
        physics_loss = constraint_layer(X_physics, predictions)
        total_loss = data_loss + 0.1 × physics_loss
    
    反向传播:
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    
    记录:
        保存loss历史
        每N个epoch保存checkpoint
    ↓
[保存模型]
checkpoint_epoch_199.pth
```

### 2. 分析阶段

```
[加载模型]
checkpoint.pth → EWPINN.load_state_dict()
    ↓
[生成测试数据]
V_test = [0, 1, 2, ..., 80]  # 电压范围
X_test = generate_features(V_test)  # 62维
    ↓
[模型预测]
predictions = model(X_test)
θ_pred = predictions[:, 0]  # 提取接触角
    ↓
[理论计算]
θ_theory = Young_Lippmann(V_test)
    ↓
[线性度分析]
cos(θ_pred) vs V²
    拟合: cos(θ) = a + b×V²
    计算: R²
    ↓
[输出结果]
R² = 0.74
图表: young_lippmann_analysis.png
报告: YOUNG_LIPPMANN_ANALYSIS.md
```

---

## 🧩 模块依赖关系

```
efd_pinns_train.py (主控)
    ├── 依赖 → ewp_pinn_model.py (模型定义)
    │   └── 依赖 → ewp_pinn_input_layer.py (输入特征)
    │   └── 依赖 → ewp_pinn_output_layer.py (输出解析)
    │
    ├── 依赖 → ewp_pinn_physics.py (物理约束)
    │   └── 依赖 → ewp_pinn_model.py (获取预测)
    │
    ├── 依赖 → ewp_pinn_optimizer.py (优化器)
    ├── 依赖 → ewp_pinn_dynamic_weight.py (动态权重)
    ├── 依赖 → ewp_pinn_training_tracker.py (训练跟踪)
    └── 依赖 → experiment_management (实验管理)

analyze_young_lippmann.py (分析)
    ├── 依赖 → ewp_pinn_model.py (加载模型)
    └── 依赖 → ewp_pinn_input_layer.py (生成测试数据)
```

---

## 🔑 关键概念

### 1. 输入特征 (62维)
```python
X = [
    # 基础时空电压 (6维)
    x, y, z, t, t_phase, V,
    
    # 几何结构 (12维)
    dist_wall_x, dist_wall_y, curvature_mean, ...
    
    # 材料界面 (10维)
    layer_position, interface_zone, wettability, ...
    
    # 电场 (8维)
    E_z, E_magnitude, field_gradient, ...
    
    # 流体动力学 (10维)
    reynolds, capillary_number, viscosity_ratio, ...
    
    # 时间动态 (6维)
    time_fourier, time_decay, velocity_trend, ...
    
    # 电润湿特性 (10维)
    electrowetting_number, young_lippmann_dev, ...
]
```

### 2. 输出物理量 (24维)
```python
Y = [
    θ,      # 接触角 (核心)
    u, v, w,  # 速度场
    p,      # 压力
    α,      # 体积分数
    κ,      # 界面曲率
    φ,      # 油墨势能
    ...     # 其他物理量
]
```

### 3. Young-Lippmann方程
```
核心物理关系:
cos(θ) = cos(θ₀) + (εε₀V²)/(2γd)

参数:
- θ₀ = 110° (初始接触角)
- ε = 3.0 (相对介电常数)
- ε₀ = 8.854e-12 (真空介电常数)
- γ = 0.0728 N/m (表面张力)
- d = 1e-6 m (介电层厚度)
- V = 0-80V (电压)

线性关系:
cos(θ) 与 V² 成线性关系
R² = 1.0 表示完美符合
```

### 4. 物理约束
```python
总物理损失 = Σ (权重ᵢ × 残差ᵢ²)

残差类型:
1. Young-Lippmann残差 (权重=5.0)
   residual = cos(θ_pred) - [cos(θ₀) + (εε₀V²)/(2γd)]

2. Navier-Stokes残差 (权重=0.01)
   continuity: ∂u/∂x + ∂v/∂y + ∂w/∂z = 0
   momentum: ρ(∂u/∂t + u·∇u) = -∇p + μ∇²u

3. 体积守恒残差 (权重=1.0)
   residual = |V_total - V_initial|

4. 界面稳定性残差 (权重=0.5)
   residual = |∇²κ|  # 曲率的拉普拉斯
```

### 5. 损失函数
```python
总损失 = 数据损失 + α × 物理损失

数据损失:
L_data = MSE(Y_pred, Y_true)
       = (1/N) Σ (Y_pred - Y_true)²

物理损失:
L_physics = Σ wᵢ × residualᵢ²

总损失:
L_total = L_data + α × L_physics
其中 α = 0.1 (物理权重)
```

---

## 🎮 训练策略

### 阶段1: 物理验证 (当前)
```
目标: 验证模型能否学习Young-Lippmann关系
配置:
  - epochs: 200
  - batch_size: 32
  - learning_rate: 0.001
  - physics_weight: 0.1
  - 数据量: 500样本
成功标准:
  - R² > 0.95
  - RMSE < 5°
  - 训练稳定 (无NaN/Inf)
```

### 阶段2: 多尺度训练 (未来)
```
目标: 处理不同尺度的物理现象
策略:
  - 粗网格 → 细网格
  - 低频 → 高频
  - 简单 → 复杂
```

### 阶段3: 完整耦合 (未来)
```
目标: 电场-流场-界面完全耦合
特点:
  - 所有物理约束同时激活
  - 自适应权重调整
  - 长时间演化
```

---

## 🐛 调试流程

### 问题: 训练不收敛
```
检查顺序:
1. 数据质量
   → 运行: python -c "验证数据的R²"
   → 期望: R² = 1.0

2. 损失平衡
   → 查看: training_tracker.log
   → 检查: data_loss vs physics_loss
   → 调整: physics_weight

3. 学习率
   → 查看: 学习率曲线
   → 调整: learning_rate, lr_scheduler

4. 梯度
   → 检查: 梯度范数
   → 调整: gradient_clipping
```

### 问题: R²太低
```
诊断:
1. 数据本身的R²
   → 如果数据R²=1.0，问题在模型
   → 如果数据R²<1.0，问题在数据生成

2. 模型复杂度
   → EWPINN太复杂 → 难学简单关系
   → 解决: 简化架构 或 接受当前结果

3. 训练不足
   → 增加epochs
   → 调整学习率
```

---

## 📈 性能指标

### 数据质量
```
理论数据: R² = 1.000000 ✅
说明: 数据完美符合Young-Lippmann方程
```

### 模型性能
```
当前: R² = 0.74
原因: 模型架构复杂，难以学习简单线性关系
建议: 保持架构用于复杂多物理场问题
```

### 训练效率
```
200 epochs: ~40分钟
显存占用: ~1.2GB
参数量: ~500K
```

---

## 🚀 快速参考

### 训练命令
```bash
conda run -n efd python efd_pinns_train.py \
    --mode train \
    --config config_stage1_physics_validation.json \
    --output-dir results \
    --epochs 200
```

### 分析命令
```bash
conda run -n efd python analyze_young_lippmann.py \
    results/experiments/*/checkpoints/checkpoint_epoch_*.pth
```

### 关键文件
```
配置: config_stage1_physics_validation.json
训练: efd_pinns_train.py
模型: ewp_pinn_model.py
物理: ewp_pinn_physics.py
分析: analyze_young_lippmann.py
```

---

**最后更新**: 2025-11-28  
**版本**: 1.0  
**状态**: 数据生成已修复，架构清晰
