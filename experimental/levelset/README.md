# 电润湿显示 (EFD) 3D Level Set PINN 仿真模型

![Status](https://img.shields.io/badge/Status-Active_Development-green)
![Python](https://img.shields.io/badge/Python-3.12%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 📖 简介

本项目实现了一个 **3D 物理信息神经网络 (PINN)**，用于模拟 **电润湿显示器 (Electrowetting Display, EFD)** 内的微流体动力学。通过求解耦合的 Navier-Stokes 方程、Level Set 输运方程和静电场方程，该模型能够预测在不同电压条件下像素内油墨和极性液体的复杂两相流动。

与传统的计算流体力学 (CFD) 方法（如 VOF 或有限元分析）相比，这种 PINN 方法具有以下优势：
1.  **无网格仿真**: 直接在点云上求解偏微分方程 (PDE)，无需复杂的网格划分。
2.  **连续表征**: 为速度场和压力场提供无限分辨率的解析解。
3.  **参数化建模**: 可以在单个训练好的模型中泛化不同的电压跳变 ($V_{from} \to V_{to}$) 和时间点 ($t$)。
4.  **可微分物理**: 允许通过反向传播对器件参数进行逆向设计和优化。

核心创新在于使用 **Level Set 方法 (符号距离函数, SDF)** 来表示油墨 (Ink) 和极性液体 (Polar Liquid) 之间的尖锐界面，并结合 **多尺度神经网络** 架构来解析接触线和角流等精细特征。

---

## 🔬 物理模型与器件结构

### 1. 器件几何尺寸
仿真模拟单个 EFD 像素，具有以下精确尺寸。

**垂直堆叠结构 (从下到上):**
1.  **基板/电极**: 基础层 (氧化铟锡 - ITO)。
2.  **绝缘层**: $0.8 \mu m$ Parylene C ($\epsilon_r \approx 3.0$)。
3.  **疏水层**: $0.4 \mu m$ ($400 nm$) Teflon AF ($\epsilon_r \approx 2.0$)。
4.  **流体域**: 油墨 ($3 \mu m$) 和 极性液体 ($17 \mu m$)。

| 组件 | 尺寸 / 数值 | 描述 |
| :--- | :--- | :--- |
| **像素区域** | $174 \mu m \times 174 \mu m$ | 内部有效显示区域 |
| **像素墙** | 高: $20 \mu m$, 宽: $5 \mu m$ | 分隔相邻像素的结构 |
| **油墨层** | 厚度: $3 \mu m$ | 彩色非极性流体 (如癸烷) |
| **极性液体** | 厚度: $17 \mu m$ | 导电液体 (如水)，填充至墙高 |
| **疏水层** | 厚度: $400 nm$ ($0.4 \mu m$) | Teflon AF，提供高接触角 |
| **绝缘层** | 厚度: $0.8 \mu m$ | Parylene C，隔离电极 |

**总流体计算域**: $174 \times 174 \times 20 \mu m^3$。

### 2. 控制方程

系统由三个物理场的耦合控制：

#### A. 流体动力学 (两相 Navier-Stokes)
两种流体均被建模为不可压缩牛顿流体：
$$ \nabla \cdot \mathbf{u} = 0 $$
$$ \rho \left( \frac{\partial \mathbf{u}}{\partial t} + \mathbf{u} \cdot \nabla \mathbf{u} \right) = -\nabla p + \mu \nabla^2 \mathbf{u} + \mathbf{F}_{st} + \mathbf{F}_{elec} $$

其中：
*   $\rho, \mu$: 密度和粘度 (相依赖: $\rho(\psi), \mu(\psi)$)。
*   $\mathbf{F}_{st} = \gamma \kappa \delta(\psi) \mathbf{n}$: 表面张力 ($\gamma=0.045 N/m$)。
*   $\mathbf{F}_{elec}$: 静电力 (麦克斯韦应力)。

#### B. 界面追踪 (Level Set 方法)
界面定义为界面高度 ψ(x, t) 的零等值面（根据 LEVELSET_PHYSICS.md）：
*   ψ > 0: 油墨相 (Ink)，值为界面高度
*   ψ = 0: 界面在 z=0，最大开口
*   ψ < 0: 不允许（违反物理约束）

界面的演化由输运方程控制：
$$ \frac{\partial \psi}{\partial t} + \mathbf{u} \cdot \nabla \psi = 0 $$

几何属性直接由 $\psi$ 导出：
*   法向量: $\mathbf{n} = \nabla \psi / |\nabla \psi|$
*   曲率: $\kappa = \nabla \cdot \mathbf{n}$

#### C. 静电学 (Young-Lippmann 效应)
接触角 $\theta$ 随施加电压 $V$ 变化：
$$ \cos\theta(V) = \cos\theta_0 + \frac{\epsilon_0 \epsilon_r}{2 \gamma d} V^2 $$
该效应驱动油墨运动。静电力通过计算麦克斯韦应力张量的散度或在 Level Set 框架下使用接触角边界条件来施加。

### 3. 材料属性

| 属性 | 油墨 (Phase 1) | 极性液体 (Phase 2) |
| :--- | :--- | :--- |
| **密度 ($\rho$)** | $730 kg/m^3$ | $997 kg/m^3$ |
| **粘度 ($\mu$)** | $0.00092 Pa \cdot s$ | $0.00089 Pa \cdot s$ |
| **相对介电常数 ($\epsilon_r$)** | 2.0 | 78.4 |

---

## 🧠 神经网络架构

本项目使用定制的 **多尺度 3D PINN**，专门设计用于捕捉 EFD 的多物理场尺度特征。

### 输入与输出
*   **输入 (6D)**: $[x, y, z, V_{start}, V_{end}, t_{elapsed}]$
    *   空间坐标 $(x,y,z)$。
    *   定义电压跳变的阶跃 ($V_{start} \to V_{end}$)。
    *   电压变化后的时间 ($t_{elapsed}$)。
*   **输出 (10D)**:
    *   第一相速度: $[u_1, v_1, w_1]$
    *   第二相速度: $[u_2, v_2, w_2]$
    *   Level Set 函数: $[\psi]$
    *   压力: $[p_1, p_2]$
    *   (可选辅助输出)

### 网络组件
1.  **主网络 (Main Network)**: 全连接层 (MLP)，使用 SiLU 激活函数。捕捉全局主体流动。
    *   结构: `[Input] -> 64 -> 64 -> 32 -> [Output]`
2.  **界面网络 (Interface Network)**: 专注于 $\psi \approx 0$ 区域的残差网络。为表面张力处理提供更高分辨率。
    *   结构: `[Input] -> 64 -> 64 -> 32 -> [Output]`
3.  **角点网络 (Corner Network)**: 专门用于处理三相接触线 (墙-油-水) 处奇异性的小型网络。
    *   结构: `[Input] -> 64 -> 64 -> 64 -> [Output]`

### 损失函数 (物理约束)
总损失 $\mathcal{L}$ 是以下各项的加权和：
1.  $\mathcal{L}_{data}$: 拟合初始条件和已知边界状态。
2.  $\mathcal{L}_{PDE}$: Navier-Stokes 和 Level Set 输运方程的残差。
3.  $\mathcal{L}_{bc}$: 边界条件 (无滑移壁面，Young-Lippmann 接触角)。
4.  $\mathcal{L}_{vol}$: **体积守恒约束** (对于封闭系统至关重要)。
5.  $\mathcal{L}_{mono}$: **开口率单调性约束** (确保物理合理的开关行为)。
6.  $\mathcal{L}_{spatial}$: $\psi$ 的空间正则化，保持 SDF 属性 ($|\nabla \psi| = 1$)。

---

## ✨ 关键特性与创新

### 1. 鲁棒的体积守恒
标准的 Level Set 方法常因数值扩散而遭受“质量损失”（油墨体积随时间收缩）。我们实施了严格的 **全局体积约束**：
$$ \mathcal{L}_{vol} = \left( \int_{\Omega} H(-\psi) dV - V_{initial} \right)^2 $$
这确保了整个仿真过程中油墨体积保持恒定（等效 $3\mu m$ 厚度）。

### 2. 开口率单调性
为了防止电压切换过程中开口率出现非物理震荡（尖峰），我们强制执行：
$$ \frac{\partial \text{Aperture}}{\partial V} \propto \text{sign}(V_{target} - V_{current}) $$
这保证了开/关曲线平滑且符合实验观察。

### 3. 动态响应与压摆率限制
评估流程包含先进的信号处理，以处理神经网络的外推伪影：
*   **状态感知平滑**: 检测“稳态”与“瞬态”。
*   **压摆率限制 (Slew Rate Limiting)**: 钳位物理上不可能的瞬时变化（例如，1ms 内开口率变化 >20%）。
*   **过渡逻辑**: 针对 0V $\to$ High V (开启) 和 High V $\to$ 0V (关闭) 使用不同的物理先验。

### 4. 自动化评估仪表板
每次训练运行都会自动生成一份综合 PDF/图片仪表板：
*   **3D 等值面**: 可视化不同时刻的油墨形状。
*   **开口率 vs 时间**: 动态响应曲线 (上升时间 / 下降时间)。
*   **Loss 收敛**: 各分量损失的详细分解。
*   **物理指标**: 体积误差百分比，界面能。

---

## 🛠️ 安装指南

### 先决条件
*   **操作系统**: Linux (推荐), Windows (WSL2), 或 macOS。
*   **Python**: 3.8+ (已在 3.12 上测试)。
*   **GPU**: NVIDIA GPU + CUDA 11.x/12.x (强烈推荐用于 3D 训练)。
    *   最低显存: 4GB (用于推理/小批量)。
    *   推荐显存: 8GB+ (用于全量训练)。

### 设置步骤
1.  **克隆仓库**:
    ```bash
    git clone https://gitee.com/EFD3D/levelset.git
    cd levelset
    ```

2.  **创建 Conda 环境**:
    ```bash
    conda create -n efd python=3.12
    conda activate efd
    ```

3.  **安装依赖**:
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install numpy matplotlib scipy pandas tqdm
    ```

---

## 🚀 使用指南

### 1. 数据生成
在训练之前，需要生成时空点云数据。这将创建一个包含流体域和边界上的配置点的 `.pt` 文件。
```bash
python generate_levelset_data.py
```
*   输出: `data/levelset_training_data_v5.5_large.pt`

### 2. 训练
运行主训练脚本。配置文件控制所有超参数。
```bash
# 标准训练 (如果有 GPU 会自动使用)
python train_levelset_3d.py --config config/v5.5_full.json
```
*   **检查点**: 保存在 `outputs_levelset_YYYYMMDD_HHMMSS/`。
*   **日志**: 实时日志在 `training.log`。
*   **恢复训练**: 要恢复训练，需手动加载 `.pt` 检查点（目前需要修改脚本）。

### 3. 评估
评估会在训练结束后自动运行。要重新评估现有模型或重新生成图表：
```bash
# 评估指定的输出目录
python evaluate.py outputs_levelset_20260127_XXXXXX

# 或仅重新生成 Loss 曲线 (快速)
python train_levelset_3d.py --plot-only --output-dir outputs_levelset_20260127_XXXXXX
```

### 4. 可视化
结果以图片形式保存在输出目录中：
*   `evaluation_dashboard_final_model.png`: 性能摘要。
*   `3d_isosurface_final_model.png`: 油墨的 3D 视图。
*   `dynamic_response.csv`: 随时间变化的开口率原始数据。

---

## 📁 项目结构

```
levelset/
├── config/                     # 配置文件 (JSON)
│   ├── v5.5_full.json          # 推荐的生产环境配置
│   └── archive/                # 历史配置
├── data/                       # 生成的训练数据 (.pt)
├── docs/                       # 文档
│   ├── EWD_PHYSICS.md          # 详细物理推导
│   ├── USER_GUIDE.md           # 可视化工具指南
│   ├── PROJECT_REVIEW.md       # 进度报告
│   └── CHANGELOG.md            # 版本历史
├── outputs_levelset_*/         # 实验结果 (日志, 模型, 图表)
├── device_parameters.py        # 物理常数的唯一真理来源
├── evaluate.py                 # 后处理与验证脚本
├── generate_levelset_data.py   # 点云生成器
├── pinn_levelset_3d.py         # PyTorch 模型定义
├── plot_training_curves.py     # Loss 可视化工具
├── train_levelset_3d.py        # 主训练循环
├── README.md                   # 本文件
└── QUICKSTART.md               # 快速开始指南
```

---

## 📊 可视化画廊

自动化评估管道会生成几个关键的可视化图表，帮助您理解模型性能。

### 1. 3D 等值面 (`3d_isosurface_final_model.png`)
*   **展示内容**: 仿真结束时 ($t=0.05s$) 油墨界面 ($\psi=0$) 的 3D 渲染图。
*   **解读**:
    *   **蓝色表面**: 油墨和极性液体的分界。
    *   **形状**: 应显示油墨被推向角落（开启状态）或平铺（关闭状态）。
    *   **伪影**: 寻找孔洞或断开的斑点，这表示拓扑错误。

### 2. 评估仪表板 (`evaluation_dashboard_final_model.png`)
一个 6 面板的综合报告：
*   **左上 (开口率)**: 最关键的指标。显示随时间变化的像素清晰区域百分比。
    *   *绿线*: 平滑后的 (物理) 响应。
    *   *红/蓝虚线*: 原始 PINN 输出。
*   **中上 (体积守恒)**: 跟踪油墨总体积。应为一条位于 1.0 (归一化) 的平线。
*   **右上 (中心高度)**: 像素中心 ($x=0, y=0$) 处的界面高度。
*   **下排 (截面图)**: 不同时间点 ($t=0, t=mid, t=end$) Level Set 场 $\psi$ 的 2D 切片。
    *   *黑线*: 界面 ($\psi=0$)。
    *   *色图*: 距离场 (蓝=油墨, 红=水)。

### 3. 训练 Loss 曲线 (`training_loss_*.png`)
*   **总 Loss**: 整体收敛趋势。
*   **分量 Loss**: 各个单独的损失 (Data, PDE, Boundary, Volume)。
    *   *注意*: 尖峰通常对应课程学习过程中的阶段转换。

---

## 📈 科学策略 (v5.5)

当前版本 (v5.5) 采用“空间约束 + 大数据”策略：
*   **数据增强**: 训练集增加到 750,000 点 (是旧版本的 2.5 倍)。
*   **Batch Size**: 大批量 (4096) 以在 GPU 上进行稳定的梯度估计。
*   **课程学习 (Curriculum Learning)**:
    *   **Stage 1 (0-5k epochs)**: 数据拟合 & SDF 初始化。
    *   **Stage 2 (5k-15k epochs)**: 引入物理约束 (输运)。
    *   **Stage 3 (15k-30k epochs)**: 逐步增加输运权重。
    *   **Stage 4 (30k-50k epochs)**: 全多物理场耦合。
    *   **Stage 5 (50k+ epochs)**: 微调。

---

## ⚙️ 配置参考

`config/v5.5_full.json` 文件控制整个训练过程。以下是关键参数的详细解释：

### 模型架构 (`model`)
*   `input_dim`: 6 (勿改)
*   `output_dim`: 10 (u1,v1,w1, u2,v2,w2, psi, p1, p2, +aux)
*   `hidden_main`: `[64, 64, 32]` - 主流场网络的层结构。
*   `hidden_interface`: `[64, 64, 32]` - 界面精细化网络的层结构。
*   `hidden_corner`: `[64, 64, 64]` - 接触线奇异性网络的层结构。
*   `psi_scale`: Level Set 输出的缩放因子 (通常为 1e-6，对应微米级)。

### 训练超参数 (`training`)
*   `epochs`: 总训练迭代次数 (如 80,000)。
*   `batch_size`: 每次梯度更新的点数 (如 4096)。
*   `learning_rate`: 初始学习率 (如 5e-5)。
*   `lr_scheduler`:
    *   `type`: "CosineAnnealingWarmRestarts" - 有助于跳出局部极小值。
    *   `T_0`: 周期长度 (如 10,000 epochs)。

### 损失权重 (`loss_weights`)
权重决定了每个物理约束的相对重要性。这些权重可以在训练阶段动态变化。
*   `data`: 拟合初始/边界数据。高权重 (100-1000)。
*   `levelset_transport`: 强制执行 $\partial\psi/\partial t + \mathbf{u}\cdot\nabla\psi = 0$。
*   `ns_1`, `ns_2`: 油墨和极性液体的 Navier-Stokes 残差。
*   `continuity_1`, `continuity_2`: 不可压缩性 ($\nabla \cdot \mathbf{u} = 0$)。
*   `volume_conservation`: 惩罚总油墨体积的偏差。
*   `aperture_monotonicity`: 惩罚非物理的开口率波动。
*   `psi_spatial`: 强制执行 $|\nabla \psi| = 1$ (Eikonal 约束)。

---

## 🧩 代码架构深度解析

### 1. `pinn_levelset_3d.py` (大脑)
此文件定义了 `LevelSet3DPINN` 类，继承自 `torch.nn.Module`。
*   **`__init__`**: 初始化三个子网络 (Main, Interface, Corner) 并设置物理常数 ($\rho, \mu, \gamma$)。
*   **`forward(x)`**: 将输入通过所有网络，并使用类似注意力的门控机制（如果启用）或直接求和来组合输出。
*   **`get_phase_properties(psi)`**: 使用平滑的 Heaviside 函数 $H(\psi)$ 计算局部密度/粘度。
*   **`compute_surface_tension_force`**: 使用自动微分计算曲率 $\kappa$，进而计算 $\mathbf{F}_{st} = \gamma \kappa \delta(\psi) \mathbf{n}$。
*   **`maxwell_stress_force`**: 基于电压梯度计算静电力。

### 2. `train_levelset_3d.py` (引擎)
主训练循环，负责数据加载、前向传播和反向传播。
*   **动态加权**: 根据配置中定义的当前 `stage` 调整损失权重。
*   **梯度累积**: 支持在较小显存的 GPU 上使用有效的大 Batch Size。
*   **日志记录**: 将所有指标记录到 `training.log` 和 TensorBoard (可选)。

### 3. `evaluate.py` (分析师)
独立的后处理套件，加载训练好的模型并执行物理验证。
*   **网格采样**: 创建 3D 网格 ($50 \times 50 \times 20$) 以采样场数据。
*   **Marching Cubes**: 提取 $\psi=0$ 等值面用于 3D 可视化。
*   **信号处理**: 对原始开口率数据应用 `slew_rate_limiter` 和 `monotonicity_constraint`。

### 4. `generate_levelset_data.py` (测量员)
生成用于训练的配置点。
*   **域**: $x,y \in [0, 174\mu m]$, $z \in [0, 20\mu m]$。
*   **采样策略**:
    *   **内部**: 流体体积内的随机均匀采样。
    *   **界面**: 在 $z=3\mu m$ (初始界面) 附近的密集采样。
    *   **墙壁**: 在无滑移边界上的采样。
    *   **时间**: 在 $t \in [0, 0.05s]$ 范围内的采样。

---

## 🗺️ 未来路线图

项目正在积极演进中。以下功能计划在未来版本 (v6.0+) 中发布：

1.  **自适应采样 (RAR)**
    *   实施 *基于残差的自适应细化 (Residual-based Adaptive Refinement)*，动态地在 PDE 误差较高的区域（如移动接触线附近）添加训练点。
2.  **接触角滞后**
    *   引入动态接触角模型 $\theta(v_{cl})$ 以考虑接触线摩擦和钉扎效应，这对精确模拟滞后现象至关重要。
3.  **多像素干扰**
    *   将域扩展到 $3 \times 3$ 像素，以研究压力波从一个像素影响邻居的“串扰”效应。
4.  **实时推理**
    *   将训练好的 PyTorch 模型导出为 ONNX/TensorRT，实现毫秒级推理，从而支持实时控制回路集成。
5.  **电荷捕获模型**
    *   在 Young-Lippmann 方程中增加时间相关项，以模拟介电层充电（可靠性物理）。

---

## 📚 科学参考文献

1.  **Physics-Informed Neural Networks (PINNs)**
    *   Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686-707.
2.  **Level Set Method**
    *   Osher, S., & Sethian, J. A. (1988). Fronts propagating with curvature-dependent speed: algorithms based on Hamilton-Jacobi formulations. *Journal of Computational Physics*, 79(1), 12-49.
3.  **Electrowetting Theory**
    *   Mugele, F., & Baret, J. C. (2005). Electrowetting: from basics to applications. *Journal of Physics: Condensed Matter*, 17(28), R705.
4.  **Immersed Boundary Method (for Contact Lines)**
    *   Peskin, C. S. (2002). The immersed boundary method. *Acta Numerica*, 11, 479-517.

---

## ❓ 常见问题 / 故障排除

**Q: 关断时 t=0 处开口率出现尖峰。**
A: 这通常是神经网络的外推伪影。`evaluate.py` 脚本现在包含一条“平滑”曲线，应用物理约束（单调性、压摆率限制）来滤除这些尖峰。请查看 `evaluation_dashboard_*.png` 对比原始数据和平滑数据。

**Q: 训练很慢。**
A: 请确保您正在 GPU 上运行（使用 `nvidia-smi` 检查）。代码已针对 CUDA 进行了优化。在 Quadro P2000 上，100 epochs 大约需要 18 秒。

**Q: 油墨体积在减少。**
A: 检查配置中的 `volume_conservation` 权重。在 v5.5 中，此权重设置为 50.0 以强烈惩罚质量损失。

---

## 📜 许可证

本项目采用 MIT 许可证 - 详情请参阅 LICENSE 文件。

---

## 👥 联系与引用

**作者**: EFD-PINNs Team (SCNU)
**日期**: 2026年1月

如果您在研究中使用了此代码，请引用：
> *3D Level Set PINN for Electrowetting Display Simulation*, Internal Report, SCNU, 2026.
