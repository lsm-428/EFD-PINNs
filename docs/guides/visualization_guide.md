# 可视化工具完整指南

**最后更新**: 2026-03-06
**版本**: v4.5

## 概述

EFD3D项目提供了多种可视化工具，用于展示器件结构、PINN流场动态和模型评估。本指南整合了所有可视化工具的使用方法、功能对比和最佳实践。

---

## 📊 可视化工具概览

### 1. `visualizer_3d/` - 3D 可视化模块

**用途**: 展示完整的器件 3D 结构和 PINN 流场动态

**功能**:
- ✅ 完整的器件材料结构（ITO、介电层、疏水层、围堰、油墨层、极性液体层）
- ✅ PyVista 高质量 3D 渲染
- ✅ 基于 PINN 模型的流场可视化
- ✅ 动画生成

**位置**: `scripts/visualizer_3d/`

**输出**:
- `aperture_3d.png` - 3D 开口率可视化
- `phi_iso_surface.png` - φ 场等值面
- `velocity_vectors_3d.png` - 速度场矢量图

### 2. `visualizer_3d/` - PINN 流场动态可视化

**用途**: 基于真实 PINN 模型的油墨开关动态可视化

**功能**:
- ✅ 从训练好的 PINN 模型读取流场
- ✅ 生成 0V → 30V → 0V 动态响应
- ✅ 包含 φ 场、速度场可视化
- ✅ 与 Stage 1 解析解对比

**位置**: `scripts/visualizer_3d/`

**输出**:
- 单帧: `aperture_3d.png`
- 动画: `frame_0000.png` ~ `frame_0029.png`
- 视频: `ink_dynamics.mp4` (需 ffmpeg)

### 3. `visualizer_3d/` - 专业 3D 可视化模块

**用途**: PINN 模型的完整 3D 可视化

**位置**: `scripts/visualizer_3d/`

**功能**:
- ✅ 基于 PINN 模型的 φ 场 3D 渲染
- ✅ 开口率动态响应
- ✅ 速度场矢量可视化
- ✅ PyVista 交互式渲染

### 4. `evaluate.py` - 模型评估和专业仪表盘

**用途**: 生成专业的4面板评估仪表盘

**功能**:
- ✅ 动态响应曲线
- ✅ 稳态扫描
- ✅ φ场切片
- ✅ 速度场矢量图

---

## 🎯 使用场景建议

### 场景1: 需要展示器件结构和 PINN 流场
**推荐**: `visualizer_3d/`

```bash
python -m scripts.visualizer_3d.visualizer_3d --demo
```

**输出**: 清晰的 3D 结构图和流场可视化，适合 PPT 和论文

### 场景2: 需要展示 PINN 流场动态
**推荐**: `visualizer_3d/`

```bash
python -m scripts.visualizer_3d.visualizer_3d --interactive
```

**输出**: 基于真实 PINN 预测的动态响应

### 场景3: 需要专业 3D 可视化（φ 场、速度场）
**推荐**: `visualizer_3d/`

```python
from scripts.visualizer_3d.visualizer_3d import PixelVisualizer
viz = PixelVisualizer(model_path="...")
viz.plot_aperture_3d(voltage=20, time=0.01)
```

**输出**: 完整的 3D 流场可视化，包含速度矢量

### 场景4: 需要模型评估和专业仪表盘
**推荐**: `evaluate.py`

```bash
python evaluate.py outputs/train/pinn_xxx
```

**输出**: 4面板专业仪表盘（动态响应、稳态扫描、φ场、速度场）

---

## 📈 功能对比表

| 特性 | `visualizer_3d/` | `evaluate.py` |
|------|------------------|---------------|
| **器件结构** | ✅ 3D 结构 | ❌ |
| **基础模型** | PINN | PINN |
| **φ 场可视化** | ✅ 3D | ✅ 2D 切片 |
| **速度场** | ✅ 3D 矢量 | ✅ 矢量图 |
| **动态响应** | ✅ PINN 预测 | ✅ PINN 预测 |
| **3D 渲染** | ✅ PyVista | ❌ |
| **动画生成** | ✅ | ❌ |
| **代码质量** | ✅ 高 | ✅ 高 |
| **主要用途** | 3D 流场可视化 | 模型评估 |

---

## 🛠️ 详细使用方法

### 1. 3D 可视化 (`visualizer_3d/`)

```bash
# 运行演示可视化
python -m scripts.visualizer_3d.visualizer_3d --demo

# 交互式 3D 探索
python -m scripts.visualizer_3d.visualizer_3d --interactive

# 指定模型路径
python -m scripts.visualizer_3d.visualizer_3d --model outputs/train/pinn_xxx/best_model.pth
```

### 2. 模型评估 (`evaluate.py`)

```bash
# 评估特定模型
python evaluate.py outputs/train/pinn_xxx

# 对比最近两个模型
python evaluate.py --compare
```

---

## 🔧 工具演变历史

### 旧版可视化工具
- ❌ `scripts/visualization/` 目录已移除
- ❌ `src/visualization/` 目录已移除
- ❌ `generate_pyvista_3d.py` 已移除
- ❌ `generate_ink_dynamics.py` 已移除

### 当前版本 (`visualizer_3d/`)
- ✅ 统一的 3D 可视化模块
- ✅ 基于 PyVista 的专业渲染
- ✅ 支持交互式探索和动画生成
- ✅ 从 PHYSICS 读取参数

---

## 💡 完整工作流程建议

### 完整的可视化工作流

```bash
# 1. 训练 PINN 模型
python train_two_phase.py --config config/v4.5-standard.json

# 2. 评估模型性能
python evaluate.py outputs/train/pinn_<timestamp>

# 3. 生成 3D 可视化
python -m scripts.visualizer_3d.visualizer_3d --demo

# 4. 交互式 3D 探索
python -m scripts.visualizer_3d.visualizer_3d --interactive
```

---

## 📂 文件组织

### 当前文件结构

```
scripts/visualizer_3d/
├── visualizer_3d.py          # 主入口
├── visualization_core.py     # 核心渲染
├── animation_utils.py        # 动画工具
└── interactive_features.py   # 交互功能

根目录:
├── evaluate.py               # 模型评估工具
└── ...
```

---

## ⚡ 性能对比

| 操作 | `visualizer_3d/` |
|------|------------------|
| 生成单帧 | ~3 秒 |
| 生成动画(30帧) | ~90 秒 |
| 内存占用 | ~400 MB |

---

## 🎓 学习路径

1. **初学者**: 从 `evaluate.py` 开始，理解模型评估
2. **进阶**: 使用 `visualizer_3d/` 生成 3D 可视化
3. **高级**: 使用 `visualizer_3d/` 交互式探索
4. **专家**: 直接使用 `visualizer_3d/` 模块，定制可视化

---

## 📞 常见问题解答

### Q1: 三个工具会冲突吗？
**A**: 不会。它们各有侧重，可以配合使用。

### Q2: 我应该用哪个？
**A**:
- 模型评估 → `evaluate.py`
- 专业 3D 分析 → `visualizer_3d/`

### Q3: 模型加载失败
```
Error: No such file or directory: 'outputs/train/pinn_xxx/best_model.pth'
```
**解决**: 检查模型路径是否正确，使用 `ls outputs/train/` 查看可用的训练输出。

### Q4: 内存不足
**解决**: 降低 `n_points` 参数或减少 `--frames` 数量。

### Q5: ffmpeg 未找到
```
ffmpeg: command not found
```
**解决**:
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS  
brew install ffmpeg

# Conda
conda install ffmpeg
```

---

## 扩展功能

### 添加物理量统计

```python
def compute_physics_metrics(self, V_from, V_to, t_since):
    """计算体积守恒、界面误差等物理量"""
    # 添加统计代码
    pass
```

---