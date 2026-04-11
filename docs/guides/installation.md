# 安装和配置指南

**最后更新**: 2026-02-04

## 系统要求

### 硬件要求
- CPU: 现代多核处理器
- 内存: 最低 8GB，推荐 16GB
- GPU: 强烈推荐 NVIDIA GPU (8GB+ 显存)，用于 Stage 2 的 60,000 Epochs 训练

### 软件要求
- Python: 3.12-3.13 (见 `pyproject.toml`)
- 包管理器: `uv` (推荐)
- 依赖管理: `pyproject.toml`

---

## 安装步骤

### 1. 克隆项目
```bash
git clone <repository_url>
cd EFD3D
```

### 2. 安装 uv (如果还没有)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3. 创建环境并安装依赖

#### 方式 1: 完整安装 (推荐)
```bash
# 创建虚拟环境并安装所有依赖
uv sync

# 激活环境
source .venv/bin/activate  # Linux/Mac
# .\.venv\Scripts\activate  # Windows
```

#### 方式 2: 最小安装 (仅核心依赖)
```bash
uv sync --no-dev
```

#### 方式 3: 安装特定可选依赖
```bash
# 安装测试依赖
uv sync --extra testing

# 安装完整依赖 (含可视化)
uv sync --extra full

# 安装 Web API 依赖
uv sync --extra web
```

### 4. 验证安装

```bash
# 验证 Python 版本
python --version  # 应该显示 3.12.x 或 3.13.x

# 验证 PyTorch 和 CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# 验证项目导入
python -c "from src.config import PHYSICS; print('EFD3D 导入成功')"

# 运行基础测试
pytest tests/test_physics_sanity.py -v
```

---

## 配置文件

### 配置文件列表

| 文件 | 说明 | 用途 |
|------|------|------|
| `config/v4.5-standard.json` | ⭐ **推荐配置** | 标准训练配置（已验证收敛，60,000 epochs） |
| `config/device_calibrated_physics.json` | 核心物理配置 | 物理参数配置 |
| `config/v4.5_lbfgs_tuned.json` | L-BFGS 微调配置 | 备选配置 |

### 推荐配置 (`device_calibrated_physics.json`)

```json
{
  "materials": {
    "epsilon_r": 12.0,
    "gamma": 0.015,
    "theta0": 120.0,
    "theta_wall": 71.0
  },
  "training": {
    "epochs": 60000,
    "batch_size": 4096,
    "learning_rate": 0.0003,
    "stage1_epochs": 1500,
    "stage2_epochs": 4000,
    "stage3_epochs": 50000
  }
}
```

详细参数说明请参见[物理理论与器件规格指南](physics_and_device_guide.md#physics-parameters) 和 [配置系统指南](configuration_guide.md)。

 详见: [配置系统指南](configuration_guide.md)

---

## 故障排除

### 模块导入失败

确保已激活虚拟环境：
```bash
source .venv/bin/activate  # Linux/Mac
# .\.venv\Scripts\activate  # Windows
```

### 依赖安装失败

```bash
# 清理缓存并重新安装
uv pip cache clear
uv sync --force-reinstall
```

### CUDA 内存不足 (OOM)

- 减少 `batch_size` (例如从 4096 减至 2048)
- 减少 `num_physics_points`
- 或者在训练命令后添加 `--cpu` 强制使用 CPU

### 测试失败

```bash
# 重新安装测试依赖
uv sync --extra testing

# 运行测试
pytest tests/ -v --tb=short
```

---

## 下一步

完成安装后：

1. [快速开始指南](quickstart.md) - 首次使用
2. [配置系统指南](configuration_guide.md) - 了解配置
3. [训练策略](training_guide.md) - 优化训练
