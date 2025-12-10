# 安装和配置指南

**最后更新**: 2025-12-08

## 系统要求

### 硬件要求
- CPU: 现代多核处理器
- 内存: 最低 8GB，推荐 16GB
- GPU: 可选，支持 CUDA 的 NVIDIA GPU

### 软件要求
- Python: 3.8+
- PyTorch: 2.0+

## 安装步骤

### 1. 克隆项目
```bash
git clone <repository_url>
cd EFD3D
```

### 2. 创建环境
```bash
conda create -n efd python=3.9
conda activate efd
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

### 4. 验证安装
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -m pytest tests/ -v
```

## 配置文件

配置文件位于 `config/` 目录：

| 文件 | 说明 |
|------|------|
| `stage6_wall_effect.json` | Stage 1 最佳配置 |
| `stage1_config.json` | 阶段1初始训练 |

### 配置结构

```json
{
  "model": {
    "input_dim": 62,
    "output_dim": 24,
    "hidden_dims": [256, 128, 64]
  },
  "training": {
    "epochs": 3000,
    "batch_size": 64,
    "learning_rate": 0.001
  },
  "materials": {
    "theta0": 120.0,
    "epsilon_r": 4.0,
    "gamma": 0.072
  }
}
```

## 快速开始

### Stage 1: 接触角预测

```python
from src.predictors import HybridPredictor

predictor = HybridPredictor()
theta = predictor.predict(voltage=30, time=0.005)
```

### Stage 2: 开口率预测

```python
from src.predictors.pinn_aperture import PINNAperturePredictor

predictor = PINNAperturePredictor()
eta = predictor.predict(voltage=30, time=0.015)
```

## 训练

### Stage 1 训练
```bash
python train_contact_angle.py --quick-run
```

### Stage 2 训练
```bash
python train_two_phase.py --epochs 30000
```

## 故障排除

### 模块导入失败
```bash
conda activate efd
```

### CUDA 内存不足
减少批次大小或使用 CPU。

### 测试失败
```bash
python -m pytest tests/ -v
```
