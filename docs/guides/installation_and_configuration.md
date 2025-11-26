# 安装和配置指南

## 系统要求

### 硬件要求
- **CPU**: 支持AVX指令集的现代处理器
- **内存**: 最低8GB，推荐16GB以上
- **GPU**: 可选，支持CUDA的NVIDIA GPU（推荐RTX 3060以上）
- **存储**: 至少10GB可用空间

### 软件要求
- **操作系统**: Linux (Ubuntu 18.04+), Windows 10+, macOS 10.15+
- **Python**: 3.8, 3.9, 3.10, 3.11
- **包管理器**: pip 20.0+

## 安装步骤

### 1. 克隆项目
```bash
git clone https://gitee.com/scnu/EFD3D.git
cd EFD3D
```

### 2. 创建虚拟环境（推荐）
```bash
# 使用 conda
conda create -n efd3d python=3.9
conda activate efd3d

# 或使用 venv
python -m venv efd3d_env
source efd3d_env/bin/activate  # Linux/macOS
# Windows: efd3d_env\Scripts\activate
```

### 3. 安装核心依赖
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy matplotlib scikit-learn pytest
```

### 4. 安装可选依赖
```bash
# ONNX支持（模型导出）
pip install onnx onnxruntime

# 3D可视化
pip install pyvista

# 数据处理
pip install pandas

# 科学计算
pip install scipy
```

### 5. 验证安装
```bash
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
python -c "import numpy; print(f'NumPy版本: {numpy.__version__}')"

# 运行简单测试
pytest -q
```

## 配置系统

### 基础配置

项目使用JSON配置文件管理训练参数。基础配置文件位于 `config/` 目录：

```json
{
    "模型": {
        "输入维度": 62,
        "输出维度": 24,
        "隐藏层": [256, 128, 64],
        "激活函数": "ReLU",
        "批标准化": true,
        "Dropout率": 0.1
    },
    "训练": {
        "渐进式训练": [
            {
                "名称": "预训练",
                "轮次": 1000,
                "学习率": 0.001,
                "批次大小": 32,
                "权重衰减": 1e-4,
                "优化器": "Adam",
                "调度策略": "cosine",
                "物理约束权重": 0.1
            },
            {
                "名称": "物理约束",
                "轮次": 2000,
                "学习率": 0.0005,
                "批次大小": 32,
                "物理约束权重": 0.5
            },
            {
                "名称": "微调",
                "轮次": 1000,
                "学习率": 0.0001,
                "批次大小": 32,
                "物理约束权重": 1.0
            }
        ],
        "早停配置": {
            "启用": true,
            "耐心值": 100,
            "最小改进": 1e-6,
            "恢复最佳模型": true
        },
        "梯度裁剪": 1.0
    },
    "数据": {
        "样本数量": 10000,
        "训练比例": 0.7,
        "验证比例": 0.15,
        "测试比例": 0.15,
        "数据增强": true
    },
    "物理约束": {
        "启用": true,
        "初始权重": 1.0,
        "物理点数量": 1000,
        "残差权重": {
            "连续性": 1.0,
            "动量_u": 0.1,
            "动量_v": 0.1,
            "动量_w": 0.1,
            "young_lippmann": 0.5,
            "接触线动力学": 0.3,
            "介电电荷": 0.2,
            "热力学": 0.2,
            "界面稳定性": 0.4,
            "频率响应": 0.3,
            "光学特性": 0.2,
            "能量效率": 0.1
        },
        "自适应权重": true
    }
}
```

### 高级配置

#### GPU配置
```json
{
    "设备": "cuda",
    "混合精度": true,
    "梯度累积步数": 1
}
```

#### 高效架构配置
```json
{
    "高效架构": true,
    "模型压缩因子": 0.8,
    "注意力机制": true,
    "残差连接": true
}
```

#### 监控配置
```json
{
    "检查点间隔": 100,
    "验证间隔": 50,
    "日志级别": "INFO",
    "可视化": true
}
```

## 环境变量配置

### 设置环境变量
```bash
# 设置PyTorch环境变量
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=1

# 设置Python路径
export PYTHONPATH=/path/to/EFD3D:$PYTHONPATH

# 设置日志级别
export LOG_LEVEL=INFO
```

### Windows环境变量
```cmd
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
set PYTHONPATH=C:\path\to\EFD3D;%PYTHONPATH%
```

## 训练配置示例

### 短训练配置
```bash
python efd_pinns_train.py --mode train \
    --config config/exp_short_config.json \
    --output-dir results_short \
    --epochs 5000 \
    --batch-size 32
```

### 长时训练配置
```bash
python efd_pinns_train.py --mode train \
    --config config/long_run_config.json \
    --output-dir results_long \
    --epochs 100000 \
    --dynamic-weight \
    --weight-strategy adaptive \
    --mixed-precision
```

### 高效架构训练
```bash
python efd_pinns_train.py --mode train \
    --config config/model_config.json \
    --efficient-architecture \
    --model-compression 0.8 \
    --output-dir results_efficient
```

## 故障排除

### 常见问题

#### 1. CUDA内存不足
```bash
# 减小批次大小
--batch-size 16

# 启用模型压缩
--model-compression 0.7

# 使用CPU训练
--device cpu
```

#### 2. 训练不稳定
```bash
# 降低学习率
--learning-rate 0.0001

# 启用梯度裁剪
--grad-clip 1.0

# 使用损失稳定器
--loss-stabilizer
```

#### 3. 依赖冲突
```bash
# 创建干净的虚拟环境
conda create -n efd3d_clean python=3.9
conda activate efd3d_clean

# 重新安装核心依赖
pip install -r requirements.txt --no-deps
```

### 性能优化

#### GPU优化
```bash
# 启用混合精度训练
--mixed-precision

# 设置CUDA设备
CUDA_VISIBLE_DEVICES=0 python efd_pinns_train.py

# 优化内存使用
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64
```

#### 训练优化
```bash
# 使用高效架构
--efficient-architecture

# 启用数据增强
--data-augmentation

# 设置合适的检查点间隔
--checkpoint-interval 500
```

## 验证安装

### 运行测试套件
```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/test_model_contract.py

# 运行训练测试
python -c "from ewp_pinn_optimized_train import progressive_training; print('训练模块导入成功')"
```

### 验证GPU支持
```python
import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU数量: {torch.cuda.device_count()}")
    print(f"当前GPU: {torch.cuda.current_device()}")
    print(f"GPU名称: {torch.cuda.get_device_name()}")
```

## 更新和维护

### 更新项目
```bash
git pull origin main
pip install --upgrade -r requirements.txt
```

### 清理环境
```bash
# 清理训练输出
rm -rf results_* outputs_*

# 清理缓存
rm -rf __pycache__ */__pycache__

# 清理检查点
rm -rf checkpoints
```

通过以上配置，您可以快速开始使用EFD3D项目进行电润湿像素的物理信息神经网络训练。