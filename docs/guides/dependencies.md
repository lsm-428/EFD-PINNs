# 依赖管理说明

**更新日期**: 2026-02-04

本项目使用 `pyproject.toml` + `uv` 进行依赖管理。

---

## 1. 依赖文件说明

### `pyproject.toml` (项目根目录)

**用途**: 项目元数据与依赖定义（现代 Python 标准）

```bash
# 安装项目及其依赖
uv sync

# 仅安装核心依赖
uv sync --no-dev

# 安装可选依赖组
uv sync --extra testing
uv sync --extra full
uv sync --extra web
```

**包含内容**:
- 核心依赖: `torch==2.7.1`, `numpy`, `scipy`, `matplotlib` 等
- 可选依赖组: `testing`, `full`, `web`, `monitoring`

---

## 2. 安装方式

### 方式 1: 完整安装 (推荐)

```bash
# 安装所有依赖（包括开发工具）
uv sync
```

### 方式 2: 最小安装 (仅核心依赖)

```bash
uv sync --no-dev
```

### 方式 3: 安装特定可选依赖

```bash
# 安装测试依赖
uv sync --extra testing

# 安装完整依赖 (含可视化)
uv sync --extra full

# 安装 Web API 依赖
uv sync --extra web
```

---

## 3. 当前环境配置

### 核心依赖 (必须)

| 包名 | 版本 | 用途 |
|------|------|------|
| Python | 3.12-3.13 | 运行环境 |
| torch | 2.7.1 | 深度学习框架 |
| torchvision | 0.22.1 | 计算机视觉工具 |
| torchaudio | 2.7.1 | 音频处理工具 |
| numpy | >=2.3.0 | 数值计算 |
| scipy | >=1.16.0 | 科学计算 |
| scikit-image | >=0.25.0 | 图像处理 |
| matplotlib | >=3.10.0 | 2D 可视化 |
| seaborn | >=0.13.0 | 统计可视化 |
| pyyaml | >=6.0 | YAML 解析 |
| tqdm | >=4.67.0 | 进度条 |

### 可选依赖

| 组名 | 包含内容 | 用途 |
|------|---------|------|
| `full` | pandas, pyvista, plotly | 完整数据分析与可视化 |
| `testing` | pytest, hypothesis | 测试框架 |
| `monitoring` | tensorboard | 训练监控 |
| `web` | fastapi, pydantic, uvicorn, streamlit | Web API 与 UI |

---

## 4. 使用 uv 管理依赖

### 安装新依赖

```bash
# 添加核心依赖
uv add package_name

# 添加开发依赖
uv add --dev package_name

# 添加可选依赖
uv add --extra testing package_name
```

### 更新依赖

```bash
# 更新单个包
uv add package_name==latest

# 升级所有包
uv sync --upgrade
```

### 导出 requirements.txt

```bash
uv export -o requirements.txt
```

---

## 5. CUDA 配置

PyTorch 通过 `pyproject.toml` 的 `[tool.uv.sections]` 配置 CUDA 支持：

```toml
[[tool.uv.index]]
name = "pytorch"
url = "https://download.pytorch.org/whl/cu118"

[tool.uv.sources]
torch = [{index = "pytorch"}]
```

安装时自动从 PyTorch 官方源下载 CUDA 11.8 版本。

---

## 6. 常见问题

### Q1: CUDA 不可用

```bash
# 检查 CUDA 是否可用
python -c "import torch; print(torch.cuda.is_available())"
```

如返回 `False`，确保使用 `uv sync` 安装（已配置 CUDA 源）。

### Q2: uv 安装缓慢

```bash
# 使用清华镜像源 (已预配置)
uv sync
```

### Q3: 安装特定版本

```bash
uv add torch==2.7.1
uv add numpy==2.3.4
```

---

## 7. 验证安装

```bash
# 验证核心依赖
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"

# 运行测试
pytest tests/ -v --tb=short
```

---

## 🔗 相关文档

- [README.md](../../README.md) - 项目快速开始
- [installation.md](installation.md) - 安装配置指南
- [configuration_guide.md](configuration_guide.md) - 配置系统指南
- [pyproject.toml](../../pyproject.toml) - 依赖定义文件

---

## 📝 变更历史

### 2026-02-04
- ✅ 重写文档，使用 `pyproject.toml` + `uv` 替代过时的 `requirements.txt`
- ✅ 更新 PyTorch 版本: 2.5.1 → 2.7.1
- ✅ 添加 uv 工作流说明
- ✅ 更新可选依赖组说明

### 2026-01-15
- 旧版文档，基于 `environment.yml` 和 `requirements.txt`
