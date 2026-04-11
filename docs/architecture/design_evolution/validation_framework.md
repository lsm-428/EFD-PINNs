# Stage 2 验证工具与流程更新说明

**最后更新**: 2025-12-22

## 1. `evaluate.py` (核心自动化工具)

为了简化验证流程并提高效率，目前所有 Stage 2 可视化和响应曲线验证统一集中到项目根目录的 `evaluate.py` 中。

### 核心功能
- **目录自动识别**: 可以直接传入输出目录（如 `outputs_pinn_<timestamp>`），或省略参数以自动选择最新的 `outputs_pinn_*` 目录；脚本会自动加载 `best_model.pth` 和 `config.json`。
- **Triad 语义支持**: 全面适配 6D 输入 `(x, y, z, V_from, V_to, t_since)`，支持稳态和动态跳变的统一验证。
- **专业可视化**:
  - 4 联图仪表盘：顶视/侧视 φ 场 + 速度矢量、0→30→0V 动态开口率曲线、稳态 C–V 曲线。
  - φ 场 7×6 网格演化图：覆盖多组升压 / 降压场景。
  - 3D φ=0.5 等值面：界面三维形状重建。

### 使用方法
```bash
# 验证特定训练结果
python evaluate.py outputs_pinn_<timestamp>

# 或省略目录参数，自动选择最新 outputs_pinn_* 目录
python evaluate.py
```

---

## 2. 物理合理性验证 (`test_pinn_complete.py`)

### 更新重点
- **6D 输入兼容性**: 所有 φ 场采样均基于 6D Triad 输入 `(x, y, z, V_from, V_to, t_since)`，与训练阶段完全一致。
- **方波响应分析**: 内置 `0V -> 30V -> 0V` 循环验证，检查升压 / 降压过程的单调性与恢复特性。
- **物理指标导出**: 自动检查 φ ∈ [0,1]、0V 开口率、时间 / 电压单调性以及典型工况下的 φ 场 MAE/MSE，并在终端与图像报告中输出结果。

---

## 3. 结果汇总（当前方案）

目前尚未提供独立的 `summarize_configs.py` 脚本。多组实验对比主要依赖以下信息源：
- `outputs_pinn_*/config.json`: 记录每次实验的配置与物理权重。
- `outputs_pinn_*/training.log`: 记录训练过程中的损失曲线和学习率调度。
- `outputs_pinn_*/pro_dashboard.png`: 由 `evaluate.py` 生成的统一仪表盘，便于肉眼对比不同实验。

后续如果引入独立的汇总脚本，可以在此处补充具体使用方式。

---

## 4. 关键物理指标验证标准 (Acceptance Criteria)

| 指标 | 合格标准 | 验证工具 |
| :--- | :--- | :--- |
| **0V 开口率** | 恒等于 0 (或 < 0.1%) | `test_pinn_complete.py` / `evaluate.py` |
| **20V 开口率** | 67% ± 5% | `evaluate.py` |
| **单调性** | η 随电压 / 时间单调递增 | `test_pinn_complete.py` |
| **响应时间** | τ < 15ms (30V) | `evaluate.py` |
| **物理残差** | φ 场 MAE/MSE 合理，NS/VOF 损失 < 0.01 | 训练日志 / `test_pinn_complete.py` |

---

## 5. 可视化规范
- **φ 场**: 0 (极性液体/透明) 到 1 (油墨/不透明)。
- **切片位置**: z 轴切片固定在 `z=2.5μm` (油墨层中心)。
- **色阶**: 黑色 (油墨) -> 白色 (透明液体)。
