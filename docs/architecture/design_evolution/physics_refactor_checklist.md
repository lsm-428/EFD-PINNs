## 高优先级 2：物理残差统一实现 Checklist

> **说明**：本 checklist 针对 PINN 训练中的物理残差计算统一，不涉及 Stage 1 解析模型（`EnhancedApertureModel` / `HybridPredictor`）。Stage 1 模型作为边界条件和目标数据源，保持独立。

> **完成状态**：✅ 主体工作已完成（2025-12-31），剩余 2 项为低优先级后续工作（已对照当前代码与测试逐条核查）
> 
> **最后更新**：2026-02-04

### 一、方程实现与统一入口 ✅ 完成

- [x] 在 `PhysicsConstraints` 中集中实现 Navier–Stokes 残差  
      （`compute_navier_stokes_residual`，含混合密度/粘度与 CSF 曲率）
- [x] 在 `PhysicsConstraints` 中集中实现 VOF 残差  
      （`_compute_vof_residual`，使用 (x, y, z, V_from, V_to, t_since) 坐标）
- [x] 在 `PhysicsConstraints` 中集中实现表面张力 / 接触角 / 体积守恒等辅助残差  
      （`compute_surface_tension_residual`、`compute_volume_conservation_residual` 等）
- [x] 在 `PhysicsConstraints` 中提供统一入口  
      `compute_core_residuals(x_phys, predictions, model)`，返回标准残差字典：
      - continuity  
      - momentum_u / momentum_v / momentum_w  
      - vof  
      - surface_tension / volume_conservation 等
- [x] **验证 Stage 1 模型不受影响**（2025-12-31）
      - 代码结构上，两者均从统一物理配置读取关键参数 ✓
      - 已通过 `tests/test_physics_sanity.py` 验证 `EnhancedApertureModel` 与 `HybridPredictor` 的关键推理路径 ✓
      - 运行结果验证与主物理配置参数（θ₀=120°, τ=5ms, ζ=0.8）一致 ✓

### 二、训练路径统一到唯一"物理真相" ✅ 完成

- [x] 在 `PhysicsLoss` 中新增统一入口（2025-12-31）：
      - `compute_all_residuals(model, points)` 调用 `compute_core_residuals`
      - `compute_total_loss(model, points, weights)` 作为推荐入口
      - `_sanitize_tensor()` 进行 NaN/Inf 清理
- [x] 在 `PINNConstraintLayer.compute_physics_loss` 中（2025-12-31）：
      - 将首个调用从  
        `compute_navier_stokes_residual(x_phys, model_predictions, model=model)`  
        调整为  
        `compute_core_residuals(x_phys, model_predictions, model=model)`  
      - 只在需要额外电学 / 热学 / 接触线等拓展约束时，再追加单独的残差计算
      - 在 `residual_weights` 中添加 `vof` 权重 (0.5)
- [x] 检查是否存在对 `compute_navier_stokes_residual`、`_compute_vof_residual` 的直接调用（2025-12-31）：
      - 经检查，这两个方法仅在 `compute_core_residuals` 内部被调用 ✓
      - 无外部直接调用，符合"统一入口"设计 ✓

### 三、PhysicsLoss 收缩为"适配层 + 数值保护壳" ✅ 完成

- [x] 在 `PhysicsLoss` 中新增以下方法（2025-12-31）：
      - `compute_all_residuals(model, points)` - 通过 `compute_core_residuals` 获取残差
      - `compute_total_loss(model, points, weights)` - 应用 log1p 缩放和加权求和
      - `_sanitize_tensor()` - NaN/Inf 清理
- [x] 在 `PhysicsLoss` 中停用以下"自带方程实现"的训练路径（2025-12-31）：
      - `continuity_residual(grads)` - 已在 docstring 中标记 deprecated ✓
      - `vof_residual(grads)` - 已在 docstring 中标记 deprecated ✓
      - `navier_stokes_residual(grads)` - 已在 docstring 中标记 deprecated ✓
      - `surface_tension_residual(grads)` - 已在 docstring 中标记 deprecated ✓
      - 已从 `_compute_physics_equation_loss` 中移除对这些方法的调用 ✓
- [x] 确认训练主流程中（2025-12-31）：
      - 物理损失统一来自 `PINNConstraintLayer.compute_physics_loss(...)` ✓
      - 已移除 `_compute_physics_equation_loss` 中对旧方法的重复调用 ✓
      - 避免同一物理方程被两套实现重复计入损失 ✓

### 四、命名与统计的一致性 ⚠️ 基本完成（1 项待检查）

- [x] 统一物理项 key 命名（2025-12-31）：
      - 训练历史中 `physics` 现在统一从 `pinn_physics` 获取 ✓
      - `pinn_physics` 内部使用 `compute_core_residuals` 的标准 key ✓
      - 标准 key: continuity, momentum_u/v/w, vof, surface_tension, volume_conservation 等
- [x] 调整训练日志与历史记录（2025-12-31）：
      - `self.history["physics"]` 现在只记录 `pinn_physics` 的值 ✓
      - 移除了对旧 key (`continuity`, `vof`, `ns`, `surface_tension`) 的聚合 ✓
- [ ] 确认损失打印 / 可视化脚本中，同一物理量只对应一个清晰的 key
      - 需要检查 evaluate.py 和其他可视化脚本

### 五、数值 sanity check ✅ 完成

- [x] 数值 sanity check（2025-12-31）：
      - 利用代表性样例验证：θ₀=120°, εᵣ=12.0, σ=0.045 在配置与约束实现中统一 ✓
      - 在 `PhysicsConstraints` 中检查 N-S + VOF + 表面张力 + 体积守恒 的集中实现 ✓
      - 通过小批量物理点验证模型参数能正确接收梯度 ✓
      - 在代表性样例下检查无 NaN/Inf ✓
      - 确认旧方法保留但不会破坏现有训练脚本 ✓
- [x] 编写一个小测试脚本或单元测试（2025-12-31）：
      - 已创建文件：`tests/test_physics_sanity.py`
      - 包含 10 个测试用例，全部通过 ✓
      - 覆盖：参数一致性、核心残差、零场景、接口兼容性、Stage 1 模型
- [x] 检查核心残差的量级（2025-12-31）：
      - 在简单物理解场景（u=v=w=0, p=常数, φ=常数）下，continuity 残差接近 0 ✓
      - 测试用例 `test_static_uniform_field` 验证通过 ✓
- [x] 若有偏大残差，针对相应方程单独排查梯度链和无量纲化是否一致
      - 当前未发现异常偏大残差，无需额外排查；如后续出现再单独处理

### 六、端到端训练验证 ⚠️ 基本完成（1 项需长期验证）

- [x] 在完成上述接口统一后，运行一次短训练（100 epoch）（2025-12-31）：
      - 确认总损失不会出现 NaN / Inf / 爆炸 ✓
      - 最终损失: 335.66，物理损失: 0.71
      - 各物理项损失正常记录（continuity, momentum, vof, volume_conservation 等）
- [ ] 对比重构前后的：
      - 开口率曲线 / 接触角响应 / 典型物理量的时间演化
      - 确认没有明显物理退化
      - 注：需要更长时间训练才能进行有意义的对比
- [x] 旧 PhysicsLoss 方程实现已标记为 deprecated（2025-12-31）：
      - `continuity_residual`, `vof_residual`, `navier_stokes_residual`, `surface_tension_residual`
      - 已从训练主路径中移除调用 ✓
      - 保留方法定义用于兼容性和 debug ✓


---

## 完成总结

### 已完成的核心工作（2025-12-31）

本 checklist 已对照 `src/physics/constraints.py`、`src/models/pinn_two_phase.py` 和 `tests/test_physics_sanity.py` 等代码与测试逐条核查，当前勾选状态与实现一致。

| 模块 | 修改内容 | 文件 |
|------|----------|------|
| PhysicsConstraints | 新增 `compute_core_residuals` 统一入口 | `src/physics/constraints.py` |
| PINNConstraintLayer | 改用 `compute_core_residuals`，添加 `vof` 权重 | `src/physics/constraints.py` |
| PhysicsLoss | 新增 `compute_all_residuals`/`compute_total_loss`，旧方法标记 deprecated | `src/models/pinn_two_phase.py` |
| Trainer | `_compute_physics_equation_loss` 移除重复调用，统一物理损失来源 | `src/models/pinn_two_phase.py` |
| 测试 | 新增 10 个 sanity check 测试用例 | `tests/test_physics_sanity.py` |

### 剩余低优先级工作

1. **四-3**：检查 `evaluate.py` 等可视化脚本中的 key 命名一致性
2. **六-2**：长时间训练后对比重构前后的物理量演化

### 验证命令

```bash
# 运行 sanity check 测试
conda activate efd
python -m pytest tests/test_physics_sanity.py -v

# 验证模块导入
python -c "from src.physics.constraints import PhysicsConstraints, PINNConstraintLayer; print('OK')"
python -c "from src.models.pinn_two_phase import PhysicsLoss, Trainer; print('OK')"

# 运行评估脚本
python evaluate.py
```
