# EWPINN 电润湿像素物理信息神经网络（EFD3D）

**项目概述**
- 面向电润湿显示像素的物理信息神经网络（PINN）研究型项目，融合数据拟合与物理约束，支持高效架构、长时训练、性能监控与结果可视化。
- 主要入口脚本覆盖训练、测试、推理与增强训练流程，输出包含模型权重、训练历史、验证结果与可视化图。

**项目背景与目标**
- 电润湿像素设计涉及几何结构、介电层、流体与电场等多物理耦合，传统仅依赖数据拟合易出现物理不一致与外推不稳
- 本项目将工艺/结构/材料参数系统化映射为 62 维输入，将可验收的工程指标定义为 24 维输出，在训练中显式纳入物理约束残差，以提升物理一致性与泛化稳定性
- 面向四类专业场景（直流阶跃、交流频扫、接触线滞后、温升工况）提供可复现的训练、诊断与报告输出，以支撑设计决策与参数选型

**方法与架构概要**
- 模型体系：基础架构与高效架构（残差与注意力），统一预测接口与字典输出，便于约束层消费
- 物理约束类别：Navier–Stokes、Young–Lippmann、接触线动力学、介电电荷、热力学、界面稳定性、频率响应、光学、能效约束，统一由 `PINNConstraintLayer` 聚合并加权
- 训练策略：渐进式训练（阶段学习率与批量）、动态物理权重（自适应平衡数据拟合与约束残差）、Warmup+CosineLR 长时训练；数值稳定策略包括安全梯度后备、裁剪与正则
- 数据与诊断：训练周期自动生成约束诊断报表与图表（残差均值/方差与权重时序），输出到 `reports/` 与 `visualizations/`，日志到 `logs/`

**约束与作用摘要**
- 连续性与动量（Navier–Stokes）：保证质量守恒与动量平衡；对速度场 `u,v,w` 与压力 `p` 的一二阶导数残差进行约束
- Young–Lippmann：电压-接触角关系的显式约束，用于电润湿响应的物理一致性校验
- 接触线动力学：前进/后退接触角滞后与钉扎效应约束，稳定边缘与响应
- 介电电荷积累：泄漏电流与松弛过程约束，限制介电层电荷密度与动态变化
- 热力学：温度对表面张力与粘度的影响、焦耳热与能量守恒的空间残差
- 界面稳定性：界面曲率、梯度、平滑性与面积变化约束，抑制不稳定与高频缺陷
- 频率响应：Debye 模型与位移/传导电流比例约束，校验交流电场下的一致性
- 光学特性：反射/透射、对比度与界面锐利度约束，辅助成像与显示指标
- 能量效率：功耗限制、能量转换效率与粘性耗散约束，提升设计的能效比

**输入/输出映射摘要**
- 输入 62 维分组：时空与电压、几何结构、材料与界面、电场与介电、流体动力学、时间动态、电润湿特异参数；单位与归一化见 `docs/feature_mapping.md`
- 输出 24 维分组：物理场（p、u/v/w、涡量、电势、电场），界面与接触线（h、曲率、斜率、接触角、半径、速度等），工程性能（响应时间、稳定性、能效）

**诊断报表解读**
- 报表字段：`loss_mean/loss_std/raw_mean/raw_std/weight_series`；键名标准见 `docs/constraints_schema.md`
- 重点观察：
  - `young_lippmann` 残差均值的下降与稳定，作为电压-接触角一致性指标
  - `frequency_response` 在目标频段的残差趋势，评估频响一致性
  - `interface_stability` 的平稳性与曲线波动，判断界面稳定性
  - 动态权重 `weight_series` 的中后期趋稳情况，避免振荡

**敏感性分析与建议**
- 使用 `scripts/sensitivity_doe.py` 扫描关键参数（如 `d/ε_r/γ/θ0`），输出 `doe_results.json`
- 建议比较不同权重策略（固定/自适应/阶段式）下指标变化，选择兼顾收敛速度与物理一致性的方案
- 将 DOE 结果与场景诊断报表结合，形成对设计参数的影响排序与选型建议

**关键指标与阈值（建议）**
- Young–Lippmann 一致性：`|cosθ_pred - cosθ_theory|` 均值 ≤ `1e-2`
- 频率响应偏差：目标频段的幅/相误差均值 ≤ `5%`
- 界面稳定性指数 ≥ `0.8`；能量效率 ≥ `0.7`

**场景策略速查**
- 直流阶跃：提高 `young_lippmann` 与 `interface_stability` 权重；中后期使用动态权重平衡
- 交流频扫：关注 `frequency_response` 与 `dielectric_charge`；长期训练启用动态权重
- 接触线滞后：强化 `contact_line_dynamics` 与稳定性；设计升/降扫描与滞后环验证
- 温升工况：强化 `thermodynamic` 与 `energy_efficiency`；关注温敏系数与能效曲线

**设计到训练的工作流**
- 选择场景模板并配置工况（电压、频率、温度、接触线速度等），将工艺与材料取值映射到 62 维输入，校验单位与范围
- 运行训练（优化/增强/长期）并启用动态权重，观察训练曲线与诊断图，检查 Young–Lippmann、一致性与频响残差的收敛趋势
- 使用 DOE/敏感性分析脚本比较参数对指标的影响度与不同权重策略的效果，形成设计建议与权衡报告

**目录结构**
- 根目录关键脚本与文档
  - `efd_pinns_train.py`：统一训练/测试/推理入口（整合短训、增强管线、长期训练、动态权重、3D映射、ONNX导出）
  - （旧脚本已迁移至 `scripts/legacy_backup/`：ewp_pinn_optimized_train.py / run_enhanced_training.py / long_term_training.py）
  - `evaluate_model.py`、`monitor_training.py`、`monitor_training_progress.py`
  - 文档：`README.md`、`README_tests.md`、`quick_start_guide.md`、`ewp_documentation.md`、`PHYSICS_CONSTRAINTS_IMPLEMENTATION.md`
- 目录与重要文件
  - `config/`：`model_config.json`、`enhanced_config.json`、`long_run_config.json`、`exp_short_config.json`、`correct_simple_config.json`
  - `scripts/`：`plot_training_history.py`、`plot_training_history_dir.py`
  - `tests/`：`tests/test_model_contract.py`
  - 结果目录：`results_enhanced/`、`results_long_run/`、`output/`（含多实验子目录）
  - 报告目录：`performance_reports/`

**技术栈与依赖**
- 必需：`torch`、`numpy`、`matplotlib`、`scikit-learn`、`pytest`
- 可选：`onnx`、`onnxruntime`（模型导出与推理）、`pyvista`（3D可视化）、`pandas`
- 安装示例：
  - `pip install torch numpy matplotlib scikit-learn pytest`
  - 可选：`pip install onnx onnxruntime pyvista pandas`

**快速开始（唯一入口）**
- 短训练
  - `python efd_pinns_train.py --mode train --config config/exp_short_config.json --output-dir results_short`
- 长训练 / 动态权重
  - `python efd_pinns_train.py --mode train --config config/long_run_config.json --output-dir results_long_run --epochs 100000 --dynamic_weight --weight_strategy adaptive`
- 高效架构与模型压缩
  - `python efd_pinns_train.py --mode train --config config/exp_short_config.json --efficient-architecture --model-compression 0.8 --output-dir results_short`
- 从检查点恢复训练
  - `python efd_pinns_train.py --mode train --config config/exp_short_config.json --resume --output-dir results_resume`
- 测试与推理
  - `python efd_pinns_train.py --mode test --model-path results_short/final_model.pth --config config/exp_short_config.json`
  - `python efd_pinns_train.py --mode infer --model-path results_short/final_model.pth`

**统一输出结构**
- 训练产物统一写入带时间戳的 `output_dir_YYYYMMDD_HHMMSS/`：
  - 顶层：`final_model.pth`、`dataset.npz`
  - `reports/training_history.json`、`reports/validation_results.json`、`reports/constraint_diagnostics_*.json`
  - `visualizations/training_curves.png`、`visualizations/constraint_*.png`
  - `checkpoints/`：阶段与最佳检查点（best.pth / latest.pth / checkpoint_epoch_*.pth）

**模型与接口**
- 主模型类与预测输出
  - `EWPINN` 主类定义位置：`ewp_pinn_model.py:459`
  - 前向返回字典，含 `main_predictions` 主输出与 `auxiliary_predictions` 辅助输出
  - 预测提取函数：`extract_predictions`（位置：`ewp_pinn_model.py:67`）统一将字典/张量转为主预测张量
- 输入/输出层
  - `ewp_pinn_input_layer.py`：生成示例输入、阶段化维度（默认62维）
  - `ewp_pinn_output_layer.py`：生成示例/随机输出（默认24维）
- 物理约束与损失
  - `ewp_pinn_physics.py`：`PINNConstraintLayer` 与 `PhysicsEnhancedLoss`（集成物理残差）
  - `ewp_consistency_validator.py`：一致性校验辅助
- 训练增强模块
  - 正则化：`ewp_pinn_regularization.py`
  - 性能监控：`ewp_pinn_performance_monitor.py`
  - 自适应超参：`ewp_pinn_adaptive_hyperoptimizer.py`
  - 高效架构：`ewp_pinn_optimized_architecture.py`

**配置说明（JSON示例）**
- 典型字段（文件：`config/model_config.json`）
  - `模型`：`输入维度`、`输出维度`、`隐藏层`、`激活函数`、`批标准化`、`Dropout率`
  - `训练`：`渐进式训练`（阶段数组，含轮次/学习率/批次大小/权重衰减/调度策略/物理约束权重）、`早停配置`、`梯度裁剪`
  - `数据`：`样本数量`、`训练比例`、`验证比例`、`测试比例`、`数据增强`
  - `物理约束`：`启用`、`初始权重`、`物理点数量`、`残差权重`、`自适应权重`
  - 以上键在脚本中被直接读取并深度合并（参考 `ewp_pinn_optimized_train.py:949` 之后的配置加载逻辑）

**输出与结果**
- 常见文件
  - `final_model.pth`：模型与训练元数据
  - `training_history.json`：训练与验证曲线数据
  - `validation_results.json`：最终测试指标（增强/长期训练脚本）
  - `visualizations/training_curves.png`：训练曲线图
  - `dataset.npz`：分割后的 `X_train/y_train/X_val/y_val/X_test/y_test/physics_points`
- 目录示例
  - `results_enhanced/`、`results_long_run/`、`output/experiments/exp_lr_*_clip_*/`

**可视化与脚本**
- 绘制训练历史
  - `python scripts/plot_training_history.py ./results_enhanced/training_history.json`
- 性能报告
  - 训练过程中自动生成报告与诊断（参考 `ewp_pinn_performance_monitor.py` 集成，由训练脚本调用）

**模型导出**
- 训练脚本支持导出ONNX（安装 `onnx` 与 `onnxruntime` 后）
  - 训练时添加 `--export-onnx`，或在保存模型时启用导出（参考 `ewp_pinn_optimized_train.py:2048` 起的 `save_model`）

**测试与契约**
- 运行测试
  - `pytest -q`
- 合同式断言（文件：`tests/test_model_contract.py`）
  - `forward` 返回包含 `main_predictions` 的字典，并与输出维度一致（参考 `tests/test_model_contract.py:16`）
  - `extract_predictions` 能从字典或张量正确抽取主预测张量（参考 `tests/test_model_contract.py:36`）

**故障排查**
- ONNX导出失败：安装 `onnx` 与 `onnxruntime`；确保 `HAS_ONNX` 为 True（参考 `ewp_pinn_optimized_train.py:2495`）
- 损失出现 NaN/Inf：脚本已集成数值稳定保护与裁剪；检查输入范围与标准化配置（参考 `LossStabilizer` 与 `DataNormalizer`）
- GPU显存不足：降低批次大小，或使用 `--model-compression <factor>` 压缩高效架构
- 路径错误：优先使用 `results_enhanced/checkpoints/best_model.pth` 或训练运行生成的目录（如 `checkpoints_optimized_*`）

**许可证与致谢**
- 许可证：MIT
- 致谢：感谢相关开源库与社区支持

**附录：关键代码位置参考**
- 模型主类：`ewp_pinn_model.py:459`
- 预测提取：`ewp_pinn_model.py:67`
- 优化训练入口：`ewp_pinn_optimized_train.py:2413`
- 增强训练入口：`run_enhanced_training.py:2420`
- 长期训练入口：`long_term_training.py:615`

**CLI 参考（优化训练）**
- 脚本：`ewp_pinn_optimized_train.py`（入口位置 `ewp_pinn_optimized_train.py:2413`）
- 参数（定义位置 `ewp_pinn_optimized_train.py:2261` 起）
  - `--mode {train,test,infer}`：运行模式
  - `--config <path>`：配置文件路径（默认 `model_config.json`）
  - `--resume`：从检查点恢复训练
  - `--checkpoint <path>`：恢复训练的检查点文件
  - `--mixed-precision`：启用混合精度训练（默认 True）
  - `--model-seed <int>`：模型初始化种子（用于集成学习）
  - `--efficient-architecture`：使用高效架构（残差+注意力）（默认 True）
  - `--model-compression <float>`：压缩因子（小于 1.0 减少参数）
  - `--model-path <path>`：测试/推理模型路径（默认 `models/best_model.pth`）
  - `--export-onnx`：导出为 ONNX
  - `--num-samples <int>`：生成样本数量（默认 200）
  - `--data-augmentation`：开启数据增强（默认 True）
  - `--output-dir <path>`：输出目录（默认 `outputs`）
  - `--device {cpu,cuda,cuda:0,...}`：运行设备

**CLI 参考（增强训练）**
- 脚本：`run_enhanced_training.py`（入口位置 `run_enhanced_training.py:2420`）
- 参数（定义位置 `run_enhanced_training.py:566` 起）
  - `--config <path>`：配置文件或 Python 配置模块路径
  - `--output_dir <path>`：输出目录（默认 `./results_enhanced`）
  - `--device {cpu,cuda,cuda:0,...}`：计算设备
  - `--num_samples <int>`：训练样本数量（默认 10000；快速模式自动改为 100）
  - `--resume`：从检查点恢复
  - `--checkpoint <path>`：恢复训练的检查点路径
  - `--mixed_precision`：混合精度训练（默认 True）
  - `--generate_data_only`：仅生成数据不训练
  - `--validate_only`：仅验证模型不训练
  - `--model_path <path>`：验证时使用的预训练模型路径
  - `--visualize`：生成训练可视化
  - `--quick_run`：极少训练量快速模式（自动创建 `output_dir/quick_run`）
  - `--clean`：清理输出目录中的训练产物
  - `--clean_all`：深度清理（含缓存与日志）
  - `--seed <int>`：随机种子（默认 42）
  - `--deterministic`：确定性训练（默认 True）
  - `--debug_capture_nan`：出现 NaN/Inf 时保存批次与模型以便调试
  - `--clip_grad <float>`：覆盖梯度裁剪阈值
  - `--override_lr <float>`：覆盖当前阶段学习率

**CLI 参考（长期训练）**
- 脚本：`long_term_training.py`（入口位置 `long_term_training.py:615`）
- 参数（定义位置 `long_term_training.py:58` 起）
  - `--output_dir <path>`：输出目录（默认 `./long_term_run`）
  - `--config <path>`：配置文件路径
  - `--epochs <int>`：训练轮次（默认 100000）
  - `--lr <float>`：初始学习率（默认 `1e-5`）
  - `--min_lr <float>`：最小学习率（默认 `1e-8`）
  - `--warmup_epochs <int>`：预热轮次（默认 1000）
  - `--batch_size <int>`：批次大小（默认 32）
  - `--physics_weight <float>`：物理损失权重（默认 2.0）
  - `--dynamic_weight`：启用动态物理权重调整
  - `--weight_strategy {adaptive,stage_based,loss_ratio}`：动态权重策略
  - `--checkpoint_interval <int>`：检查点保存间隔（默认 5000）
  - `--validation_interval <int>`：验证间隔（默认 1000）
  - `--resume <path>`：从检查点恢复
  - `--seed <int>`：随机种子（默认 42）
  - `--device {auto,cpu,cuda}`：设备选择（默认 `auto`）

**配置字段详解（优化训练 JSON）**
- 文件示例：`config/model_config.json`；加载与合并逻辑参考 `ewp_pinn_optimized_train.py:949`
- `模型`
  - `输入维度`：默认 62
  - `输出维度`：默认 24
  - `隐藏层`：如 `[128,64,32]`
  - `激活函数`：`ReLU|LeakyReLU|GELU|SiLU`
  - `批标准化`：`true|false`
  - `Dropout率`：如 `0.1`
- `训练`
  - `渐进式训练`：阶段数组，每项包含 `名称`、`轮次`、`学习率`、`批次大小`、`权重衰减`、`优化器`、`调度策略`、`调度参数`、`描述`、`物理约束权重`
  - `早停配置`：`启用`、`耐心值`、`最小改进`、`恢复最佳模型`
  - `梯度裁剪`：如 `1.0`
  - `梯度累积步数`：如 `1`
- `数据`
  - `样本数量`、`数据增强`、`训练/验证/测试比例`
- `物理约束`
  - `启用`、`初始权重`、`权重衰减`、`物理点数量`
  - `残差权重`：如 `{连续性:1.0, 动量_u:0.1, ...}`
  - `自适应权重`：`true|false`
- `正则化配置`（可选）
  - `L1正则化系数`、`L2正则化系数`、`Dropout率`、`使用权重裁剪`、`权重裁剪阈值`、`使用谱归一化`、`启用早停`、`早停耐心值`
- 生成数据细节与标准化器默认：参考 `ewp_pinn_optimized_train.py:630` 与 `ewp_pinn_optimized_train.py:347`

**数据与结果文件格式**
- `dataset.npz`
  - `X_train,y_train,X_val,y_val,X_test,y_test,physics_points`
  - 生成位置：增强训练 `run_enhanced_training.py:1031`，长期训练 `long_term_training.py:250`
- `training_history.json`
  - 增强训练：包含 `train_losses,val_losses,physics_losses,lr_history,epochs_completed`（生成位置 `run_enhanced_training.py:1985`）
  - 长期训练：包含 `train_losses,val_losses,physics_losses,lr_history,epochs_completed`（生成位置 `long_term_training.py:607`）
- `validation_results.json`
  - 键：`test_loss,physics_loss,normalized_loss,test_samples,timestamp`（生成位置 `run_enhanced_training.py:2407`）
- `final_model.pth`
  - 保存模型 `state_dict`、训练历史与配置信息（参考各脚本保存段落）
- 可视化
  - `visualizations/training_curves.png`（增强训练 `run_enhanced_training.py:2002`；优化训练 `ewp_pinn_optimized_train.py:1859`）

**模块概览与关键函数**
- 模型与工具
  - `ewp_pinn_model.py`：`EWPINN` 主类（`ewp_pinn_model.py:459`），`extract_predictions`（`ewp_pinn_model.py:67`）
  - `ewp_pinn_optimized_architecture.py`：高效架构构造与建议
  - `ewp_pinn_regularization.py`：高级正则化与 DropConnect 应用
  - `ewp_pinn_performance_monitor.py`：训练监控、诊断与报告导出
  - `ewp_pinn_adaptive_hyperoptimizer.py`：自适应超参数优化器
  - `ewp_pinn_physics.py`：`PINNConstraintLayer` 与 `PhysicsEnhancedLoss`
  - `ewp_pinn_input_layer.py`、`ewp_pinn_output_layer.py`：输入/输出层
  - `ewp_data_interface.py`：统一数据集与 `DataLoader` 适配器
- 训练脚本
  - 优化训练：`ewp_pinn_optimized_train.py`（渐进式训练 `progressive_training`）
  - 增强训练：`run_enhanced_training.py`（四阶段训练与可视化）
  - 长期训练：`long_term_training.py`（Warmup+CosineLR、动态物理权重）

**工作流示例（优化训练）**
- 准备配置 `config/model_config.json`
- 运行训练：`python ewp_pinn_optimized_train.py --mode train --config config/model_config.json --output-dir results_enhanced`
- 观察输出：`results_enhanced` 内 `final_model.pth`、`training_history.json`、`visualizations/training_curves.png`
- 测试/推理：使用 `--mode test`、`--mode infer` 加载 `best_model.pth`

**代码契约与测试**
- 合同测试文件：`tests/test_model_contract.py`
  - 要求 `forward` 返回包含 `main_predictions` 键的字典，并且维度匹配（参考 `tests/test_model_contract.py:16`）
  - `extract_predictions` 能从字典或张量抽取主预测（参考 `tests/test_model_contract.py:36`）

**物理约束与实现说明**
- 约束说明：`PHYSICS_CONSTRAINTS_IMPLEMENTATION.md`
- 物理损失集成：优化训练（`ewp_pinn_optimized_train.py:1085` 起）、增强训练（`run_enhanced_training.py:221` 起）
- 动态权重（长期训练）：`ewp_pinn_dynamic_weight.py` 与 `long_term_training.py:327`

**常见问题与排查**
- ONNX 导出失败：安装 `onnx` 与 `onnxruntime`，确保 `HAS_ONNX` 为 True（`ewp_pinn_optimized_train.py:2495`）
- 损失出现 NaN/Inf：启用 `--mixed-precision`；检查输入范围与标准化器（`DataNormalizer` 定义 `ewp_pinn_optimized_train.py:347`）；必要时降低学习率或增大耐心值
- 显存不足：降低批量、关闭注意力或使用 `--model-compression`；使用 CPU 作为后备
- 路径与数据集：优先使用运行生成的结果目录（如 `results_enhanced/`、`results_long_run/`、`output/experiments/`），`dataset.npz` 含分割数据与物理点

**文档索引**
- 设计指南：`docs/design_guide.md`
- 特征映射详表：`docs/feature_mapping.md`
- 约束键与诊断字段：`docs/constraints_schema.md`
- 场景配置模板：`docs/config_samples/`
  - 直流阶跃：`docs/config_samples/dc_step_config.json`
  - 交流频扫：`docs/config_samples/ac_sweep_config.json`
  - 接触线滞后：`docs/config_samples/contact_line_hysteresis_config.json`
  - 温升工况：`docs/config_samples/thermal_rise_config.json`
- 约束诊断：`scripts/generate_constraint_report.py`
- 约束诊断可视化：`scripts/visualize_constraint_report.py`
- 场景试验协议：`docs/experiment_protocols.md`
- 物理约束实现：`docs/physics_constraints.md`
- CLI 参考：`docs/cli_reference.md`
- Runbook：`docs/runbooks.md`
- 文档总索引：`docs/index.md`

**更新说明**
- 新增专业文档与脚本：
  - `docs/design_guide.md` 电润湿像素设计指南（输入62维/输出24维映射、参数建议、频扫设计与模板索引）
  - `docs/feature_mapping.md` 输入输出特征映射与单位规范（数据接口标准化）
  - `docs/constraints_schema.md` 约束残差与诊断字段标准（保证报表结构一致）
  - `docs/experiment_protocols.md` 场景试验协议（直流、交流、滞后、温升）
  - `scripts/generate_constraint_report.py` 生成约束诊断报表（残差均值/方差与权重时序）
  - `scripts/visualize_constraint_report.py` 约束诊断图表输出（统计与权重时序）
  - `scripts/run_scenario.py` 一键场景运行（拷贝模板→训练→诊断→可视化闭环）
  - `scripts/sensitivity_doe.py` DOE/敏感性分析（参数扫描与设计建议）
- 清理不必要文档以聚焦专业内容：`training_comparison_report.md`、`experiment_summary_report.md`、`training_tracking_report.md`、`quick_start_guide.md`、`ewp_documentation.md`
- 直流阶跃模板训练已完成并验证：模型与训练历史输出于 `results_enhanced/`，诊断报表与图表输出于 `results_enhanced/consistency_data/`

**场景模板运行示例（按专业建议）**
- 直流阶跃响应（强化 Young–Lippmann 与界面稳定性）
  - `cp docs/config_samples/dc_step_config.json config/model_config.json`
  - `python ewp_pinn_optimized_train.py --mode train --config config/model_config.json --output-dir results_enhanced`
  - `python scripts/generate_constraint_report.py --model_path results_enhanced/checkpoints/best_model.pth --config_path config/model_config.json --dataset_path results_enhanced/dataset.npz --output_dir results_enhanced/consistency_data --applied_voltage 20.0`
- 交流频扫（关注频率响应与介电电荷）
  - `cp docs/config_samples/ac_sweep_config.json config/model_config.json`
  - `python long_term_training.py --output_dir results_long_run --epochs 30000 --dynamic_weight --weight_strategy adaptive`
  - `python scripts/generate_constraint_report.py --model_path results_long_run/final_model.pth --config_path config/model_config.json --dataset_path results_long_run/dataset.npz --output_dir results_long_run/consistency_data --applied_voltage 5.0 --time 1.0`
  - `python scripts/visualize_constraint_report.py --report_path results_long_run/consistency_data/constraint_diagnostics.json --output_dir results_long_run/consistency_data`
  - 报表键定义参考：`docs/constraints_schema.md`
- 接触线滞后验证（强化接触线动力学与稳定性）
  - `cp docs/config_samples/contact_line_hysteresis_config.json config/model_config.json`
  - `python ewp_pinn_optimized_train.py --mode train --config config/model_config.json --output-dir results_enhanced`
  - `python scripts/generate_constraint_report.py --model_path results_enhanced/checkpoints/best_model.pth --config_path config/model_config.json --dataset_path results_enhanced/dataset.npz --output_dir results_enhanced/consistency_data --contact_line_velocity 0.02`
  - `python scripts/visualize_constraint_report.py --report_path results_enhanced/consistency_data/constraint_diagnostics.json --output_dir results_enhanced/consistency_data`
- 温升工况（强化热力学与能效）
  - `cp docs/config_samples/thermal_rise_config.json config/model_config.json`
  - `python long_term_training.py --output_dir results_long_run --epochs 30000 --dynamic_weight --weight_strategy adaptive`
  - `python scripts/generate_constraint_report.py --model_path results_long_run/final_model.pth --config_path config/model_config.json --dataset_path results_long_run/dataset.npz --output_dir results_long_run/consistency_data --temperature 330.0`
  - `python scripts/visualize_constraint_report.py --report_path results_long_run/consistency_data/constraint_diagnostics.json --output_dir results_long_run/consistency_data`
