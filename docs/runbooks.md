# 场景 Runbook（操作手册）

## 直流阶跃响应
- 拷贝模板：`cp docs/config_samples/dc_step_config.json config/model_config.json`
- 训练：`python ewp_pinn_optimized_train.py --mode train --config config/model_config.json --output-dir results_enhanced`
- 诊断：`python scripts/generate_constraint_report.py ... --applied_voltage 20.0`
- 可视化：`python scripts/visualize_constraint_report.py ...`
- 验收：Young–Lippmann 残差均值≤1e-2，稳定性指数≥0.8

## 交流频扫
- 拷贝模板：`cp docs/config_samples/ac_sweep_config.json config/model_config.json`
- 长期训练：`python long_term_training.py --output_dir results_long_run --epochs 30000 --dynamic_weight --weight_strategy adaptive`
- 诊断：`python scripts/generate_constraint_report.py ... --applied_voltage 5.0 --time 1.0`
- 可视化：`python scripts/visualize_constraint_report.py ...`
- 验收：目标频段幅/相误差均值≤5%

## 100k 长期训练计划（Warmup+CosineLR + 动态权重）
- 配置建议：
  - 轮次：`--epochs 100000`，预热 `--warmup_epochs 1000`
  - 学习率：`--lr 1e-5 --min_lr 1e-8`
  - 批次：`--batch_size 32`
  - 物理权重：`--physics_weight 2.0 --dynamic_weight --weight_strategy adaptive`
  - 检查点与验证：`--checkpoint_interval 5000 --validation_interval 1000`
- 运行：
  - `python long_term_training.py --output_dir results_long_run --epochs 100000 --lr 1e-5 --min_lr 1e-8 --warmup_epochs 1000 --batch_size 32 --physics_weight 2.0 --dynamic_weight --weight_strategy adaptive --checkpoint_interval 5000 --validation_interval 1000`
- 输出归类：
  - 报表：`results_long_run/reports/`（周期诊断 JSON 与 training_history.json）
  - 可视化：`results_long_run/visualizations/`（训练曲线与约束图）
  - 日志：`results_long_run/logs/`
- 验收与监控：
  - 每 1000 轮检查约束诊断的 `young_lippmann` 与 `frequency_response` 残差是否收敛；曲线平稳后再调低学习率
  - 观察 `dynamic_weight` 时序是否在中后期趋于稳定；如振荡增大，降低 `adjustment_factor`

## 训练稳健性修复与证据
- 批次物理点采样与对齐：每个 batch 采样/重复 `physics_points` 为 `batch_phys`，大小与 `pred.shape[0]` 一致，设置 `requires_grad_(True)` 并与预测设备对齐
- 梯度与二阶项稳健化：二阶项采用拉普拉斯近似，梯度失败时回退零张量（安全后备函数）
- 坐标子集策略：所有物理约束计算仅用 `coords3 = x[:, :3]` 的空间维度，避免 62 维坐标误用
- 频率响应容错：材料参数为标量时直接使用；电压张量化并按 batch 广播；仅在提供频率时计算 Debye 模型
- 日志与诊断：周期在 `reports/` 输出 JSON 报表并在 `visualizations/` 输出图表；训练日志在 `logs/` 持续记录

## 短训（3D映射与双相软约束）
- 命令示例：
  - `python long_term_training.py --output_dir results_short_run_3d --epochs 20 --lr 5e-5 --min_lr 1e-6 --warmup_epochs 5 --batch_size 32 --physics_weight 1.0 --dynamic_weight --weight_strategy adaptive --checkpoint_interval 10 --validation_interval 10 --device cpu --use_3d_mapping --num_samples 1000`
- 要点：
  - 启用 `--use_3d_mapping`，从 `generate_pyvista_3d.py` 生成阶段3输入；用 `--num_samples` 控制规模以避免OOM
  - 约束层包含双相软约束：`volume_conservation` 与 `volume_consistency`，默认权重 0.05
  - 输出归类至 `results_short_run_3d/reports` 与 `visualizations`；查看 `constraint_residual_stats.png` 与 `constraint_weight_series.png`
- 复现记录：
  - 修复 `np` 加载冲突、将设备切至 CPU、降低样本数，短训流程跑通，物理路径执行、图表生成正常

## 接触线滞后
- 拷贝模板：`cp docs/config_samples/contact_line_hysteresis_config.json config/model_config.json`
- 训练：`python ewp_pinn_optimized_train.py --mode train --config config/model_config.json --output-dir results_enhanced`
- 诊断：`python scripts/generate_constraint_report.py ... --contact_line_velocity 0.02`
- 可视化：`python scripts/visualize_constraint_report.py ...`
- 验收：滞后差值与钉扎阈值在目标范围内，稳定性指数≥0.8

## 温升工况
- 拷贝模板：`cp docs/config_samples/thermal_rise_config.json config/model_config.json`
- 长期训练：`python long_term_training.py --output_dir results_long_run --epochs 30000 --dynamic_weight --weight_strategy adaptive`
- 诊断：`python scripts/generate_constraint_report.py ... --temperature 330.0`
- 可视化：`python scripts/visualize_constraint_report.py ...`
- 验收：能效≥0.7，温敏系数稳定
