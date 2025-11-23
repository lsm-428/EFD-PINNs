# 场景试验协议（EWPINN 电润湿像素）

**直流阶跃响应**
- 步骤：设定电压阶跃（如 0→20 V），记录 `phi,E_x,h,theta` 的时域响应；窗口 0–200 ms。
- 指标：响应时间常数、稳态接触角、界面稳定性指数。
- 模型建议：提高 `young_lippmann` 与 `interface_stability` 权重，训练轮次 1–2e4。

**交流频扫**
- 步骤：对数频点覆盖 10 Hz–100 kHz；每频点记录幅/相响应，输入包含 `frequency_hz` 与电压幅值。
- 指标：幅频/相频曲线拐点、介电弛豫时间常数、能效曲线。
- 模型建议：提高 `frequency_response` 与 `dielectric_charge` 权重，训练轮次 2–3e4。

**接触线滞后验证**
- 步骤：电压升/降扫描，记录前进/后退接触角与接触线速度；统计滞后环面积。
- 指标：滞后差值、钉扎阈值、稳定性指数。
- 模型建议：提高 `contact_line_dynamics` 与 `interface_stability` 权重，训练轮次 2e4。

**温升工况**
- 步骤：设定环境或器件温度上升（如 300→330 K），记录 `theta,h,energy_efficiency` 随温度的变化。
- 指标：温敏系数、能效曲线、稳定性裕度。
- 模型建议：提高 `thermodynamic` 与 `energy_efficiency` 权重，训练轮次 2–3e4。

**数据记录与验收**
- 输入规范：按 `docs/feature_mapping.md` 进行归一化与字段齐备检查。
- 输出记录：保存物理单位或明确归一化规则，确保跨场景可比较。
- 诊断：使用 `scripts/generate_constraint_report.py` 生成约束诊断报表，并用 `scripts/visualize_constraint_report.py` 出具图表。
- 敏感性与DOE：使用 `scripts/sensitivity_doe.py` 输出参数-指标关系与设计建议，结合不同权重策略对比。
