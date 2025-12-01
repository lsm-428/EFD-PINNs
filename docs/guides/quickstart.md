# EFD-PINNs 快速开始指南

**最后更新**: 2025-12-01

## 🚀 快速上手

### 1. 环境准备

```bash
# 激活conda环境
conda activate efd

# 验证环境
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

### 2. 基础训练

```bash
# 激活conda环境
conda activate efd

# 运行训练 (使用当前配置)
python efd_pinns_train.py --mode train --config config_stage2_10k.json --epochs 200
```

**预期输出：**
- 训练进度显示
- 损失曲线实时更新
- 最终模型保存到 `outputs_*/final_model.pth`

### 3. 分析结果

```bash
# 动态响应分析
python analyze_dynamic_response.py --model outputs_*/final_model.pth --output outputs_*/

# 参数验证
python verify_parameters.py
```

## 📋 完整工作流程

### 步骤1：配置训练参数

创建自定义配置文件 `my_config.json`：

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
        "轮次": 1000,
        "学习率": 0.001,
        "批次大小": 32,
        "物理约束权重": 0.1
      }
    ]
  }
}
```

### 步骤2：执行训练

```bash
# 基础训练
python efd_pinns_train.py --mode train --config my_config.json --output-dir my_results

# 高效架构训练（推荐）
python efd_pinns_train.py --mode train --config my_config.json --efficient-architecture --model-compression 0.8 --output-dir my_results

# 长时训练
python efd_pinns_train.py --mode train --config config/long_run_config.json --epochs 100000 --dynamic-weight --output-dir results_long
```

### 步骤3：监控训练进度

```bash
# 实时监控训练进度
python monitor_training.py --log-dir my_results/logs/

# 绘制训练历史
python scripts/plot_training_history.py my_results/training_history.json
```

### 步骤4：结果分析

训练完成后，检查以下文件：

- `my_results/final_model.pth` - 训练好的模型
- `my_results/training_history.json` - 训练历史数据
- `my_results/visualizations/` - 可视化图表
- `my_results/reports/` - 性能报告

## 🔧 常用命令速查

### 训练相关
```bash
# 从检查点恢复训练
python efd_pinns_train.py --mode train --config my_config.json --resume --output-dir my_results

# 启用混合精度训练
python efd_pinns_train.py --mode train --config my_config.json --mixed-precision --output-dir my_results

# 指定GPU设备
python efd_pinns_train.py --mode train --config my_config.json --device cuda:0 --output-dir my_results
```

### 测试与推理
```bash
# 批量测试
python efd_pinns_train.py --mode test --model-path my_results/final_model.pth --config my_config.json

# 单样本推理
python efd_pinns_train.py --mode infer --model-path my_results/final_model.pth --input-data sample_input.json

# 导出ONNX模型
python efd_pinns_train.py --mode train --config my_config.json --export-onnx --output-dir my_results
```

### 性能优化
```bash
# 使用高效架构（推荐）
python efd_pinns_train.py --mode train --efficient-architecture --model-compression 0.8

# 启用梯度检查点（内存优化）
python efd_pinns_train.py --mode train --gradient-checkpointing

# 数据增强
python efd_pinns_train.py --mode train --data-augmentation
```

## 🎯 场景化配置

### 直流阶跃场景
```bash
python efd_pinns_train.py --mode train --config config/dc_step_config.json --output-dir results_dc
```

### 交流频扫场景
```bash
python efd_pinns_train.py --mode train --config config/ac_sweep_config.json --output-dir results_ac
```

### 接触线滞后场景
```bash
python efd_pinns_train.py --mode train --config config/contact_line_config.json --output-dir results_cl
```

## 🚨 故障排除

### 常见问题

**问题1：CUDA内存不足**
```bash
# 解决方案：降低批次大小或启用模型压缩
python efd_pinns_train.py --mode train --batch-size 16 --model-compression 0.7
```

**问题2：训练不稳定（NaN损失）**
```bash
# 解决方案：启用数值稳定化
python efd_pinns_train.py --mode train --safe-training --gradient-clip 1.0
```

**问题3：依赖冲突**
```bash
# 解决方案：创建干净的虚拟环境
python -m venv clean-env
source clean-env/bin/activate
pip install -r requirements.txt
```

### 性能优化建议

1. **GPU训练**：优先使用CUDA设备加速训练
2. **混合精度**：启用混合精度减少内存占用
3. **高效架构**：使用残差连接和注意力机制
4. **模型压缩**：适当压缩模型大小保持性能
5. **数据预处理**：确保输入数据正确归一化

## 📊 结果解读

训练完成后，重点关注以下指标：

- **训练损失**：应平稳下降并收敛
- **验证损失**：应与训练损失趋势一致
- **物理约束残差**：各物理方程的残差应逐渐减小
- **训练时间**：记录训练耗时用于性能评估

## 🎉 下一步

完成基础训练后，您可以：

1. **探索高级功能**：查看[API文档](../api/)了解详细接口
2. **定制模型架构**：参考[架构说明](../architecture/model_architecture.md)
3. **优化训练策略**：学习[训练策略指南](./training_strategies.md)
4. **部署应用**：使用[部署优化指南](./deployment_optimization.md)

---

**需要帮助？** 查看[故障排除指南](./troubleshooting_debugging.md)或提交Issue。