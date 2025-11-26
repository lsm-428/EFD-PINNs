# EFD3D - 电润湿像素物理信息神经网络

EFD3D是一个面向电润湿显示像素的物理信息神经网络（PINN）研究项目，融合数据拟合与物理约束，支持高效架构、长时训练、性能监控与结果可视化。

## 项目特色

- **物理约束驱动**：集成Navier-Stokes、Young-Lippmann、接触线动力学等物理方程
- **高效神经网络架构**：残差连接与注意力机制，支持模型压缩
- **渐进式训练策略**：多阶段训练，动态物理权重调整
- **全面监控与可视化**：自动生成训练曲线、约束诊断和性能报告

## 快速开始

### 安装依赖
```bash
pip install torch numpy matplotlib scikit-learn pytest
# 可选依赖
pip install onnx onnxruntime pyvista pandas
```

### 基础训练
```bash
# 短训练模式
python efd_pinns_train.py --mode train --config config/exp_short_config.json --output-dir results_short

# 高效架构训练
python efd_pinns_train.py --mode train --config config/exp_short_config.json --efficient-architecture --model-compression 0.8 --output-dir results_efficient

# 长时训练（推荐）
python efd_pinns_train.py --mode train --config config/long_run_config.json --output-dir results_long_run --epochs 100000 --dynamic_weight --weight_strategy adaptive
```

### 测试与推理
```bash
# 模型测试
python efd_pinns_train.py --mode test --model-path results_short/final_model.pth --config config/exp_short_config.json

# 模型推理
python efd_pinns_train.py --mode infer --model-path results_short/final_model.pth
```

## 核心模块

### 神经网络架构
- **EfficientEWPINN**：高效神经网络主类，支持残差连接和注意力机制
- **EWPINNInputLayer**：62维输入特征处理，支持阶段化实现
- **EWPINNOutputLayer**：24维输出特征处理，包含物理范围配置

### 物理约束系统
- **PhysicsConstraints**：物理约束核心类，实现多种物理方程
- **PINNConstraintLayer**：物理约束层，统一管理约束权重

### 训练系统
- **progressive_training**：渐进式训练主函数
- **LossStabilizer**：高级损失稳定器，防止训练不稳定
- **unified_progressive_training**：统一训练函数，整合多种训练模式

## 输入输出映射

### 输入特征（62维）
- 时空与电压参数
- 几何结构参数
- 材料与界面特性
- 电场与介电参数
- 流体动力学参数
- 时间动态参数
- 电润湿特异参数

### 输出特征（24维）
- 物理场（压力、速度、电势、电场等）
- 界面与接触线特性
- 工程性能指标（响应时间、稳定性、能效等）

## 配置系统

项目使用JSON配置文件管理训练参数：

```json
{
    "模型": {
        "输入维度": 62,
        "输出维度": 24,
        "隐藏层": [128, 64, 32],
        "激活函数": "ReLU"
    },
    "训练": {
        "渐进式训练": [
            {
                "名称": "预训练",
                "轮次": 1000,
                "学习率": 0.001,
                "物理约束权重": 0.1
            }
        ]
    }
}
```

## 输出结构

训练完成后，输出目录包含：
- `final_model.pth`：训练完成的模型
- `training_history.json`：训练历史数据
- `validation_results.json`：验证结果
- `visualizations/`：训练曲线和诊断图表
- `reports/`：性能报告和约束诊断

## 文档导航

- [API参考文档](./api/) - 详细的类和函数API文档
- [架构说明](./architecture/) - 神经网络架构详解
- [使用指南](./guides/) - 配置、部署和故障排除指南
- [案例研究](./case_studies/) - 实际应用案例

## 技术支持

- **问题报告**：检查现有文档和故障排除指南
- **性能优化**：使用高效架构和模型压缩
- **物理一致性**：监控约束诊断报告，调整物理权重

## 许可证

MIT License

## 致谢

感谢相关开源库和社区的支持。