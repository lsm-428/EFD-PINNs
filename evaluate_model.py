#!/usr/bin/env python3
"""
模型评估脚本
对训练好的模型进行全面评估，包括:
1. 加载训练好的模型
2. 在测试集上评估模型性能
3. 分析预测结果
4. 生成评估报告和可视化
"""

import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from datetime import datetime
import argparse

# 导入模型相关模块
from ewp_pinn_physics import PINNConstraintLayer
from run_enhanced_training import EWPINNOptimizerManager
from ewp_pinn_model import EWPINN


def load_model(model_path, config):
    """加载训练好的模型"""
    print(f"加载模型从: {model_path}")
    
    # 创建模型实例
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EWPINN(
        input_dim=config['模型架构']['输入维度'],
        output_dim=config['模型架构']['输出维度'],
        device=device,
        config=config
    )
    
    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    # 处理不同的保存格式
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


def load_test_data(data_path):
    """加载测试数据"""
    print(f"加载测试数据从: {data_path}")
    
    if not os.path.exists(data_path):
        print(f"警告: 测试数据文件 {data_path} 不存在，使用随机生成的数据进行演示")
        # 生成随机测试数据
        np.random.seed(42)
        x_test = np.random.randn(1000, 10).astype(np.float32)
        y_test = np.random.randn(1000, 24).astype(np.float32)
        return x_test, y_test
    
    data = np.load(data_path)
    # 检查数据文件中的键
    print(f"数据文件中的键: {list(data.keys())}")
    
    # 尝试加载测试数据
    if 'X_test' in data and 'y_test' in data:
        x_test = data['X_test']
        y_test = data['y_test']
    elif 'x_test' in data and 'y_test' in data:
        x_test = data['x_test']
        y_test = data['y_test']
    elif 'X' in data and 'y' in data:
        # 如果没有专门的测试集，使用全部数据的一部分作为测试集
        x = data['X']
        y = data['y']
        split_idx = int(0.8 * len(x))
        x_test = x[split_idx:]
        y_test = y[split_idx:]
    elif 'x' in data and 'y' in data:
        # 如果没有专门的测试集，使用全部数据的一部分作为测试集
        x = data['x']
        y = data['y']
        split_idx = int(0.8 * len(x))
        x_test = x[split_idx:]
        y_test = y[split_idx:]
    else:
        raise ValueError("无法从数据文件中识别测试数据")
    
    print(f"测试数据形状: X={x_test.shape}, y={y_test.shape}")
    return x_test, y_test


def evaluate_model(model, x_test, y_test, device='cpu'):
    """评估模型性能"""
    print("开始评估模型...")
    
    # 将数据转换为PyTorch张量
    x_tensor = torch.FloatTensor(x_test).to(device)
    y_tensor = torch.FloatTensor(y_test).to(device)
    
    # 创建数据加载器
    batch_size = 32
    dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # 评估指标
    total_loss = 0.0
    total_samples = 0
    predictions = []
    targets = []
    
    # 损失函数
    criterion = nn.MSELoss()
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            # 前向传播
            outputs = model(batch_x)
            
            # 处理模型输出 - 可能是字典或张量
            if isinstance(outputs, dict):
                # 尝试从字典中提取主要预测
                if 'main_predictions' in outputs:
                    pred_tensor = outputs['main_predictions']
                elif 'predictions' in outputs:
                    pred_tensor = outputs['predictions']
                else:
                    # 使用第一个张量值
                    pred_tensor = next(iter(outputs.values()))
            else:
                pred_tensor = outputs
            
            # 确保预测张量与目标张量的形状匹配
            if pred_tensor.shape != batch_y.shape:
                # 如果形状不匹配，尝试调整
                if pred_tensor.shape[0] == batch_y.shape[0] and pred_tensor.shape[1] > batch_y.shape[1]:
                    pred_tensor = pred_tensor[:, :batch_y.shape[1]]
                elif pred_tensor.shape[0] == batch_y.shape[0] and pred_tensor.shape[1] < batch_y.shape[1]:
                    # 填充零
                    padding = torch.zeros(pred_tensor.shape[0], batch_y.shape[1] - pred_tensor.shape[1], device=pred_tensor.device)
                    pred_tensor = torch.cat([pred_tensor, padding], dim=1)
            
            # 计算损失
            loss = criterion(pred_tensor, batch_y)
            
            # 累计损失
            total_loss += loss.item() * batch_x.size(0)
            total_samples += batch_x.size(0)
            
            # 保存预测和目标
            predictions.append(pred_tensor.cpu().numpy())
            targets.append(batch_y.cpu().numpy())
    
    # 计算平均损失
    avg_loss = total_loss / total_samples
    
    # 合并所有预测和目标
    predictions = np.vstack(predictions)
    targets = np.vstack(targets)
    
    # 计算其他评估指标
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    
    # 计算每个输出维度的R²分数
    r2_scores = []
    for i in range(targets.shape[1]):
        ss_res = np.sum((targets[:, i] - predictions[:, i]) ** 2)
        ss_tot = np.sum((targets[:, i] - np.mean(targets[:, i])) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        r2_scores.append(r2)
    
    avg_r2 = np.mean(r2_scores)
    
    # 返回评估结果
    results = {
        'mse_loss': avg_loss,
        'mae': mae,
        'rmse': rmse,
        'r2_scores': r2_scores,
        'avg_r2': avg_r2,
        'predictions': predictions,
        'targets': targets
    }
    
    return results


def generate_evaluation_report(results, output_dir):
    """生成评估报告"""
    print("生成评估报告...")
    
    # 创建报告字典
    report = {
        'evaluation_timestamp': datetime.now().isoformat(),
        'metrics': {
            'mse_loss': float(results['mse_loss']),
            'mae': float(results['mae']),
            'rmse': float(results['rmse']),
            'avg_r2': float(results['avg_r2'])
        },
        'r2_scores_per_output': [float(score) for score in results['r2_scores']]
    }
    
    # 保存报告
    report_path = os.path.join(output_dir, 'evaluation_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"评估报告已保存到: {report_path}")
    
    # 打印关键指标
    print("\n=== 模型评估结果 ===")
    print(f"MSE损失: {results['mse_loss']:.6f}")
    print(f"MAE: {results['mae']:.6f}")
    print(f"RMSE: {results['rmse']:.6f}")
    print(f"平均R²分数: {results['avg_r2']:.6f}")
    print(f"最小R²分数: {min(results['r2_scores']):.6f}")
    print(f"最大R²分数: {max(results['r2_scores']):.6f}")
    
    return report


def create_visualizations(results, output_dir):
    """创建可视化图表"""
    print("创建可视化图表...")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 预测值 vs 真实值散点图
    plt.figure(figsize=(10, 8))
    predictions = results['predictions']
    targets = results['targets']
    
    # 随机选择一些样本进行可视化
    n_samples = min(500, len(predictions))
    indices = np.random.choice(len(predictions), n_samples, replace=False)
    
    # 选择几个输出维度进行可视化
    n_dims = min(4, predictions.shape[1])
    dim_indices = np.linspace(0, predictions.shape[1]-1, n_dims, dtype=int)
    
    for i, dim in enumerate(dim_indices):
        plt.subplot(2, 2, i+1)
        plt.scatter(targets[indices, dim], predictions[indices, dim], alpha=0.5)
        plt.plot([targets[indices, dim].min(), targets[indices, dim].max()], 
                 [targets[indices, dim].min(), targets[indices, dim].max()], 'r--')
        plt.title(f'输出维度 {dim}')
        plt.xlabel('真实值')
        plt.ylabel('预测值')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_vs_actual.png'), dpi=300)
    plt.close()
    
    # 2. 残差分布图
    plt.figure(figsize=(12, 8))
    residuals = predictions - targets
    
    for i, dim in enumerate(dim_indices):
        plt.subplot(2, 2, i+1)
        plt.hist(residuals[:, dim], bins=30, alpha=0.7, edgecolor='black')
        plt.title(f'输出维度 {dim} 的残差分布')
        plt.xlabel('残差值')
        plt.ylabel('频数')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'residual_distribution.png'), dpi=300)
    plt.close()
    
    # 3. R²分数条形图
    plt.figure(figsize=(12, 6))
    r2_scores = results['r2_scores']
    dims = np.arange(len(r2_scores))
    
    plt.bar(dims, r2_scores, alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('各输出维度的R²分数')
    plt.xlabel('输出维度')
    plt.ylabel('R²分数')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'r2_scores.png'), dpi=300)
    plt.close()
    
    print(f"可视化图表已保存到: {output_dir}")


def compare_with_training_history(training_history_path, evaluation_results, output_dir):
    """与训练历史进行比较"""
    print("与训练历史进行比较...")
    
    if not os.path.exists(training_history_path):
        print(f"警告: 训练历史文件 {training_history_path} 不存在")
        return
    
    # 加载训练历史
    with open(training_history_path, 'r') as f:
        training_history = json.load(f)
    
    # 提取验证损失
    val_losses = training_history.get('val_losses', [])
    
    if not val_losses:
        print("训练历史中没有验证损失数据")
        return
    
    # 创建比较图
    plt.figure(figsize=(10, 6))
    epochs = np.arange(1, len(val_losses) + 1)
    
    plt.plot(epochs, val_losses, label='验证损失')
    plt.axhline(y=evaluation_results['mse_loss'], color='r', linestyle='--', 
                label=f'测试损失: {evaluation_results["mse_loss"]:.6f}')
    
    plt.title('训练过程中的验证损失 vs 最终测试损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失值')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'train_vs_test_loss.png'), dpi=300)
    plt.close()
    
    print("训练与测试比较图已保存")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='评估训练好的PINN模型')
    parser.add_argument('--model_path', type=str, default='results_long_run/final_model.pth',
                        help='训练好的模型路径')
    parser.add_argument('--data_path', type=str, default='results_long_run/dataset.npz',
                        help='测试数据路径')
    parser.add_argument('--config_path', type=str, default='config/long_run_config.json',
                        help='模型配置文件路径')
    parser.add_argument('--training_history_path', type=str, default='results_long_run/training_history.json',
                        help='训练历史文件路径')
    parser.add_argument('--output_dir', type=str, default='results_long_run/evaluation',
                        help='评估结果输出目录')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载配置
    with open(args.config_path, 'r') as f:
        config = json.load(f)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    model = load_model(args.model_path, config)
    model = model.to(device)
    
    # 加载测试数据
    x_test, y_test = load_test_data(args.data_path)
    print(f"测试数据形状: x={x_test.shape}, y={y_test.shape}")
    
    # 评估模型
    results = evaluate_model(model, x_test, y_test, device)
    
    # 生成评估报告
    report = generate_evaluation_report(results, args.output_dir)
    
    # 创建可视化
    create_visualizations(results, args.output_dir)
    
    # 与训练历史比较
    compare_with_training_history(args.training_history_path, results, args.output_dir)
    
    print("\n=== 评估完成 ===")
    print(f"评估结果已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()