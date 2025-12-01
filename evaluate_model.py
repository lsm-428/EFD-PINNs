#!/usr/bin/env python3
"""
模型评估脚本
对训练好的EFD-PINNs模型进行全面评估，包括:
1. 加载训练好的模型
2. 在测试集上评估模型性能
3. 分析动态响应特性（超调、响应时间等）
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
from efd_pinns_train import OptimizedEWPINN


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def load_model(model_path, config):
    """加载训练好的模型"""
    from efd_pinns_train import DataNormalizer
    
    print(f"📦 加载模型: {model_path}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载checkpoint获取实际模型配置
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # 从checkpoint中获取保存的配置
    saved_config = checkpoint.get('config', config)
    model_config = saved_config.get('model', config.get('model', {}))
    
    input_dim = model_config.get('input_dim', 62)
    output_dim = model_config.get('output_dim', 24)
    
    # 从state_dict推断hidden_dims
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    # 分析网络结构 - 找出所有Linear层的输出维度
    linear_layers = []
    for key, value in sorted(state_dict.items()):
        if 'main_layers' in key and '.weight' in key and 'running' not in key:
            if len(value.shape) == 2:  # Linear层
                linear_layers.append((key, value.shape))
    
    # 提取hidden_dims (除了最后一层输出层)
    hidden_dims = []
    for key, shape in linear_layers[:-1]:  # 排除最后的输出层
        hidden_dims.append(shape[0])
    
    if not hidden_dims:
        hidden_dims = [128, 128, 128]  # 默认值
    
    print(f"   推断的hidden_dims: {hidden_dims}")
    print(f"   input_dim: {input_dim}, output_dim: {output_dim}")
    
    # 创建模型实例
    model = OptimizedEWPINN(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        activation=model_config.get('activation', 'gelu'),
        config=saved_config
    )
    
    # 加载权重
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"   Loss: {checkpoint.get('loss', 'N/A'):.4f}" if checkpoint.get('loss') else "")
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    model.to(device)
    
    # 加载输出归一化器
    output_normalizer = None
    if 'output_normalizer' in checkpoint and checkpoint['output_normalizer'] is not None:
        output_normalizer = DataNormalizer(method="standard")
        output_normalizer.load_state_dict(checkpoint['output_normalizer'])
        print(f"   ✅ 已加载输出归一化器")
    else:
        print(f"   ⚠️ 未找到输出归一化器")
    
    return model, device, output_normalizer, saved_config


def generate_test_data(config, num_samples=200):
    """生成测试数据"""
    print(f"🔧 生成测试数据，样本数: {num_samples}")
    
    # 导入数据生成函数
    from efd_pinns_train import generate_dynamic_ewod_data
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 临时修改样本数
    config_copy = config.copy()
    if 'data' not in config_copy:
        config_copy['data'] = {}
    config_copy['data']['num_samples'] = num_samples
    
    # 生成数据 - 返回: X_train, Y_train, X_val, Y_val, X_test, Y_test, physics_points, normalizers
    result = generate_dynamic_ewod_data(config_copy, device)
    
    # 解包结果
    X_train, Y_train, X_val, Y_val, X_test, Y_test, physics_points, normalizers = result
    
    # 转换为numpy
    X = X_test.cpu().numpy() if torch.is_tensor(X_test) else X_test
    y = Y_test.cpu().numpy() if torch.is_tensor(Y_test) else Y_test
    
    return X, y, None


def evaluate_predictions(model, X, y, device):
    """评估模型预测性能"""
    print("📊 评估模型预测...")
    
    model.eval()
    X_tensor = torch.FloatTensor(X).to(device)
    y_tensor = torch.FloatTensor(y).to(device)
    
    with torch.no_grad():
        outputs = model(X_tensor)
        
        # 处理输出格式
        if isinstance(outputs, dict):
            predictions = outputs.get('main_predictions', outputs.get('predictions', None))
            if predictions is None:
                predictions = list(outputs.values())[0]
        else:
            predictions = outputs
    
    predictions = predictions.cpu().numpy()
    targets = y
    
    # 计算指标
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    rmse = np.sqrt(mse)
    
    # 计算每个输出的R²
    r2_scores = []
    for i in range(targets.shape[1]):
        ss_res = np.sum((targets[:, i] - predictions[:, i]) ** 2)
        ss_tot = np.sum((targets[:, i] - np.mean(targets[:, i])) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0
        r2_scores.append(r2)
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2_scores': r2_scores,
        'avg_r2': np.mean(r2_scores),
        'predictions': predictions,
        'targets': targets
    }


def analyze_dynamic_response(model, config, device, output_normalizer=None):
    """分析动态响应特性"""
    print("⚡ 分析动态响应...")
    
    # 生成阶跃响应测试数据
    num_time_steps = 100
    T_total = 0.02
    time = np.linspace(0, T_total, num_time_steps)  # 0-20ms
    
    # 获取材料和几何参数
    materials = config.get('materials', {})
    geometry = config.get('geometry', {})
    data_config = config.get('data', {})
    
    epsilon_r = materials.get('epsilon_r', 4.0)
    gamma = materials.get('gamma', 0.072)
    d = materials.get('dielectric_thickness', 4e-7)
    theta0 = materials.get('theta0', 110.0)
    
    Lx = geometry.get('Lx', 184e-6)
    Ly = geometry.get('Ly', 184e-6)
    Lz = geometry.get('Lz', 20.855e-6)
    
    # 从配置读取电压范围
    voltage_range = data_config.get('voltage_range', [0, 30])
    V_step = voltage_range[1] if isinstance(voltage_range, list) else 30
    
    results = {
        'time': time,
        'voltage': V_step,
        'contact_angles': [],
        'response_metrics': {}
    }
    
    model.eval()
    contact_angles = []
    
    for t_current in time:
        # 构建输入特征 (62维) - 与训练数据生成一致
        features = np.zeros(62, dtype=np.float32)
        
        # 归一化的空间坐标
        features[0] = 0.5  # x归一化
        features[1] = 0.5  # y归一化
        features[2] = 0.5  # z归一化
        features[3] = t_current / T_total  # t归一化
        features[4] = np.sin(2 * np.pi * t_current / T_total)
        features[5] = V_step / 30.0  # V归一化
        features[6] = 0.5  # 到边界距离
        features[7] = 0.0  # 到中心距离
        
        X = torch.FloatTensor(features).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(X)
            if isinstance(outputs, dict):
                pred = outputs.get('main_predictions', list(outputs.values())[0])
            else:
                pred = outputs
            
            # 反归一化输出
            if output_normalizer is not None:
                pred_np = pred.cpu().numpy()
                pred_denorm = output_normalizer.inverse_transform(pred_np)
                # 接触角在索引10 (弧度)
                theta_rad = pred_denorm[0, 10]
            else:
                # 如果没有归一化器，假设输出已经是物理值
                theta_rad = pred[0, 10].item()
            
            # 转换为度
            theta_deg = np.degrees(theta_rad)
            theta_deg = np.clip(theta_deg, 50, 130)  # 物理约束
            contact_angles.append(theta_deg)
    
    contact_angles = np.array(contact_angles)
    results['contact_angles'] = contact_angles
    
    # 计算响应指标
    theta_initial = contact_angles[0]
    theta_final = contact_angles[-1]
    theta_change = theta_final - theta_initial
    
    if abs(theta_change) > 1e-6:
        # 归一化响应
        normalized = (contact_angles - theta_initial) / theta_change
        
        # 响应时间 t90 (达到90%变化)
        t90_idx = np.where(np.abs(normalized) >= 0.9)[0]
        t90 = time[t90_idx[0]] * 1000 if len(t90_idx) > 0 else time[-1] * 1000
        
        # 超调
        if theta_change > 0:
            overshoot = (np.max(contact_angles) - theta_final) / abs(theta_change) * 100
        else:
            overshoot = (theta_final - np.min(contact_angles)) / abs(theta_change) * 100
        overshoot = max(0, overshoot)
        
        # 稳定时间 (进入±5%范围)
        settling_idx = np.where(np.abs(normalized - 1.0) <= 0.05)[0]
        if len(settling_idx) > 0:
            # 找到最后一次离开±5%范围后的时间
            for i in range(len(settling_idx) - 1, -1, -1):
                if settling_idx[i] == len(time) - 1 or all(np.abs(normalized[settling_idx[i]:] - 1.0) <= 0.05):
                    settling_time = time[settling_idx[i]] * 1000
                    break
            else:
                settling_time = time[-1] * 1000
        else:
            settling_time = time[-1] * 1000
    else:
        t90 = 0
        overshoot = 0
        settling_time = 0
        normalized = np.zeros_like(contact_angles)
    
    results['response_metrics'] = {
        'theta_initial': float(theta_initial),
        'theta_final': float(theta_final),
        'theta_change': float(theta_change),
        't90_ms': float(t90),
        'overshoot_percent': float(overshoot),
        'settling_time_ms': float(settling_time)
    }
    
    return results


def create_visualizations(eval_results, dynamic_results, output_dir):
    """创建可视化图表"""
    print("📈 生成可视化...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 预测 vs 真实值
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    predictions = eval_results['predictions']
    targets = eval_results['targets']
    
    # 选择关键输出维度
    key_dims = [0, 6, 12, 18]  # u, theta, 其他
    dim_names = ['速度u', '接触角θ', '压力p', '界面位置']
    
    for i, (dim, name) in enumerate(zip(key_dims, dim_names)):
        if dim < predictions.shape[1]:
            ax = axes[i // 2, i % 2]
            ax.scatter(targets[:, dim], predictions[:, dim], alpha=0.5, s=10)
            
            # 对角线
            lims = [min(targets[:, dim].min(), predictions[:, dim].min()),
                    max(targets[:, dim].max(), predictions[:, dim].max())]
            ax.plot(lims, lims, 'r--', label='理想')
            
            ax.set_xlabel('真实值')
            ax.set_ylabel('预测值')
            ax.set_title(f'{name} (R²={eval_results["r2_scores"][dim]:.3f})')
            ax.grid(True, alpha=0.3)
            ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'prediction_vs_true.png'), dpi=150)
    plt.close()
    
    # 2. 动态响应曲线
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    time_ms = dynamic_results['time'] * 1000
    contact_angles = dynamic_results['contact_angles']
    metrics = dynamic_results['response_metrics']
    
    # 接触角响应
    ax1 = axes[0]
    ax1.plot(time_ms, contact_angles, 'b-', linewidth=2, label='接触角响应')
    ax1.axhline(y=metrics['theta_final'], color='g', linestyle='--', label=f'稳态值: {metrics["theta_final"]:.2f}')
    ax1.axvline(x=metrics['t90_ms'], color='r', linestyle=':', label=f't90: {metrics["t90_ms"]:.2f}ms')
    ax1.set_xlabel('时间 (ms)')
    ax1.set_ylabel('接触角')
    ax1.set_title(f'阶跃响应 (V={dynamic_results["voltage"]}V)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 响应指标条形图
    ax2 = axes[1]
    metrics_names = ['t90 (ms)', '超调 (%)', '稳定时间 (ms)']
    metrics_values = [metrics['t90_ms'], metrics['overshoot_percent'], metrics['settling_time_ms']]
    colors = ['blue', 'red' if metrics['overshoot_percent'] > 15 else 'green', 'orange']
    
    bars = ax2.bar(metrics_names, metrics_values, color=colors, alpha=0.7)
    ax2.set_ylabel('值')
    ax2.set_title('动态响应指标')
    
    # 添加目标线
    ax2.axhline(y=15, color='r', linestyle='--', alpha=0.5, label='超调目标 (<15%)')
    ax2.legend()
    
    # 在条形上显示数值
    for bar, val in zip(bars, metrics_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dynamic_response.png'), dpi=150)
    plt.close()
    
    # 3. R²分数分布
    plt.figure(figsize=(12, 5))
    r2_scores = eval_results['r2_scores']
    x = np.arange(len(r2_scores))
    colors = ['green' if r2 > 0.8 else 'orange' if r2 > 0.5 else 'red' for r2 in r2_scores]
    
    plt.bar(x, r2_scores, color=colors, alpha=0.7)
    plt.axhline(y=0.8, color='g', linestyle='--', alpha=0.5, label='良好 (R²>0.8)')
    plt.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='一般 (R²>0.5)')
    plt.xlabel('输出维度')
    plt.ylabel('R²分数')
    plt.title(f'各输出维度R²分数 (平均: {eval_results["avg_r2"]:.3f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'r2_scores.png'), dpi=150)
    plt.close()
    
    print(f"   可视化保存到: {output_dir}")


def generate_report(eval_results, dynamic_results, output_dir):
    """生成评估报告"""
    print("📋 生成评估报告...")
    
    metrics = dynamic_results['response_metrics']
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'prediction_metrics': {
            'mse': float(eval_results['mse']),
            'mae': float(eval_results['mae']),
            'rmse': float(eval_results['rmse']),
            'avg_r2': float(eval_results['avg_r2']),
            'min_r2': float(min(eval_results['r2_scores'])),
            'max_r2': float(max(eval_results['r2_scores']))
        },
        'dynamic_response': {
            'voltage': dynamic_results['voltage'],
            'theta_initial': metrics['theta_initial'],
            'theta_final': metrics['theta_final'],
            'theta_change': metrics['theta_change'],
            't90_ms': metrics['t90_ms'],
            'overshoot_percent': metrics['overshoot_percent'],
            'settling_time_ms': metrics['settling_time_ms']
        },
        'quality_assessment': {
            'prediction_quality': 'good' if eval_results['avg_r2'] > 0.8 else 'fair' if eval_results['avg_r2'] > 0.5 else 'poor',
            'overshoot_target_met': metrics['overshoot_percent'] < 15,
            'response_time_reasonable': 1 < metrics['t90_ms'] < 10
        }
    }
    
    report_path = os.path.join(output_dir, 'evaluation_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # 打印摘要
    print("\n" + "="*60)
    print("📊 模型评估结果摘要")
    print("="*60)
    print(f"\n【预测性能】")
    print(f"  MSE:  {eval_results['mse']:.6f}")
    print(f"  MAE:  {eval_results['mae']:.6f}")
    print(f"  RMSE: {eval_results['rmse']:.6f}")
    print(f"  平均R²: {eval_results['avg_r2']:.4f}")
    
    print(f"\n【动态响应】(V={dynamic_results['voltage']}V 阶跃)")
    print(f"  响应时间 t90: {metrics['t90_ms']:.2f} ms")
    print(f"  超调量: {metrics['overshoot_percent']:.2f}%", end="")
    print(" ✅" if metrics['overshoot_percent'] < 15 else " ❌ (目标<15%)")
    print(f"  稳定时间: {metrics['settling_time_ms']:.2f} ms")
    print(f"  接触角变化: {metrics['theta_initial']:.2f}° → {metrics['theta_final']:.2f}°")
    
    print("\n" + "="*60)
    
    return report


def main():
    parser = argparse.ArgumentParser(description='评估EFD-PINNs模型')
    parser.add_argument('--model', type=str, default=None,
                        help='模型路径 (默认: 自动查找最新)')
    parser.add_argument('--config', type=str, default='config_stage2_10k.json',
                        help='配置文件路径')
    parser.add_argument('--output', type=str, default=None,
                        help='输出目录 (默认: 模型目录/evaluation)')
    parser.add_argument('--num_samples', type=int, default=200,
                        help='测试样本数')
    
    args = parser.parse_args()
    
    print("="*60)
    print("🔬 EFD-PINNs 模型评估")
    print("="*60)
    
    # 自动查找模型
    if args.model is None:
        from pathlib import Path
        output_dirs = sorted(Path('.').glob('outputs_*'), key=lambda p: p.stat().st_mtime, reverse=True)
        for d in output_dirs:
            model_path = d / 'final_model.pth'
            if model_path.exists():
                args.model = str(model_path)
                break
        if args.model is None:
            print("❌ 未找到模型文件")
            return
    
    # 设置输出目录
    if args.output is None:
        model_dir = os.path.dirname(args.model)
        args.output = os.path.join(model_dir, 'evaluation')
    
    # 加载配置和模型
    config = load_config(args.config)
    model, device, output_normalizer, saved_config = load_model(args.model, config)
    
    # 使用保存的配置（如果有）
    config = saved_config if saved_config else config
    
    # 生成测试数据
    X_test, y_test, _ = generate_test_data(config, args.num_samples)
    
    # 评估预测性能
    eval_results = evaluate_predictions(model, X_test, y_test, device)
    
    # 分析动态响应
    dynamic_results = analyze_dynamic_response(model, config, device, output_normalizer)
    
    # 生成可视化
    create_visualizations(eval_results, dynamic_results, args.output)
    
    # 生成报告
    report = generate_report(eval_results, dynamic_results, args.output)
    
    print(f"\n✅ 评估完成！结果保存在: {args.output}")


if __name__ == "__main__":
    main()
