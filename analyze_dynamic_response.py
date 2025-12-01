#!/usr/bin/env python3
"""
动态响应分析工具
分析训练好的模型的时间演化预测能力

使用方法:
    conda activate efd
    python analyze_dynamic_response.py --model outputs_xxx/final_model.pth
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import argparse

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_model(model_path, config_path=None):
    """加载训练好的模型"""
    from efd_pinns_train import OptimizedEWPINN, DataNormalizer
    
    print(f"📦 加载模型: {model_path}")
    # 强制使用CPU避免GPU内存问题
    device = torch.device('cpu')
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # 获取配置
    if 'config' in checkpoint:
        config = checkpoint['config']
    elif config_path:
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {}
    
    model_config = config.get('model', {})
    input_dim = model_config.get('input_dim', 62)
    output_dim = model_config.get('output_dim', 24)
    
    # 从state_dict推断hidden_dims
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    linear_layers = []
    for key, value in sorted(state_dict.items()):
        if 'main_layers' in key and '.weight' in key and 'running' not in key:
            if len(value.shape) == 2:
                linear_layers.append((key, value.shape))
    
    hidden_dims = [shape[0] for key, shape in linear_layers[:-1]] if linear_layers else [128, 128, 128]
    
    print(f"   hidden_dims: {hidden_dims}")
    print(f"   input_dim: {input_dim}, output_dim: {output_dim}")
    
    model = OptimizedEWPINN(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        activation=model_config.get('activation', 'gelu'),
        config=config
    )
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
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
        print(f"   ⚠️ 未找到输出归一化器，将使用物理范围估算")
    
    return model, config, device, output_normalizer


def predict_with_model(model, config, device, t, V_seq, output_normalizer=None):
    """使用训练好的模型预测动态响应"""
    materials = config.get('materials', {})
    geometry = config.get('geometry', {})
    
    epsilon_r = materials.get('epsilon_r', 4.0)
    gamma = materials.get('gamma', 0.072)  # 癸烷+染料与水的界面张力
    d = materials.get('dielectric_thickness', 4e-7)
    theta0 = materials.get('theta0', 110.0)  # Teflon AF 1600X 初始接触角
    
    Lx = geometry.get('Lx', 184e-6)
    Ly = geometry.get('Ly', 184e-6)
    Lz = geometry.get('Lz', 20.855e-6)
    
    T_total = t[-1]
    contact_angles = []
    
    model.eval()
    with torch.no_grad():
        for ti, (t_current, V_current) in enumerate(zip(t, V_seq)):
            # 构建输入特征 (62维)
            features = np.zeros(62, dtype=np.float32)
            features[0] = 0.5  # x归一化
            features[1] = 0.5  # y归一化
            features[2] = 0.5  # z归一化
            features[3] = t_current / T_total  # t归一化
            features[4] = np.sin(2 * np.pi * t_current / T_total)
            features[5] = V_current / 30.0  # V归一化 (工作电压0-30V)
            features[6] = 0.5  # 到边界距离
            features[7] = 0.0  # 到中心距离
            
            X = torch.FloatTensor(features).unsqueeze(0).to(device)
            output = model(X)
            
            # 反归一化输出
            if output_normalizer is not None:
                output_np = output.cpu().numpy()
                output_denorm = output_normalizer.inverse_transform(output_np)
                # 接触角在索引10 (弧度)
                theta_rad = output_denorm[0, 10]
            else:
                # 如果没有归一化器，使用物理范围估算
                # 接触角范围: 60° - 120° (弧度: 1.05 - 2.09)
                theta_normalized = output[0, 10].item()
                # 假设归一化到[-1, 1]或[0, 1]范围
                if -2 < theta_normalized < 2:
                    # 标准化输出，映射到物理范围
                    theta_min_rad = np.radians(60)
                    theta_max_rad = np.radians(120)
                    theta_rad = theta_min_rad + (theta_normalized + 1) * (theta_max_rad - theta_min_rad) / 2
                else:
                    theta_rad = theta_normalized
            
            # 转换为度
            theta_deg = np.degrees(theta_rad)
            
            # 物理约束: 接触角应在合理范围内
            theta_deg = np.clip(theta_deg, 50, 130)
            
            contact_angles.append(theta_deg)
    
    return np.array(contact_angles)


def predict_theoretical(t, V_seq, config):
    """使用理论模型预测（作为对比）"""
    materials = config.get('materials', {})
    data_config = config.get('data', {})
    dynamics_params = data_config.get('dynamics_params', {})
    
    theta0 = materials.get('theta0', 110.0)
    epsilon_r = materials.get('epsilon_r', 4.0)
    gamma = materials.get('gamma', 0.072)
    d = materials.get('dielectric_thickness', 4e-7)
    epsilon_0 = 8.854e-12
    
    # 从配置读取动力学参数
    tau = dynamics_params.get('tau', 5e-3)  # 默认5ms
    zeta = dynamics_params.get('zeta', 0.85)  # 默认0.85
    
    # 二阶欠阻尼参数
    omega_0 = 2 * np.pi / tau
    omega_d = omega_0 * np.sqrt(1 - zeta**2)
    
    theta_pred = []
    theta_prev = theta0
    theta_start = theta0  # 初始化
    t_since_change = 0
    V_prev = 0
    dt = t[1] - t[0] if len(t) > 1 else 1e-4
    
    for ti, (t_current, V_current) in enumerate(zip(t, V_seq)):
        # 计算平衡接触角
        cos_theta0_rad = np.cos(np.radians(theta0))
        term = (epsilon_0 * epsilon_r * V_current**2) / (2 * gamma * d)
        cos_theta_eq = np.clip(cos_theta0_rad + term, -1, 1)
        theta_eq = np.degrees(np.arccos(cos_theta_eq))
        
        if ti == 0:
            theta_current = theta0
            theta_start = theta0
        else:
            # 检测电压变化
            if V_current != V_prev:
                t_since_change = 0
                theta_start = theta_prev
            else:
                t_since_change += dt
            
            # 二阶欠阻尼响应
            exp_term = np.exp(-zeta * omega_0 * t_since_change)
            cos_term = np.cos(omega_d * t_since_change)
            sin_term = np.sin(omega_d * t_since_change)
            damping_factor = zeta / np.sqrt(1 - zeta**2)
            
            theta_current = theta_eq + (theta_start - theta_eq) * exp_term * (
                cos_term + damping_factor * sin_term
            )
        
        theta_pred.append(theta_current)
        theta_prev = theta_current
        V_prev = V_current
    
    return np.array(theta_pred)


def analyze_response_metrics(t, V_seq, theta_pred):
    """分析动态响应指标"""
    rising_edges = np.where(np.diff(V_seq) > 0)[0]
    falling_edges = np.where(np.diff(V_seq) < 0)[0]
    
    rising_edge = rising_edges[0] if len(rising_edges) > 0 else 20
    falling_edge = falling_edges[0] if len(falling_edges) > 0 else 60
    
    theta_initial = theta_pred[rising_edge]
    theta_final = np.mean(theta_pred[min(rising_edge+15, falling_edge-5):falling_edge-2])
    
    # 响应时间 t90
    theta_change = theta_initial - theta_final
    if abs(theta_change) > 0.1:
        theta_90 = theta_initial - 0.9 * theta_change
        try:
            t_90_idx = np.where(theta_pred[rising_edge:falling_edge] <= theta_90)[0]
            t_90 = (t[rising_edge + t_90_idx[0]] - t[rising_edge]) * 1000 if len(t_90_idx) > 0 else np.nan
        except:
            t_90 = np.nan
    else:
        t_90 = np.nan
    
    # 超调
    theta_min = np.min(theta_pred[rising_edge:falling_edge])
    if abs(theta_change) > 0.1:
        overshoot = max(0, (theta_final - theta_min) / abs(theta_change) * 100)
    else:
        overshoot = 0
    
    # 稳定时间
    settling_band = 0.05 * abs(theta_change) if abs(theta_change) > 0.1 else 1.0
    try:
        settled = np.where(np.abs(theta_pred[rising_edge:falling_edge] - theta_final) < settling_band)[0]
        t_settle = (t[rising_edge + settled[0]] - t[rising_edge]) * 1000 if len(settled) > 0 else np.nan
    except:
        t_settle = np.nan
    
    # 下降响应
    theta_low = np.mean(theta_pred[min(falling_edge+15, len(theta_pred)-1):])
    try:
        theta_90_fall = theta_final + 0.9 * (theta_low - theta_final)
        t_90_fall_idx = np.where(theta_pred[falling_edge:] >= theta_90_fall)[0]
        t_90_fall = (t[falling_edge + t_90_fall_idx[0]] - t[falling_edge]) * 1000 if len(t_90_fall_idx) > 0 else np.nan
    except:
        t_90_fall = np.nan
    
    return {
        'theta_initial': theta_initial,
        'theta_final': theta_final,
        'theta_low': theta_low,
        'theta_change': theta_change,
        't_90_rise': t_90,
        't_90_fall': t_90_fall,
        'overshoot': overshoot,
        't_settle': t_settle,
        'rising_edge_time': t[rising_edge] * 1000,
        'falling_edge_time': t[falling_edge] * 1000
    }


def plot_dynamic_response(t, V_seq, theta_model, theta_theory, metrics, output_path):
    """绘制动态响应分析图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 图1: 电压输入
    ax1 = axes[0, 0]
    ax1.plot(t*1000, V_seq, 'b-', linewidth=2)
    ax1.axvline(metrics['rising_edge_time'], color='g', linestyle='--', alpha=0.5, label='Rising')
    ax1.axvline(metrics['falling_edge_time'], color='r', linestyle='--', alpha=0.5, label='Falling')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Voltage (V)')
    ax1.set_title('Input Voltage')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 图2: 接触角响应对比
    ax2 = axes[0, 1]
    ax2.plot(t*1000, theta_model, 'r-', linewidth=2, label='Model Prediction')
    ax2.plot(t*1000, theta_theory, 'b--', linewidth=1.5, alpha=0.7, label='Theoretical')
    ax2.axhline(metrics['theta_initial'], color='gray', linestyle=':', alpha=0.5)
    ax2.axhline(metrics['theta_final'], color='green', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Contact Angle (deg)')
    ax2.set_title('Dynamic Response')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 图3: 上升响应细节
    ax3 = axes[1, 0]
    rising_start = int(metrics['rising_edge_time'] / (t[-1]*1000) * len(t))
    rising_end = min(rising_start + 50, len(t))
    t_rise = t[rising_start:rising_end] * 1000
    
    ax3.plot(t_rise, theta_model[rising_start:rising_end], 'r-', linewidth=2, label='Model')
    ax3.plot(t_rise, theta_theory[rising_start:rising_end], 'b--', linewidth=1.5, label='Theory')
    ax3.axhline(metrics['theta_final'], color='g', linestyle='--', alpha=0.5)
    if not np.isnan(metrics['t_90_rise']):
        ax3.axvline(metrics['rising_edge_time'] + metrics['t_90_rise'], 
                   color='orange', linestyle='--', label=f't90={metrics["t_90_rise"]:.2f}ms')
    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('Contact Angle (deg)')
    ax3.set_title('Rising Response Detail')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 图4: 性能指标
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    t90_status = "PASS" if 1 <= metrics['t_90_rise'] <= 10 else "FAIL"
    overshoot_status = "PASS" if metrics['overshoot'] < 15 else "FAIL"
    
    summary = f"""
    Dynamic Response Metrics
    ========================
    
    Initial angle:  {metrics['theta_initial']:.1f} deg
    Final angle:    {metrics['theta_final']:.1f} deg
    Angle change:   {metrics['theta_change']:.1f} deg
    
    Response time (t90): {metrics['t_90_rise']:.2f} ms  [{t90_status}]
    Overshoot:           {metrics['overshoot']:.1f}%    [{overshoot_status}]
    Settling time:       {metrics['t_settle']:.2f} ms
    
    Target Criteria:
    - Response time: 1-10 ms
    - Overshoot: < 15%
    """
    
    ax4.text(0.1, 0.5, summary, fontsize=11, family='monospace',
             verticalalignment='center', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"📈 Plot saved: {output_path}")
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='Analyze dynamic response of trained model')
    parser.add_argument('--model', type=str, default=None, help='Model path')
    parser.add_argument('--config', type=str, default='config_stage2_optimized.json', help='Config path')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    args = parser.parse_args()
    
    print("=" * 60)
    print("🔬 Dynamic Response Analysis")
    print("=" * 60)
    
    # 查找模型
    if args.model is None:
        output_dirs = sorted(Path('.').glob('outputs_*'), key=lambda p: p.stat().st_mtime, reverse=True)
        for d in output_dirs:
            model_path = d / 'final_model.pth'
            if model_path.exists():
                args.model = str(model_path)
                break
    
    if args.model is None or not Path(args.model).exists():
        print("❌ No model found. Using theoretical model only.")
        use_model = False
        config = {}
        output_normalizer = None
        if Path(args.config).exists():
            with open(args.config, 'r') as f:
                config = json.load(f)
    else:
        use_model = True
        model, config, device, output_normalizer = load_model(args.model, args.config)
    
    # 设置输出目录
    if args.output is None:
        args.output = str(Path(args.model).parent) if args.model else '.'
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    # 生成测试序列
    print("\n📊 Generating step response test...")
    nt = 100
    T_total = 20e-3
    t = np.linspace(0, T_total, nt)
    
    # 从配置读取电压范围
    data_config = config.get('data', {})
    voltage_range = data_config.get('voltage_range', [0, 30])
    V_max = voltage_range[1] if isinstance(voltage_range, list) else 30
    
    V_seq = np.zeros(nt)
    V_seq[20:60] = V_max  # 阶跃电压 at 4-12ms
    
    # 理论预测
    print("🔮 Computing theoretical response...")
    theta_theory = predict_theoretical(t, V_seq, config)
    
    # 模型预测
    if use_model:
        print("🤖 Computing model prediction...")
        theta_model = predict_with_model(model, config, device, t, V_seq, output_normalizer)
    else:
        theta_model = theta_theory.copy()
    
    # 分析指标
    print("\n📈 Analyzing response metrics...")
    metrics = analyze_response_metrics(t, V_seq, theta_model)
    
    # 打印结果
    print("\n" + "=" * 60)
    print("📊 Dynamic Response Metrics")
    print("=" * 60)
    print(f"Response time (t90): {metrics['t_90_rise']:.2f} ms")
    print(f"Overshoot:           {metrics['overshoot']:.1f}%")
    print(f"Settling time:       {metrics['t_settle']:.2f} ms")
    print(f"Initial angle:       {metrics['theta_initial']:.1f} deg")
    print(f"Final angle:         {metrics['theta_final']:.1f} deg")
    print(f"Angle change:        {metrics['theta_change']:.1f} deg")
    print("=" * 60)
    
    # 评估
    print("\n🎯 Evaluation:")
    if 1 <= metrics['t_90_rise'] <= 10:
        print("   ✅ Response time: PASS (1-10 ms)")
    else:
        print(f"   ❌ Response time: FAIL ({metrics['t_90_rise']:.2f} ms)")
    
    if metrics['overshoot'] < 15:
        print("   ✅ Overshoot: PASS (< 15%)")
    else:
        print(f"   ❌ Overshoot: FAIL ({metrics['overshoot']:.1f}%)")
    
    # 绘图
    print("\n📈 Generating plots...")
    plot_path = Path(args.output) / 'dynamic_response_analysis.png'
    plot_dynamic_response(t, V_seq, theta_model, theta_theory, metrics, str(plot_path))
    
    # 保存报告
    report = {
        'model_path': args.model,
        'metrics': {k: float(v) if isinstance(v, (int, float, np.floating)) else v 
                   for k, v in metrics.items()},
        'pass_criteria': {
            'response_time': bool(1 <= metrics['t_90_rise'] <= 10),
            'overshoot': bool(metrics['overshoot'] < 15)
        }
    }
    
    report_path = Path(args.output) / 'dynamic_response_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"📄 Report saved: {report_path}")
    
    print("\n✅ Analysis complete!")


if __name__ == '__main__':
    main()
