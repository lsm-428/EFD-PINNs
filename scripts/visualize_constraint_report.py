#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

def load_report(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def plot_residual_stats(report, out_dir=None, save_path=None):
    """绘制残差统计图表，支持两种调用方式：
    1. 传入out_dir参数，将图表保存到指定目录
    2. 传入save_path参数，将图表保存到指定路径
    兼容report参数为字典或列表的情况
    """
    # 处理report为字典或列表的情况
    if isinstance(report, dict):
        # 处理完整的约束诊断报告字典
        keys = list(report.keys())
        # 尝试不同的键名格式
        if 'loss_mean' in report.get(list(keys)[0], {}):
            loss_means = [report[k]['loss_mean'] for k in keys]
            loss_stds = [report[k]['loss_std'] for k in keys]
            raw_means = [report[k]['raw_mean'] for k in keys]
            raw_stds = [report[k]['raw_std'] for k in keys]
        elif 'mean' in report.get(list(keys)[0], {}):
            loss_means = [report[k]['mean'] * report[k].get('weight', 1.0) for k in keys]
            loss_stds = [report[k]['std'] * report[k].get('weight', 1.0) for k in keys]
            raw_means = [report[k]['mean'] for k in keys]
            raw_stds = [report[k]['std'] for k in keys]
        else:
            # 回退方案：使用键名作为标签，值的均值作为数据
            loss_means = [np.mean(v) for v in report.values()]
            loss_stds = [np.std(v) for v in report.values()]
            raw_means = loss_means.copy()
            raw_stds = loss_stds.copy()
    else:
        # 处理physics_loss列表
        keys = ["physics_loss"]
        loss_means = [np.mean(report)]
        loss_stds = [np.std(report)]
        raw_means = loss_means.copy()
        raw_stds = loss_stds.copy()

    plt.figure(figsize=(14, 6))
    x = np.arange(len(keys))
    plt.subplot(1, 2, 1)
    plt.bar(x, loss_means, yerr=loss_stds, alpha=0.7, capsize=4)
    plt.title('Weighted Residual Loss Mean ± Std')
    plt.ylabel('Weighted Loss')
    plt.xticks(x, keys, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.bar(x, raw_means, yerr=raw_stds, alpha=0.7, capsize=4, color='orange')
    plt.title('Raw Squared Residual Mean ± Std')
    plt.ylabel('Mean(residual^2)')
    plt.xticks(x, keys, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 确定保存路径
    if save_path:
        # 确保保存目录存在
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_path, dpi=300)
        print(f'Saved: {save_path}')
    elif out_dir:
        os.makedirs(out_dir, exist_ok=True)
        p = os.path.join(out_dir, 'constraint_residual_stats.png')
        plt.savefig(p, dpi=300)
        print(f'Saved: {p}')
    
    plt.close()

def plot_weight_series(report, out_dir=None, save_path=None):
    """绘制权重时序图，支持两种调用方式：
    1. 传入out_dir参数，将图表保存到指定目录
    2. 传入save_path参数，将图表保存到指定路径
    """
    plt.figure(figsize=(14, 8))
    i = 1
    for k, v in report.items():
        # 支持不同格式的权重历史数据
        if isinstance(v, list):
            series = v
        else:
            series = v.get('weight_series', [])
        
        if not series:
            continue
        plt.subplot(3, 3, i)
        plt.plot(series, '-o', markersize=2)
        plt.title(f'{k} Weight Series')
        plt.xlabel('Batch Index')
        plt.ylabel('Weight')
        plt.grid(True, alpha=0.3)
        i += 1
        if i > 9:
            break
    plt.tight_layout()
    
    # 确定保存路径
    if save_path:
        # 确保保存目录存在
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        plt.savefig(save_path, dpi=300)
        print(f'Saved: {save_path}')
    elif out_dir:
        os.makedirs(out_dir, exist_ok=True)
        p = os.path.join(out_dir, 'constraint_weight_series.png')
        plt.savefig(p, dpi=300)
        print(f'Saved: {p}')
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Constraint Diagnostic Report Visualization')
    parser.add_argument('--report_path', type=str, default='results_enhanced/consistency_data/constraint_diagnostics.json')
    parser.add_argument('--output_dir', type=str, default='results_enhanced/consistency_data')
    args = parser.parse_args()

    report = load_report(args.report_path)
    plot_residual_stats(report, args.output_dir)
    plot_weight_series(report, args.output_dir)

if __name__ == '__main__':
    main()

