#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

def load_report(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def plot_residual_stats(report, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    keys = list(report.keys())
    loss_means = [report[k]['loss_mean'] for k in keys]
    loss_stds = [report[k]['loss_std'] for k in keys]
    raw_means = [report[k]['raw_mean'] for k in keys]
    raw_stds = [report[k]['raw_std'] for k in keys]

    plt.figure(figsize=(14, 6))
    x = np.arange(len(keys))
    plt.subplot(1, 2, 1)
    plt.bar(x, loss_means, yerr=loss_stds, alpha=0.7, capsize=4)
    plt.title('加权残差损失均值±方差')
    plt.ylabel('Weighted Loss')
    plt.xticks(x, keys, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.bar(x, raw_means, yerr=raw_stds, alpha=0.7, capsize=4, color='orange')
    plt.title('原始平方残差均值±方差')
    plt.ylabel('Mean(residual^2)')
    plt.xticks(x, keys, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    p = os.path.join(out_dir, 'constraint_residual_stats.png')
    plt.savefig(p, dpi=300)
    plt.close()
    print(f'Saved: {p}')

def plot_weight_series(report, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(14, 8))
    i = 1
    for k, v in report.items():
        series = v.get('weight_series', [])
        if not series:
            continue
        plt.subplot(3, 3, i)
        plt.plot(series, '-o', markersize=2)
        plt.title(f'{k} 权重时序')
        plt.xlabel('Batch Index')
        plt.ylabel('Weight')
        plt.grid(True, alpha=0.3)
        i += 1
        if i > 9:
            break
    plt.tight_layout()
    p = os.path.join(out_dir, 'constraint_weight_series.png')
    plt.savefig(p, dpi=300)
    plt.close()
    print(f'Saved: {p}')

def main():
    parser = argparse.ArgumentParser(description='约束诊断报表可视化')
    parser.add_argument('--report_path', type=str, default='results_enhanced/consistency_data/constraint_diagnostics.json')
    parser.add_argument('--output_dir', type=str, default='results_enhanced/consistency_data')
    args = parser.parse_args()

    report = load_report(args.report_path)
    plot_residual_stats(report, args.output_dir)
    plot_weight_series(report, args.output_dir)

if __name__ == '__main__':
    main()

