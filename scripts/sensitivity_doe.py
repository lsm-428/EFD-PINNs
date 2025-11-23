#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np
import torch

from ewp_pinn_model import EWPINN, extract_predictions

PARAMS = [
    ('dielectric_thickness_d', [0.5, 1.0, 2.0, 5.0]),
    ('epsilon_r', [2.0, 3.0, 4.0]),
    ('surface_tension_gamma', [20.0, 35.0, 50.0]),
    ('theta0', [100.0, 110.0, 120.0])
]

def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def proxy_metric(pred):
    # 示例：以接触角与高度场的 MSE 作为设计敏感性代理
    if isinstance(pred, torch.Tensor):
        y = pred
    else:
        y = pred.get('main_predictions') if isinstance(pred, dict) else pred
    return torch.mean(y[:, :2]**2).item()

def main():
    parser = argparse.ArgumentParser(description='DOE/敏感性分析')
    parser.add_argument('--model_path', type=str, default='results_enhanced/final_model.pth')
    parser.add_argument('--config_path', type=str, default='config/model_config.json')
    parser.add_argument('--output_dir', type=str, default='results_enhanced/doe')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    cfg = load_config(args.config_path)
    model = EWPINN(input_dim=cfg.get('模型', {}).get('输入维度', 62), output_dim=cfg.get('模型', {}).get('输出维度', 24), device=device, config=cfg)
    state = torch.load(args.model_path, map_location=device)
    if isinstance(state, dict) and 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'], strict=False)
    else:
        model.load_state_dict(state, strict=False)
    model.eval()

    # 构造基础输入（随机或从配置派生），再逐个参数扫描
    base = torch.randn(128, model.input_dim, device=device)
    results = []
    for name, grid in PARAMS:
        for val in grid:
            x = base.clone()
            # 简化：将参数占位映射到输入向量的若干列（实际可通过 input_layer 接口）
            col = 0 if name == 'dielectric_thickness_d' else 1 if name == 'epsilon_r' else 2 if name == 'surface_tension_gamma' else 3
            x[:, col] = float(val)
            with torch.no_grad():
                pred = model(x)
                m = proxy_metric(pred)
            results.append({'param': name, 'value': float(val), 'metric': float(m)})

    out_json = os.path.join(args.output_dir, 'doe_results.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'Saved DOE results: {out_json}')

if __name__ == '__main__':
    main()

