#!/usr/bin/env python3
import os
import sys
import json
import argparse
import numpy as np
import torch

# 确保仓库根目录在路径中
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from ewp_pinn_physics import PINNConstraintLayer
from ewp_pinn_model import EWPINN, extract_predictions
try:
    from efd_pinns_train import OptimizedEWPINN
except Exception:
    from scripts.legacy_backup.ewp_pinn_optimized_train import OptimizedEWPINN
from ewp_pinn_optimized_architecture import EfficientEWPINN

def load_checkpoint(model_path, device):
    # 安全加载，仅权重（兼容旧版本）
    import torch.serialization
    import __main__
    try:
        from efd_pinns_train import DataNormalizer
    except Exception:
        from scripts.legacy_backup.ewp_pinn_optimized_train import DataNormalizer
    try:
        torch.serialization.add_safe_globals([DataNormalizer])
    except Exception:
        pass
    # 兼容使用 __main__.DataNormalizer 保存的检查点
    try:
        setattr(__main__, 'DataNormalizer', DataNormalizer)
    except Exception:
        pass
    try:
        ckpt = torch.load(model_path, map_location=device, weights_only=True)
    except Exception:
        ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        state = ckpt['model_state_dict']
    else:
        state = ckpt
    return state, ckpt

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def build_model_for_checkpoint(ckpt, cfg, device):
    info = ckpt.get('model_info', {}) if isinstance(ckpt, dict) else {}
    input_dim = info.get('input_dim', cfg.get('模型', {}).get('输入维度', 62))
    output_dim = info.get('output_dim', cfg.get('模型', {}).get('输出维度', 24))
    hidden_layers = info.get('hidden_layers', cfg.get('模型', {}).get('隐藏层', [128,64,32]))
    activation = info.get('activation', cfg.get('模型', {}).get('激活函数', 'ReLU'))
    batch_norm = info.get('batch_norm', cfg.get('模型', {}).get('批标准化', True))
    dropout_rate = info.get('dropout_rate', cfg.get('模型', {}).get('Dropout率', 0.1))
    compression_factor = info.get('compression_factor', 1.0)
    arch = info.get('architecture', 'auto')

    candidates = []
    try:
        candidates.append(('efficient', EfficientEWPINN(input_dim=input_dim, output_dim=output_dim, hidden_layers=hidden_layers, dropout_rate=dropout_rate, activation=activation, batch_norm=batch_norm, device=device, compression_factor=compression_factor, use_residual=True, use_attention=True)))
    except Exception:
        pass
    try:
        # 适配本仓库的 OptimizedEWPINN 签名
        model_cfg = {'use_batch_norm': batch_norm, 'use_residual': True, 'use_attention': False}
        optimized = OptimizedEWPINN(input_dim=input_dim, hidden_dims=hidden_layers, output_dim=output_dim, activation=activation.lower(), config=model_cfg)
        optimized = optimized.to(device)
        candidates.append(('optimized', optimized))
    except Exception:
        try:
            # 适配旧版脚本的 OptimizedEWPINN
            candidates.append(('optimized_legacy', OptimizedEWPINN(input_dim=input_dim, output_dim=output_dim, hidden_layers=hidden_layers, dropout_rate=dropout_rate, activation=activation, batch_norm=batch_norm, device=device)))
        except Exception:
            pass
    try:
        candidates.append(('base', EWPINN(input_dim=input_dim, output_dim=output_dim, device=device, config=cfg)))
    except Exception:
        pass
    return candidates

def load_dataset(npz_path, device, input_dim=None):
    try:
        data = np.load(npz_path)
        X = data.get('X_val', data.get('X_test', data.get('X_train')))
        y = data.get('y_val', data.get('y_test', data.get('y_train')))
        physics_points = data.get('physics_points', None)
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        y_t = torch.tensor(y, dtype=torch.float32, device=device)
        P_t = torch.tensor(physics_points, dtype=torch.float32, device=device) if physics_points is not None else None
        return X_t, y_t, P_t
    except Exception:
        n = 1024
        d = input_dim or 62
        X_t = torch.randn(n, d, device=device)
        y_t = torch.zeros(n, 1, device=device)
        P_t = None
        return X_t, y_t, P_t

def compute_constraint_stats(model, X, y, P, device, applied_voltage=None, contact_line_velocity=None, time=None, temperature=None, batch_size=64):
    layer = PINNConstraintLayer()
    model.eval()
    stats = {}
    weights_series = {}
    with torch.no_grad():
        n = X.shape[0]
        for i in range(0, n, batch_size):
            xb = X[i:i+batch_size]
            yb = y[i:i+batch_size]
            out = model(xb)
            pred = extract_predictions(out)
            # 使用物理点匹配批次大小；后备使用数据点
            if P is not None:
                # 随机采样等量物理点
                idx = torch.randperm(P.shape[0])[:pred.shape[0]]
                Pb = P[idx]
            else:
                Pb = xb
            # 计算一次物理损失，获取加权残差详情
            physics_loss, weighted = layer.compute_physics_loss(
                Pb, pred,
                applied_voltage=applied_voltage,
                contact_line_velocity=contact_line_velocity,
                time=time,
                temperature=temperature
            )
            # 聚合统计
            schema_keys = [
                'continuity','momentum_u','momentum_v','momentum_w','young_lippmann','contact_line_dynamics',
                'dielectric_charge','thermodynamic','interface_stability','frequency_response','optical_properties',
                'energy_efficiency','data_fit'
            ]
            for k in schema_keys:
                if k not in weighted:
                    continue
                v = weighted[k]
                s = stats.setdefault(k, {"loss": [], "weight": [], "raw": []})
                s["loss"].append(float(v.get("loss", 0.0)))
                w = v.get("weight", 0.0)
                w_val = w.item() if hasattr(w, 'item') else float(w)
                s["weight"].append(w_val)
                s["raw"].append(float(v.get("raw_value", 0.0)))
    # 计算均值/方差与权重时序
    report = {}
    for k, s in stats.items():
        report[k] = {
            "loss_mean": float(np.mean(s["loss"])) if s["loss"] else 0.0,
            "loss_std": float(np.std(s["loss"])) if s["loss"] else 0.0,
            "raw_mean": float(np.mean(s["raw"])) if s["raw"] else 0.0,
            "raw_std": float(np.std(s["raw"])) if s["raw"] else 0.0,
            "weight_series": s["weight"]
        }
    # 结构化补充缺失键，保持一致性
    for k in [
        'continuity','momentum_u','momentum_v','momentum_w','young_lippmann','contact_line_dynamics',
        'dielectric_charge','thermodynamic','interface_stability','frequency_response','optical_properties',
        'energy_efficiency','data_fit'
    ]:
        if k not in report:
            report[k] = {"loss_mean":0.0,"loss_std":0.0,"raw_mean":0.0,"raw_std":0.0,"weight_series":[]}
    return report

def main():
    parser = argparse.ArgumentParser(description="生成约束诊断报表")
    parser.add_argument('--model_path', type=str, default='results_enhanced/checkpoints/best_model.pth')
    parser.add_argument('--config_path', type=str, default='config/model_config.json')
    parser.add_argument('--dataset_path', type=str, default='results_enhanced/dataset.npz')
    parser.add_argument('--output_dir', type=str, default='results_enhanced/consistency_data')
    parser.add_argument('--applied_voltage', type=float, default=None)
    parser.add_argument('--contact_line_velocity', type=float, default=None)
    parser.add_argument('--time', type=float, default=None)
    parser.add_argument('--temperature', type=float, default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    cfg = load_config(args.config_path)
    state, full = load_checkpoint(args.model_path, device)
    candidates = build_model_for_checkpoint(full if isinstance(full, dict) else {}, cfg, device)
    loaded = False
    for name, model in candidates:
        try:
            model.load_state_dict(state, strict=False)
            loaded = True
            break
        except Exception:
            continue
    if not loaded:
        # 回退：直接创建基础模型并忽略加载失败
        model = candidates[-1][1]
    # 先构建模型，再依据模型输入维度加载/生成数据
    X, y, P = load_dataset(args.dataset_path, device, getattr(model, 'input_dim', None))

    report = compute_constraint_stats(
        model, X, y, P, device,
        applied_voltage=args.applied_voltage,
        contact_line_velocity=args.contact_line_velocity,
        time=args.time,
        temperature=args.temperature
    )

    out_path = os.path.join(args.output_dir, 'constraint_diagnostics.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"约束诊断报表已保存: {out_path}")

if __name__ == '__main__':
    main()
