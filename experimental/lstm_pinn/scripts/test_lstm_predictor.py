#!/usr/bin/env python3
"""
测试 LSTM-Hybrid-PINN 预测器

用法：
    python scripts/test_lstm_predictor.py --model outputs/train/lstm_hybrid_xxx/best_model.pth
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="测试 LSTM-Hybrid-PINN 预测器")
    parser.add_argument("--model", type=str, required=True, help="模型路径")
    parser.add_argument("--device", type=str, default="cpu", help="设备")
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"错误: 模型文件不存在: {args.model}")
        return 1

    print(f"加载模型: {args.model}")

    from src.predictors import get_lstm_hybrid_predictor

    LSTMHybridPredictor = get_lstm_hybrid_predictor()

    predictor = LSTMHybridPredictor(args.model, device=args.device)

    print("\n" + "=" * 60)
    print("测试 1: 单步升压预测 (0 → 30V)")
    print("=" * 60)

    voltage_seq = [(0, 30, 0.01)]
    phi = predictor.predict_phi(voltage_seq, time=0.005)
    print(f"  电压序列: {voltage_seq}")
    print(f"  预测 φ: {phi:.4f}")

    print("\n" + "=" * 60)
    print("测试 2: 多步电压预测 (0 → 20 → 30)")
    print("=" * 60)

    voltage_seq = [(0, 20, 0.01), (20, 30, 0.005)]
    phi = predictor.predict_phi(voltage_seq, time=0.002)
    print(f"  电压序列: {voltage_seq}")
    print(f"  预测 φ: {phi:.4f}")

    print("\n" + "=" * 60)
    print("测试 3: 电压响应曲线 (0 → 30V)")
    print("=" * 60)

    times, apertures = predictor.voltage_response(
        V_from=0, V_to=30, t_duration=0.05, n_steps=10
    )

    for t, phi in zip(times, apertures):
        print(f"  t={t * 1000:6.1f}ms: φ={phi:.4f}")

    print("\n" + "=" * 60)
    print("测试 4: 多步响应 (0 → 20 → 30 → 20 → 0)")
    print("=" * 60)

    voltage_steps = [
        (0, 20, 0.01),
        (20, 30, 0.01),
        (30, 20, 0.01),
        (20, 0, 0.01),
    ]

    result = predictor.multi_step_response(voltage_steps, n_points_per_step=5)

    print(
        f"  时间范围: {result['times'][0] * 1000:.1f}ms - {result['times'][-1] * 1000:.1f}ms"
    )
    print(
        f"  φ 范围: {result['apertures'].min():.4f} - {result['apertures'].max():.4f}"
    )

    print("\n✅ 预测器测试完成!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
