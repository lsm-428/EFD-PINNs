#!/usr/bin/env python3
"""
监控 LevelSet 训练进度和体积守恒改进效果
"""

import sys
import os
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from evaluate import LevelSetEvaluator
import torch


def monitor_training(output_dir: str, interval: int = 300):
    """
    监控训练进度

    Args:
        output_dir: 训练输出目录
        interval: 检查间隔（秒）
    """
    evaluator = LevelSetEvaluator()
    output_path = Path(output_dir)

    print(f"开始监控训练: {output_dir}")
    print(f"检查间隔: {interval} 秒")
    print("=" * 60)

    last_epoch = -1
    while True:
        try:
            # 检查是否有新的检查点
            checkpoint_files = list(output_path.glob("checkpoint_epoch_*.pt"))
            if not checkpoint_files:
                print("等待检查点生成...")
                time.sleep(interval)
                continue

            # 找到最新的检查点
            latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.stem.split('_')[-1]))
            current_epoch = int(latest_checkpoint.stem.split('_')[-1])

            if current_epoch <= last_epoch:
                print(f"当前 epoch: {current_epoch} (无新进展)")
                time.sleep(interval)
                continue

            print(f"\n🔍 检测到新检查点: epoch {current_epoch}")

            # 加载模型并验证体积守恒
            model, config = evaluator.load_model(str(output_path), str(latest_checkpoint))

            print("计算体积守恒...")
            vol_0V = evaluator.compute_volume(model, 0.0, 0.02)
            vol_30V = evaluator.compute_volume(model, 30.0, 0.02)
            vol_loss = (vol_30V - vol_0V) / vol_0V * 100 if vol_0V > 0 else 0

            theoretical_volume = 3e-6 / 20e-6  # 15%

            print(f"    Ink Vol (0V):  {vol_0V*100:.2f}% (理论: {theoretical_volume*100:.2f}%)")
            print(f"    Ink Vol (30V): {vol_30V*100:.2f}% (理论: {theoretical_volume*100:.2f}%)")
            print(f"    Volume Error:  {vol_loss:+.2f}% (目标: <1%)")

            # 保存结果
            result_file = output_path / f"volume_check_epoch_{current_epoch}.txt"
            with open(result_file, 'w') as f:
                f.write(f"Epoch: {current_epoch}\n")
                f.write(f"Ink Vol (0V): {vol_0V*100:.2f}%\n")
                f.write(f"Ink Vol (30V): {vol_30V*100:.2f}%\n")
                f.write(f"Volume Error: {vol_loss:+.2f}%\n")
                f.write(f"Theoretical Volume: {theoretical_volume*100:.2f}%\n")

            print(f"✅ 结果已保存: {result_file}")

            last_epoch = current_epoch

            # 如果达到目标，可以提前停止
            if abs(vol_loss) < 1.0:
                print("🎉 体积守恒目标达成！误差 < 1%")
                break

        except Exception as e:
            print(f"⚠️  监控出错: {e}")

        time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="监控 LevelSet 训练进度")
    parser.add_argument("--output-dir", required=True, help="训练输出目录")
    parser.add_argument("--interval", type=int, default=300, help="检查间隔（秒，默认300秒）")

    args = parser.parse_args()

    monitor_training(args.output_dir, args.interval)