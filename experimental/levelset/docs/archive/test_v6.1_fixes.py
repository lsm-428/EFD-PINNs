#!/usr/bin/env python3
"""
v6.1 修复验证脚本
===============

验证所有 v6.1 修复是否正确实施
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import json


def test_volume_constraint():
    """测试体积守恒约束"""
    print("测试 1: 体积守恒约束...")

    from pinn_levelset_3d import LevelSet3DPINN

    config = {
        "input_dim": 6,
        "output_dim": 10,
        "hidden_main": [64, 64, 32],
        "hidden_interface": [64, 64, 32],
        "hidden_corner": [64, 64, 64],
        "target_volume_fraction": 0.15,
        "volume_correction_strength": 0.1,
        "volume_constraint_enabled": True,
    }

    model = LevelSet3DPINN(config)

    x = torch.randn(100, 6)
    output = model(x)

    assert hasattr(model, "target_volume_fraction"), "缺少 target_volume_fraction 属性"
    assert hasattr(model, "volume_correction_strength"), (
        "缺少 volume_correction_strength 属性"
    )
    assert hasattr(model, "volume_constraint_enabled"), (
        "缺少 volume_constraint_enabled 属性"
    )
    assert hasattr(model, "enforce_volume_conservation"), (
        "缺少 enforce_volume_conservation 方法"
    )

    psi = output[:, 6:7]
    vol = torch.sigmoid(-psi).mean().item()

    print(f"  目标体积: {model.target_volume_fraction:.3f}")
    print(f"  当前体积: {vol:.3f}")
    print(f"  差异: {abs(vol - model.target_volume_fraction):.6f}")

    print("✅ 测试 1 通过\n")
    return True


def test_nan_detection():
    """测试 NaN 检测参数"""
    print("测试 2: NaN 检测参数...")

    from train_levelset_3d import LevelSet3DTrainer
    import tempfile

    config = {
        "input_dim": 6,
        "output_dim": 10,
        "epochs": 100,
        "learning_rate": 1e-3,
        "hidden_main": [32, 32],
        "hidden_interface": [32, 32],
        "hidden_corner": [32, 32],
        "num_interior": 100,
        "num_boundary": 20,
        "num_interface": 20,
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        trainer = LevelSet3DTrainer(config, output_dir)

        assert hasattr(trainer, "last_valid_state"), "缺少 last_valid_state 属性"
        assert hasattr(trainer, "nan_recovery_count"), "缺少 nan_recovery_count 属性"
        assert hasattr(trainer, "save_state_interval"), "缺少 save_state_interval 属性"

        assert trainer.last_valid_state is None, "last_valid_state 初始应为 None"
        assert trainer.nan_recovery_count == 0, "nan_recovery_count 初始应为 0"

        print("✅ 测试 2 通过\n")
        return True


def test_config_file():
    """测试 v6.1 配置文件"""
    print("测试 3: v6.1 配置文件...")

    config_path = Path(__file__).parent / "config" / "v6.1_fixed.json"

    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        return False

    with open(config_path) as f:
        config = json.load(f)

    assert "model" in config, "配置文件缺少 model 部分"
    assert "training" in config, "配置文件缺少 training 部分"

    model_config = config["model"]
    assert "target_volume_fraction" in model_config, "缺少 target_volume_fraction"
    assert "volume_correction_strength" in model_config, (
        "缺少 volume_correction_strength"
    )
    assert "volume_constraint_enabled" in model_config, "缺少 volume_constraint_enabled"

    assert model_config["target_volume_fraction"] == 0.15
    assert model_config["volume_correction_strength"] == 0.1
    assert model_config["volume_constraint_enabled"] == True

    print("✅ 测试 3 通过\n")
    return True


def main():
    print("=" * 60)
    print("v6.1 修复验证")
    print("=" * 60)
    print()

    results = []

    try:
        results.append(("体积守恒约束", test_volume_constraint()))
    except Exception as e:
        print(f"❌ 测试 1 失败: {e}\n")
        results.append(("体积守恒约束", False))

    try:
        results.append(("NaN 检测参数", test_nan_detection()))
    except Exception as e:
        print(f"❌ 测试 2 失败: {e}\n")
        results.append(("NaN 检测参数", False))

    try:
        results.append(("v6.1 配置文件", test_config_file()))
    except Exception as e:
        print(f"❌ 测试 3 失败: {e}\n")
        results.append(("v6.1 配置文件", False))

    print("=" * 60)
    print("验证结果汇总")
    print("=" * 60)

    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{name}: {status}")

    all_passed = all(r[1] for r in results)

    print()
    if all_passed:
        print("🎉 所有测试通过! v6.1 修复已正确实施")
        print()
        print("下一步:")
        print("  python3 train_levelset_3d.py --config config/v6.1_fixed.json")
        return 0
    else:
        print("⚠️  部分测试失败，请检查上述错误")
        return 1


if __name__ == "__main__":
    sys.exit(main())
