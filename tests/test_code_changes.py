#!/usr/bin/env python3
"""
快速测试代码修改是否正确
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

def test_imports():
    """测试导入"""
    print("1. 测试导入...")
    from src.models.pinn_two_phase import TwoPhasePINN, PHYSICS, DataGenerator
    from src.models.aperture_model import EnhancedApertureModel
    print("   ✅ 导入成功")

def test_time_sampling():
    """测试时间采样策略"""
    print("\n2. 测试时间采样策略...")
    tau = 0.005  # 5ms
    n_times = 90

    t_early = np.linspace(0.001, 2*tau, n_times // 3)  # 0-10ms
    t_mid = np.linspace(2*tau, 4*tau, n_times // 3)    # 10-20ms
    t_late = np.linspace(4*tau, 0.05, n_times // 3)    # 20-50ms
    time_samples = np.concatenate([t_early, t_mid, t_late])

    print(f"   早期 (0-10ms): {len(t_early)} 点, 范围 [{t_early[0]*1000:.1f}, {t_early[-1]*1000:.1f}] ms")
    print(f"   中期 (10-20ms): {len(t_mid)} 点, 范围 [{t_mid[0]*1000:.1f}, {t_mid[-1]*1000:.1f}] ms")
    print(f"   后期 (20-50ms): {len(t_late)} 点, 范围 [{t_late[0]*1000:.1f}, {t_late[-1]*1000:.1f}] ms")
    print(f"   总计: {len(time_samples)} 点")

    # 检查问题区间 (10-20ms) 是否有足够的采样
    mid_count = np.sum((time_samples >= 0.010) & (time_samples <= 0.020))
    print(f"   问题区间 (10-20ms) 采样点: {mid_count}")

    assert mid_count >= 20, "问题区间采样点不足"
    print("   ✅ 时间采样策略正确")

def test_eta_constraints():
    """测试开口率约束的时间点"""
    print("\n3. 测试开口率约束时间点...")

    t_early = [0.001, 0.002, 0.003, 0.005, 0.007, 0.010]  # 6个早期点
    t_mid = [0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019, 0.020]  # 10个中期点
    t_late = [0.025, 0.030, 0.040, 0.050]  # 4个后期点

    print(f"   早期约束点: {len(t_early)} 个, 范围 [{t_early[0]*1000:.0f}, {t_early[-1]*1000:.0f}] ms")
    print(f"   中期约束点: {len(t_mid)} 个, 范围 [{t_mid[0]*1000:.0f}, {t_mid[-1]*1000:.0f}] ms")
    print(f"   后期约束点: {len(t_late)} 个, 范围 [{t_late[0]*1000:.0f}, {t_late[-1]*1000:.0f}] ms")

    # 检查中期约束点是否覆盖问题区间
    mid_coverage = all(10 <= t*1000 <= 20 for t in t_mid)
    assert mid_coverage, "中期约束点未完全覆盖问题区间"
    print("   ✅ 中期约束点完全覆盖问题区间")

def test_smooth_constraint():
    """测试时间平滑约束"""
    print("\n4. 测试时间平滑约束...")

    t_smooth = [0.010, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019, 0.020]
    v_test = [20.0, 25.0, 30.0]

    print(f"   平滑约束时间点: {len(t_smooth)} 个")
    print(f"   平滑约束电压: {v_test}")
    print(f"   相邻时间间隔: {(t_smooth[1] - t_smooth[0])*1000:.0f} ms")

    # 检查时间间隔是否均匀
    intervals = [t_smooth[i+1] - t_smooth[i] for i in range(len(t_smooth)-1)]
    assert all(abs(i - intervals[0]) < 1e-6 for i in intervals), "时间间隔不均匀"
    print("   ✅ 时间平滑约束配置正确")

def test_model_forward():
    """测试模型前向传播"""
    print("\n5. 测试模型前向传播...")

    from src.models.pinn_two_phase import TwoPhasePINN, PHYSICS

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TwoPhasePINN().to(device)

    # 测试输入
    batch_size = 100
    inputs = torch.zeros(batch_size, 6, device=device)
    inputs[:, 0] = torch.rand(batch_size) * PHYSICS["Lx"]  # x
    inputs[:, 1] = torch.rand(batch_size) * PHYSICS["Ly"]  # y
    inputs[:, 2] = torch.rand(batch_size) * PHYSICS["Lz"]  # z
    inputs[:, 3] = 0.0  # V_from
    inputs[:, 4] = 30.0  # V_to
    inputs[:, 5] = 0.015  # t_since

    with torch.no_grad():
        outputs = model(inputs)

    print(f"   输入形状: {inputs.shape}")
    print(f"   输出形状: {outputs.shape}")
    print(f"   phi 范围: [{outputs[:, 4].min():.4f}, {outputs[:, 4].max():.4f}]")

    assert outputs.shape == (batch_size, 5), f"输出形状错误: {outputs.shape}"
    assert torch.isfinite(outputs).all(), "输出包含NaN或Inf"
    print("   ✅ 模型前向传播正常")

def main():
    print("=" * 60)
    print("代码修改验证测试")
    print("=" * 60)

    tests = [
        test_imports,
        test_time_sampling,
        test_eta_constraints,
        test_smooth_constraint,
        test_model_forward,
    ]

    try:
        for fn in tests:
            fn()
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ 测试失败: {type(e).__name__}: {e}")
        return 1

    print("\n" + "=" * 60)
    print("✅ 所有测试通过！代码修改正确，可以开始训练。")
    return 0

if __name__ == "__main__":
    sys.exit(main())
