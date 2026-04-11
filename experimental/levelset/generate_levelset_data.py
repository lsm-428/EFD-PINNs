#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 3D Level Set PINN 训练数据
===============================

根据 LEVELSET_PHYSICS.md:
- ψ 是界面高度，始终 >= 0
- ψ > 0: 有油墨，值为界面高度
- ψ = 0: 界面在 z=0，最大开口
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from device_parameters import PHYSICS
from src.models.aperture_model import EnhancedApertureModel

CONFIG_PATH = Path(__file__).parent / "config" / "v5.5_full.json"
_APERTURE_MODEL = EnhancedApertureModel(config_path=str(CONFIG_PATH))
# _APERTURE_MODEL = None


def sample_continuous_times(n_samples: int, t_max: float) -> np.ndarray:
    """
    连续时间采样 (Beta 分布)

    使用 Beta(0.5, 1.0) 分布，自然地在早期 (t=0) 附近采样更多点，
    同时平滑覆盖整个时间域。

    数学特性:
    - Beta(0.5, 1.0) 在 t=0 处有奇点，密度趋于无穷大
    - 约 31.6% 的采样点会落在前 10% 的时间域内
    - 相比均匀分布，更有效地捕捉早期快速动态

    Args:
        n_samples: 采样点数量
        t_max: 最大时间 (秒)

    Returns:
        时间采样数组 (形状: [n_samples])
    """
    t_samples = np.random.beta(0.5, 1.0, n_samples) * t_max
    return t_samples


def compute_opening_rate(V: float, t: float) -> float:
    """
    计算开口率 η(V, t)

    基于 Young-Lippmann 方程和动态响应

    ⚠️ 注意：此函数固定 V_from=0，仅支持升压场景
    建议使用 compute_opening_rate_from_triad 支持完整场景
    """
    _, eta = _APERTURE_MODEL.theta_eta_from_triad(0.0, float(V), float(t))
    eta_max = float(PHYSICS.get("eta_max", 0.85))
    return float(np.clip(eta, 0.0, eta_max))


def compute_opening_rate_from_triad(
    V_from: float, V_to: float, t_since: float
) -> float:
    """
    计算开口率 η(V_from, V_to, t_since)

    支持三种电压场景：
    1. 稳态: V_from = V_to
    2. 升压: V_to > V_from
    3. 降压: V_to < V_from

    Args:
        V_from: 起始电压
        V_to: 目标电压
        t_since: 跳变后的时间

    Returns:
        开口率 η ∈ [0, 1]
    """
    V_from_val = float(V_from)
    V_to_val = float(V_to)
    t_val = max(0.0, float(t_since))

    # 使用 EnhancedApertureModel 的 theta_eta_from_triad 接口
    theta, eta = _APERTURE_MODEL.theta_eta_from_triad(V_from_val, V_to_val, t_val)

    eta_max = float(PHYSICS.get("eta_max", 0.85))
    return float(np.clip(eta, 0.0, eta_max))


def normalize_psi(psi_physical: float, scale: float = 3e-6) -> float:
    """
    将物理单位的 ψ 归一化到 [0, 1] 范围

    根据 LEVELSET_PHYSICS.md: ψ 始终 >= 0，表示界面高度

    Args:
        psi_physical: 物理单位的 ψ 值（0 到 ~3e-6）
        scale: 归一化尺度因子，默认 3μm（油墨层厚度）

    Returns:
        归一化到 [0, 1] 的 ψ 值
    """
    psi_normalized = psi_physical / scale
    return float(np.clip(psi_normalized, 0.0, 1.0))


def denormalize_psi(psi_normalized: float, scale: float = 3e-6) -> float:
    """
    将归一化的 ψ 转换回物理单位

    Args:
        psi_normalized: [0, 1] 范围的 ψ 值
        scale: 归一化尺度因子，必须与 normalize_psi 一致

    Returns:
        物理单位的 ψ 值
    """
    return psi_normalized * scale


def _sample_point_by_eta(eta: float, Lx: float, Ly: float, Lz: float, h_ink: float):
    """根据开口率 η 在界面附近采样点"""
    if eta < 0.01:
        x = np.random.uniform(0, Lx)
        y = np.random.uniform(0, Ly)
        z = h_ink + np.random.normal(0, h_ink * 0.2)
        z = np.clip(z, 0, Lz)
    else:
        r_open = np.sqrt(eta * Lx * Ly / np.pi)
        theta = np.random.uniform(0, 2 * np.pi)
        r = r_open + np.random.normal(0, 2e-6)
        r = np.clip(r, 0, np.sqrt(2) * Lx / 2)

        x = Lx / 2 + r * np.cos(theta)
        y = Ly / 2 + r * np.sin(theta)
        x = np.clip(x, 0, Lx)
        y = np.clip(y, 0, Ly)
        z = np.random.uniform(0, Lz)

    return x, y, z


def _sample_ink_point_by_eta(eta: float, Lx: float, Ly: float, Lz: float, h_ink: float):
    """在油墨区域采样点（ψ < 0）"""
    if eta < 0.01:
        x = np.random.uniform(0, Lx)
        y = np.random.uniform(0, Ly)
        z = np.random.uniform(0, h_ink * 0.8)
    else:
        r_open = np.sqrt(eta * Lx * Ly / np.pi)
        r_min = r_open + 3e-6
        r_max = np.sqrt(2) * Lx / 2

        if r_min < r_max:
            r = np.random.uniform(r_min, r_max)
            theta = np.random.uniform(0, 2 * np.pi)
            x = Lx / 2 + r * np.cos(theta)
            y = Ly / 2 + r * np.sin(theta)
            x = np.clip(x, 0, Lx)
            y = np.clip(y, 0, Ly)
        else:
            x = np.random.uniform(0, Lx)
            y = np.random.uniform(0, Ly)

        ink_area = Lx * Ly - np.pi * r_open**2
        h_ink_edge = Lx * Ly * h_ink / max(ink_area, 1e-12)
        h_ink_edge = min(h_ink_edge, Lz * 0.8)
        z = np.random.uniform(0, min(Lz, h_ink_edge * 0.8))

    return x, y, z


def _sample_polar_point_by_eta(
    eta: float, Lx: float, Ly: float, Lz: float, h_ink: float
):
    """在极性液体区域采样点（ψ > 0）

    修复版本 v5.1: 平衡方法 - 稳定性与物理准确性的平衡

    修复历史:
    - v5: 95% Z=0 → 数值不稳定 (LevelSet Loss 1e+7, Epoch 25400 NaN)
    - v5.1: 70% Z=0 → 平衡点

    v5.1 策略:
    - 降低固定 Z=0 比例: 95% → 70%
    - 添加微小噪声避免数值奇点
    - 保持 3D 空间多样性
    - 预期 30V 底部 Polar: 60-70% (vs v4: 30%, v5: 85%)
    """
    if eta < 0.01:
        # 低电压/无开口：极性液体在顶部
        x = np.random.uniform(0, Lx)
        y = np.random.uniform(0, Ly)
        z = np.random.uniform(h_ink * 1.2, Lz)
    else:
        # 有开口：极性液体区域（修复 v5.2：扩大采样范围）
        r_open = np.sqrt(eta * Lx * Ly / np.pi)

        # v5.2 修复：采样范围应该是整个像素，而不仅仅是开口区域
        # 这样模型才能学习到边界处也是极性液体（高电压时）
        # 采样策略：
        #   - 高开口 (eta > 0.5)：整个像素都是极性液体
        #   - 中等开口 (0.2 < eta <= 0.5)：大部分区域
        #   - 低开口 (eta <= 0.2)：仅在开口区域附近

        if eta > 0.5:
            # 高开口：采样整个像素（包括边界）
            r_max = Lx / 2 * 1.414  # 对角线的一半
        elif eta > 0.2:
            # 中等开口：采样到开口区域外侧
            r_max = r_open * 1.5
        else:
            # 低开口：仅在开口内
            r_max = r_open * 0.8

        r = np.random.uniform(0, r_max)
        theta = np.random.uniform(0, 2 * np.pi)
        x = Lx / 2 + r * np.cos(theta)
        y = Ly / 2 + r * np.sin(theta)
        x = np.clip(x, 0, Lx)
        y = np.clip(y, 0, Ly)

        # 修复 v5.1：平衡方法 - 稳定性优先
        # 关键：避免过度集中 Z=0 导致数值不稳定
        high_opening = eta > 0.3  # 阈值保持 30%

        if high_opening:
            # 高电压：70% 固定 Z=0（vs v5 的 95%）
            # 添加微小噪声避免数值奇点
            if np.random.random() < 0.70:  # 70% 固定 Z=0
                z = np.random.uniform(0, 1e-7)  # [0, 0.1μm] 微小扰动
            else:
                z = np.random.uniform(0, h_ink * 0.5)  # 近底部范围扩大
        else:
            # 低电压：可以采样不同高度
            if np.random.random() < 0.5:
                z = np.random.uniform(0, h_ink * 0.5)
            else:
                z = np.random.uniform(0, Lz)

    return x, y, z


def compute_psi(x: float, y: float, z: float, V: float, t: float) -> float:
    """
    计算 Level Set 函数 ψ(x, y, z, V, t)

    根据 LEVELSET_PHYSICS.md:
    - ψ 是界面高度，始终 >= 0
    - ψ > 0: 有油墨，值为界面高度
    - ψ = 0: 界面在 z=0，最大开口

    注意：ψ 不再是 "到界面的距离"，而是直接表示界面的 z 坐标
    """
    Lx = PHYSICS["Lx"]
    Ly = PHYSICS["Ly"]
    Lz = PHYSICS["Lz"]
    h_ink = PHYSICS.get("h_ink", 3e-6)

    cx, cy = Lx / 2, Ly / 2

    eta = compute_opening_rate(V, t)

    if eta < 0.01:
        interface_height = h_ink
        return interface_height

    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r_open = np.sqrt(eta * Lx * Ly / np.pi)

    ink_area = Lx * Ly - np.pi * r_open**2
    h_ink_edge = Lx * Ly * h_ink / max(ink_area, 1e-12)
    h_ink_edge = min(h_ink_edge, Lz * 0.8)

    if r < r_open:
        return 0.0
    elif r > r_open + 2e-6:
        return h_ink_edge
    else:
        return h_ink_edge * 0.5


def generate_training_data(
    num_interface: int = 50000,
    num_ink: int = 30000,
    num_polar: int = 30000,
    output_path: str = "data/levelset_training_data.pt",
):
    """
    生成训练数据（改进版：连续时间采样 + 完整电压场景）

    改进点：
    1. 使用 Beta 分布采样时间（重点覆盖早期动态）
    2. 支持三种电压场景：稳态(40%)、升压(30%)、降压(30%)
    3. 完整利用三元组输入格式 (V_from, V_to, t_since)
    """
    print("=" * 60)
    print("生成 3D Level Set 训练数据（改进版：连续时间模式）")
    print("=" * 60)

    Lx = PHYSICS["Lx"]
    Ly = PHYSICS["Ly"]
    Lz = PHYSICS["Lz"]
    h_ink = PHYSICS.get("h_ink", 3e-6)
    t_max = PHYSICS.get("t_max", 0.05)

    # ============================================================
    # 1. 界面点（ψ ≈ 0 附近）
    # ============================================================
    print(f"\n生成界面点 ({num_interface})...")
    interface_points = []
    interface_targets = []

    # 1.1 稳态 (30%)
    n_steady = int(num_interface * 0.3)
    V_steady = np.random.uniform(0, 30.0, n_steady)  # 连续电压
    t_steady = sample_continuous_times(n_steady, t_max)

    for V, t in zip(V_steady, t_steady):
        eta = compute_opening_rate_from_triad(V, V, t)  # V_from = V_to
        x, y, z = _sample_point_by_eta(eta, Lx, Ly, Lz, h_ink)
        psi = compute_psi(x, y, z, V, t)  # 传目标电压 V
        psi_normalized = normalize_psi(psi)
        interface_points.append([x, y, z, V, V, t])  # 三元组格式
        interface_targets.append(psi_normalized)

    print(f"   稳态: {n_steady}")

    # 1.2 升压 (30%)
    n_up = int(num_interface * 0.3)
    V_up = np.random.uniform(1.0, 30.0, n_up)
    t_up = sample_continuous_times(n_up, t_max)

    for V, t in zip(V_up, t_up):
        eta = compute_opening_rate_from_triad(0.0, V, t)
        x, y, z = _sample_point_by_eta(eta, Lx, Ly, Lz, h_ink)
        psi = compute_psi(x, y, z, V, t)  # 传目标电压 V
        psi_normalized = normalize_psi(psi)
        interface_points.append([x, y, z, 0.0, V, t])
        interface_targets.append(psi_normalized)

    print(f"   升压: {n_up}")

    # 1.3 降压 (30%)
    n_down = num_interface - n_steady - n_up
    V_down = np.random.uniform(1.0, 30.0, n_down)
    t_down = sample_continuous_times(n_down, t_max)

    for V, t in zip(V_down, t_down):
        eta = compute_opening_rate_from_triad(V, 0.0, t)
        x, y, z = _sample_point_by_eta(eta, Lx, Ly, Lz, h_ink)
        psi = compute_psi(x, y, z, 0.0, t)  # 传目标电压 0
        psi_normalized = normalize_psi(psi)
        interface_points.append([x, y, z, V, 0.0, t])
        interface_targets.append(psi_normalized)

    print(f"   降压: {n_down}")

    # ============================================================
    # 2. 油墨区域点（ψ < 0）
    # ============================================================
    print(f"\n生成油墨区域点 ({num_ink})...")
    ink_points = []
    ink_targets = []

    # 离散电压采样 + Triad 输入
    voltages = [0, 5, 10, 15, 20, 25, 30]
    n_ink_steady = int(num_ink * 0.3)
    n_ink_up = int(num_ink * 0.3)
    n_ink_down = num_ink - n_ink_steady - n_ink_up

    # 稳态：从离散电压列表中随机选择
    V_ink_steady = np.random.choice(voltages, n_ink_steady)
    t_ink_steady = sample_continuous_times(n_ink_steady, t_max)
    for V, t in zip(V_ink_steady, t_ink_steady):
        eta = compute_opening_rate_from_triad(V, V, t)
        x, y, z = _sample_ink_point_by_eta(eta, Lx, Ly, Lz, h_ink)
        psi = compute_psi(x, y, z, V, t)
        psi_normalized = normalize_psi(psi)
        ink_points.append([x, y, z, V, V, t])
        ink_targets.append(psi_normalized)

    # 升压：从离散电压列表中随机选择
    V_ink_up = np.random.choice(voltages, n_ink_up)
    t_ink_up = sample_continuous_times(n_ink_up, t_max)
    for V, t in zip(V_ink_up, t_ink_up):
        eta = compute_opening_rate_from_triad(0.0, V, t)
        x, y, z = _sample_ink_point_by_eta(eta, Lx, Ly, Lz, h_ink)
        psi = compute_psi(x, y, z, V, t)
        psi_normalized = normalize_psi(psi)
        ink_points.append([x, y, z, 0.0, V, t])
        ink_targets.append(psi_normalized)

    # 降压：从离散电压列表中随机选择
    V_ink_down = np.random.choice(voltages, n_ink_down)
    t_ink_down = sample_continuous_times(n_ink_down, t_max)
    for V, t in zip(V_ink_down, t_ink_down):
        eta = compute_opening_rate_from_triad(V, 0.0, t)
        x, y, z = _sample_ink_point_by_eta(eta, Lx, Ly, Lz, h_ink)
        psi = compute_psi(x, y, z, 0.0, t)
        psi_normalized = normalize_psi(psi)
        ink_points.append([x, y, z, V, 0.0, t])
        ink_targets.append(psi_normalized)

    print(f"   稳态: {n_ink_steady}, 升压: {n_ink_up}, 降压: {n_ink_down}")

    # ============================================================
    # 3. 极性液体区域点（ψ > 0）
    # ============================================================
    print(f"\n生成极性液体区域点 ({num_polar})...")
    polar_points = []
    polar_targets = []

    # 离散电压采样 + Triad 输入
    voltages = [0, 5, 10, 15, 20, 25, 30]
    n_polar_steady = int(num_polar * 0.3)
    n_polar_up = int(num_polar * 0.3)
    n_polar_down = num_polar - n_polar_steady - n_polar_up

    # 稳态：从离散电压列表中随机选择
    V_polar_steady = np.random.choice(voltages, n_polar_steady)
    t_polar_steady = sample_continuous_times(n_polar_steady, t_max)
    for V, t in zip(V_polar_steady, t_polar_steady):
        eta = compute_opening_rate_from_triad(V, V, t)
        x, y, z = _sample_polar_point_by_eta(eta, Lx, Ly, Lz, h_ink)
        psi = compute_psi(x, y, z, V, t)
        psi_normalized = normalize_psi(psi)
        polar_points.append([x, y, z, V, V, t])
        polar_targets.append(psi_normalized)

    # 升压：从离散电压列表中随机选择
    V_polar_up = np.random.choice(voltages, n_polar_up)
    t_polar_up = sample_continuous_times(n_polar_up, t_max)
    for V, t in zip(V_polar_up, t_polar_up):
        eta = compute_opening_rate_from_triad(0.0, V, t)
        x, y, z = _sample_polar_point_by_eta(eta, Lx, Ly, Lz, h_ink)
        psi = compute_psi(x, y, z, V, t)
        psi_normalized = normalize_psi(psi)
        polar_points.append([x, y, z, 0.0, V, t])
        polar_targets.append(psi_normalized)

    # 降压：从离散电压列表中随机选择
    V_polar_down = np.random.choice(voltages, n_polar_down)
    t_polar_down = sample_continuous_times(n_polar_down, t_max)
    for V, t in zip(V_polar_down, t_polar_down):
        eta = compute_opening_rate_from_triad(V, 0.0, t)
        x, y, z = _sample_polar_point_by_eta(eta, Lx, Ly, Lz, h_ink)
        psi = compute_psi(x, y, z, 0.0, t)
        psi_normalized = normalize_psi(psi)
        polar_points.append([x, y, z, V, 0.0, t])
        polar_targets.append(psi_normalized)

    print(f"   稳态: {n_polar_steady}, 升压: {n_polar_up}, 降压: {n_polar_down}")

    # 转换为张量
    data = {
        "interface_points": torch.tensor(
            interface_points, dtype=torch.float32, requires_grad=True
        ),
        "interface_targets": torch.tensor(interface_targets, dtype=torch.float32),
        "ink_points": torch.tensor(ink_points, dtype=torch.float32, requires_grad=True),
        "ink_targets": torch.tensor(ink_targets, dtype=torch.float32),
        "polar_points": torch.tensor(
            polar_points, dtype=torch.float32, requires_grad=True
        ),
        "polar_targets": torch.tensor(polar_targets, dtype=torch.float32),
    }

    # ========================================================================
    # 归一化到 [-1, 1] 范围（用于神经网络训练）
    # psi ∈ [-2e-6, 2e-6] → normalized ∈ [-1, 1]
    # ========================================================================
    print("\n归一化 ψ 到 [-1, 1]...")
    scale = 2e-6

    for key in ["interface_targets", "ink_targets", "polar_targets"]:
        data[key] = torch.clamp(data[key] / scale, -1.0, 1.0)

    # 保存
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, output_path)

    print(f"\n✅ 训练数据已保存: {output_path}")
    print(f"   界面点: {len(interface_points)}")
    print(f"   油墨点: {len(ink_points)}")
    print(f"   极性液体点: {len(polar_points)}")
    print(f"   总计: {len(interface_points) + len(ink_points) + len(polar_points)}")

    # 统计和验证
    print(f"\n数据统计 (归一化到 [-1, 1]):")
    print(
        f"   界面 ψ 范围: [{min(interface_targets):.4f}, {max(interface_targets):.4f}]"
    )
    print(f"   油墨 ψ 范围: [{min(ink_targets):.4f}, {max(ink_targets):.4f}]")
    print(f"   极性液体 ψ 范围: [{min(polar_targets):.4f}, {max(polar_targets):.4f}]")

    ink_negative = sum(1 for t in ink_targets if t < 0)
    polar_negative = sum(1 for t in polar_targets if t < 0)

    print(f"\n✅ 符号约定验证:")
    print(f"   油墨区域: {ink_negative}/{len(ink_targets)} 负值 (应始终为 0)")
    print(f"   极性液体: {polar_negative}/{len(polar_targets)} 负值 (应始终为 0)")

    if ink_negative > 0:
        print(f"   ⚠️  错误: 油墨区域有 {ink_negative} 个负值，违反 ψ >= 0 约束")
    if polar_negative > 0:
        print(f"   ⚠️  错误: 极性液体区域有 {polar_negative} 个负值，违反 ψ >= 0 约束")

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成 Level Set 训练数据")
    parser.add_argument(
        "--num-interface", type=int, default=50000, help="界面点数量 (默认: 50000)"
    )
    parser.add_argument(
        "--num-ink", type=int, default=30000, help="油墨区域点数量 (默认: 30000)"
    )
    parser.add_argument(
        "--num-polar", type=int, default=30000, help="极性液体区域点数量 (默认: 30000)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/levelset_training_data.pt",
        help="输出文件路径 (默认: data/levelset_training_data.pt)",
    )

    args = parser.parse_args()

    generate_training_data(
        num_interface=args.num_interface,
        num_ink=args.num_ink,
        num_polar=args.num_polar,
        output_path=args.output,
    )
