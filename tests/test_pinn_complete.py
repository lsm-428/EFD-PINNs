#!/usr/bin/env python3
"""
PINN 模型完整测试套件
====================

整合三大测试模块：
1. 物理验证 - φ 场范围、边界条件、守恒定律
2. 解析对比 - 与解析模型的一致性
3. 鲁棒性测试 - 噪声、外推、边界情况

使用方法:
    python test_pinn_complete.py                    # 测试最新模型
    python test_pinn_complete.py outputs_pinn_xxx   # 测试指定目录
    python test_pinn_complete.py --all              # 测试所有模型

作者: EFD-PINNs Team
"""

import os
import sys
import glob
import json
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.pinn_two_phase import TwoPhasePINN, PHYSICS, DEFAULT_CONFIG
from matplotlib.colors import LinearSegmentedColormap

# 设置字体
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 专业 EWD 色图
EWD_CMAP = LinearSegmentedColormap.from_list("EWD", ["#E0FFFF", "#FF00FF"])


# ============================================================================
# 模型加载
# ============================================================================

def load_model(checkpoint_path: str) -> Tuple[TwoPhasePINN, torch.device, Dict]:
    """加载模型和配置"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    config = checkpoint.get("config", DEFAULT_CONFIG)
    model = TwoPhasePINN(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    info = {
        "epoch": checkpoint.get("epoch", "N/A"),
        "best_loss": checkpoint.get("best_loss", float("inf")),
        "config": config
    }
    
    return model, device, info


def predict_phi(model, device, x, y, z, V_from, V_to, t_since) -> np.ndarray:
    """预测 φ 场（支持批量）"""
    x = np.atleast_1d(x).astype(np.float32)
    y = np.atleast_1d(y).astype(np.float32)
    z = np.atleast_1d(z).astype(np.float32)
    
    n = len(x)
    V_from_arr = np.full(n, V_from, dtype=np.float32)
    V_to_arr = np.full(n, V_to, dtype=np.float32)
    t_arr = np.full(n, t_since, dtype=np.float32)
    
    inputs = np.stack([x, y, z, V_from_arr, V_to_arr, t_arr], axis=1)
    
    with torch.no_grad():
        outputs = model(torch.tensor(inputs, device=device))
        phi = outputs[:, 4].cpu().numpy()
    
    return phi


def compute_aperture(model, device, V_from, V_to, t_since, n=50) -> Tuple[float, np.ndarray]:
    """计算开口率"""
    Lx, Ly, h_ink = PHYSICS["Lx"], PHYSICS["Ly"], PHYSICS["h_ink"]
    
    x = np.linspace(0, Lx, n)
    y = np.linspace(0, Ly, n)
    X, Y = np.meshgrid(x, y)
    
    phi = predict_phi(model, device, X.flatten(), Y.flatten(), 
                      np.full(n*n, h_ink/2), V_from, V_to, t_since)
    
    eta = np.mean(phi < 0.5)
    return eta, phi.reshape(n, n)


# ============================================================================
# 1. 物理验证测试
# ============================================================================

class PhysicsValidator:
    """物理合理性验证"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.results = {}
    
    def test_phi_range(self) -> Dict:
        """测试 φ 值范围 [0, 1]"""
        print("\n[1.1] φ 值范围测试")
        
        voltages = [0, 10, 20, 30]
        times = [0.001, 0.005, 0.01, 0.02]
        
        violations = []
        for V in voltages:
            for t in times:
                _, phi = compute_aperture(self.model, self.device, 0, V, t, n=30)
                phi_min, phi_max = phi.min(), phi.max()
                
                if phi_min < -0.05 or phi_max > 1.05:
                    violations.append({"V": V, "t": t, "min": phi_min, "max": phi_max})
        
        passed = len(violations) == 0
        print(f"  {'✅' if passed else '❌'} φ ∈ [0,1]: {16-len(violations)}/16 通过")
        
        self.results["phi_range"] = {"passed": passed, "violations": violations}
        return self.results["phi_range"]
    
    def test_initial_condition(self) -> Dict:
        """测试初始条件：t=0 时 φ≈1（油墨覆盖）"""
        print("\n[1.2] 初始条件测试")
        
        errors = []
        for V in [0, 10, 20, 30]:
            eta, phi = compute_aperture(self.model, self.device, V, V, 0.0001, n=30)
            mean_phi = phi.mean()
            
            if mean_phi < 0.9:
                errors.append({"V": V, "mean_phi": mean_phi, "eta": eta})
        
        passed = len(errors) == 0
        print(f"  {'✅' if passed else '❌'} t≈0 时 φ≈1: {4-len(errors)}/4 通过")
        
        self.results["initial_condition"] = {"passed": passed, "errors": errors}
        return self.results["initial_condition"]
    
    def test_zero_voltage(self) -> Dict:
        """测试零电压：V=0 时 φ≈1（油墨不动）"""
        print("\n[1.3] 零电压测试")
        
        errors = []
        for t in [0.005, 0.01, 0.02, 0.04]:
            eta, phi = compute_aperture(self.model, self.device, 0, 0, t, n=30)
            
            if eta > 0.05:
                errors.append({"t": t, "eta": eta})
        
        passed = len(errors) == 0
        print(f"  {'✅' if passed else '❌'} V=0 时 η≈0: {4-len(errors)}/4 通过")
        
        self.results["zero_voltage"] = {"passed": passed, "errors": errors}
        return self.results["zero_voltage"]
    
    def test_monotonicity(self) -> Dict:
        """测试单调性：η 随时间单调增加（升压时）"""
        print("\n[1.4] 时间单调性测试")
        
        violations = []
        for V in [15, 20, 25, 30]:
            times = [0.002, 0.005, 0.008, 0.012, 0.015]
            etas = [compute_aperture(self.model, self.device, 0, V, t, n=30)[0] for t in times]
            
            for i in range(len(etas)-1):
                if etas[i+1] < etas[i] - 0.02:  # 允许小波动
                    violations.append({"V": V, "t1": times[i], "t2": times[i+1], 
                                       "eta1": etas[i], "eta2": etas[i+1]})
        
        passed = len(violations) == 0
        print(f"  {'✅' if passed else '⚠️'} η(t) 单调增: {4-len(violations)}/4 通过")
        
        self.results["monotonicity"] = {"passed": passed, "violations": violations}
        return self.results["monotonicity"]
    
    def test_voltage_response(self) -> Dict:
        """测试电压响应：η 随电压单调增加"""
        print("\n[1.5] 电压响应测试")
        
        violations = []
        t = 0.015
        voltages = [5, 10, 15, 20, 25, 30]
        etas = [compute_aperture(self.model, self.device, 0, V, t, n=30)[0] for V in voltages]
        
        for i in range(len(etas)-1):
            if etas[i+1] < etas[i] - 0.02:
                violations.append({"V1": voltages[i], "V2": voltages[i+1],
                                   "eta1": etas[i], "eta2": etas[i+1]})
        
        passed = len(violations) == 0
        print(f"  {'✅' if passed else '⚠️'} η(V) 单调增: {5-len(violations)}/5 通过")
        
        self.results["voltage_response"] = {"passed": passed, "violations": violations}
        return self.results["voltage_response"]
    
    def test_phi_field_error(self) -> Dict:
        """测试 φ 场与目标值的 MAE/MSE"""
        print("\n[1.6] φ 场误差测试 (MAE/MSE)")
        
        from src.models.pinn_two_phase import DataGenerator, DEFAULT_CONFIG
        
        # 创建数据生成器获取目标值
        data_gen = DataGenerator(DEFAULT_CONFIG, self.device)
        
        Lx, Ly, h_ink = PHYSICS["Lx"], PHYSICS["Ly"], PHYSICS["h_ink"]
        n = 40
        x = np.linspace(0, Lx, n)
        y = np.linspace(0, Ly, n)
        X, Y = np.meshgrid(x, y)
        z = h_ink / 2
        
        # 测试多个条件
        test_cases = [
            (0, 0, 0.01, "V=0, t=10ms"),
            (0, 15, 0.01, "0->15V, t=10ms"),
            (0, 30, 0.01, "0->30V, t=10ms"),
            (0, 30, 0.02, "0->30V, t=20ms"),
            (30, 0, 0.01, "30->0V, t=10ms"),
        ]
        
        all_mae = []
        all_mse = []
        details = []
        
        for V_from, V_to, t, label in test_cases:
            # PINN 预测
            phi_pinn = predict_phi(self.model, self.device, X.flatten(), Y.flatten(),
                                   np.full(n*n, z), V_from, V_to, t)
            phi_pinn = phi_pinn.reshape(n, n)
            
            # 目标值
            phi_target = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    phi_target[i, j] = data_gen.target_phi_3d(x[j], y[i], z, t, V_to, V_prev=V_from)
            
            # 计算误差
            mae = np.mean(np.abs(phi_pinn - phi_target))
            mse = np.mean((phi_pinn - phi_target)**2)
            
            all_mae.append(mae)
            all_mse.append(mse)
            details.append({"case": label, "mae": float(mae), "mse": float(mse)})
        
        avg_mae = np.mean(all_mae)
        avg_mse = np.mean(all_mse)
        
        passed = avg_mae < 0.15
        print(f"  {'✅' if passed else '⚠️'} 平均 MAE: {avg_mae:.4f}, MSE: {avg_mse:.6f}")
        for d in details:
            print(f"      {d['case']}: MAE={d['mae']:.4f}, MSE={d['mse']:.6f}")
        
        self.results["phi_field_error"] = {
            "passed": passed, 
            "avg_mae": float(avg_mae), 
            "avg_mse": float(avg_mse),
            "details": details
        }
        return self.results["phi_field_error"]
    
    def run_all(self) -> Dict:
        """运行所有物理验证"""
        print("\n" + "="*60)
        print("1. 物理验证测试")
        print("="*60)
        
        self.test_phi_range()
        self.test_initial_condition()
        self.test_zero_voltage()
        self.test_monotonicity()
        self.test_voltage_response()
        self.test_phi_field_error()
        
        total = len(self.results)
        passed = sum(1 for r in self.results.values() if r["passed"])
        print(f"\n物理验证总结: {passed}/{total} 通过")
        
        return self.results


# ============================================================================
# 2. 解析对比测试
# ============================================================================

class AnalyticalComparator:
    """与解析模型对比"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.results = {}
    
    def _analytical_aperture(self, V, t) -> float:
        """简化解析模型计算开口率"""
        if V <= 0:
            return 0.0
        
        theta0 = PHYSICS.get("theta0", 120)
        tau = PHYSICS.get("tau", 0.005)
        
        # Young-Lippmann 稳态接触角
        epsilon_0 = 8.854e-12
        epsilon_r = PHYSICS.get("epsilon_r", 12)
        d = PHYSICS.get("d_dielectric", 4e-7)
        gamma = PHYSICS.get("sigma", 0.045)
        
        cos_theta0 = np.cos(np.radians(theta0))
        ew_term = (epsilon_0 * epsilon_r * V**2) / (2 * gamma * d)
        cos_theta_eq = np.clip(cos_theta0 + ew_term, -1, 1)
        theta_eq = np.degrees(np.arccos(cos_theta_eq))
        
        # 动态响应
        theta_t = theta_eq + (theta0 - theta_eq) * np.exp(-t / tau)
        
        # 接触角到开口率
        theta_min = 60  # 最小接触角
        eta = np.clip((theta0 - theta_t) / (theta0 - theta_min), 0, 0.9)
        
        return eta
    
    def test_steady_state(self) -> Dict:
        """测试稳态开口率"""
        print("\n[2.1] 稳态开口率对比")
        
        t = 0.020  # 稳态时间
        voltages = [10, 15, 20, 25, 30]
        
        errors = []
        for V in voltages:
            eta_pinn, _ = compute_aperture(self.model, self.device, 0, V, t, n=40)
            eta_anal = self._analytical_aperture(V, t)
            error = abs(eta_pinn - eta_anal)
            errors.append({"V": V, "pinn": eta_pinn, "analytical": eta_anal, "error": error})
        
        mae = np.mean([e["error"] for e in errors])
        passed = mae < 0.15
        print(f"  {'✅' if passed else '⚠️'} 稳态 MAE: {mae:.4f}")
        
        self.results["steady_state"] = {"passed": passed, "mae": mae, "details": errors}
        return self.results["steady_state"]
    
    def test_dynamic_response(self) -> Dict:
        """测试动态响应"""
        print("\n[2.2] 动态响应对比")
        
        V = 30
        times = [0.002, 0.005, 0.008, 0.012, 0.015, 0.020]
        
        errors = []
        for t in times:
            eta_pinn, _ = compute_aperture(self.model, self.device, 0, V, t, n=40)
            eta_anal = self._analytical_aperture(V, t)
            error = abs(eta_pinn - eta_anal)
            errors.append({"t": t, "pinn": eta_pinn, "analytical": eta_anal, "error": error})
        
        mae = np.mean([e["error"] for e in errors])
        passed = mae < 0.15
        print(f"  {'✅' if passed else '⚠️'} 动态 MAE: {mae:.4f}")
        
        self.results["dynamic_response"] = {"passed": passed, "mae": mae, "details": errors}
        return self.results["dynamic_response"]
    
    def test_recovery(self) -> Dict:
        """测试降压恢复"""
        print("\n[2.3] 降压恢复测试")
        
        # 从 30V 降到 0V
        times = [0.001, 0.005, 0.010, 0.020, 0.030]
        
        etas = []
        for t in times:
            eta, _ = compute_aperture(self.model, self.device, 30, 0, t, n=40)
            etas.append(eta)
        
        # 检查恢复趋势
        decreasing = all(etas[i] >= etas[i+1] - 0.05 for i in range(len(etas)-1))
        final_low = etas[-1] < 0.1
        
        passed = decreasing and final_low
        print(f"  {'✅' if passed else '⚠️'} 降压恢复: η 从 {etas[0]:.3f} 降到 {etas[-1]:.3f}")
        
        self.results["recovery"] = {"passed": passed, "etas": etas, "times": times}
        return self.results["recovery"]
    
    def run_all(self) -> Dict:
        """运行所有解析对比"""
        print("\n" + "="*60)
        print("2. 解析对比测试")
        print("="*60)
        
        self.test_steady_state()
        self.test_dynamic_response()
        self.test_recovery()
        
        total = len(self.results)
        passed = sum(1 for r in self.results.values() if r["passed"])
        print(f"\n解析对比总结: {passed}/{total} 通过")
        
        return self.results


# ============================================================================
# 3. 鲁棒性测试
# ============================================================================

class RobustnessValidator:
    """鲁棒性测试"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.results = {}
    
    def test_input_noise(self) -> Dict:
        """测试输入噪声敏感性"""
        print("\n[3.1] 输入噪声敏感性")
        
        Lx, Ly, h_ink = PHYSICS["Lx"], PHYSICS["Ly"], PHYSICS["h_ink"]
        V, t = 20, 0.015
        
        # 基准预测
        n = 30
        x = np.linspace(0, Lx, n)
        y = np.linspace(0, Ly, n)
        X, Y = np.meshgrid(x, y)
        
        phi_base = predict_phi(self.model, self.device, X.flatten(), Y.flatten(),
                               np.full(n*n, h_ink/2), 0, V, t)
        
        # 添加噪声
        noise_levels = [0.01, 0.05, 0.10]  # 相对噪声
        sensitivities = []
        
        for noise in noise_levels:
            x_noisy = X.flatten() * (1 + np.random.randn(n*n) * noise)
            y_noisy = Y.flatten() * (1 + np.random.randn(n*n) * noise)
            x_noisy = np.clip(x_noisy, 0, Lx)
            y_noisy = np.clip(y_noisy, 0, Ly)
            
            phi_noisy = predict_phi(self.model, self.device, x_noisy, y_noisy,
                                    np.full(n*n, h_ink/2), 0, V, t)
            
            mae = np.mean(np.abs(phi_noisy - phi_base))
            sensitivities.append({"noise": noise, "mae": mae})
        
        # 检查敏感性是否合理（噪声增加，误差应该适度增加）
        max_sensitivity = max(s["mae"] for s in sensitivities)
        passed = max_sensitivity < 0.3
        
        print(f"  {'✅' if passed else '⚠️'} 最大敏感性: {max_sensitivity:.4f}")
        
        self.results["input_noise"] = {"passed": passed, "sensitivities": sensitivities}
        return self.results["input_noise"]
    
    def test_high_voltage_stability(self) -> Dict:
        """测试高电压稳定性 (实际工作: 0-20V, 训练: 0-30V)"""
        print("\n[3.2] 高电压稳定性测试 (20-30V)")
        
        t = 0.015
        eta_max = PHYSICS.get("eta_max", 0.85)
        
        # 实际工作边界
        eta_20, _ = compute_aperture(self.model, self.device, 0, 20, t, n=30)
        
        # 训练范围内的高电压 (20-30V，已超出实际工作范围)
        eta_25, _ = compute_aperture(self.model, self.device, 0, 25, t, n=30)
        eta_30, _ = compute_aperture(self.model, self.device, 0, 30, t, n=30)
        
        # 检查高电压区域是否稳定（单调递增，不突变，不超过物理约束）
        reasonable = all(0 <= eta <= 1 for eta in [eta_20, eta_25, eta_30])
        monotonic = eta_25 >= eta_20 - 0.05 and eta_30 >= eta_25 - 0.05
        within_physics = all(eta <= eta_max + 0.05 for eta in [eta_20, eta_25, eta_30])
        
        passed = reasonable and monotonic and within_physics
        print(f"  实际边界 V=20V: η={eta_20:.3f} (物理上限: {eta_max})")
        status = '✅' if passed else '❌'
        print(f"  {status} 高电压 V=25V: η={eta_25:.3f}, V=30V: η={eta_30:.3f}")
        if not within_physics:
            print(f"  ⚠️ 警告: 开口率超过物理约束 eta_max={eta_max}")
        
        self.results["high_voltage_stability"] = {
            "passed": passed, 
            "eta_max": eta_max,
            "within_physics": within_physics,
            "eta_20": float(eta_20),
            "eta_25": float(eta_25),
            "eta_30": float(eta_30)
        }
        return self.results["high_voltage_stability"]
    
    def test_time_extrapolation(self) -> Dict:
        """测试时间外推 (训练范围: 0-50ms)"""
        print("\n[3.3] 时间外推测试 (Train: 0-50ms)")
        
        V = 30
        eta_max = PHYSICS.get("eta_max", 0.85)
        
        # 训练边界
        eta_50ms, _ = compute_aperture(self.model, self.device, 0, V, 0.050, n=30)
        
        # 外推到更长时间 (>50ms)
        eta_60ms, _ = compute_aperture(self.model, self.device, 0, V, 0.060, n=30)
        eta_80ms, _ = compute_aperture(self.model, self.device, 0, V, 0.080, n=30)
        eta_100ms, _ = compute_aperture(self.model, self.device, 0, V, 0.100, n=30)
        
        # 检查外推是否合理：
        # 1. 稳态应该保持（不应大幅波动）
        # 2. 不应超过物理约束 eta_max
        stable = abs(eta_60ms - eta_50ms) < 0.1 and abs(eta_100ms - eta_50ms) < 0.15
        within_physics = all(eta <= eta_max + 0.05 for eta in [eta_50ms, eta_60ms, eta_80ms, eta_100ms])
        reasonable = all(0 <= eta <= 1 for eta in [eta_60ms, eta_80ms, eta_100ms])
        
        passed = stable and reasonable and within_physics
        
        print(f"  边界 t=50ms: η={eta_50ms:.3f} (物理上限: {eta_max})")
        status = '✅' if passed else '❌'
        print(f"  {status} 外推 t=60ms: η={eta_60ms:.3f}, t=80ms: η={eta_80ms:.3f}, t=100ms: η={eta_100ms:.3f}")
        if not within_physics:
            print(f"  ⚠️ 警告: 开口率超过物理约束 eta_max={eta_max}")
        
        self.results["time_extrapolation"] = {
            "passed": passed,
            "eta_max": eta_max,
            "within_physics": within_physics,
            "train_boundary": {"t_ms": 50, "eta": float(eta_50ms)},
            "extrapolation": [
                {"t_ms": 60, "eta": float(eta_60ms)},
                {"t_ms": 80, "eta": float(eta_80ms)},
                {"t_ms": 100, "eta": float(eta_100ms)}
            ]
        }
        return self.results["time_extrapolation"]
    
    def test_boundary_cases(self) -> Dict:
        """测试边界情况"""
        print("\n[3.4] 边界情况测试")
        
        Lx, Ly, h_ink = PHYSICS["Lx"], PHYSICS["Ly"], PHYSICS["h_ink"]
        
        # 测试边界点
        boundary_points = [
            (0, Ly/2, "x=0"),
            (Lx, Ly/2, "x=Lx"),
            (Lx/2, 0, "y=0"),
            (Lx/2, Ly, "y=Ly"),
            (0, 0, "corner (0,0)"),
            (Lx, Ly, "corner (Lx,Ly)"),
        ]
        
        errors = []
        for x, y, name in boundary_points:
            phi = predict_phi(self.model, self.device, [x], [y], [h_ink/2], 0, 30, 0.015)
            
            if not (0 <= phi[0] <= 1):
                errors.append({"point": name, "phi": phi[0]})
        
        passed = len(errors) == 0
        print(f"  {'✅' if passed else '❌'} 边界点 φ∈[0,1]: {6-len(errors)}/6 通过")
        
        self.results["boundary_cases"] = {"passed": passed, "errors": errors}
        return self.results["boundary_cases"]
    
    def test_intermediate_voltages(self) -> Dict:
        """测试中间电压（训练数据稀疏区域）"""
        print("\n[3.5] 中间电压测试")
        
        t = 0.015
        
        # 测试非整数电压
        test_voltages = [7.5, 12.5, 17.5, 22.5, 27.5]
        
        errors = []
        for V in test_voltages:
            eta, phi = compute_aperture(self.model, self.device, 0, V, t, n=30)
            
            # 检查是否在相邻整数电压之间
            V_low, V_high = int(V), int(V) + 1
            eta_low, _ = compute_aperture(self.model, self.device, 0, V_low, t, n=30)
            eta_high, _ = compute_aperture(self.model, self.device, 0, V_high, t, n=30)
            
            # 应该在两者之间（允许小误差）
            in_range = (eta_low - 0.05) <= eta <= (eta_high + 0.05)
            
            if not in_range:
                errors.append({"V": V, "eta": eta, "eta_low": eta_low, "eta_high": eta_high})
        
        passed = len(errors) <= 1  # 允许1个异常
        print(f"  {'✅' if passed else '⚠️'} 中间电压插值: {5-len(errors)}/5 通过")
        
        self.results["intermediate_voltages"] = {"passed": passed, "errors": errors}
        return self.results["intermediate_voltages"]
    
    def run_all(self) -> Dict:
        """运行所有鲁棒性测试"""
        print("\n" + "="*60)
        print("3. 鲁棒性测试")
        print("="*60)
        
        self.test_input_noise()
        self.test_high_voltage_stability()
        self.test_time_extrapolation()
        self.test_boundary_cases()
        self.test_intermediate_voltages()
        
        total = len(self.results)
        passed = sum(1 for r in self.results.values() if r["passed"])
        print(f"\n鲁棒性测试总结: {passed}/{total} 通过")
        
        return self.results


# ============================================================================
# 可视化报告
# ============================================================================

def generate_report(model, device, output_dir: str, results: Dict):
    """生成可视化测试报告"""
    print("\n" + "="*60)
    print("生成测试报告")
    print("="*60)
    
    fig = plt.figure(figsize=(20, 16))
    
    # 1. 开口率 vs 电压（稳态）
    ax1 = fig.add_subplot(2, 3, 1)
    voltages = np.linspace(0, 35, 36)
    t = 0.015
    etas = [compute_aperture(model, device, 0, V, t, n=30)[0] for V in voltages]
    ax1.plot(voltages, etas, 'b-o', markersize=4, linewidth=2)
    ax1.axvline(x=30, color='r', linestyle='--', alpha=0.5, label='Train Boundary')
    ax1.set_xlabel('Voltage (V)')
    ax1.set_ylabel('Aperture Ratio η')
    ax1.set_title('η vs V (t=15ms)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 35)
    ax1.set_ylim(0, 1)
    
    # 2. 开口率 vs 时间
    ax2 = fig.add_subplot(2, 3, 2)
    times = np.linspace(0, 0.05, 50)
    for V in [10, 20, 30]:
        etas = [compute_aperture(model, device, 0, V, t, n=30)[0] for t in times]
        ax2.plot(times*1000, etas, label=f'V={V}V', linewidth=2)
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Aperture Ratio η')
    ax2.set_title('η vs t (Rise)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 50)
    ax2.set_ylim(0, 1)
    
    # 3. 降压恢复
    ax3 = fig.add_subplot(2, 3, 3)
    times = np.linspace(0, 0.04, 40)
    for V_prev in [15, 20, 25, 30]:
        etas = [compute_aperture(model, device, V_prev, 0, t, n=30)[0] for t in times]
        ax3.plot(times*1000, etas, label=f'{V_prev}V->0V', linewidth=2)
    ax3.set_xlabel('Time since fall (ms)')
    ax3.set_ylabel('Aperture Ratio η')
    ax3.set_title('Recovery (Fall)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 40)
    ax3.set_ylim(0, 1)
    
    # 4. φ 场分布 (V=30V, t=15ms)
    ax4 = fig.add_subplot(2, 3, 4)
    _, phi = compute_aperture(model, device, 0, 30, 0.015, n=80)
    Lx, Ly = PHYSICS["Lx"], PHYSICS["Ly"]
    x = np.linspace(0, Lx, 80)
    y = np.linspace(0, Ly, 80)
    X, Y = np.meshgrid(x, y)
    im = ax4.contourf(X*1e6, Y*1e6, phi, levels=20, cmap=EWD_CMAP, vmin=0, vmax=1)
    ax4.contour(X*1e6, Y*1e6, phi, levels=[0.5], colors='black', linewidths=2)
    ax4.set_xlabel('x (μm)')
    ax4.set_ylabel('y (μm)')
    ax4.set_title('φ Field (V=30V, t=15ms)')
    ax4.set_aspect('equal')
    plt.colorbar(im, ax=ax4, label='φ')
    
    # 5. 测试结果汇总
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.axis('off')
    
    summary_text = "Test Results Summary\n" + "="*40 + "\n\n"
    
    for category, tests in results.items():
        passed = sum(1 for t in tests.values() if t.get("passed", False))
        total = len(tests)
        status = "PASS" if passed == total else "WARN"
        summary_text += f"[{status}] {category}: {passed}/{total}\n"
    
    # 总体评分
    all_tests = []
    for tests in results.values():
        all_tests.extend(tests.values())
    total_passed = sum(1 for t in all_tests if t.get("passed", False))
    total_tests = len(all_tests)
    score = total_passed / total_tests * 100 if total_tests > 0 else 0
    
    summary_text += "\n" + "="*40 + "\n"
    summary_text += f"Overall Score: {score:.1f}% ({total_passed}/{total_tests})\n"
    
    if score >= 90:
        summary_text += "Grade: Excellent ***"
    elif score >= 70:
        summary_text += "Grade: Good **"
    elif score >= 50:
        summary_text += "Grade: Pass *"
    else:
        summary_text += "Grade: Needs Improvement"
    
    ax5.text(0.1, 0.9, summary_text, transform=ax5.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 6. 鲁棒性雷达图
    ax6 = fig.add_subplot(2, 3, 6, projection='polar')
    
    if "robustness" in results:
        categories = list(results["robustness"].keys())
        values = [1 if results["robustness"][c].get("passed", False) else 0 for c in categories]
        
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]
        
        ax6.plot(angles, values, 'o-', linewidth=2)
        ax6.fill(angles, values, alpha=0.25)
        ax6.set_xticks(angles[:-1])
        ax6.set_xticklabels([c[:10] for c in categories], size=8)
        ax6.set_ylim(0, 1)
        ax6.set_title('Robustness Tests')
    
    plt.suptitle('PINN Model Complete Test Report', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    report_path = os.path.join(output_dir, "test_report.png")
    plt.savefig(report_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 测试报告已保存: {report_path}")
    
    # 保存 JSON 结果
    json_path = os.path.join(output_dir, "test_results.json")
    
    # 转换为可序列化格式
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        return obj
    
    with open(json_path, 'w') as f:
        json.dump(convert_to_serializable(results), f, indent=2)
    
    print(f"✅ 测试结果已保存: {json_path}")
    
    return score


# ============================================================================
# 主函数
# ============================================================================

def run_complete_test(model_dir: str) -> Dict:
    """运行完整测试"""
    print("\n" + "#"*60)
    print(f"# PINN 模型完整测试")
    print(f"# 目录: {model_dir}")
    print("#"*60)
    
    # 查找模型文件
    checkpoint_path = None
    for name in ["best_model.pth", "final_model.pth"]:
        path = os.path.join(model_dir, name)
        if os.path.exists(path):
            checkpoint_path = path
            break
    
    if checkpoint_path is None:
        print(f"❌ 未找到模型文件")
        return None
    
    print(f"\n📁 模型文件: {checkpoint_path}")
    
    # 加载模型
    try:
        model, device, info = load_model(checkpoint_path)
        print(f"✅ 模型加载成功")
        print(f"   训练轮数: {info['epoch']}")
        print(f"   最佳损失: {info['best_loss']:.4e}")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None
    
    # 运行测试
    results = {}
    
    # 1. 物理验证
    physics_validator = PhysicsValidator(model, device)
    results["physics"] = physics_validator.run_all()
    
    # 2. 解析对比
    analytical_comparator = AnalyticalComparator(model, device)
    results["analytical"] = analytical_comparator.run_all()
    
    # 3. 鲁棒性测试
    robustness_validator = RobustnessValidator(model, device)
    results["robustness"] = robustness_validator.run_all()
    
    # 生成报告
    score = generate_report(model, device, model_dir, results)
    
    # 打印总结
    print("\n" + "="*60)
    print("测试完成")
    print("="*60)
    print(f"总体评分: {score:.1f}%")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="PINN 模型完整测试套件")
    parser.add_argument("model_dir", nargs="?", default=None, 
                        help="模型目录（默认自动查找最新）")
    parser.add_argument("--all", action="store_true", 
                        help="测试所有 outputs_pinn_* 目录")
    
    args = parser.parse_args()
    
    # 确定要测试的目录
    if args.all:
        model_dirs = sorted(glob.glob("outputs_pinn_*"))
        if not model_dirs:
            print("❌ 未找到 outputs_pinn_* 目录")
            return
        print(f"🔍 找到 {len(model_dirs)} 个模型目录")
    elif args.model_dir:
        model_dirs = [args.model_dir]
    else:
        # 自动查找最新
        dirs = sorted(glob.glob("outputs_pinn_*"))
        if not dirs:
            print("❌ 未找到 outputs_pinn_* 目录")
            return
        model_dirs = [dirs[-1]]
        print(f"🔍 自动选择最新目录: {model_dirs[0]}")
    
    # 运行测试
    all_results = {}
    for model_dir in model_dirs:
        if os.path.isdir(model_dir):
            results = run_complete_test(model_dir)
            if results:
                all_results[model_dir] = results
    
    # 如果测试多个模型，打印对比
    if len(all_results) > 1:
        print("\n" + "="*60)
        print("多模型对比")
        print("="*60)
        
        for model_dir, results in all_results.items():
            total_passed = 0
            total_tests = 0
            for category in results.values():
                for test in category.values():
                    total_tests += 1
                    if test.get("passed", False):
                        total_passed += 1
            
            score = total_passed / total_tests * 100 if total_tests > 0 else 0
            print(f"{model_dir}: {score:.1f}% ({total_passed}/{total_tests})")


if __name__ == "__main__":
    main()
