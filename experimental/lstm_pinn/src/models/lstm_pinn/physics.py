"""
LSTM-PINN 电润湿物理模型

复用现有的物理模型实现，提供：
- Young-Lippmann 方程
- 升压动力学（二阶欠阻尼）
- 降压动力学（一阶指数衰减）
- 接触角到开口率映射
"""

import numpy as np
from typing import Optional, Dict, Any


class ElectrowettingPhysics:
    """
    电润湿物理模型
    
    基于 config/device_calibrated_physics.json 的校准参数
    """
    
    # 默认物理参数（从 device_calibrated_physics.json 同步）
    DEFAULT_PARAMS = {
        # 材料参数
        "theta0": 120.0,           # 初始接触角 (度)
        "theta_30V": 67.5,         # 30V 时的饱和接触角 (度)
        "theta_wall": 71.0,        # 像素墙接触角 (度)
        "epsilon_r": 12.0,         # 有效介电常数
        "epsilon_hydrophobic": 1.9, # 疏水层介电常数
        "gamma": 0.015,            # 极性液体表面张力 (N/m)
        "dielectric_thickness": 4e-7,  # 介电层厚度 (m)
        "hydrophobic_thickness": 4e-7, # 疏水层厚度 (m)
        "V_threshold": 3.0,        # 阈值电压 (V)
        "aperture_max": 0.85,      # 最大开口率
        
        # 动力学参数
        "tau": 0.005,              # 电润湿响应时间常数 (s)
        "tau_recovery": 0.0075,    # 表面张力恢复时间常数 (s)
        "zeta": 0.8,               # 阻尼比
        
        # 物理常数
        "epsilon_0": 8.854e-12,    # 真空介电常数
    }
    
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        初始化物理模型
        
        Args:
            params: 物理参数字典，覆盖默认值
        """
        self.params = self.DEFAULT_PARAMS.copy()
        if params:
            self.params.update(params)
        
        # 预计算常用值
        self._precompute()
    
    def _precompute(self):
        """预计算常用值"""
        tau = self.params["tau"]
        zeta = self.params["zeta"]
        
        self.omega_0 = 1.0 / tau
        if zeta < 1:
            self.omega_d = self.omega_0 * np.sqrt(1 - zeta**2)
        else:
            self.omega_d = 0
    
    def effective_voltage(self, V: float) -> float:
        """
        计算有效电压（考虑阈值效应）
        
        V_eff = max(0, V - V_threshold)
        
        Args:
            V: 施加电压 (V)
        
        Returns:
            有效电压 (V)
        """
        V_threshold = self.params["V_threshold"]
        return max(0, V - V_threshold)
    
    def young_lippmann(self, V: float) -> float:
        """
        Young-Lippmann 方程计算平衡接触角
        
        cos(θ) = cos(θ₀) + ε₀εᵣV_eff² / (2γd)
        
        Args:
            V: 电压 (V)
        
        Returns:
            平衡接触角 (度)
        """
        V_eff = self.effective_voltage(V)
        
        if V_eff <= 0:
            return self.params["theta0"]
        
        theta0 = self.params["theta0"]
        epsilon_0 = self.params["epsilon_0"]
        epsilon_r = self.params["epsilon_r"]
        gamma = self.params["gamma"]
        d = self.params["dielectric_thickness"]
        
        cos_theta0 = np.cos(np.radians(theta0))
        ew_term = (epsilon_0 * epsilon_r * V_eff**2) / (2 * gamma * d)
        cos_theta = np.clip(cos_theta0 + ew_term, -1, 1)
        theta = np.degrees(np.arccos(cos_theta))
        
        # 应用饱和效应
        theta_min = self.params["theta_30V"]
        return max(theta, theta_min)
    
    def contact_angle_rise(
        self,
        V_to: float,
        t_since: float,
        V_from: float = 0.0
    ) -> float:
        """
        升压过程接触角动力学（二阶欠阻尼系统）
        
        物理机制：电润湿力驱动极性液体铺展，推动油墨
        
        θ(t) = θ_eq + (θ_0 - θ_eq) × e^(-ζω₀t) × [cos(ω_d·t) + ζ/√(1-ζ²)·sin(ω_d·t)]
        
        Args:
            V_to: 目标电压 (V)
            t_since: 电压变化后经过的时间 (s)
            V_from: 初始电压 (V)
        
        Returns:
            当前接触角 (度)
        """
        # 目标平衡接触角
        theta_eq = self.young_lippmann(V_to)
        
        # 初始接触角
        if V_from > 0:
            theta_0 = self.young_lippmann(V_from)
        else:
            theta_0 = self.params["theta0"]
        
        # 如果已经达到平衡
        if abs(theta_0 - theta_eq) < 0.1:
            return theta_eq
        
        zeta = self.params["zeta"]
        
        if zeta >= 1:
            # 临界阻尼或过阻尼
            exp_term = np.exp(-t_since / self.params["tau"])
            return theta_eq + (theta_0 - theta_eq) * exp_term
        else:
            # 欠阻尼
            exp_term = np.exp(-zeta * self.omega_0 * t_since)
            damping = zeta / np.sqrt(1 - zeta**2)
            
            theta = theta_eq + (theta_0 - theta_eq) * exp_term * (
                np.cos(self.omega_d * t_since) + damping * np.sin(self.omega_d * t_since)
            )
            
            # 确保不低于饱和值
            return max(theta, self.params["theta_30V"])
    
    def aperture_fall(
        self,
        V_from: float,
        t_since: float,
        V_to: float = 0.0
    ) -> float:
        """
        降压过程开口率动力学（一阶指数衰减）
        
        物理机制：电场消失，表面张力驱动油墨铺展回中心
        
        η(t) = η_target + (η_initial - η_target) × exp(-t/τ_recovery)
        
        Args:
            V_from: 降压前电压 (V)
            t_since: 降压后经过的时间 (s)
            V_to: 降压后电压 (V)
        
        Returns:
            当前开口率
        """
        # 降压前的稳态开口率
        theta_from = self.young_lippmann(V_from)
        eta_initial = self.contact_angle_to_aperture(theta_from)
        
        # 降压后的目标开口率
        if V_to > 0:
            theta_to = self.young_lippmann(V_to)
            eta_target = self.contact_angle_to_aperture(theta_to)
        else:
            eta_target = 0.0
        
        # 一阶指数衰减
        tau_recovery = self.params["tau_recovery"]
        eta = eta_target + (eta_initial - eta_target) * np.exp(-t_since / tau_recovery)
        
        return np.clip(eta, 0, self.params["aperture_max"])
    
    def contact_angle_to_aperture(self, theta: float) -> float:
        """
        接触角映射到开口率
        
        基于几何模型和体积守恒
        
        Args:
            theta: 接触角 (度)
        
        Returns:
            开口率 η ∈ [0, aperture_max]
        """
        theta0 = self.params["theta0"]
        theta_min = self.params["theta_30V"]
        eta_max = self.params["aperture_max"]
        
        if theta >= theta0:
            return 0.0
        elif theta <= theta_min:
            return eta_max
        else:
            # 基于 cos(θ) 变化的非线性映射
            cos_change = np.cos(np.radians(theta)) - np.cos(np.radians(theta0))
            cos_max_change = np.cos(np.radians(theta_min)) - np.cos(np.radians(theta0))
            eta = eta_max * np.tanh(2 * cos_change / cos_max_change)
            return np.clip(eta, 0, eta_max)
    
    def aperture_to_contact_angle(self, eta: float) -> float:
        """
        开口率反推接触角（逆映射）
        
        Args:
            eta: 开口率
        
        Returns:
            接触角 (度)
        """
        theta0 = self.params["theta0"]
        theta_min = self.params["theta_30V"]
        eta_max = self.params["aperture_max"]
        
        if eta <= 0:
            return theta0
        elif eta >= eta_max:
            return theta_min
        else:
            # 反解 tanh 映射
            cos_theta0 = np.cos(np.radians(theta0))
            cos_theta_min = np.cos(np.radians(theta_min))
            cos_max_change = cos_theta_min - cos_theta0
            
            # eta = eta_max * tanh(2 * cos_change / cos_max_change)
            # tanh_arg = eta / eta_max
            # cos_change = arctanh(tanh_arg) * cos_max_change / 2
            tanh_arg = np.clip(eta / eta_max, -0.999, 0.999)
            cos_change = np.arctanh(tanh_arg) * cos_max_change / 2
            cos_theta = cos_theta0 + cos_change
            cos_theta = np.clip(cos_theta, -1, 1)
            
            return np.degrees(np.arccos(cos_theta))
    
    def get_steady_state_aperture(self, V: float) -> float:
        """
        获取稳态开口率
        
        Args:
            V: 电压 (V)
        
        Returns:
            稳态开口率
        """
        theta = self.young_lippmann(V)
        return self.contact_angle_to_aperture(theta)
    
    def get_dynamic_aperture(
        self,
        V: float,
        t: float,
        V_prev: Optional[float] = None,
        t_step: float = 0.0
    ) -> float:
        """
        获取动态开口率（支持升压和降压）
        
        Args:
            V: 当前电压 (V)
            t: 当前时间 (s)
            V_prev: 之前的电压 (V)，默认等于 V
            t_step: 电压变化时刻 (s)
        
        Returns:
            当前开口率
        """
        if V_prev is None:
            V_prev = V
        
        t_since = max(0, t - t_step)
        
        if V > V_prev:
            # 升压
            theta = self.contact_angle_rise(V, t_since, V_prev)
            return self.contact_angle_to_aperture(theta)
        elif V < V_prev:
            # 降压
            return self.aperture_fall(V_prev, t_since, V)
        else:
            # 恒定电压
            theta = self.contact_angle_rise(V, t_since, 0)
            return self.contact_angle_to_aperture(theta)
    
    @classmethod
    def from_config(cls, config_path: str) -> "ElectrowettingPhysics":
        """
        从配置文件创建实例
        
        Args:
            config_path: JSON 配置文件路径
        
        Returns:
            ElectrowettingPhysics 实例
        """
        import json
        from pathlib import Path
        
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 提取材料和动力学参数
        params = {}
        
        materials = config.get("materials", {})
        params.update({
            "theta0": materials.get("theta0", 120.0),
            "theta_30V": materials.get("theta_30V", 67.5),
            "theta_wall": materials.get("theta_wall", 71.0),
            "epsilon_r": materials.get("epsilon_r", 12.0),
            "gamma": materials.get("gamma", 0.015),
            "dielectric_thickness": materials.get("dielectric_thickness", 4e-7),
            "V_threshold": materials.get("V_threshold", 3.0),
            "aperture_max": materials.get("aperture_max", 0.85),
        })
        
        dynamics = config.get("dynamics", config.get("data", {}).get("dynamics_params", {}))
        params.update({
            "tau": dynamics.get("tau", 0.005),
            "tau_recovery": dynamics.get("tau_recovery", 0.0075),
            "zeta": dynamics.get("zeta", 0.8),
        })
        
        return cls(params)
