#!/usr/bin/env python3
"""
EWP 混合预测器
==============

职责：电压 → 接触角（Young-Lippmann + 动力学）

与 EnhancedApertureModel 的分工：
- HybridPredictor: 电压 → 接触角（本模块）
- EnhancedApertureModel: 接触角 → 开口率 + 油墨分布 + 可视化

核心功能：
- Young-Lippmann 方程计算稳态接触角
- 二阶欠阻尼响应计算动态过渡
- 电压扫描响应（升压/降压）
- 方波响应（含表面张力恢复）

物理机制：
- 升压（水推油走）：电润湿驱动，接触角渐变（二阶欠阻尼）
- 降压（油推水走）：接触角瞬间恢复，开口率渐变（油墨铺展）

使用方法:
    from src.predictors import HybridPredictor
    from src.config import CONFIG_PATH
    
    predictor = HybridPredictor(config_path=str(CONFIG_PATH))
    theta = predictor.predict(voltage=30, time=0.005)  # 30V, 5ms
    
    # 电压扫描（含回退）
    voltages, theta, aperture = predictor.voltage_sweep_response(V_max=30)

作者: EFD-PINNs Team
日期: 2025-12-02
"""

import numpy as np
import torch
from pathlib import Path
from typing import Tuple, Optional, Dict, Union
import json


class HybridPredictor:
    """
    混合预测器：Stage 6 模型 + 解析公式
    
    物理基础：
    1. Young-Lippmann 方程: cos(θ) = cos(θ₀) + ε₀εᵣV²/(2γd)
    2. 二阶欠阻尼响应: θ(t) = θ_eq + (θ₀-θ_eq)·e^(-ζω₀t)·[cos(ω_d·t) + ζ/√(1-ζ²)·sin(ω_d·t)]
    """
    
    def __init__(
        self,
        model_path: str = 'outputs_20251201_212735/final_model.pth',
        config_path: Optional[str] = None,
        use_model_for_steady_state: bool = False,  # 默认使用解析公式
        device: str = 'cpu',
        aperture_model: Optional['EnhancedApertureModel'] = None  # 依赖注入
    ):
        """
        初始化混合预测器
        
        Args:
            model_path: Stage 6 模型路径
            config_path: 配置文件路径（可选，会从checkpoint读取）
            use_model_for_steady_state: 是否使用模型预测稳态（False则纯解析）
            device: 计算设备
            aperture_model: 可选的 EnhancedApertureModel 实例（依赖注入，避免重复创建）
        """
        self.device = torch.device(device)
        self.use_model = use_model_for_steady_state
        self._aperture_model = aperture_model  # 依赖注入或 None（延迟初始化）
        
        # 从统一配置获取默认物理参数
        try:
            from src.config import get_physics_config
            _config = get_physics_config()
            self.params = _config.to_predictor_params()
        except ImportError:
            # 回退：使用本地默认值
            self.params = {
                'theta0': 120.0,        # 初始接触角 (度)
                'epsilon_0': 8.854e-12, # 真空介电常数
                'gamma': 0.015,         # 极性液体表面张力 (N/m)
                'epsilon_r': 12.0,      # 有效相对介电常数
                'd': 4e-7,              # 介电层厚度 (m) = 400nm
                'epsilon_h': 1.9,       # Teflon 相对介电常数
                'd_h': 4e-7,            # Teflon 厚度 (m) = 400nm
                'tau': 0.005,           # 电润湿响应时间常数 (s)
                'tau_recovery': 0.0075, # 表面张力恢复时间常数 (s)
                'zeta': 0.8,            # 阻尼比
                'V_max': 30.0,          # 最大电压 (V)
                'V_threshold': 3.0,     # 阈值电压 (V)
            }
        
        # 加载模型和配置
        if use_model_for_steady_state and Path(model_path).exists():
            self._load_model(model_path, config_path)
        else:
            self.model = None
            self.use_model = False
            if config_path and Path(config_path).exists():
                self._load_config(config_path)
        
        # 计算派生参数
        self._update_derived_params()
        
        print(f"✅ HybridPredictor 初始化完成")
        print(f"   模式: 解析公式 (Young-Lippmann + 二阶欠阻尼)")
        print(f"   θ₀={self.params['theta0']}°, τ={self.params['tau']*1000:.1f}ms, ζ={self.params['zeta']}")
    
    def _update_derived_params(self):
        """更新派生参数"""
        tau = self.params['tau']
        zeta = self.params['zeta']
        self.omega_0 = 1.0 / tau
        self.omega_d = self.omega_0 * np.sqrt(max(0, 1 - zeta**2))
    
    def _load_config(self, config_path: str):
        """从配置文件加载参数"""
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        materials = config.get('materials', {})
        data_config = config.get('data', {})
        dynamics = data_config.get('dynamics_params', {})
        
        self.params.update({
            'theta0': materials.get('theta0', self.params['theta0']),
            'epsilon_r': materials.get('epsilon_r', self.params['epsilon_r']),
            'gamma': materials.get('gamma', self.params['gamma']),
            'd': materials.get('dielectric_thickness', self.params['d']),
            # Teflon 疏水层参数
            'epsilon_h': materials.get('epsilon_hydrophobic', self.params['epsilon_h']),
            'd_h': materials.get('hydrophobic_thickness', self.params['d_h']),
            # 动力学参数
            'tau': dynamics.get('tau', self.params['tau']),
            'tau_recovery': dynamics.get('tau_recovery', self.params['tau'] * 1.5),
            'zeta': dynamics.get('zeta', self.params['zeta']),
        })
    
    def _load_model(self, model_path: str, config_path: Optional[str]):
        """加载 PINN 模型"""
        from src.models.optimized_ewpinn import OptimizedEWPINN
        from src.training.components import DataNormalizer
        
        print(f"📦 加载模型: {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        
        # 获取配置
        config = checkpoint.get('config', {})
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
        
        # 更新物理参数
        materials = config.get('materials', {})
        data_config = config.get('data', {})
        dynamics = data_config.get('dynamics_params', {})
        
        self.params.update({
            'theta0': materials.get('theta0', self.params['theta0']),
            'epsilon_r': materials.get('epsilon_r', self.params['epsilon_r']),
            'gamma': materials.get('gamma', self.params['gamma']),
            'd': materials.get('dielectric_thickness', self.params['d']),
            'tau': dynamics.get('tau', self.params['tau']),
            'zeta': dynamics.get('zeta', self.params['zeta']),
        })
        
        # 构建模型
        model_config = config.get('model', {})
        input_dim = model_config.get('input_dim', 62)
        output_dim = model_config.get('output_dim', 24)
        
        # 从 state_dict 推断 hidden_dims
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        linear_layers = []
        for key, value in sorted(state_dict.items()):
            if 'main_layers' in key and '.weight' in key and 'running' not in key:
                if len(value.shape) == 2:
                    linear_layers.append(value.shape[0])
        hidden_dims = linear_layers[:-1] if linear_layers else [256, 256, 128, 64]
        
        self.model = OptimizedEWPINN(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            activation=model_config.get('activation', 'gelu'),
            config=config
        )
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.eval()
        self.model.to(self.device)
        
        # 加载归一化器
        self.input_normalizer = None
        self.output_normalizer = None
        
        if 'normalizer' in checkpoint and checkpoint['normalizer'] is not None:
            self.input_normalizer = DataNormalizer(method="standard")
            self.input_normalizer.load_state_dict(checkpoint['normalizer'])
        
        if 'output_normalizer' in checkpoint and checkpoint['output_normalizer'] is not None:
            self.output_normalizer = DataNormalizer(method="standard")
            self.output_normalizer.load_state_dict(checkpoint['output_normalizer'])
        
        self.config = config
    
    def young_lippmann(self, V: float) -> float:
        """
        Young-Lippmann 方程计算平衡接触角（含阈值电压修正）
        
        物理机制：
        - 电润湿作用在极性液体上，改变其在疏水层上的接触角
        - 接触角减小 → 极性液体铺展 → 挤开油墨 → 形成透明开口
        
        考虑 SU-8 + Teflon 串联电容结构：
        cos(θ) = cos(θ₀) + C·V²/(2γ)
        
        其中 C 是单位面积电容（SU-8 + Teflon 串联）：
        1/C = 1/C_SU8 + 1/C_Teflon = d_SU8/(ε₀ε_SU8) + d_Teflon/(ε₀ε_Teflon)
        
        注意：极性液体有导电性，电压降在介电层上，流体层不参与串联
        
        Args:
            V: 电压 (V)
        
        Returns:
            平衡接触角 (度) - 极性液体在疏水层上的接触角
        """
        V_threshold = self.params.get('V_threshold', 5.0)
        
        # 有效电压 = max(0, V - V_T)
        V_eff = max(0, V - V_threshold)
        
        # 串联电容
        # C_SU8 = ε₀ε_SU8 / d_SU8
        # C_Teflon = ε₀ε_Teflon / d_Teflon
        # 1/C_total = 1/C_SU8 + 1/C_Teflon
        epsilon_0 = self.params['epsilon_0']
        epsilon_r = self.params.get('epsilon_r', 3.0)  # SU-8
        d = self.params.get('d', 4e-7)  # SU-8 厚度
        epsilon_h = self.params.get('epsilon_h', 1.9)  # Teflon
        d_h = self.params.get('d_h', 4e-7)  # Teflon 厚度
        
        # 单位面积电容 (F/m²)
        C_su8 = epsilon_0 * epsilon_r / d
        C_teflon = epsilon_0 * epsilon_h / d_h
        C_total = 1.0 / (1.0 / C_su8 + 1.0 / C_teflon)
        
        cos_theta0 = np.cos(np.radians(self.params['theta0']))
        # Young-Lippmann: cos(θ) = cos(θ₀) + C·V²/(2γ)
        ew_term = C_total * V_eff**2 / (2 * self.params['gamma'])
        cos_theta = np.clip(cos_theta0 + ew_term, -1, 1)
        return np.degrees(np.arccos(cos_theta))
    
    def dynamic_response(
        self, 
        t: float, 
        theta_start: float, 
        theta_eq: float
    ) -> float:
        """
        二阶欠阻尼动态响应（电润湿驱动）
        
        θ(t) = θ_eq + (θ_start - θ_eq) · e^(-ζω₀t) · [cos(ω_d·t) + ζ/√(1-ζ²)·sin(ω_d·t)]
        
        Args:
            t: 时间 (s)
            theta_start: 初始角度 (度)
            theta_eq: 平衡角度 (度)
        
        Returns:
            当前角度 (度)
        """
        zeta = self.params['zeta']
        
        if zeta >= 1:
            # 临界阻尼或过阻尼
            return theta_eq + (theta_start - theta_eq) * np.exp(-t / self.params['tau'])
        else:
            # 欠阻尼
            exp_term = np.exp(-zeta * self.omega_0 * t)
            damping_factor = zeta / np.sqrt(1 - zeta**2)
            return theta_eq + (theta_start - theta_eq) * exp_term * (
                np.cos(self.omega_d * t) + damping_factor * np.sin(self.omega_d * t)
            )
    
    def surface_tension_recovery(
        self,
        t: float,
        theta_start: float,
        theta_eq: float = None
    ) -> float:
        """
        表面张力恢复动力学（电压撤除后）
        
        物理机制：
        - 电压撤除后，电润湿力消失
        - 油墨仅靠表面张力恢复到初始状态
        - 恢复过程是一阶指数衰减（过阻尼，无振荡）
        
        θ(t) = θ₀ + (θ_start - θ₀) · e^(-t/τ_recovery)
        
        其中 τ_recovery 是表面张力恢复时间常数，通常比电润湿响应慢
        
        Args:
            t: 时间 (s)，从电压撤除时刻开始
            theta_start: 电压撤除时的接触角 (度)
            theta_eq: 恢复目标角度 (度)，默认为 θ₀
        
        Returns:
            当前角度 (度)
        """
        if theta_eq is None:
            theta_eq = self.params['theta0']  # 恢复到初始接触角
        
        tau_recovery = self.params.get('tau_recovery', self.params['tau'] * 1.5)
        
        # 一阶指数恢复（纯表面张力驱动，无振荡）
        return theta_eq + (theta_start - theta_eq) * np.exp(-t / tau_recovery)
    
    def predict_steady_state(self, V: float) -> float:
        """
        预测稳态接触角
        
        如果有模型，使用模型预测；否则使用 Young-Lippmann 方程
        
        Args:
            V: 电压 (V)
        
        Returns:
            稳态接触角 (度)
        """
        if not self.use_model or self.model is None:
            return self.young_lippmann(V)
        
        # 使用模型预测稳态（t >> tau）
        return self._model_predict(V, t=0.1, t_step=0.0)
    
    def _model_predict(self, V: float, t: float, t_step: float) -> float:
        """使用模型进行单点预测"""
        # 构建输入特征
        features = self._build_features(V, t, t_step)
        
        # 应用输入归一化
        if self.input_normalizer is not None:
            features = self.input_normalizer.transform(features.reshape(1, -1)).flatten()
        
        # 模型推理
        with torch.no_grad():
            X = torch.FloatTensor(features).unsqueeze(0).to(self.device)
            output = self.model(X)
        
        # 反归一化输出
        if self.output_normalizer is not None:
            output_np = output.cpu().numpy()
            output_denorm = self.output_normalizer.inverse_transform(output_np)
            theta_rad = output_denorm[0, 10]  # 接触角在索引10
        else:
            theta_rad = output[0, 10].item()
        
        return np.clip(np.degrees(theta_rad), 50, 130)
    
    def _build_features(self, V: float, t: float, t_step: float) -> np.ndarray:
        """构建62维输入特征"""
        features = np.zeros(62, dtype=np.float32)
        
        T_total = 0.02
        V_max = self.params['V_max']
        tau = self.params['tau']
        zeta = self.params['zeta']
        
        # 空间坐标
        features[0:3] = 0.5
        
        # 时间特征
        features[3] = t / T_total
        features[4] = np.sin(2 * np.pi * t / T_total)
        features[5] = np.cos(2 * np.pi * t / T_total)
        
        # 电压特征
        features[6] = V / V_max
        features[7] = (V / V_max) ** 2
        
        # 动态响应特征
        features[8] = t_step / T_total
        features[9] = max(0, t - t_step) / T_total
        features[10] = max(0, t - t_step) / tau
        
        # 电压变化信息
        V_before = 0 if V > 0 else V_max
        V_after = V
        features[11] = V_before / V_max
        features[12] = V_after / V_max
        features[13] = (V_after - V_before) / V_max
        
        # 角度信息
        theta_before = self.young_lippmann(V_before)
        theta_after = self.young_lippmann(V_after)
        features[14] = np.radians(theta_before) / np.pi
        features[15] = np.radians(theta_after) / np.pi
        features[16] = np.radians(theta_after - theta_before) / np.pi
        
        # 动力学参数
        features[17] = tau * 1000
        features[18] = zeta
        features[19] = self.omega_0 / 1000
        
        # 材料参数
        features[20] = self.params['epsilon_r'] / 10.0
        features[21] = self.params['gamma'] / 0.1
        features[22] = self.params['d'] / 1e-6
        features[23] = self.params['theta0'] / 180.0
        
        # 几何参数
        features[24:27] = [184e-6/1e-3, 184e-6/1e-3, 20.855e-6/1e-4]
        
        # 响应阶段
        if t < t_step:
            features[27] = 0.0
        elif t < t_step + tau:
            features[27] = 0.5
        else:
            features[27] = 1.0
        
        # 响应进度
        if t >= t_step:
            t_since = t - t_step
            features[28] = 1.0 - np.exp(-zeta * self.omega_0 * t_since)
        
        return features
    
    def predict(
        self, 
        voltage: float, 
        time: float, 
        V_initial: float = 0.0,
        t_step: float = 0.0
    ) -> float:
        """
        混合预测：模型稳态 + 解析动态
        
        物理机制：
        - 升压（V_initial < voltage）：电润湿驱动，接触角渐变（二阶欠阻尼）
        - 降压（V_initial > voltage）：接触角瞬间恢复到目标值
        
        Args:
            voltage: 当前电压 (V)
            time: 当前时间 (s)
            V_initial: 初始电压 (V)
            t_step: 电压阶跃时间 (s)
        
        Returns:
            预测的接触角 (度)
        """
        # 获取稳态角度
        theta_eq = self.predict_steady_state(voltage)
        theta_start = self.predict_steady_state(V_initial)
        
        # 计算动态响应
        if time < t_step:
            return theta_start
        else:
            # 判断升压还是降压
            if voltage >= V_initial:
                # 升压：电润湿驱动，接触角渐变（二阶欠阻尼）
                t_since = time - t_step
                return self.dynamic_response(t_since, theta_start, theta_eq)
            else:
                # 降压：接触角瞬间恢复到目标值
                # Young-Lippmann 是瞬态方程，电场消失 → 接触角立即回到本征值
                return theta_eq
    
    def step_response(
        self,
        V_start: float = 0.0,
        V_end: float = 30.0,
        duration: float = 0.02,
        t_step: float = 0.002,
        num_points: int = 500
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算阶跃响应
        
        Args:
            V_start: 初始电压 (V)
            V_end: 最终电压 (V)
            duration: 总时长 (s)
            t_step: 阶跃时间 (s)
            num_points: 采样点数
        
        Returns:
            (时间数组, 接触角数组)
        """
        t = np.linspace(0, duration, num_points)
        theta = np.zeros(num_points)
        
        theta_start = self.predict_steady_state(V_start)
        theta_end = self.predict_steady_state(V_end)
        
        for i, ti in enumerate(t):
            if ti < t_step:
                theta[i] = theta_start
            else:
                t_since = ti - t_step
                theta[i] = self.dynamic_response(t_since, theta_start, theta_end)
        
        return t, theta
    
    def square_wave_response(
        self,
        V_low: float = 0.0,
        V_high: float = 30.0,
        duration: float = 0.02,
        t_rise: float = 0.002,
        t_fall: float = 0.012,
        num_points: int = 500
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        计算方波响应（含表面张力恢复）
        
        物理过程：
        - 升压阶段 (t_rise → t_fall): 
          * 电润湿驱动极性液体铺展（水推油走）
          * 接触角：二阶欠阻尼响应
          * 开口率：随接触角变化
        - 降压阶段 (t_fall → end): 
          * 接触角：瞬间恢复到 θ₀（Young-Lippmann 是瞬态的）
          * 开口率：油墨靠表面张力慢慢铺回去（油推水走，一阶指数衰减）
        
        Args:
            V_low: 低电压 (V)
            V_high: 高电压 (V)
            duration: 总时长 (s)
            t_rise: 上升沿时间 (s)
            t_fall: 下降沿时间 (s)
            num_points: 采样点数
        
        Returns:
            (时间数组, 电压数组, 接触角数组, 开口率数组)
        """
        t = np.linspace(0, duration, num_points)
        V = np.where((t >= t_rise) & (t < t_fall), V_high, V_low)
        theta = np.zeros(num_points)
        eta = np.zeros(num_points)
        
        theta_low = self.predict_steady_state(V_low)  # 120°
        theta_high = self.predict_steady_state(V_high)  # ~67.5° at 30V
        tau_recovery = self.params.get('tau_recovery', self.params['tau'] * 1.5)
        
        # 计算稳态开口率
        eta_low = self._theta_to_aperture(np.array([theta_low]))[0]  # ~0
        eta_high = self._theta_to_aperture(np.array([theta_high]))[0]  # ~83%
        
        eta_at_fall = None  # 电压撤除时刻的开口率
        
        for i, ti in enumerate(t):
            if ti < t_rise:
                # 初始状态
                theta[i] = theta_low
                eta[i] = eta_low
            elif ti < t_fall:
                # 升压阶段：电润湿驱动（水推油走）
                t_since = ti - t_rise
                theta[i] = self.dynamic_response(t_since, theta_low, theta_high)
                # 开口率随接触角变化
                eta[i] = self._theta_to_aperture(np.array([theta[i]]))[0]
                eta_at_fall = eta[i]  # 记录最后的开口率
            else:
                # 降压阶段：油墨表面张力驱动（油推水走）
                t_since = ti - t_fall
                
                # 接触角：瞬间恢复到 θ₀
                theta[i] = theta_low
                
                # 开口率：油墨慢慢铺回去（一阶指数衰减）
                if eta_at_fall is not None:
                    eta[i] = eta_at_fall * np.exp(-t_since / tau_recovery)
                else:
                    eta[i] = eta_low
        
        return t, V, theta, eta
    
    def voltage_sweep_response(
        self,
        V_max: float = 30.0,
        v_step: float = 1.0,
        t_per_step: float = 0.001
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        计算电压扫描响应 (0→V_max→0V)
        
        物理过程：
        - 升压阶段 (0→V_max): 电润湿驱动极性液体铺展，挤开油墨
        - 降压阶段 (V_max→0): 
          * 接触角瞬间恢复到 θ₀（Young-Lippmann 是瞬态的）
          * 油墨靠表面张力慢慢铺回去（油推水走，油墨亲疏水层）
        
        关态物理机制：
        - 油墨亲疏水层（Teflon），愿意铺在底部
        - 极性液体不亲疏水层，被油墨挤开
        - 所以是"油推水走"，油墨通过表面张力铺展回去
        
        Args:
            V_max: 最大电压 (V)
            v_step: 电压步长 (V)
            t_per_step: 每个电压步的时间 (s)
        
        Returns:
            (电压数组, 接触角数组, 开口率数组)
        """
        # 升压阶段：电润湿驱动（水推油走）
        voltages_up = np.arange(0, V_max + v_step, v_step)
        theta_up = np.array([self.predict_steady_state(V) for V in voltages_up])
        aperture_up = self._theta_to_aperture(theta_up)
        
        # 降压阶段：油墨表面张力驱动（油推水走）
        voltages_down = np.arange(V_max - v_step, -v_step, -v_step)
        n_down = len(voltages_down)
        
        theta_0 = self.params['theta0']  # 120°
        eta_at_max = aperture_up[-1]  # V_max 时的开口率
        tau_recovery = self.params.get('tau_recovery', self.params['tau'] * 1.5)
        
        theta_down = np.zeros(n_down)
        aperture_down = np.zeros(n_down)
        
        for i in range(n_down):
            # 接触角：电压撤除后瞬间恢复到 θ₀
            # Young-Lippmann 是瞬态方程，电场消失 → 接触角立即回到本征值
            theta_down[i] = theta_0
            
            # 开口率：油墨靠表面张力慢慢铺回去（指数衰减）
            # 恢复进度映射到 5τ 时间，确保完全恢复
            progress = (i + 1) / n_down
            t_recovery = progress * 5 * tau_recovery
            # η(t) = η_max × exp(-t/τ_recovery)
            aperture_down[i] = eta_at_max * np.exp(-t_recovery / tau_recovery)
        
        # 合并
        voltages = np.concatenate([voltages_up, voltages_down])
        theta = np.concatenate([theta_up, theta_down])
        aperture = np.concatenate([aperture_up, aperture_down])
        
        return voltages, theta, aperture
    
    def _theta_to_aperture(self, theta: np.ndarray) -> np.ndarray:
        """
        接触角转开口率（使用 EnhancedApertureModel 的方法）
        
        与 train_contact_angle.py 和 EnhancedApertureModel 保持同步
        
        Args:
            theta: 接触角数组 (度)
        
        Returns:
            开口率数组 (0-1)
        """
        # 延迟初始化：仅在首次调用且未注入时创建
        if self._aperture_model is None:
            try:
                from src.models.aperture_model import EnhancedApertureModel
                from src.config import CONFIG_PATH
                self._aperture_model = EnhancedApertureModel(
                    config_path=str(CONFIG_PATH)
                )
            except ImportError:
                # 回退到简化映射
                theta_0 = self.params['theta0']
                theta_min = 60.0
                eta_max = 0.85
                return np.clip((theta_0 - theta) / (theta_0 - theta_min) * eta_max, 0, eta_max)
        
        model = self._aperture_model
        if np.isscalar(theta):
            return model.contact_angle_to_aperture_ratio(theta)
        else:
            return np.array([model.contact_angle_to_aperture_ratio(t) for t in theta])
    
    def get_response_metrics(
        self,
        t: np.ndarray,
        theta: np.ndarray,
        t_step: float = 0.002
    ) -> Dict[str, float]:
        """
        计算响应指标
        
        Args:
            t: 时间数组
            theta: 接触角数组
            t_step: 阶跃时间
        
        Returns:
            指标字典
        """
        # 找到阶跃点
        step_idx = np.searchsorted(t, t_step)
        
        theta_initial = theta[step_idx]
        theta_final = theta[-1]
        theta_change = theta_initial - theta_final
        
        # t90 响应时间
        if abs(theta_change) > 0.1:
            theta_90 = theta_initial - 0.9 * theta_change
            t_90_idx = np.where(theta[step_idx:] <= theta_90)[0]
            t_90 = (t[step_idx + t_90_idx[0]] - t_step) * 1000 if len(t_90_idx) > 0 else np.nan
        else:
            t_90 = np.nan
        
        # 超调
        theta_min = np.min(theta[step_idx:])
        if abs(theta_change) > 0.1:
            overshoot = max(0, (theta_final - theta_min) / abs(theta_change) * 100)
        else:
            overshoot = 0
        
        return {
            'theta_initial': theta_initial,
            'theta_final': theta_final,
            'theta_change': theta_change,
            't_90_ms': t_90,
            'overshoot_percent': overshoot,
        }


def demo():
    """演示混合预测器的使用"""
    import matplotlib.pyplot as plt
    from src.config import CONFIG_PATH
    
    print("=" * 60)
    print("🔬 EWP 混合预测器演示 (含表面张力恢复)")
    print("=" * 60)
    
    # 创建预测器 (使用解析公式，从配置文件读取参数)
    predictor = HybridPredictor(
        config_path=str(CONFIG_PATH),
        use_model_for_steady_state=False
    )
    
    # 显示动力学参数
    print(f"\n动力学参数:")
    print(f"   τ (电润湿响应): {predictor.params['tau']*1000:.1f} ms")
    print(f"   τ_recovery (表面张力恢复): {predictor.params.get('tau_recovery', predictor.params['tau']*1.5)*1000:.1f} ms")
    print(f"   ζ (阻尼比): {predictor.params['zeta']}")
    
    # 1. 稳态预测 (Young-Lippmann)
    print("\n📊 稳态预测 (Young-Lippmann 方程):")
    print("-" * 40)
    print(f"{'电压(V)':<10} {'接触角(°)':<12} {'角度变化(°)':<12}")
    print("-" * 40)
    
    theta_0 = predictor.young_lippmann(0)
    for V in [0, 10, 20, 30]:
        theta = predictor.young_lippmann(V)
        delta = theta_0 - theta
        print(f"{V:<10} {theta:<12.1f} {delta:<12.1f}")
    
    # 2. 电压扫描响应 (0→30→0V)
    print("\n📈 电压扫描响应 (0V → 30V → 0V):")
    print("   升压: 电润湿驱动")
    print("   降压: 表面张力恢复 (油墨平铺回去)")
    
    voltages, theta_sweep, aperture_sweep = predictor.voltage_sweep_response(
        V_max=30, v_step=5, t_per_step=0.002
    )
    
    n_up = 7  # 0, 5, 10, 15, 20, 25, 30
    print("\n   升压阶段:")
    for i in range(n_up):
        print(f"      V={voltages[i]:2.0f}V: θ={theta_sweep[i]:.1f}°, η={aperture_sweep[i]:.1%}")
    
    print("   降压阶段:")
    for i in range(n_up, len(voltages)):
        print(f"      V={voltages[i]:2.0f}V: θ={theta_sweep[i]:.1f}°, η={aperture_sweep[i]:.1%}")
    
    # 3. 方波响应
    print("\n📈 方波响应 (0V → 30V → 0V, 0-100ms):")
    print("   升压: 接触角渐变（电润湿动力学）")
    print("   降压: 接触角瞬间恢复，开口率渐变（油墨铺展）")
    t_sq, V_sq, theta_sq, eta_sq = predictor.square_wave_response(
        V_low=0, V_high=30, duration=0.1, t_rise=0.002, t_fall=0.04
    )
    
    # 绘图 - 4 个子图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 方波电压
    ax1 = axes[0, 0]
    ax1.plot(t_sq * 1000, V_sq, 'b-', linewidth=2)
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Voltage (V)')
    ax1.set_title('Square Wave Voltage')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 100)
    
    # 2. 方波接触角响应（降压时瞬间恢复）
    ax2 = axes[0, 1]
    ax2.plot(t_sq * 1000, theta_sq, 'r-', linewidth=2, label='Contact Angle')
    ax2.axhline(predictor.young_lippmann(0), color='gray', linestyle='--', alpha=0.5, label='θ(0V)=120°')
    ax2.axhline(predictor.young_lippmann(30), color='green', linestyle='--', alpha=0.5, label='θ(30V)≈67°')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Contact Angle θ (°)')
    ax2.set_title('Contact Angle: Instant Recovery at V=0')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 100)
    
    # 3. 方波开口率响应（降压时渐变恢复）
    ax3 = axes[1, 0]
    ax3.plot(t_sq * 1000, eta_sq * 100, 'b-', linewidth=2, label='Aperture Ratio')
    ax3.axvline(x=2, color='gray', linestyle=':', alpha=0.5, label='V↑')
    ax3.axvline(x=40, color='gray', linestyle='--', alpha=0.5, label='V↓')
    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('Aperture Ratio η (%)')
    ax3.set_title('Aperture: Gradual Recovery (Ink Spreading)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 100)
    ax3.set_ylim(0, 100)
    
    # 4. 电压扫描 - 开口率 (迟滞环)
    ax4 = axes[1, 1]
    ax4.plot(voltages[:n_up], aperture_sweep[:n_up]*100, 'b-o', markersize=6, linewidth=2, label='↑ Electrowetting (water pushes oil)')
    ax4.plot(voltages[n_up-1:], aperture_sweep[n_up-1:]*100, 'r-s', markersize=6, linewidth=2, label='↓ Recovery (oil pushes water)')
    ax4.set_xlabel('Voltage (V)')
    ax4.set_ylabel('Aperture Ratio η (%)')
    ax4.set_title('Voltage Sweep: Aperture Ratio (Hysteresis)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 30)
    ax4.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig('hybrid_predictor_demo.png', dpi=150)
    print(f"\n📊 图表已保存: hybrid_predictor_demo.png")
    
    print("\n✅ 演示完成!")


if __name__ == '__main__':
    demo()
