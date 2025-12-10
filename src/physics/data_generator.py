#!/usr/bin/env python3
"""
基于物理的电润湿训练数据生成器
使用 Young-Lippmann 方程和二阶欠阻尼动态响应生成真实的训练数据
"""

import numpy as np
import torch
import logging
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)


class ElectrowettingPhysicsGenerator:
    """基于物理的电润湿数据生成器"""
    
    def __init__(self, config: dict):
        """
        初始化生成器
        
        Args:
            config: 配置字典，包含材料参数、几何参数和动力学参数
        """
        # 物理常数
        self.epsilon_0 = 8.854e-12  # 真空介电常数 (F/m)
        
        # 材料参数
        materials = config.get('materials', {})
        self.epsilon_r = materials.get('epsilon_r', 4.0)  # 相对介电常数
        self.gamma = materials.get('gamma', 0.072)  # 界面张力 (N/m)
        self.d = materials.get('dielectric_thickness', 4e-7)  # 介电层厚度 (m)
        self.theta0 = materials.get('theta0', 110.0)  # 疏水层初始接触角 (度)
        
        # 像素墙参数 (新增)
        self.theta_wall = materials.get('theta_wall', 70.0)  # 像素墙接触角 (度)
        self.wall_influence_threshold = materials.get('wall_influence_threshold', 0.7)  # 液滴展开到多少比例时开始受墙影响
        
        # 几何参数
        geometry = config.get('geometry', {})
        self.Lx = geometry.get('Lx', 184e-6)  # x方向尺寸 (m)
        self.Ly = geometry.get('Ly', 184e-6)  # y方向尺寸 (m)
        self.Lz = geometry.get('Lz', 20.855e-6)  # z方向尺寸 (m)
        
        # 数据配置
        data_config = config.get('data', {})
        self.voltage_range = data_config.get('voltage_range', [0, 30])
        self.time_range = data_config.get('time_range', [0, 0.02])
        self.spatial_resolution = data_config.get('spatial_resolution', [10, 10, 5])
        self.temporal_resolution = data_config.get('temporal_resolution', 100)
        
        # 动力学参数
        dynamics = data_config.get('dynamics_params', {})
        self.tau = dynamics.get('tau', 0.005)  # 时间常数 (s)
        self.zeta = dynamics.get('zeta', 0.85)  # 阻尼比
        
        # 计算二阶系统参数
        self.omega_0 = 1.0 / self.tau  # 自然频率
        self.omega_d = self.omega_0 * np.sqrt(max(0, 1 - self.zeta**2))  # 阻尼频率
        
        logger.info(f"电润湿数据生成器初始化:")
        logger.info(f"  材料: ε_r={self.epsilon_r}, γ={self.gamma}, d={self.d*1e6:.2f}μm")
        logger.info(f"  接触角: 疏水层θ₀={self.theta0}°, 像素墙θ_wall={self.theta_wall}°")
        logger.info(f"  动力学: τ={self.tau*1000:.1f}ms, ζ={self.zeta}")
        logger.info(f"  电压范围: {self.voltage_range[0]}-{self.voltage_range[1]}V")
    
    def young_lippmann(self, V: np.ndarray, include_wall_effect: bool = True) -> np.ndarray:
        """
        Young-Lippmann 方程计算平衡接触角 (含像素墙边界效应)
        
        基础方程: cos(θ) = cos(θ₀) + ε₀ε_r V² / (2γd)
        
        边界效应: 当液滴展开接近像素墙时，有效接触角受墙影响
        θ_eff = θ_YL * (1 - w) + θ_wall * w
        其中 w 是墙影响权重，与液滴展开程度相关
        
        Args:
            V: 电压数组 (V)
            include_wall_effect: 是否包含像素墙效应
            
        Returns:
            接触角数组 (度)
        """
        theta0_rad = np.radians(self.theta0)
        cos_theta0 = np.cos(theta0_rad)
        
        # 电润湿项
        ew_term = self.epsilon_0 * self.epsilon_r * V**2 / (2 * self.gamma * self.d)
        
        cos_theta = cos_theta0 + ew_term
        cos_theta = np.clip(cos_theta, -1, 1)
        
        theta_rad = np.arccos(cos_theta)
        theta_yl = np.degrees(theta_rad)  # 纯 Young-Lippmann 角度
        
        if not include_wall_effect:
            return theta_yl
        
        # 计算像素墙影响
        # 液滴展开程度：角度越小，展开越大
        # 展开比例 = (θ₀ - θ) / (θ₀ - θ_min)
        theta_min = 60.0  # 最小可能角度 (饱和)
        spread_ratio = (self.theta0 - theta_yl) / (self.theta0 - theta_min + 1e-6)
        spread_ratio = np.clip(spread_ratio, 0, 1)
        
        # 墙影响权重：当展开超过阈值时开始影响
        # 使用平滑过渡函数
        wall_weight = np.where(
            spread_ratio > self.wall_influence_threshold,
            (spread_ratio - self.wall_influence_threshold) / (1 - self.wall_influence_threshold),
            0.0
        )
        wall_weight = np.clip(wall_weight, 0, 0.5)  # 最大影响50%
        
        # 混合接触角
        theta_eff = theta_yl * (1 - wall_weight) + self.theta_wall * wall_weight
        
        return theta_eff
    
    def young_lippmann_pure(self, V: np.ndarray) -> np.ndarray:
        """纯 Young-Lippmann 方程，不含边界效应 (用于对比)"""
        return self.young_lippmann(V, include_wall_effect=False)
    
    def dynamic_response(self, t: np.ndarray, theta_start: float, theta_eq: float) -> np.ndarray:
        """
        二阶欠阻尼系统动态响应
        
        θ(t) = θ_eq + (θ_start - θ_eq) * exp(-ζω₀t) * [cos(ω_d*t) + (ζ/√(1-ζ²))sin(ω_d*t)]
        
        Args:
            t: 时间数组 (s)
            theta_start: 初始角度 (度)
            theta_eq: 平衡角度 (度)
            
        Returns:
            接触角数组 (度)
        """
        if self.zeta >= 1:
            # 过阻尼或临界阻尼：一阶指数响应
            theta = theta_eq + (theta_start - theta_eq) * np.exp(-t / self.tau)
        else:
            # 欠阻尼：振荡响应
            exp_term = np.exp(-self.zeta * self.omega_0 * t)
            damping_factor = self.zeta / np.sqrt(1 - self.zeta**2)
            oscillation = np.cos(self.omega_d * t) + damping_factor * np.sin(self.omega_d * t)
            theta = theta_eq + (theta_start - theta_eq) * exp_term * oscillation
        
        return theta

    def generate_step_response_data(self, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成阶跃响应训练数据 - 包含完整时间序列
        
        关键改进：
        1. 每个阶跃场景生成多个时间点的数据，让模型学习时间演化
        2. 增加高电压稳态样本，改善高电压预测精度
        3. 更均衡的电压分布
        
        Args:
            num_samples: 样本数量
            
        Returns:
            X: 输入特征 (num_samples, 62)
            y: 输出标签 (num_samples, 24)
        """
        V_min, V_max = self.voltage_range
        T_min, T_max = self.time_range
        
        X_list = []
        y_list = []
        
        # 每个阶跃场景生成多个时间点
        num_scenarios = num_samples // self.temporal_resolution
        
        # 场景分配 (优化比例) - 重点增加方波响应
        # 40% 方波 (上升+下降), 20% 上升阶跃, 20% 下降阶跃, 20% 稳态
        n_square = int(num_scenarios * 0.4)
        n_rise = int(num_scenarios * 0.2)
        n_fall = int(num_scenarios * 0.2)
        n_steady = num_scenarios - n_square - n_rise - n_fall
        
        # 场景1: 方波响应 (0V -> 30V -> 0V) - 完整开关周期
        for i in range(n_square):
            V_high = V_min + 15 + (V_max - V_min - 15) * (i + 0.5) / n_square
            V_high += np.random.uniform(-2, 2)
            V_high = np.clip(V_high, 20, V_max)
            t_rise = 0.002  # 上升时刻 2ms
            t_fall = 0.012  # 下降时刻 12ms
            X_seq, y_seq = self._generate_square_wave_sequence(0, V_high, t_rise, t_fall, T_max)
            X_list.append(X_seq)
            y_list.append(y_seq)
        
        # 场景2: 上升阶跃 (0V -> V_target)
        for i in range(n_rise):
            V_target = V_min + 10 + (V_max - V_min - 10) * (i + 0.5) / n_rise
            V_target += np.random.uniform(-2, 2)
            V_target = np.clip(V_target, V_min + 10, V_max)
            t_step = 0.002
            X_seq, y_seq = self._generate_full_sequence(0, V_target, t_step, T_max)
            X_list.append(X_seq)
            y_list.append(y_seq)
        
        # 场景3: 下降阶跃 (V_start -> 0V)
        for i in range(n_fall):
            V_start = V_min + 10 + (V_max - V_min - 10) * (i + 0.5) / n_fall
            V_start += np.random.uniform(-2, 2)
            V_start = np.clip(V_start, V_min + 10, V_max)
            t_step = 0.002
            X_seq, y_seq = self._generate_full_sequence(V_start, 0, t_step, T_max)
            X_list.append(X_seq)
            y_list.append(y_seq)
        
        # 场景4: 稳态 (各种电压)
        for i in range(n_steady):
            V_steady = V_min + (V_max - V_min) * (i + 0.5) / n_steady
            V_steady += np.random.uniform(-1, 1)
            V_steady = np.clip(V_steady, V_min, V_max)
            X_seq, y_seq = self._generate_steady_sequence(V_steady, T_max)
            X_list.append(X_seq)
            y_list.append(y_seq)
        
        X = np.vstack(X_list)
        y = np.vstack(y_list)
        
        # 打乱数据
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        # 截断到请求的样本数
        X = X[:num_samples]
        y = y[:num_samples]
        
        logger.info(f"生成训练数据: X.shape={X.shape}, y.shape={y.shape}")
        logger.info(f"  接触角范围: {np.degrees(y[:, 10].min()):.1f}° - {np.degrees(y[:, 10].max()):.1f}°")
        
        return X.astype(np.float32), y.astype(np.float32)
    
    def _generate_square_wave_sequence(self, V_low: float, V_high: float, 
                                        t_rise: float, t_fall: float, T_total: float) -> Tuple[np.ndarray, np.ndarray]:
        """生成方波响应序列 - 包含上升和下降过程"""
        theta_low = self.young_lippmann(np.array([V_low]))[0]
        theta_high = self.young_lippmann(np.array([V_high]))[0]
        
        nt = self.temporal_resolution
        t = np.linspace(0, T_total, nt)
        
        # 计算动态响应
        theta = np.zeros(nt)
        V = np.zeros(nt)
        theta_current = theta_low
        theta_target = theta_low
        t_last_change = 0
        
        for i, ti in enumerate(t):
            if ti < t_rise:
                # 初始低电压状态
                V[i] = V_low
                theta[i] = theta_low
                theta_current = theta_low
            elif ti < t_fall:
                # 高电压状态 (上升响应)
                V[i] = V_high
                t_since = ti - t_rise
                theta[i] = self.dynamic_response(np.array([t_since]), theta_low, theta_high)[0]
                theta_current = theta[i]
                t_last_change = t_rise
                theta_target = theta_high
            else:
                # 低电压状态 (下降响应/关态)
                V[i] = V_low
                t_since = ti - t_fall
                # 从当前角度（高电压稳态附近）回到低电压稳态
                theta_at_fall = self.dynamic_response(np.array([t_fall - t_rise]), theta_low, theta_high)[0]
                theta[i] = self.dynamic_response(np.array([t_since]), theta_at_fall, theta_low)[0]
                t_last_change = t_fall
                theta_target = theta_low
        
        # 生成特征和标签
        X_seq = []
        y_seq = []
        for i in range(nt):
            ti = t[i]
            # 确定当前阶段的参数
            if ti < t_rise:
                t_step = t_rise
                V_before = V_low
                V_after = V_high
                theta_before = theta_low
            elif ti < t_fall:
                t_step = t_rise
                V_before = V_low
                V_after = V_high
                theta_before = theta_low
            else:
                t_step = t_fall
                V_before = V_high
                V_after = V_low
                theta_before = self.dynamic_response(np.array([t_fall - t_rise]), theta_low, theta_high)[0]
            
            x = self._build_input_features_dynamic(
                ti, V[i], T_total, t_step, V_before, V_after, theta_before
            )
            y = self._build_output_labels(theta[i], V[i])
            X_seq.append(x)
            y_seq.append(y)
        
        return np.array(X_seq), np.array(y_seq)
    
    def _generate_steady_sequence(self, V_steady: float, T_total: float) -> Tuple[np.ndarray, np.ndarray]:
        """生成稳态时间序列 - 电压恒定，接触角为平衡值"""
        theta_eq = self.young_lippmann(np.array([V_steady]))[0]
        
        nt = self.temporal_resolution
        t = np.linspace(0, T_total, nt)
        
        X_seq = []
        y_seq = []
        for i in range(nt):
            # 稳态：没有阶跃，电压始终不变
            x = self._build_input_features_dynamic(
                t[i], V_steady, T_total, 
                t_step=0,  # 没有阶跃
                V_before=V_steady, V_after=V_steady,
                theta_before=theta_eq
            )
            y = self._build_output_labels(theta_eq, V_steady)
            X_seq.append(x)
            y_seq.append(y)
        
        return np.array(X_seq), np.array(y_seq)
    
    def _generate_full_sequence(self, V_start: float, V_end: float, t_step: float, T_total: float) -> Tuple[np.ndarray, np.ndarray]:
        """生成完整的时间序列数据 - 让模型学习动态过程"""
        # 计算平衡角度
        theta_start = self.young_lippmann(np.array([V_start]))[0]
        theta_end = self.young_lippmann(np.array([V_end]))[0]
        
        # 生成时间序列
        nt = self.temporal_resolution
        t = np.linspace(0, T_total, nt)
        
        # 计算动态响应
        theta = np.zeros(nt)
        for i, ti in enumerate(t):
            if ti < t_step:
                theta[i] = theta_start
            else:
                t_since_step = ti - t_step
                theta[i] = self.dynamic_response(np.array([t_since_step]), theta_start, theta_end)[0]
        
        # 电压序列
        V = np.where(t < t_step, V_start, V_end)
        
        # 为每个时间点生成特征和标签
        X_seq = []
        y_seq = []
        for i in range(nt):
            # 添加关键的时间相关特征
            x = self._build_input_features_dynamic(t[i], V[i], T_total, t_step, V_start, V_end, theta_start)
            y = self._build_output_labels(theta[i], V[i])
            X_seq.append(x)
            y_seq.append(y)
        
        return np.array(X_seq), np.array(y_seq)
    
    def _generate_steady_state(self, V: float) -> Tuple[np.ndarray, np.ndarray]:
        """生成稳态数据点"""
        theta_eq = self.young_lippmann(np.array([V]))[0]
        t = np.random.uniform(self.time_range[0], self.time_range[1])
        T_total = self.time_range[1]
        
        x = self._build_input_features_dynamic(t, V, T_total, 0, V, V, theta_eq)
        y = self._build_output_labels(theta_eq, V)
        
        return x.reshape(1, -1), y.reshape(1, -1)
    
    def _build_input_features_dynamic(self, t: float, V: float, T_total: float, 
                                       t_step: float, V_before: float, V_after: float,
                                       theta_before: float) -> np.ndarray:
        """
        构建62维输入特征 - 包含动态响应所需的关键信息
        
        关键改进：添加阶跃时间、电压变化量、阶跃后经过时间等特征
        """
        features = np.zeros(62, dtype=np.float32)
        V_max = self.voltage_range[1]
        
        # 空间坐标 (随机采样)
        features[0] = np.random.uniform(0, 1)  # x
        features[1] = np.random.uniform(0, 1)  # y
        features[2] = np.random.uniform(0, 1)  # z
        
        # 时间特征 - 关键！
        features[3] = t / T_total  # 归一化时间
        features[4] = np.sin(2 * np.pi * t / T_total)
        features[5] = np.cos(2 * np.pi * t / T_total)
        
        # 电压特征
        features[6] = V / V_max  # 当前电压
        features[7] = (V / V_max) ** 2  # 电压平方项
        
        # ⭐ 动态响应关键特征
        features[8] = t_step / T_total  # 阶跃发生时间
        features[9] = max(0, t - t_step) / T_total  # 阶跃后经过的时间 (关键!)
        features[10] = max(0, t - t_step) / self.tau  # 以时间常数为单位的时间
        
        # 电压变化信息
        features[11] = V_before / V_max  # 阶跃前电压
        features[12] = V_after / V_max   # 阶跃后电压
        features[13] = (V_after - V_before) / V_max  # 电压变化量
        
        # 角度变化信息
        theta_after = self.young_lippmann(np.array([V_after]))[0]
        features[14] = np.radians(theta_before) / np.pi  # 初始角度
        features[15] = np.radians(theta_after) / np.pi   # 目标角度
        features[16] = np.radians(theta_after - theta_before) / np.pi  # 角度变化量
        
        # 动力学参数
        features[17] = self.tau * 1000  # 时间常数 (ms)
        features[18] = self.zeta  # 阻尼比
        features[19] = self.omega_0 / 1000  # 自然频率
        
        # 材料参数
        features[20] = self.epsilon_r / 10.0
        features[21] = self.gamma / 0.1
        features[22] = self.d / 1e-6
        features[23] = self.theta0 / 180.0
        
        # 几何参数
        features[24] = self.Lx / 1e-3
        features[25] = self.Ly / 1e-3
        features[26] = self.Lz / 1e-4
        
        # 动态响应阶段指示
        if t < t_step:
            features[27] = 0.0  # 阶跃前
        elif t < t_step + self.tau:
            features[27] = 0.5  # 快速响应阶段
        else:
            features[27] = 1.0  # 稳定阶段
        
        # 理论响应进度 (0-1)
        if t >= t_step:
            t_since = t - t_step
            if self.zeta < 1:
                exp_decay = np.exp(-self.zeta * self.omega_0 * t_since)
                features[28] = 1.0 - exp_decay  # 响应进度
            else:
                features[28] = 1.0 - np.exp(-t_since / self.tau)
        else:
            features[28] = 0.0
        
        # ⭐ 新增：角速度和角加速度特征 (帮助学习动态响应)
        theta_after = self.young_lippmann(np.array([V_after]))[0]
        delta_theta = theta_after - theta_before  # 角度变化量 (度)
        
        if t >= t_step and abs(delta_theta) > 0.1:
            t_since = t - t_step
            # 二阶欠阻尼系统的角速度
            # dθ/dt = (θ_eq - θ_0) * ω_0 * exp(-ζω_0t) * sin(ω_d*t) / sqrt(1-ζ²)
            if self.zeta < 1:
                omega_d = self.omega_0 * np.sqrt(1 - self.zeta**2)
                exp_term = np.exp(-self.zeta * self.omega_0 * t_since)
                # 角速度 (度/秒)
                angular_velocity = -delta_theta * self.omega_0 * exp_term * np.sin(omega_d * t_since) / np.sqrt(1 - self.zeta**2)
                # 归一化角速度
                features[29] = angular_velocity / (abs(delta_theta) * self.omega_0 + 1e-6)
                
                # 角加速度 (简化)
                features[30] = -self.zeta * self.omega_0 * features[29]
            else:
                features[29] = 0.0
                features[30] = 0.0
        else:
            features[29] = 0.0
            features[30] = 0.0
        
        # 电压变化方向 (上升=1, 下降=-1, 不变=0)
        features[31] = np.sign(V_after - V_before)
        
        # 当前角度与目标角度的差距比例
        if abs(delta_theta) > 0.1 and t >= t_step:
            t_since = t - t_step
            current_theta = self.dynamic_response(np.array([t_since]), theta_before, theta_after)[0]
            features[32] = (current_theta - theta_after) / (theta_before - theta_after + 1e-6)
        else:
            features[32] = 0.0 if t >= t_step else 1.0
        
        # ⭐ 新增：理论预测的当前角度 (让模型学习修正)
        if t >= t_step:
            t_since = t - t_step
            theta_theory = self.dynamic_response(np.array([t_since]), theta_before, theta_after)[0]
        else:
            theta_theory = theta_before
        features[33] = np.radians(theta_theory) / np.pi  # 理论角度
        
        # 理论角度与目标角度的差
        features[34] = np.radians(theta_theory - theta_after) / np.pi
        
        # 响应完成度 (基于理论)
        if abs(delta_theta) > 0.1:
            features[35] = (theta_theory - theta_before) / (theta_after - theta_before + 1e-6)
        else:
            features[35] = 1.0
        
        # 时间常数的倍数
        if t >= t_step:
            features[36] = (t - t_step) / self.tau
        else:
            features[36] = 0.0
        
        # 填充剩余特征
        for i in range(37, 62):
            features[i] = np.random.uniform(-0.01, 0.01)
        
        return features
    
    def _build_input_features(self, t: float, V: float, T_total: float) -> np.ndarray:
        """
        构建62维输入特征
        
        特征布局:
        [0-2]: 空间坐标 (x, y, z) 归一化
        [3]: 时间 t/T_total
        [4]: sin(2πt/T)
        [5]: 电压 V/V_max
        [6]: 到边界距离
        [7]: 到中心距离
        [8-61]: 其他特征 (材料参数、几何参数等)
        """
        features = np.zeros(62, dtype=np.float32)
        
        # 空间坐标 (随机采样)
        features[0] = np.random.uniform(0, 1)  # x归一化
        features[1] = np.random.uniform(0, 1)  # y归一化
        features[2] = np.random.uniform(0, 1)  # z归一化
        
        # 时间特征
        features[3] = t / T_total
        features[4] = np.sin(2 * np.pi * t / T_total)
        features[5] = np.cos(2 * np.pi * t / T_total)
        
        # 电压特征
        V_max = self.voltage_range[1]
        features[6] = V / V_max
        features[7] = (V / V_max) ** 2  # 电压平方项 (Young-Lippmann)
        
        # 空间特征
        x_pos = features[0] * self.Lx
        y_pos = features[1] * self.Ly
        features[8] = min(x_pos, self.Lx - x_pos) / self.Lx  # 到x边界距离
        features[9] = min(y_pos, self.Ly - y_pos) / self.Ly  # 到y边界距离
        features[10] = np.sqrt((features[0] - 0.5)**2 + (features[1] - 0.5)**2)  # 到中心距离
        
        # 材料参数 (归一化)
        features[11] = self.epsilon_r / 10.0
        features[12] = self.gamma / 0.1
        features[13] = self.d / 1e-6
        features[14] = self.theta0 / 180.0
        
        # 几何参数 (归一化)
        features[15] = self.Lx / 1e-3
        features[16] = self.Ly / 1e-3
        features[17] = self.Lz / 1e-4
        
        # 动力学参数
        features[18] = self.tau / 0.01
        features[19] = self.zeta
        
        # 时间导数特征 (用于动态响应)
        features[20] = np.cos(2 * np.pi * t / T_total)
        features[21] = -2 * np.pi / T_total * np.sin(2 * np.pi * t / T_total)
        
        # 电压变化率特征 (简化)
        features[22] = 0  # dV/dt (需要更多上下文)
        
        # 填充剩余特征
        for i in range(23, 62):
            features[i] = np.random.uniform(-0.1, 0.1)  # 小随机噪声
        
        return features

    def _build_output_labels(self, theta_deg: float, V: float) -> np.ndarray:
        """
        构建24维输出标签
        
        输出布局:
        [0-2]: 速度场 (u, v, w)
        [3-5]: 压力和压力梯度
        [6-9]: 界面位置和形状
        [10]: 接触角 (弧度) - 主要目标
        [11-14]: 界面曲率和法向量
        [15-17]: 表面张力分量
        [18-20]: 电场分量
        [21-23]: 其他物理量
        """
        labels = np.zeros(24, dtype=np.float32)
        
        theta_rad = np.radians(theta_deg)
        
        # 速度场 (简化：稳态时接近零)
        labels[0] = 0.0  # u
        labels[1] = 0.0  # v
        labels[2] = 0.0  # w
        
        # 压力 (与电压相关)
        V_max = self.voltage_range[1]
        labels[3] = 0.5 * self.epsilon_0 * self.epsilon_r * (V / V_max)**2  # 电场压力
        labels[4] = 0.0  # dp/dx
        labels[5] = 0.0  # dp/dy
        
        # 界面位置 (简化)
        labels[6] = 0.5  # 界面x位置
        labels[7] = 0.5  # 界面y位置
        labels[8] = self.Lz * np.cos(theta_rad)  # 界面高度
        labels[9] = 0.0  # 界面速度
        
        # 接触角 (弧度) - 主要输出
        labels[10] = theta_rad
        
        # 界面曲率 (与接触角相关)
        # 简化：假设球冠形状，曲率 κ ≈ 2*cos(θ)/R
        R_base = self.Lx / 2  # 基底半径
        labels[11] = 2 * np.cos(theta_rad) / R_base  # 平均曲率
        labels[12] = np.sin(theta_rad)  # 法向量x分量
        labels[13] = 0.0  # 法向量y分量
        labels[14] = np.cos(theta_rad)  # 法向量z分量
        
        # 表面张力分量
        labels[15] = self.gamma * np.sin(theta_rad)  # 水平分量
        labels[16] = 0.0
        labels[17] = self.gamma * np.cos(theta_rad)  # 垂直分量
        
        # 电场分量 (简化)
        E_field = V / self.d if self.d > 0 else 0
        labels[18] = 0.0  # Ex
        labels[19] = 0.0  # Ey
        labels[20] = E_field / 1e6  # Ez (归一化)
        
        # 其他物理量
        labels[21] = np.cos(theta_rad)  # cos(θ) 用于Young-Lippmann验证
        labels[22] = V / V_max  # 归一化电压
        labels[23] = (theta_deg - self.theta0) / self.theta0  # 相对角度变化
        
        return labels
    
    def generate_full_dataset(self, num_samples: int, train_ratio: float = 0.7, 
                              val_ratio: float = 0.15) -> Dict[str, np.ndarray]:
        """
        生成完整的训练/验证/测试数据集
        
        Args:
            num_samples: 总样本数
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            
        Returns:
            包含所有数据集的字典
        """
        X, y = self.generate_step_response_data(num_samples)
        
        # 分割数据
        n_train = int(num_samples * train_ratio)
        n_val = int(num_samples * val_ratio)
        
        X_train, y_train = X[:n_train], y[:n_train]
        X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
        X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]
        
        logger.info(f"数据集分割: 训练={len(X_train)}, 验证={len(X_val)}, 测试={len(X_test)}")
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }


def generate_physics_based_data(config: dict, num_samples: int, device: torch.device) -> Tuple:
    """
    生成基于物理的训练数据 (供 efd_pinns_train.py 调用)
    
    Args:
        config: 配置字典
        num_samples: 样本数量
        device: PyTorch设备
        
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test, physics_points, normalizer
    """
    from src.training.components import DataNormalizer
    
    generator = ElectrowettingPhysicsGenerator(config)
    
    data_config = config.get('data', {})
    train_ratio = data_config.get('train_ratio', 0.7)
    val_ratio = data_config.get('val_ratio', 0.15)
    
    dataset = generator.generate_full_dataset(num_samples, train_ratio, val_ratio)
    
    # 创建归一化器
    input_normalizer = DataNormalizer(method='standard')
    input_normalizer.fit(dataset['X_train'])
    
    output_normalizer = DataNormalizer(method='standard')
    output_normalizer.fit(dataset['y_train'])
    
    # 归一化输入
    X_train_norm = input_normalizer.transform(dataset['X_train'])
    X_val_norm = input_normalizer.transform(dataset['X_val'])
    X_test_norm = input_normalizer.transform(dataset['X_test'])
    
    # 归一化输出
    y_train_norm = output_normalizer.transform(dataset['y_train'])
    y_val_norm = output_normalizer.transform(dataset['y_val'])
    y_test_norm = output_normalizer.transform(dataset['y_test'])
    
    # 生成物理约束点
    physics_samples = min(1000, num_samples // 5)
    X_physics, _ = generator.generate_step_response_data(physics_samples)
    X_physics_norm = input_normalizer.transform(X_physics)
    physics_points = torch.tensor(X_physics_norm, dtype=torch.float32, device=device)
    
    # 转换为张量
    X_train = torch.tensor(X_train_norm, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train_norm, dtype=torch.float32, device=device)
    X_val = torch.tensor(X_val_norm, dtype=torch.float32, device=device)
    y_val = torch.tensor(y_val_norm, dtype=torch.float32, device=device)
    X_test = torch.tensor(X_test_norm, dtype=torch.float32, device=device)
    y_test = torch.tensor(y_test_norm, dtype=torch.float32, device=device)
    
    logger.info(f"✅ 基于物理的数据生成完成")
    logger.info(f"   训练集: {X_train.shape}, 验证集: {X_val.shape}, 测试集: {X_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, physics_points, input_normalizer, output_normalizer


if __name__ == '__main__':
    # 测试数据生成
    import json
    
    logging.basicConfig(level=logging.INFO)
    
    with open('config_stage2_10k.json', 'r') as f:
        config = json.load(f)
    
    generator = ElectrowettingPhysicsGenerator(config)
    
    # 测试 Young-Lippmann
    print("\n=== Young-Lippmann 方程测试 ===")
    voltages = np.array([0, 10, 20, 30])
    angles = generator.young_lippmann(voltages)
    for V, theta in zip(voltages, angles):
        print(f"  V={V:2d}V → θ={theta:.1f}°")
    
    # 测试数据生成
    print("\n=== 数据生成测试 ===")
    X, y = generator.generate_step_response_data(1000)
    print(f"  X.shape: {X.shape}")
    print(f"  y.shape: {y.shape}")
    print(f"  接触角范围: {np.degrees(y[:, 10].min()):.1f}° - {np.degrees(y[:, 10].max()):.1f}°")
    print(f"  接触角均值: {np.degrees(y[:, 10].mean()):.1f}°")
    print(f"  接触角标准差: {np.degrees(y[:, 10].std()):.1f}°")
