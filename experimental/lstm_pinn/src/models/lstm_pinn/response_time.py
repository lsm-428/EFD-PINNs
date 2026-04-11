"""
LSTM-PINN Response Time 计算模块

Response Time 定义：
- 升压滞后时间：0V → V_target 达到稳态最大开口率的时间
- 降压恢复时间：V_target 最大开口 → 0V 完全关闭的时间
- 总响应时间 = 升压滞后 + 降压恢复
"""

import numpy as np
import torch
from typing import Dict, Optional, Tuple, Union

from .physics import ElectrowettingPhysics


class ResponseTimeCalculator:
    """
    Response Time 计算器
    
    基于物理模型计算升压滞后时间和降压恢复时间
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化计算器
        
        Args:
            config: 配置字典
        """
        self.physics = ElectrowettingPhysics(config)
        
        # 响应时间阈值（达到稳态值的百分比）
        self.rise_threshold = 0.95  # 升压：达到 95% 稳态值
        self.fall_threshold = 0.05  # 降压：降到 5% 初始值
        
        # 时间分辨率
        self.dt = 0.0001  # 0.1ms
        self.t_max = 0.1   # 100ms 最大搜索时间
    
    def compute_rise_time(
        self,
        V_from: float,
        V_to: float,
        threshold: Optional[float] = None
    ) -> Dict[str, float]:
        """
        计算升压滞后时间
        
        Args:
            V_from: 初始电压 (V)
            V_to: 目标电压 (V)
            threshold: 达到稳态的阈值（默认 0.95）
        
        Returns:
            包含 rise_time, eta_initial, eta_final, eta_target 的字典
        """
        if threshold is None:
            threshold = self.rise_threshold
        
        # 初始和目标开口率
        eta_initial = self.physics.get_steady_state_aperture(V_from)
        eta_final = self.physics.get_steady_state_aperture(V_to)
        
        # 目标开口率（达到稳态的 threshold 比例）
        eta_target = eta_initial + threshold * (eta_final - eta_initial)
        
        # 搜索达到目标的时间
        rise_time = None
        t = 0.0
        
        while t < self.t_max:
            theta = self.physics.contact_angle_rise(V_to, t, V_from)
            eta = self.physics.contact_angle_to_aperture(theta)
            
            if eta >= eta_target:
                rise_time = t
                break
            
            t += self.dt
        
        if rise_time is None:
            rise_time = self.t_max  # 未达到目标
        
        return {
            "rise_time": rise_time,
            "rise_time_ms": rise_time * 1000,
            "eta_initial": eta_initial,
            "eta_final": eta_final,
            "eta_target": eta_target,
            "V_from": V_from,
            "V_to": V_to
        }
    
    def compute_fall_time(
        self,
        V_from: float,
        V_to: float = 0.0,
        threshold: Optional[float] = None
    ) -> Dict[str, float]:
        """
        计算降压恢复时间
        
        Args:
            V_from: 初始电压 (V)
            V_to: 目标电压 (V)，默认 0V
            threshold: 降到初始值的阈值（默认 0.05）
        
        Returns:
            包含 fall_time, eta_initial, eta_final, eta_target 的字典
        """
        if threshold is None:
            threshold = self.fall_threshold
        
        # 初始和目标开口率
        eta_initial = self.physics.get_steady_state_aperture(V_from)
        eta_final = self.physics.get_steady_state_aperture(V_to)
        
        # 目标开口率（降到初始值的 threshold 比例）
        eta_target = eta_final + threshold * (eta_initial - eta_final)
        
        # 搜索达到目标的时间
        fall_time = None
        t = 0.0
        
        while t < self.t_max:
            eta = self.physics.aperture_fall(V_from, t, V_to)
            
            if eta <= eta_target:
                fall_time = t
                break
            
            t += self.dt
        
        if fall_time is None:
            fall_time = self.t_max  # 未达到目标
        
        return {
            "fall_time": fall_time,
            "fall_time_ms": fall_time * 1000,
            "eta_initial": eta_initial,
            "eta_final": eta_final,
            "eta_target": eta_target,
            "V_from": V_from,
            "V_to": V_to
        }
    
    def compute_total_response_time(
        self,
        V_target: float,
        V_initial: float = 0.0
    ) -> Dict[str, float]:
        """
        计算总响应时间（升压 + 降压）
        
        Args:
            V_target: 目标电压 (V)
            V_initial: 初始电压 (V)，默认 0V
        
        Returns:
            包含 total_time, rise_time, fall_time 等的字典
        """
        # 升压时间
        rise_result = self.compute_rise_time(V_initial, V_target)
        
        # 降压时间
        fall_result = self.compute_fall_time(V_target, V_initial)
        
        # 总时间
        total_time = rise_result["rise_time"] + fall_result["fall_time"]
        
        return {
            "total_time": total_time,
            "total_time_ms": total_time * 1000,
            "rise_time": rise_result["rise_time"],
            "rise_time_ms": rise_result["rise_time_ms"],
            "fall_time": fall_result["fall_time"],
            "fall_time_ms": fall_result["fall_time_ms"],
            "V_target": V_target,
            "V_initial": V_initial,
            "eta_max": rise_result["eta_final"]
        }
    
    def compute_partial_transition_time(
        self,
        V_from: float,
        V_to: float
    ) -> Dict[str, float]:
        """
        计算部分跳变的响应时间
        
        支持任意升压（如 10V → 20V）或降压（如 30V → 20V）
        
        Args:
            V_from: 初始电压 (V)
            V_to: 目标电压 (V)
        
        Returns:
            响应时间信息
        """
        if V_to > V_from:
            # 升压
            return self.compute_rise_time(V_from, V_to)
        else:
            # 降压
            return self.compute_fall_time(V_from, V_to)
    
    def get_response_time_curve(
        self,
        voltages: Optional[list] = None
    ) -> Dict[str, np.ndarray]:
        """
        获取不同电压的响应时间曲线
        
        Args:
            voltages: 电压列表，默认 [5, 10, 15, 20, 25, 30]
        
        Returns:
            包含 voltages, rise_times, fall_times, total_times 的字典
        """
        if voltages is None:
            voltages = [5, 10, 15, 20, 25, 30]
        
        rise_times = []
        fall_times = []
        total_times = []
        eta_max_values = []
        
        for V in voltages:
            result = self.compute_total_response_time(V)
            rise_times.append(result["rise_time_ms"])
            fall_times.append(result["fall_time_ms"])
            total_times.append(result["total_time_ms"])
            eta_max_values.append(result["eta_max"])
        
        return {
            "voltages": np.array(voltages),
            "rise_times_ms": np.array(rise_times),
            "fall_times_ms": np.array(fall_times),
            "total_times_ms": np.array(total_times),
            "eta_max": np.array(eta_max_values)
        }


def compute_response_time(
    V_target: float,
    config: Optional[Dict] = None
) -> Dict[str, float]:
    """
    便捷函数：计算指定电压的响应时间
    
    Args:
        V_target: 目标电压 (V)
        config: 配置字典
    
    Returns:
        响应时间信息
    """
    calculator = ResponseTimeCalculator(config)
    return calculator.compute_total_response_time(V_target)
