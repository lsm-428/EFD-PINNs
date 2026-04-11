"""
LSTM-PINN 可视化模块

包含：
- φ 场俯视图可视化
- 开口率动态曲线可视化
- C-V 曲线可视化
- 多步序列响应可视化
"""

import numpy as np
import torch
from typing import Dict, Optional, List, Tuple, Union
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from .physics import ElectrowettingPhysics
from .response_time import ResponseTimeCalculator


# 自定义颜色映射（油墨为深色，透明为浅色）
INK_CMAP = LinearSegmentedColormap.from_list(
    'ink', ['white', 'darkblue']
)


class LSTMPINNVisualizer:
    """
    LSTM-PINN 可视化器
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化可视化器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.physics = ElectrowettingPhysics(config)
        self.response_calc = ResponseTimeCalculator(config)
        
        # 几何参数
        geometry = self.config.get("geometry", {})
        self.Lx = geometry.get("Lx", 174e-6)
        self.Ly = geometry.get("Ly", 174e-6)
        self.Lz = geometry.get("Lz", 20e-6)
        self.h_ink = geometry.get("ink_thickness", 3e-6)
        
        # 默认图形设置
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
    
    def plot_phi_topview(
        self,
        phi_field: np.ndarray,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        title: str = "φ Field Top View",
        ax: Optional[plt.Axes] = None,
        cmap: str = 'RdYlBu_r',
        show_colorbar: bool = True
    ) -> plt.Axes:
        """
        绘制 φ 场俯视图
        
        Args:
            phi_field: (nx, ny) φ 值网格
            x_coords: (nx,) x 坐标
            y_coords: (ny,) y 坐标
            title: 图标题
            ax: matplotlib axes
            cmap: 颜色映射
            show_colorbar: 是否显示颜色条
        
        Returns:
            matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))
        
        # 转换坐标为 μm
        x_um = x_coords * 1e6
        y_um = y_coords * 1e6
        
        # 绘制
        im = ax.pcolormesh(
            x_um, y_um, phi_field.T,
            cmap=cmap, vmin=0, vmax=1, shading='auto'
        )
        
        ax.set_xlabel('x (μm)')
        ax.set_ylabel('y (μm)')
        ax.set_title(title)
        ax.set_aspect('equal')
        
        if show_colorbar:
            plt.colorbar(im, ax=ax, label='φ (oil fraction)')
        
        return ax
    
    def plot_aperture_dynamics(
        self,
        voltages: List[float] = [10, 20, 30],
        t_max: float = 0.05,
        n_points: int = 500,
        include_fall: bool = True,
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        """
        绘制开口率动态曲线
        
        Args:
            voltages: 电压列表
            t_max: 最大时间 (s)
            n_points: 时间点数
            include_fall: 是否包含降压过程
            ax: matplotlib axes
        
        Returns:
            matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        t = np.linspace(0, t_max, n_points)
        t_ms = t * 1000
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(voltages)))
        
        for V, color in zip(voltages, colors):
            # 升压过程
            eta_rise = []
            for ti in t:
                theta = self.physics.contact_angle_rise(V, ti, 0)
                eta = self.physics.contact_angle_to_aperture(theta)
                eta_rise.append(eta)
            
            ax.plot(t_ms, eta_rise, color=color, label=f'{V}V rise', linewidth=2)
            
            if include_fall:
                # 降压过程（从稳态开始）
                eta_fall = []
                for ti in t:
                    eta = self.physics.aperture_fall(V, ti, 0)
                    eta_fall.append(eta)
                
                ax.plot(t_ms, eta_fall, color=color, linestyle='--', 
                       label=f'{V}V fall', linewidth=2, alpha=0.7)
        
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Aperture Ratio η')
        ax.set_title('Aperture Dynamics')
        ax.legend(loc='best', ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, t_max * 1000)
        ax.set_ylim(0, 1)
        
        return ax
    
    def plot_cv_curve(
        self,
        v_max: float = 30,
        n_points: int = 100,
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        """
        绘制 C-V 曲线（稳态开口率 vs 电压）
        
        Args:
            v_max: 最大电压
            n_points: 电压点数
            ax: matplotlib axes
        
        Returns:
            matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        voltages = np.linspace(0, v_max, n_points)
        apertures = [self.physics.get_steady_state_aperture(V) for V in voltages]
        
        ax.plot(voltages, apertures, 'b-', linewidth=2)
        ax.fill_between(voltages, 0, apertures, alpha=0.2)
        
        # 标记关键点
        V_threshold = self.physics.params.get("V_threshold", 3.0)
        ax.axvline(V_threshold, color='r', linestyle='--', alpha=0.5, 
                  label=f'V_threshold = {V_threshold}V')
        
        ax.set_xlabel('Voltage (V)')
        ax.set_ylabel('Steady-State Aperture Ratio η')
        ax.set_title('C-V Curve (Capacitance-Voltage)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, v_max)
        ax.set_ylim(0, 1)
        
        return ax
    
    def plot_multi_step_response(
        self,
        voltage_sequence: List[Tuple[float, float, float]],
        n_points_per_step: int = 100,
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        """
        绘制多步序列响应
        
        Args:
            voltage_sequence: [(t_start, t_end, V), ...] 电压序列
            n_points_per_step: 每步的时间点数
            ax: matplotlib axes
        
        Returns:
            matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
        
        all_t = []
        all_eta = []
        all_V = []
        
        V_prev = 0
        
        for t_start, t_end, V in voltage_sequence:
            t = np.linspace(t_start, t_end, n_points_per_step)
            
            for ti in t:
                t_since = ti - t_start
                
                if V > V_prev:
                    # 升压
                    theta = self.physics.contact_angle_rise(V, t_since, V_prev)
                    eta = self.physics.contact_angle_to_aperture(theta)
                elif V < V_prev:
                    # 降压
                    eta = self.physics.aperture_fall(V_prev, t_since, V)
                else:
                    # 恒定
                    eta = self.physics.get_steady_state_aperture(V)
                
                all_t.append(ti)
                all_eta.append(eta)
                all_V.append(V)
            
            V_prev = V
        
        # 绘制开口率
        ax.plot(np.array(all_t) * 1000, all_eta, 'b-', linewidth=2, label='Aperture η')
        
        # 绘制电压（次坐标轴）
        ax2 = ax.twinx()
        ax2.plot(np.array(all_t) * 1000, all_V, 'r--', linewidth=1.5, alpha=0.7, label='Voltage')
        ax2.set_ylabel('Voltage (V)', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Aperture Ratio η', color='b')
        ax.tick_params(axis='y', labelcolor='b')
        ax.set_title('Multi-Step Voltage Response')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # 合并图例
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='best')
        
        return ax
    
    def plot_response_time_summary(
        self,
        voltages: List[float] = [5, 10, 15, 20, 25, 30],
        ax: Optional[plt.Axes] = None
    ) -> plt.Axes:
        """
        绘制响应时间汇总图
        
        Args:
            voltages: 电压列表
            ax: matplotlib axes
        
        Returns:
            matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        curve = self.response_calc.get_response_time_curve(voltages)
        
        x = np.arange(len(voltages))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, curve["rise_times_ms"], width, 
                      label='Rise Time', color='steelblue')
        bars2 = ax.bar(x + width/2, curve["fall_times_ms"], width,
                      label='Fall Time', color='coral')
        
        ax.set_xlabel('Voltage (V)')
        ax.set_ylabel('Response Time (ms)')
        ax.set_title('Response Time Summary')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{v}V' for v in voltages])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
        
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
        
        return ax
    
    def create_summary_figure(
        self,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        创建汇总图（4 子图）
        
        Args:
            save_path: 保存路径
        
        Returns:
            matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. C-V 曲线
        self.plot_cv_curve(ax=axes[0, 0])
        
        # 2. 开口率动态
        self.plot_aperture_dynamics(ax=axes[0, 1])
        
        # 3. 响应时间汇总
        self.plot_response_time_summary(ax=axes[1, 0])
        
        # 4. 多步序列响应
        sequence = [
            (0, 0.01, 0),      # 0-10ms: 0V
            (0.01, 0.02, 20),  # 10-20ms: 20V
            (0.02, 0.03, 30),  # 20-30ms: 30V
            (0.03, 0.04, 20),  # 30-40ms: 20V
            (0.04, 0.05, 0),   # 40-50ms: 0V
        ]
        self.plot_multi_step_response(sequence, ax=axes[1, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
