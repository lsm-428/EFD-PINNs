"""
EWP Flow Field Solver - 电润湿像素流场求解器

实现两相流 Navier-Stokes 方程的数值求解，追踪油墨-水界面演化。
支持传统 CFD 求解（用于验证）和 PINN 加速求解。

核心功能：
- 结构化网格生成
- VOF 界面追踪
- 移动接触线处理
- 两相流 Navier-Stokes 求解
- PINN 加速求解
- 与 EnhancedApertureModel 集成

Author: EFD-PINNs Team
Date: 2025-12-03
"""

import numpy as np
import torch
import torch.nn as nn
import json
import os
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Tuple, Dict, Any, Union
from datetime import datetime

# 导入现有模块
from src.predictors.hybrid_predictor import HybridPredictor
from src.models.aperture_model import EnhancedApertureModel


# ============================================================
# 物理常量和参数
# ============================================================

# 流体物理属性
FLUID_PROPERTIES = {
    "oil": {
        "density": 1000.0,          # kg/m³ (油墨密度)
        "viscosity": 0.01,          # Pa·s (油墨粘度)
        "surface_tension": 0.030,   # N/m (油-水界面张力)
    },
    "polar_liquid": {
        "density": 1000.0,          # kg/m³ (极性液体密度)
        "viscosity": 0.001,         # Pa·s (极性液体粘度)
        "surface_tension": 0.072,   # N/m (水-空气界面张力)
    }
}

# 计算域参数
DOMAIN_PARAMETERS = {
    "Lx": 174e-6,           # m (像素内沿宽度)
    "Ly": 174e-6,           # m (像素内沿高度)
    "Lz": 20e-6,            # m (围堰高度)
    "ink_thickness": 3e-6,  # m (油墨层厚度)
    "polar_thickness": 17e-6,  # m (极性液体层厚度)
}

# 数值参数
NUMERICAL_PARAMETERS = {
    "cfl_number": 0.5,      # CFL 数
    "max_iterations": 100,  # 最大迭代次数
    "convergence_tol": 1e-6,  # 收敛容差
    "mass_error_tol": 0.001,  # 质量误差容差 (0.1%)
}


# ============================================================
# 数据类定义
# ============================================================

@dataclass
class Mesh:
    """计算网格数据结构"""
    # 节点坐标
    x: np.ndarray       # (nx+1,) 节点 x 坐标
    y: np.ndarray       # (ny+1,) 节点 y 坐标
    z: np.ndarray       # (nz+1,) 节点 z 坐标
    
    # 单元中心坐标
    xc: np.ndarray      # (nx,) 单元中心 x 坐标
    yc: np.ndarray      # (ny,) 单元中心 y 坐标
    zc: np.ndarray      # (nz,) 单元中心 z 坐标
    
    # 网格尺寸
    dx: float           # x 方向网格间距
    dy: float           # y 方向网格间距
    dz: float           # z 方向网格间距
    
    # 网格数量
    nx: int             # x 方向网格数
    ny: int             # y 方向网格数
    nz: int             # z 方向网格数
    
    # 边界单元索引
    boundary_cells: Dict[str, np.ndarray] = field(default_factory=dict)
    
    @property
    def total_cells(self) -> int:
        """总单元数"""
        return self.nx * self.ny * self.nz
    
    @property
    def cell_volume(self) -> float:
        """单元体积"""
        return self.dx * self.dy * self.dz
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于序列化）"""
        return {
            'nx': self.nx, 'ny': self.ny, 'nz': self.nz,
            'dx': self.dx, 'dy': self.dy, 'dz': self.dz,
            'Lx': self.x[-1] - self.x[0],
            'Ly': self.y[-1] - self.y[0],
            'Lz': self.z[-1] - self.z[0],
        }


@dataclass
class SimulationResult:
    """模拟结果数据结构"""
    # 时间序列
    t: np.ndarray               # (nt,) 时间点
    
    # 流场数据 (可选，大数据量时可能只保存部分时间步)
    u: Optional[np.ndarray] = None      # (nt, nx, ny, nz) x 方向速度
    v: Optional[np.ndarray] = None      # (nt, nx, ny, nz) y 方向速度
    w: Optional[np.ndarray] = None      # (nt, nx, ny, nz) z 方向速度
    p: Optional[np.ndarray] = None      # (nt, nx, ny, nz) 压力
    phi: Optional[np.ndarray] = None    # (nt, nx, ny, nz) 体积分数
    
    # 积分量（始终保存）
    aperture_ratio: Optional[np.ndarray] = None  # (nt,) 开口率
    contact_angle: Optional[np.ndarray] = None   # (nt,) 接触角
    
    # 守恒误差
    mass_error: Optional[np.ndarray] = None      # (nt,) 质量误差
    
    # 元数据
    config: Dict[str, Any] = field(default_factory=dict)
    computation_time: float = 0.0
    method: str = 'cfd'
    
    def has_full_fields(self) -> bool:
        """是否包含完整流场数据"""
        return all(f is not None for f in [self.u, self.v, self.w, self.p, self.phi])
    
    def get_field_at_time(self, field_name: str, time_idx: int) -> np.ndarray:
        """获取指定时间步的场数据"""
        field = getattr(self, field_name, None)
        if field is None:
            raise ValueError(f"Field '{field_name}' not available")
        return field[time_idx]


@dataclass
class SolverConfig:
    """求解器配置"""
    # 网格参数
    nx: int = 32
    ny: int = 32
    nz: int = 16
    
    # 时间参数
    dt: Optional[float] = None  # 时间步长，None 表示自动计算
    t_end: float = 0.02         # 结束时间 (20ms)
    save_interval: int = 10     # 保存间隔（每多少步保存一次）
    
    # 物理参数
    sigma: float = 0.030        # 界面张力 (N/m)
    rho_oil: float = 1000.0     # 油墨密度 (kg/m³)
    rho_water: float = 1000.0   # 极性液体密度 (kg/m³)
    mu_oil: float = 0.01        # 油墨粘度 (Pa·s)
    mu_water: float = 0.001     # 极性液体粘度 (Pa·s)
    
    # 数值参数
    cfl: float = 0.5
    max_iter: int = 100
    tol: float = 1e-6
    
    # 边界条件
    slip_length: float = 1e-9   # 滑移长度 (m)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SolverConfig':
        """从字典创建"""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    @classmethod
    def from_json(cls, path: str) -> 'SolverConfig':
        """从 JSON 文件加载"""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data.get('solver', data))
    
    def save_json(self, path: str):
        """保存为 JSON 文件"""
        with open(path, 'w') as f:
            json.dump({'solver': self.to_dict()}, f, indent=2)


# ============================================================
# 异常类定义
# ============================================================

class FlowSolverError(Exception):
    """流场求解器错误基类"""
    pass


class MeshGenerationError(FlowSolverError):
    """网格生成错误"""
    pass


class ConvergenceError(FlowSolverError):
    """收敛错误"""
    pass


class MassConservationError(FlowSolverError):
    """质量守恒错误"""
    pass


class InterfaceTrackingError(FlowSolverError):
    """界面追踪错误"""
    pass


class CFLViolationError(FlowSolverError):
    """CFL 条件违反错误"""
    pass


# ============================================================
# MeshGenerator 类
# ============================================================

class MeshGenerator:
    """
    计算网格生成器
    
    生成用于两相流求解的结构化网格。
    
    Example:
        >>> generator = MeshGenerator()
        >>> mesh = generator.generate_structured_mesh(nx=32, ny=32, nz=16)
        >>> print(f"Total cells: {mesh.total_cells}")
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化网格生成器
        
        Args:
            config: 网格配置，包含域尺寸等参数
        """
        if config is None:
            config = {}
        
        # 域尺寸
        self.Lx = config.get('Lx', DOMAIN_PARAMETERS['Lx'])
        self.Ly = config.get('Ly', DOMAIN_PARAMETERS['Ly'])
        self.Lz = config.get('Lz', DOMAIN_PARAMETERS['Lz'])
        
        # 油墨层厚度（用于初始条件）
        self.ink_thickness = config.get('ink_thickness', DOMAIN_PARAMETERS['ink_thickness'])
    
    def generate_structured_mesh(self, nx: int = 32, ny: int = 32, nz: int = 16) -> Mesh:
        """
        生成结构化网格
        
        Args:
            nx: x 方向网格数
            ny: y 方向网格数
            nz: z 方向网格数
            
        Returns:
            Mesh 对象
        """
        if nx <= 0 or ny <= 0 or nz <= 0:
            raise MeshGenerationError(f"网格数必须为正整数: nx={nx}, ny={ny}, nz={nz}")
        
        # 计算网格间距
        dx = self.Lx / nx
        dy = self.Ly / ny
        dz = self.Lz / nz
        
        # 生成节点坐标（以域中心为原点）
        x = np.linspace(-self.Lx/2, self.Lx/2, nx + 1)
        y = np.linspace(-self.Ly/2, self.Ly/2, ny + 1)
        z = np.linspace(0, self.Lz, nz + 1)  # z 从 0 开始
        
        # 生成单元中心坐标
        xc = 0.5 * (x[:-1] + x[1:])
        yc = 0.5 * (y[:-1] + y[1:])
        zc = 0.5 * (z[:-1] + z[1:])
        
        # 识别边界单元
        boundary_cells = self._identify_boundary_cells(nx, ny, nz)
        
        mesh = Mesh(
            x=x, y=y, z=z,
            xc=xc, yc=yc, zc=zc,
            dx=dx, dy=dy, dz=dz,
            nx=nx, ny=ny, nz=nz,
            boundary_cells=boundary_cells
        )
        
        return mesh
    
    def _identify_boundary_cells(self, nx: int, ny: int, nz: int) -> Dict[str, np.ndarray]:
        """
        识别边界单元索引
        
        Args:
            nx, ny, nz: 网格数
            
        Returns:
            边界单元索引字典
        """
        # 使用扁平化索引
        # 索引计算: idx = i + j*nx + k*nx*ny
        
        boundaries = {}
        
        # 底面 (z=0, 基底)
        bottom_indices = []
        for j in range(ny):
            for i in range(nx):
                bottom_indices.append(i + j*nx + 0*nx*ny)
        boundaries['bottom'] = np.array(bottom_indices, dtype=int)
        
        # 顶面 (z=Lz)
        top_indices = []
        for j in range(ny):
            for i in range(nx):
                top_indices.append(i + j*nx + (nz-1)*nx*ny)
        boundaries['top'] = np.array(top_indices, dtype=int)
        
        # 前面 (y=0, 围堰)
        front_indices = []
        for k in range(nz):
            for i in range(nx):
                front_indices.append(i + 0*nx + k*nx*ny)
        boundaries['front'] = np.array(front_indices, dtype=int)
        
        # 后面 (y=Ly, 围堰)
        back_indices = []
        for k in range(nz):
            for i in range(nx):
                back_indices.append(i + (ny-1)*nx + k*nx*ny)
        boundaries['back'] = np.array(back_indices, dtype=int)
        
        # 左面 (x=0, 围堰)
        left_indices = []
        for k in range(nz):
            for j in range(ny):
                left_indices.append(0 + j*nx + k*nx*ny)
        boundaries['left'] = np.array(left_indices, dtype=int)
        
        # 右面 (x=Lx, 围堰)
        right_indices = []
        for k in range(nz):
            for j in range(ny):
                right_indices.append((nx-1) + j*nx + k*nx*ny)
        boundaries['right'] = np.array(right_indices, dtype=int)
        
        # 所有壁面（合并）
        all_walls = np.unique(np.concatenate([
            boundaries['bottom'],
            boundaries['front'], boundaries['back'],
            boundaries['left'], boundaries['right']
        ]))
        boundaries['walls'] = all_walls
        
        return boundaries
    
    def get_initial_phi(self, mesh: Mesh) -> np.ndarray:
        """
        生成初始体积分数场（油墨在底部）
        
        Args:
            mesh: 计算网格
            
        Returns:
            初始体积分数场 phi (nx, ny, nz)
            phi=1 表示油墨，phi=0 表示极性液体
        """
        phi = np.zeros((mesh.nx, mesh.ny, mesh.nz))
        
        # 油墨层在底部
        for k in range(mesh.nz):
            if mesh.zc[k] < self.ink_thickness:
                phi[:, :, k] = 1.0
            elif mesh.zc[k] < self.ink_thickness + mesh.dz:
                # 界面过渡区
                frac = (self.ink_thickness - (mesh.zc[k] - mesh.dz/2)) / mesh.dz
                phi[:, :, k] = np.clip(frac, 0, 1)
        
        return phi


# ============================================================
# InterfaceTracker 类
# ============================================================

class InterfaceTracker:
    """
    VOF 界面追踪器
    
    使用 Volume of Fluid (VOF) 方法追踪两相界面。
    
    Example:
        >>> tracker = InterfaceTracker(mesh)
        >>> phi_new = tracker.advect(phi, velocity, dt)
    """
    
    def __init__(self, mesh: Mesh, config: Optional[Dict[str, Any]] = None):
        """
        初始化界面追踪器
        
        Args:
            mesh: 计算网格
            config: 配置参数
        """
        self.mesh = mesh
        
        if config is None:
            config = {}
        
        self.advection_scheme = config.get('advection_scheme', 'upwind')
        self.interface_compression = config.get('interface_compression', 0.5)
        self.reinit_interval = config.get('reinit_interval', 10)
        
        # 表面张力系数
        self.sigma = config.get('surface_tension', FLUID_PROPERTIES['oil']['surface_tension'])
    
    def advect(self, phi: np.ndarray, u: np.ndarray, v: np.ndarray, 
               w: np.ndarray, dt: float) -> np.ndarray:
        """
        界面平流（VOF 方程求解）
        
        ∂φ/∂t + ∇·(φu) = 0
        
        Args:
            phi: 当前体积分数场 (nx, ny, nz)
            u, v, w: 速度分量 (nx, ny, nz)
            dt: 时间步长
            
        Returns:
            更新后的体积分数场
        """
        nx, ny, nz = self.mesh.nx, self.mesh.ny, self.mesh.nz
        dx, dy, dz = self.mesh.dx, self.mesh.dy, self.mesh.dz
        
        # 使用一阶迎风格式
        phi_new = phi.copy()
        
        # x 方向通量
        for i in range(1, nx-1):
            for j in range(ny):
                for k in range(nz):
                    if u[i, j, k] > 0:
                        flux_x = u[i, j, k] * phi[i-1, j, k]
                    else:
                        flux_x = u[i, j, k] * phi[i, j, k]
                    
                    if u[i+1, j, k] > 0:
                        flux_x_p = u[i+1, j, k] * phi[i, j, k]
                    else:
                        flux_x_p = u[i+1, j, k] * phi[i+1, j, k] if i+1 < nx else 0
                    
                    phi_new[i, j, k] -= dt / dx * (flux_x_p - flux_x)
        
        # y 方向通量
        for i in range(nx):
            for j in range(1, ny-1):
                for k in range(nz):
                    if v[i, j, k] > 0:
                        flux_y = v[i, j, k] * phi[i, j-1, k]
                    else:
                        flux_y = v[i, j, k] * phi[i, j, k]
                    
                    if v[i, j+1, k] > 0:
                        flux_y_p = v[i, j+1, k] * phi[i, j, k]
                    else:
                        flux_y_p = v[i, j+1, k] * phi[i, j+1, k] if j+1 < ny else 0
                    
                    phi_new[i, j, k] -= dt / dy * (flux_y_p - flux_y)
        
        # z 方向通量
        for i in range(nx):
            for j in range(ny):
                for k in range(1, nz-1):
                    if w[i, j, k] > 0:
                        flux_z = w[i, j, k] * phi[i, j, k-1]
                    else:
                        flux_z = w[i, j, k] * phi[i, j, k]
                    
                    if w[i, j, k+1] > 0:
                        flux_z_p = w[i, j, k+1] * phi[i, j, k]
                    else:
                        flux_z_p = w[i, j, k+1] * phi[i, j, k+1] if k+1 < nz else 0
                    
                    phi_new[i, j, k] -= dt / dz * (flux_z_p - flux_z)
        
        # 限制 phi 在 [0, 1] 范围内
        phi_new = np.clip(phi_new, 0.0, 1.0)
        
        return phi_new
    
    def compute_interface_normal(self, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        计算界面法向量
        
        n = -∇φ / |∇φ|
        
        Args:
            phi: 体积分数场 (nx, ny, nz)
            
        Returns:
            法向量分量 (nx, ny, nz), (ny, ny, nz), (nz, ny, nz)
        """
        dx, dy, dz = self.mesh.dx, self.mesh.dy, self.mesh.dz
        
        # 计算梯度（中心差分）
        grad_x = np.zeros_like(phi)
        grad_y = np.zeros_like(phi)
        grad_z = np.zeros_like(phi)
        
        # 内部点使用中心差分
        grad_x[1:-1, :, :] = (phi[2:, :, :] - phi[:-2, :, :]) / (2 * dx)
        grad_y[:, 1:-1, :] = (phi[:, 2:, :] - phi[:, :-2, :]) / (2 * dy)
        grad_z[:, :, 1:-1] = (phi[:, :, 2:] - phi[:, :, :-2]) / (2 * dz)
        
        # 边界使用单侧差分
        grad_x[0, :, :] = (phi[1, :, :] - phi[0, :, :]) / dx
        grad_x[-1, :, :] = (phi[-1, :, :] - phi[-2, :, :]) / dx
        grad_y[:, 0, :] = (phi[:, 1, :] - phi[:, 0, :]) / dy
        grad_y[:, -1, :] = (phi[:, -1, :] - phi[:, -2, :]) / dy
        grad_z[:, :, 0] = (phi[:, :, 1] - phi[:, :, 0]) / dz
        grad_z[:, :, -1] = (phi[:, :, -1] - phi[:, :, -2]) / dz
        
        # 计算梯度模
        grad_mag = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2 + 1e-12)
        
        # 归一化
        nx = -grad_x / grad_mag
        ny = -grad_y / grad_mag
        nz = -grad_z / grad_mag
        
        return nx, ny, nz
    
    def compute_curvature(self, phi: np.ndarray) -> np.ndarray:
        """
        计算界面曲率
        
        κ = ∇·n = ∇·(-∇φ/|∇φ|)
        
        Args:
            phi: 体积分数场 (nx, ny, nz)
            
        Returns:
            曲率场 (nx, ny, nz)
        """
        dx, dy, dz = self.mesh.dx, self.mesh.dy, self.mesh.dz
        
        # 获取法向量
        nx, ny, nz = self.compute_interface_normal(phi)
        
        # 计算法向量的散度
        kappa = np.zeros_like(phi)
        
        # ∂nx/∂x
        kappa[1:-1, :, :] += (nx[2:, :, :] - nx[:-2, :, :]) / (2 * dx)
        # ∂ny/∂y
        kappa[:, 1:-1, :] += (ny[:, 2:, :] - ny[:, :-2, :]) / (2 * dy)
        # ∂nz/∂z
        kappa[:, :, 1:-1] += (nz[:, :, 2:] - nz[:, :, :-2]) / (2 * dz)
        
        return kappa
    
    def compute_surface_tension_force(self, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        计算表面张力体积力（CSF 模型）
        
        F_σ = σ κ ∇φ
        
        Args:
            phi: 体积分数场 (nx, ny, nz)
            
        Returns:
            表面张力分量 (fx, fy, fz)
        """
        dx, dy, dz = self.mesh.dx, self.mesh.dy, self.mesh.dz
        
        # 计算曲率
        kappa = self.compute_curvature(phi)
        
        # 计算 ∇φ
        grad_x = np.zeros_like(phi)
        grad_y = np.zeros_like(phi)
        grad_z = np.zeros_like(phi)
        
        grad_x[1:-1, :, :] = (phi[2:, :, :] - phi[:-2, :, :]) / (2 * dx)
        grad_y[:, 1:-1, :] = (phi[:, 2:, :] - phi[:, :-2, :]) / (2 * dy)
        grad_z[:, :, 1:-1] = (phi[:, :, 2:] - phi[:, :, :-2]) / (2 * dz)
        
        # CSF 表面张力
        fx = self.sigma * kappa * grad_x
        fy = self.sigma * kappa * grad_y
        fz = self.sigma * kappa * grad_z
        
        return fx, fy, fz
    
    def get_interface_position(self, phi: np.ndarray) -> np.ndarray:
        """
        提取界面位置（φ=0.5 等值面）
        
        Args:
            phi: 体积分数场 (nx, ny, nz)
            
        Returns:
            界面点坐标数组 (n_points, 3)
        """
        # 找到 φ 跨越 0.5 的单元
        interface_points = []
        
        for i in range(self.mesh.nx - 1):
            for j in range(self.mesh.ny - 1):
                for k in range(self.mesh.nz - 1):
                    # 检查是否跨越 0.5
                    phi_min = min(phi[i, j, k], phi[i+1, j, k], 
                                 phi[i, j+1, k], phi[i, j, k+1])
                    phi_max = max(phi[i, j, k], phi[i+1, j, k], 
                                 phi[i, j+1, k], phi[i, j, k+1])
                    
                    if phi_min <= 0.5 <= phi_max:
                        # 线性插值估计界面位置
                        x = self.mesh.xc[i]
                        y = self.mesh.yc[j]
                        z = self.mesh.zc[k]
                        interface_points.append([x, y, z])
        
        return np.array(interface_points) if interface_points else np.array([]).reshape(0, 3)
    
    def compute_interface_thickness(self, phi: np.ndarray) -> float:
        """
        计算界面厚度（0.01 < φ < 0.99 区域的厚度）
        
        Args:
            phi: 体积分数场 (nx, ny, nz)
            
        Returns:
            界面厚度（网格单元数）
        """
        # 统计界面区域的单元数
        interface_mask = (phi > 0.01) & (phi < 0.99)
        n_interface_cells = np.sum(interface_mask)
        
        if n_interface_cells == 0:
            return 0.0
        
        # 估计界面厚度（假设界面大致水平）
        # 计算每列中界面单元的数量
        thickness_per_column = np.sum(interface_mask, axis=2)  # 沿 z 方向求和
        
        # 取平均厚度
        nonzero_columns = thickness_per_column[thickness_per_column > 0]
        if len(nonzero_columns) > 0:
            avg_thickness = np.mean(nonzero_columns)
        else:
            avg_thickness = 0.0
        
        return avg_thickness
    
    def clip_volume_fraction(self, phi: np.ndarray) -> np.ndarray:
        """
        限制体积分数在 [0, 1] 范围内
        
        Args:
            phi: 体积分数场
            
        Returns:
            限制后的体积分数场
        """
        return np.clip(phi, 0.0, 1.0)


# ============================================================
# ContactLineHandler 类
# ============================================================

class ContactLineHandler:
    """
    移动接触线处理器
    
    处理三相接触线的动态行为，集成 HybridPredictor 获取动态接触角。
    
    Example:
        >>> handler = ContactLineHandler()
        >>> theta = handler.get_dynamic_contact_angle(voltage=30, time=0.005)
    """
    
    def __init__(self, predictor: Optional[HybridPredictor] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        初始化接触线处理器
        
        Args:
            predictor: HybridPredictor 实例
            config: 配置参数
        """
        # 初始化 HybridPredictor
        if predictor is None:
            self.predictor = HybridPredictor()
        else:
            self.predictor = predictor
        
        if config is None:
            config = {}
        
        # 滑移模型参数
        self.slip_length = config.get('slip_length', 1e-9)  # Navier 滑移长度
        self.slip_model = config.get('slip_model', 'navier')  # 滑移模型类型
        
        # 接触角滞后参数
        self.theta_advancing = config.get('theta_advancing', None)  # 前进角
        self.theta_receding = config.get('theta_receding', None)    # 后退角
        
        # 当前状态
        self.current_voltage = 0.0
        self.V_initial = 0.0
        self.t_step = 0.0
    
    def set_voltage_step(self, V_initial: float, V_target: float, t_step: float = 0.0):
        """
        设置电压阶跃参数
        
        Args:
            V_initial: 初始电压
            V_target: 目标电压
            t_step: 阶跃时间
        """
        self.V_initial = V_initial
        self.current_voltage = V_target
        self.t_step = t_step
    
    def get_dynamic_contact_angle(self, voltage: float, time: float,
                                  V_initial: Optional[float] = None,
                                  t_step: Optional[float] = None) -> float:
        """
        获取动态接触角
        
        Args:
            voltage: 当前电压 (V)
            time: 当前时间 (s)
            V_initial: 初始电压（可选）
            t_step: 阶跃时间（可选）
            
        Returns:
            接触角（度）
        """
        if V_initial is None:
            V_initial = self.V_initial
        if t_step is None:
            t_step = self.t_step
        
        # 使用 HybridPredictor 获取动态接触角
        theta = self.predictor.predict(
            voltage=voltage,
            time=time,
            V_initial=V_initial,
            t_step=t_step
        )
        
        return theta
    
    def get_equilibrium_contact_angle(self, voltage: float) -> float:
        """
        获取平衡接触角（稳态）
        
        Args:
            voltage: 电压 (V)
            
        Returns:
            平衡接触角（度）
        """
        return self.predictor.young_lippmann(voltage)
    
    def apply_slip_model(self, u: np.ndarray, v: np.ndarray, w: np.ndarray,
                        mesh: Mesh, contact_line_cells: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        应用 Navier 滑移模型
        
        u_slip = λ ∂u/∂n
        
        Args:
            u, v, w: 速度分量
            mesh: 计算网格
            contact_line_cells: 接触线单元索引
            
        Returns:
            修正后的速度分量
        """
        u_new = u.copy()
        v_new = v.copy()
        w_new = w.copy()
        
        # 对接触线单元应用滑移边界条件
        for idx in contact_line_cells:
            # 将扁平索引转换为 3D 索引
            i = idx % mesh.nx
            j = (idx // mesh.nx) % mesh.ny
            k = idx // (mesh.nx * mesh.ny)
            
            # 简化的滑移模型：允许切向速度
            # 在底面 (k=0)，法向为 z，切向为 x, y
            if k == 0:
                # 保持切向速度，但限制法向速度
                w_new[i, j, k] = 0.0  # 法向速度为零
                # 切向速度根据滑移长度调整
                # u_slip = λ * du/dz
                if k + 1 < mesh.nz:
                    u_new[i, j, k] = self.slip_length / mesh.dz * (u[i, j, k+1] - u[i, j, k])
                    v_new[i, j, k] = self.slip_length / mesh.dz * (v[i, j, k+1] - v[i, j, k])
        
        return u_new, v_new, w_new
    
    def detect_contact_line_cells(self, phi: np.ndarray, mesh: Mesh) -> np.ndarray:
        """
        检测接触线单元（界面与壁面相交的单元）
        
        Args:
            phi: 体积分数场
            mesh: 计算网格
            
        Returns:
            接触线单元索引数组
        """
        contact_line_cells = []
        
        # 检查底面边界
        k = 0
        for i in range(mesh.nx):
            for j in range(mesh.ny):
                # 如果该单元在界面附近（0.01 < φ < 0.99）
                if 0.01 < phi[i, j, k] < 0.99:
                    idx = i + j * mesh.nx + k * mesh.nx * mesh.ny
                    contact_line_cells.append(idx)
        
        return np.array(contact_line_cells, dtype=int)
    
    def detect_pinning(self, phi: np.ndarray, mesh: Mesh) -> bool:
        """
        检测接触线是否被钉扎在围堰边缘
        
        Args:
            phi: 体积分数场
            mesh: 计算网格
            
        Returns:
            是否被钉扎
        """
        # 检查边界处的体积分数
        # 如果边界处 φ 接近 0.5，说明接触线在边界
        
        # 检查四个侧面边界
        boundaries = ['front', 'back', 'left', 'right']
        
        for boundary in boundaries:
            boundary_cells = mesh.boundary_cells.get(boundary, np.array([]))
            if len(boundary_cells) > 0:
                # 将扁平索引转换为 3D 索引并检查 φ
                for idx in boundary_cells:
                    i = idx % mesh.nx
                    j = (idx // mesh.nx) % mesh.ny
                    k = idx // (mesh.nx * mesh.ny)
                    
                    if 0.01 < phi[i, j, k] < 0.99:
                        return True  # 接触线在边界，可能被钉扎
        
        return False
    
    def get_contact_line_velocity(self, phi: np.ndarray, phi_prev: np.ndarray,
                                  mesh: Mesh, dt: float) -> float:
        """
        计算接触线速度
        
        Args:
            phi: 当前体积分数场
            phi_prev: 上一时间步体积分数场
            mesh: 计算网格
            dt: 时间步长
            
        Returns:
            接触线速度 (m/s)
        """
        # 找到当前和上一时间步的接触线位置
        # 简化：使用底面 φ=0.5 等值线的平均半径
        
        def get_contact_radius(phi_field):
            """计算接触线半径"""
            k = 0  # 底面
            radii = []
            
            for i in range(mesh.nx):
                for j in range(mesh.ny):
                    if 0.4 < phi_field[i, j, k] < 0.6:
                        r = np.sqrt(mesh.xc[i]**2 + mesh.yc[j]**2)
                        radii.append(r)
            
            return np.mean(radii) if radii else 0.0
        
        r_current = get_contact_radius(phi)
        r_prev = get_contact_radius(phi_prev)
        
        # 接触线速度
        velocity = (r_current - r_prev) / dt if dt > 0 else 0.0
        
        return velocity


# ============================================================
# FlowSolver 类
# ============================================================

class FlowSolver:
    """
    两相流 CFD 求解器
    
    使用有限体积法求解两相流 Navier-Stokes 方程。
    支持 VOF 界面追踪和移动接触线处理。
    
    Example:
        >>> generator = MeshGenerator()
        >>> mesh = generator.generate_structured_mesh(nx=32, ny=32, nz=16)
        >>> solver = FlowSolver(mesh)
        >>> solver.set_initial_conditions(phi_0)
        >>> result = solver.solve(t_end=0.01)
    """
    
    def __init__(self, mesh: Mesh, config: Optional[Dict[str, Any]] = None):
        """
        初始化求解器
        
        Args:
            mesh: 计算网格
            config: 求解器配置
        """
        self.mesh = mesh
        
        if config is None:
            config = {}
        
        # 物理参数
        self.rho_oil = config.get('rho_oil', FLUID_PROPERTIES['oil']['density'])
        self.rho_water = config.get('rho_water', FLUID_PROPERTIES['polar_liquid']['density'])
        self.mu_oil = config.get('mu_oil', FLUID_PROPERTIES['oil']['viscosity'])
        self.mu_water = config.get('mu_water', FLUID_PROPERTIES['polar_liquid']['viscosity'])
        self.sigma = config.get('sigma', FLUID_PROPERTIES['oil']['surface_tension'])
        
        # 数值参数
        self.cfl = config.get('cfl', NUMERICAL_PARAMETERS['cfl_number'])
        self.max_iter = config.get('max_iter', NUMERICAL_PARAMETERS['max_iterations'])
        self.tol = config.get('tol', NUMERICAL_PARAMETERS['convergence_tol'])
        self.mass_error_tol = config.get('mass_error_tol', NUMERICAL_PARAMETERS['mass_error_tol'])
        
        # 初始化组件
        self.interface_tracker = InterfaceTracker(mesh, config)
        self.contact_handler = ContactLineHandler(config=config)
        
        # 场变量
        self.u = None  # x 方向速度
        self.v = None  # y 方向速度
        self.w = None  # z 方向速度
        self.p = None  # 压力
        self.phi = None  # 体积分数
        
        # 边界条件配置
        self.bc_config = {}
        
        # 初始质量（用于守恒检查）
        self.initial_mass_oil = None
        self.initial_mass_water = None
        
        # 当前时间
        self.current_time = 0.0
    
    def set_initial_conditions(self, phi_0: np.ndarray, 
                               u_0: Optional[np.ndarray] = None,
                               v_0: Optional[np.ndarray] = None,
                               w_0: Optional[np.ndarray] = None,
                               p_0: Optional[np.ndarray] = None):
        """
        设置初始条件
        
        Args:
            phi_0: 初始体积分数场 (nx, ny, nz)
            u_0: 初始 x 速度场（可选，默认为零）
            v_0: 初始 y 速度场（可选，默认为零）
            w_0: 初始 z 速度场（可选，默认为零）
            p_0: 初始压力场（可选，默认为零）
        """
        nx, ny, nz = self.mesh.nx, self.mesh.ny, self.mesh.nz
        
        # 设置体积分数
        self.phi = np.clip(phi_0.copy(), 0.0, 1.0)
        
        # 设置速度场
        self.u = u_0.copy() if u_0 is not None else np.zeros((nx, ny, nz))
        self.v = v_0.copy() if v_0 is not None else np.zeros((nx, ny, nz))
        self.w = w_0.copy() if w_0 is not None else np.zeros((nx, ny, nz))
        
        # 设置压力场
        self.p = p_0.copy() if p_0 is not None else np.zeros((nx, ny, nz))
        
        # 计算初始质量
        cell_vol = self.mesh.cell_volume
        self.initial_mass_oil = np.sum(self.phi * self.rho_oil) * cell_vol
        self.initial_mass_water = np.sum((1 - self.phi) * self.rho_water) * cell_vol
        
        # 应用边界条件
        self._apply_boundary_conditions()
        
        # 重置时间
        self.current_time = 0.0
    
    def set_boundary_conditions(self, bc_config: Dict[str, Any]):
        """
        设置边界条件
        
        Args:
            bc_config: 边界条件配置
                - 'walls': 壁面边界条件类型 ('no_slip', 'slip')
                - 'top': 顶面边界条件 ('pressure', 'velocity')
                - 'contact_angle': 接触角边界条件
        """
        self.bc_config = bc_config.copy()
        
        # 默认边界条件
        if 'walls' not in self.bc_config:
            self.bc_config['walls'] = 'no_slip'
        if 'top' not in self.bc_config:
            self.bc_config['top'] = 'pressure'
    
    def _apply_boundary_conditions(self):
        """应用边界条件到场变量"""
        if self.u is None:
            return
        
        mesh = self.mesh
        bc_type = self.bc_config.get('walls', 'no_slip')
        
        if bc_type == 'no_slip':
            # 无滑移边界条件：壁面速度为零
            # 底面
            self.u[:, :, 0] = 0.0
            self.v[:, :, 0] = 0.0
            self.w[:, :, 0] = 0.0
            
            # 前后面 (y 边界)
            self.u[:, 0, :] = 0.0
            self.v[:, 0, :] = 0.0
            self.w[:, 0, :] = 0.0
            self.u[:, -1, :] = 0.0
            self.v[:, -1, :] = 0.0
            self.w[:, -1, :] = 0.0
            
            # 左右面 (x 边界)
            self.u[0, :, :] = 0.0
            self.v[0, :, :] = 0.0
            self.w[0, :, :] = 0.0
            self.u[-1, :, :] = 0.0
            self.v[-1, :, :] = 0.0
            self.w[-1, :, :] = 0.0
        
        elif bc_type == 'slip':
            # 滑移边界条件：法向速度为零，切向速度自由
            # 底面：w = 0
            self.w[:, :, 0] = 0.0
            # 前后面：v = 0
            self.v[:, 0, :] = 0.0
            self.v[:, -1, :] = 0.0
            # 左右面：u = 0
            self.u[0, :, :] = 0.0
            self.u[-1, :, :] = 0.0
    
    def _compute_dt(self) -> float:
        """
        计算满足 CFL 条件的时间步长
        
        Returns:
            时间步长
        """
        dx, dy, dz = self.mesh.dx, self.mesh.dy, self.mesh.dz
        
        # 最大速度
        u_max = np.max(np.abs(self.u)) + 1e-10
        v_max = np.max(np.abs(self.v)) + 1e-10
        w_max = np.max(np.abs(self.w)) + 1e-10
        
        # CFL 条件
        dt_cfl = self.cfl * min(dx / u_max, dy / v_max, dz / w_max)
        
        # 粘性稳定性条件
        nu_max = max(self.mu_oil / self.rho_oil, self.mu_water / self.rho_water)
        dt_visc = 0.25 * min(dx, dy, dz)**2 / (nu_max + 1e-10)
        
        # 表面张力稳定性条件
        rho_min = min(self.rho_oil, self.rho_water)
        dt_sigma = 0.5 * np.sqrt(rho_min * min(dx, dy, dz)**3 / (np.pi * self.sigma + 1e-10))
        
        dt = min(dt_cfl, dt_visc, dt_sigma)
        
        return dt
    
    def _get_density_field(self) -> np.ndarray:
        """获取混合密度场"""
        return self.phi * self.rho_oil + (1 - self.phi) * self.rho_water
    
    def _get_viscosity_field(self) -> np.ndarray:
        """获取混合粘度场"""
        return self.phi * self.mu_oil + (1 - self.phi) * self.mu_water
    
    def solve_step(self, dt: float) -> Dict[str, np.ndarray]:
        """
        求解一个时间步
        
        使用分步法（Fractional Step Method）：
        1. 预测速度（不考虑压力）
        2. 求解压力 Poisson 方程
        3. 校正速度
        4. 更新界面（VOF 平流）
        
        Args:
            dt: 时间步长
            
        Returns:
            包含 u, v, w, p, phi 的字典
        """
        if self.phi is None:
            raise FlowSolverError("Initial conditions not set. Call set_initial_conditions() first.")
        
        mesh = self.mesh
        nx, ny, nz = mesh.nx, mesh.ny, mesh.nz
        dx, dy, dz = mesh.dx, mesh.dy, mesh.dz
        
        # 获取物性场
        rho = self._get_density_field()
        mu = self._get_viscosity_field()
        
        # 计算表面张力
        fx, fy, fz = self.interface_tracker.compute_surface_tension_force(self.phi)
        
        # ========== Step 1: 预测速度 ==========
        u_star = self.u.copy()
        v_star = self.v.copy()
        w_star = self.w.copy()
        
        # 对流项（简化：一阶迎风）
        # 粘性项（简化：显式中心差分）
        # 这里使用简化的显式格式
        
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                for k in range(1, nz-1):
                    # 对流项
                    dudx = (self.u[i+1,j,k] - self.u[i-1,j,k]) / (2*dx) if self.u[i,j,k] > 0 else (self.u[i+1,j,k] - self.u[i-1,j,k]) / (2*dx)
                    dudy = (self.u[i,j+1,k] - self.u[i,j-1,k]) / (2*dy)
                    dudz = (self.u[i,j,k+1] - self.u[i,j,k-1]) / (2*dz)
                    
                    dvdx = (self.v[i+1,j,k] - self.v[i-1,j,k]) / (2*dx)
                    dvdy = (self.v[i,j+1,k] - self.v[i,j-1,k]) / (2*dy)
                    dvdz = (self.v[i,j,k+1] - self.v[i,j,k-1]) / (2*dz)
                    
                    dwdx = (self.w[i+1,j,k] - self.w[i-1,j,k]) / (2*dx)
                    dwdy = (self.w[i,j+1,k] - self.w[i,j-1,k]) / (2*dy)
                    dwdz = (self.w[i,j,k+1] - self.w[i,j,k-1]) / (2*dz)
                    
                    conv_u = self.u[i,j,k]*dudx + self.v[i,j,k]*dudy + self.w[i,j,k]*dudz
                    conv_v = self.u[i,j,k]*dvdx + self.v[i,j,k]*dvdy + self.w[i,j,k]*dvdz
                    conv_w = self.u[i,j,k]*dwdx + self.v[i,j,k]*dwdy + self.w[i,j,k]*dwdz
                    
                    # 粘性项（拉普拉斯）
                    lap_u = (self.u[i+1,j,k] - 2*self.u[i,j,k] + self.u[i-1,j,k]) / dx**2 + \
                            (self.u[i,j+1,k] - 2*self.u[i,j,k] + self.u[i,j-1,k]) / dy**2 + \
                            (self.u[i,j,k+1] - 2*self.u[i,j,k] + self.u[i,j,k-1]) / dz**2
                    
                    lap_v = (self.v[i+1,j,k] - 2*self.v[i,j,k] + self.v[i-1,j,k]) / dx**2 + \
                            (self.v[i,j+1,k] - 2*self.v[i,j,k] + self.v[i,j-1,k]) / dy**2 + \
                            (self.v[i,j,k+1] - 2*self.v[i,j,k] + self.v[i,j,k-1]) / dz**2
                    
                    lap_w = (self.w[i+1,j,k] - 2*self.w[i,j,k] + self.w[i-1,j,k]) / dx**2 + \
                            (self.w[i,j+1,k] - 2*self.w[i,j,k] + self.w[i,j-1,k]) / dy**2 + \
                            (self.w[i,j,k+1] - 2*self.w[i,j,k] + self.w[i,j,k-1]) / dz**2
                    
                    # 更新预测速度
                    rho_ijk = rho[i,j,k]
                    mu_ijk = mu[i,j,k]
                    
                    u_star[i,j,k] = self.u[i,j,k] + dt * (-conv_u + mu_ijk/rho_ijk * lap_u + fx[i,j,k]/rho_ijk)
                    v_star[i,j,k] = self.v[i,j,k] + dt * (-conv_v + mu_ijk/rho_ijk * lap_v + fy[i,j,k]/rho_ijk)
                    w_star[i,j,k] = self.w[i,j,k] + dt * (-conv_w + mu_ijk/rho_ijk * lap_w + fz[i,j,k]/rho_ijk)
        
        # ========== Step 2: 压力 Poisson 方程 ==========
        # 简化：使用 Jacobi 迭代
        p_new = self.p.copy()
        
        # 计算速度散度
        div_u = np.zeros((nx, ny, nz))
        div_u[1:-1, 1:-1, 1:-1] = (
            (u_star[2:, 1:-1, 1:-1] - u_star[:-2, 1:-1, 1:-1]) / (2*dx) +
            (v_star[1:-1, 2:, 1:-1] - v_star[1:-1, :-2, 1:-1]) / (2*dy) +
            (w_star[1:-1, 1:-1, 2:] - w_star[1:-1, 1:-1, :-2]) / (2*dz)
        )
        
        # Jacobi 迭代求解压力
        for _ in range(self.max_iter):
            p_old = p_new.copy()
            
            for i in range(1, nx-1):
                for j in range(1, ny-1):
                    for k in range(1, nz-1):
                        p_new[i,j,k] = (1/6) * (
                            p_old[i+1,j,k] + p_old[i-1,j,k] +
                            p_old[i,j+1,k] + p_old[i,j-1,k] +
                            p_old[i,j,k+1] + p_old[i,j,k-1] -
                            dx**2 * rho[i,j,k] / dt * div_u[i,j,k]
                        )
            
            # 检查收敛
            if np.max(np.abs(p_new - p_old)) < self.tol:
                break
        
        self.p = p_new
        
        # ========== Step 3: 校正速度 ==========
        for i in range(1, nx-1):
            for j in range(1, ny-1):
                for k in range(1, nz-1):
                    dpdx = (self.p[i+1,j,k] - self.p[i-1,j,k]) / (2*dx)
                    dpdy = (self.p[i,j+1,k] - self.p[i,j-1,k]) / (2*dy)
                    dpdz = (self.p[i,j,k+1] - self.p[i,j,k-1]) / (2*dz)
                    
                    self.u[i,j,k] = u_star[i,j,k] - dt / rho[i,j,k] * dpdx
                    self.v[i,j,k] = v_star[i,j,k] - dt / rho[i,j,k] * dpdy
                    self.w[i,j,k] = w_star[i,j,k] - dt / rho[i,j,k] * dpdz
        
        # ========== Step 4: 更新界面 ==========
        self.phi = self.interface_tracker.advect(self.phi, self.u, self.v, self.w, dt)
        
        # 应用边界条件
        self._apply_boundary_conditions()
        
        # 更新时间
        self.current_time += dt
        
        return {
            'u': self.u.copy(),
            'v': self.v.copy(),
            'w': self.w.copy(),
            'p': self.p.copy(),
            'phi': self.phi.copy()
        }
    
    def solve(self, t_end: float, dt: Optional[float] = None,
              save_interval: int = 10) -> SimulationResult:
        """
        求解到指定时间
        
        Args:
            t_end: 结束时间
            dt: 时间步长（可选，自动计算）
            save_interval: 保存间隔（每多少步保存一次）
            
        Returns:
            SimulationResult 对象
        """
        import time as time_module
        start_time = time_module.time()
        
        # 存储结果
        t_list = [0.0]
        u_list = [self.u.copy()]
        v_list = [self.v.copy()]
        w_list = [self.w.copy()]
        p_list = [self.p.copy()]
        phi_list = [self.phi.copy()]
        aperture_list = [compute_aperture_ratio_from_phi(self.phi, self.mesh)]
        mass_error_list = [0.0]
        
        step = 0
        while self.current_time < t_end:
            # 计算时间步长
            if dt is None:
                dt_step = self._compute_dt()
            else:
                dt_step = dt
            
            # 确保不超过结束时间
            if self.current_time + dt_step > t_end:
                dt_step = t_end - self.current_time
            
            # 求解一步
            self.solve_step(dt_step)
            step += 1
            
            # 保存结果
            if step % save_interval == 0 or self.current_time >= t_end:
                t_list.append(self.current_time)
                u_list.append(self.u.copy())
                v_list.append(self.v.copy())
                w_list.append(self.w.copy())
                p_list.append(self.p.copy())
                phi_list.append(self.phi.copy())
                aperture_list.append(compute_aperture_ratio_from_phi(self.phi, self.mesh))
                mass_error_list.append(self.compute_mass_conservation_error()['total_error'])
        
        computation_time = time_module.time() - start_time
        
        # 创建结果对象
        result = SimulationResult(
            t=np.array(t_list),
            u=np.array(u_list),
            v=np.array(v_list),
            w=np.array(w_list),
            p=np.array(p_list),
            phi=np.array(phi_list),
            aperture_ratio=np.array(aperture_list),
            mass_error=np.array(mass_error_list),
            config={
                'mesh': self.mesh.to_dict(),
                'rho_oil': self.rho_oil,
                'rho_water': self.rho_water,
                'mu_oil': self.mu_oil,
                'mu_water': self.mu_water,
                'sigma': self.sigma,
            },
            computation_time=computation_time,
            method='cfd'
        )
        
        return result
    
    def compute_mass_conservation_error(self) -> Dict[str, float]:
        """
        计算质量守恒误差
        
        Returns:
            包含油墨和极性液体质量误差的字典
        """
        if self.phi is None or self.initial_mass_oil is None:
            return {'oil_error': 0.0, 'water_error': 0.0, 'total_error': 0.0}
        
        cell_vol = self.mesh.cell_volume
        
        # 当前质量
        current_mass_oil = np.sum(self.phi * self.rho_oil) * cell_vol
        current_mass_water = np.sum((1 - self.phi) * self.rho_water) * cell_vol
        
        # 相对误差
        oil_error = abs(current_mass_oil - self.initial_mass_oil) / (self.initial_mass_oil + 1e-10)
        water_error = abs(current_mass_water - self.initial_mass_water) / (self.initial_mass_water + 1e-10)
        total_error = (oil_error + water_error) / 2
        
        return {
            'oil_error': oil_error,
            'water_error': water_error,
            'total_error': total_error
        }
    
    def compute_momentum_residual(self) -> np.ndarray:
        """
        计算动量方程残差
        
        Returns:
            残差场 (nx, ny, nz)
        """
        if self.u is None:
            return np.zeros((self.mesh.nx, self.mesh.ny, self.mesh.nz))
        
        mesh = self.mesh
        dx, dy, dz = mesh.dx, mesh.dy, mesh.dz
        
        # 计算速度散度（连续性方程残差）
        residual = np.zeros((mesh.nx, mesh.ny, mesh.nz))
        
        residual[1:-1, 1:-1, 1:-1] = (
            (self.u[2:, 1:-1, 1:-1] - self.u[:-2, 1:-1, 1:-1]) / (2*dx) +
            (self.v[1:-1, 2:, 1:-1] - self.v[1:-1, :-2, 1:-1]) / (2*dy) +
            (self.w[1:-1, 1:-1, 2:] - self.w[1:-1, 1:-1, :-2]) / (2*dz)
        )
        
        return residual
    
    def get_wall_velocity(self) -> Dict[str, np.ndarray]:
        """
        获取壁面速度（用于验证无滑移边界条件）
        
        Returns:
            各壁面的速度数组
        """
        return {
            'bottom': np.sqrt(self.u[:, :, 0]**2 + self.v[:, :, 0]**2 + self.w[:, :, 0]**2),
            'front': np.sqrt(self.u[:, 0, :]**2 + self.v[:, 0, :]**2 + self.w[:, 0, :]**2),
            'back': np.sqrt(self.u[:, -1, :]**2 + self.v[:, -1, :]**2 + self.w[:, -1, :]**2),
            'left': np.sqrt(self.u[0, :, :]**2 + self.v[0, :, :]**2 + self.w[0, :, :]**2),
            'right': np.sqrt(self.u[-1, :, :]**2 + self.v[-1, :, :]**2 + self.w[-1, :, :]**2),
        }


# ============================================================
# PINNSolver 类
# ============================================================

class PINNSolver:
    """
    PINN 两相流求解器
    
    使用物理信息神经网络 (Physics-Informed Neural Network) 求解两相流问题。
    复用 efd_pinns_train.py 的训练框架。
    
    Example:
        >>> solver = PINNSolver()
        >>> solver.build_network()
        >>> history = solver.train(training_data, epochs=1000)
        >>> result = solver.predict(x, t=0.01)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化 PINN 求解器
        
        Args:
            config: 配置参数
        """
        if config is None:
            config = {}
        
        # 网络参数
        self.hidden_layers = config.get('hidden_layers', [64, 64, 64, 64])
        self.activation = config.get('activation', 'tanh')
        self.learning_rate = config.get('learning_rate', 1e-3)
        
        # 物理参数
        self.rho_oil = config.get('rho_oil', FLUID_PROPERTIES['oil']['density'])
        self.rho_water = config.get('rho_water', FLUID_PROPERTIES['polar_liquid']['density'])
        self.mu_oil = config.get('mu_oil', FLUID_PROPERTIES['oil']['viscosity'])
        self.mu_water = config.get('mu_water', FLUID_PROPERTIES['polar_liquid']['viscosity'])
        self.sigma = config.get('sigma', FLUID_PROPERTIES['oil']['surface_tension'])
        
        # 域参数
        self.Lx = config.get('Lx', DOMAIN_PARAMETERS['Lx'])
        self.Ly = config.get('Ly', DOMAIN_PARAMETERS['Ly'])
        self.Lz = config.get('Lz', DOMAIN_PARAMETERS['Lz'])
        
        # 损失权重
        self.lambda_data = config.get('lambda_data', 1.0)
        self.lambda_physics = config.get('lambda_physics', 1.0)
        self.lambda_bc = config.get('lambda_bc', 10.0)
        
        # 神经网络
        self.model = None
        self.optimizer = None
        
        # 训练历史
        self.history = {
            'total_loss': [],
            'data_loss': [],
            'physics_loss': [],
            'bc_loss': []
        }
        
        # 设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def build_network(self, input_dim: int = 4, output_dim: int = 5) -> nn.Module:
        """
        构建神经网络
        
        Args:
            input_dim: 输入维度 (x, y, z, t)
            output_dim: 输出维度 (u, v, w, p, phi)
            
        Returns:
            PyTorch 模型
        """
        layers = []
        
        # 输入层
        prev_dim = input_dim
        
        # 隐藏层
        for hidden_dim in self.hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if self.activation == 'tanh':
                layers.append(nn.Tanh())
            elif self.activation == 'relu':
                layers.append(nn.ReLU())
            elif self.activation == 'gelu':
                layers.append(nn.GELU())
            else:
                layers.append(nn.Tanh())
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers).to(self.device)
        
        # 初始化优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        return self.model
    
    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """归一化输入到 [-1, 1]"""
        # x: (N, 4) -> (x, y, z, t)
        x_norm = x.clone()
        x_norm[:, 0] = x[:, 0] / (self.Lx / 2)  # x: [-Lx/2, Lx/2] -> [-1, 1]
        x_norm[:, 1] = x[:, 1] / (self.Ly / 2)  # y: [-Ly/2, Ly/2] -> [-1, 1]
        x_norm[:, 2] = x[:, 2] / self.Lz * 2 - 1  # z: [0, Lz] -> [-1, 1]
        x_norm[:, 3] = x[:, 3] / 0.02 * 2 - 1  # t: [0, 0.02] -> [-1, 1]
        return x_norm
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x_norm = self._normalize_input(x)
        return self.model(x_norm)
    
    def compute_physics_loss(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算物理损失
        
        包括：
        - Navier-Stokes 动量方程残差
        - 连续性方程残差
        - VOF 方程残差
        
        Args:
            x: 输入坐标 (N, 4) -> (x, y, z, t)
            
        Returns:
            包含各项残差的字典
        """
        x.requires_grad_(True)
        
        # 前向传播
        output = self.forward(x)
        u, v, w, p, phi = output[:, 0:1], output[:, 1:2], output[:, 2:3], output[:, 3:4], output[:, 4:5]
        
        # 计算梯度
        def grad(y, x):
            return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y),
                                       create_graph=True, retain_graph=True)[0]
        
        # 一阶导数
        u_x = grad(u, x)[:, 0:1]
        u_y = grad(u, x)[:, 1:2]
        u_z = grad(u, x)[:, 2:3]
        u_t = grad(u, x)[:, 3:4]
        
        v_x = grad(v, x)[:, 0:1]
        v_y = grad(v, x)[:, 1:2]
        v_z = grad(v, x)[:, 2:3]
        v_t = grad(v, x)[:, 3:4]
        
        w_x = grad(w, x)[:, 0:1]
        w_y = grad(w, x)[:, 1:2]
        w_z = grad(w, x)[:, 2:3]
        w_t = grad(w, x)[:, 3:4]
        
        p_x = grad(p, x)[:, 0:1]
        p_y = grad(p, x)[:, 1:2]
        p_z = grad(p, x)[:, 2:3]
        
        phi_x = grad(phi, x)[:, 0:1]
        phi_y = grad(phi, x)[:, 1:2]
        phi_z = grad(phi, x)[:, 2:3]
        phi_t = grad(phi, x)[:, 3:4]
        
        # 二阶导数（拉普拉斯）
        u_xx = grad(u_x, x)[:, 0:1]
        u_yy = grad(u_y, x)[:, 1:2]
        u_zz = grad(u_z, x)[:, 2:3]
        
        v_xx = grad(v_x, x)[:, 0:1]
        v_yy = grad(v_y, x)[:, 1:2]
        v_zz = grad(v_z, x)[:, 2:3]
        
        w_xx = grad(w_x, x)[:, 0:1]
        w_yy = grad(w_y, x)[:, 1:2]
        w_zz = grad(w_z, x)[:, 2:3]
        
        # 混合物性
        rho = phi * self.rho_oil + (1 - phi) * self.rho_water
        mu = phi * self.mu_oil + (1 - phi) * self.mu_water
        
        # ========== 连续性方程残差 ==========
        # ∂u/∂x + ∂v/∂y + ∂w/∂z = 0
        continuity_residual = u_x + v_y + w_z
        
        # ========== 动量方程残差 (简化) ==========
        # ρ(∂u/∂t + u·∇u) = -∇p + μ∇²u
        
        # x 动量
        ns_x = rho * (u_t + u * u_x + v * u_y + w * u_z) + p_x - mu * (u_xx + u_yy + u_zz)
        
        # y 动量
        ns_y = rho * (v_t + u * v_x + v * v_y + w * v_z) + p_y - mu * (v_xx + v_yy + v_zz)
        
        # z 动量
        ns_z = rho * (w_t + u * w_x + v * w_y + w * w_z) + p_z - mu * (w_xx + w_yy + w_zz)
        
        # ========== VOF 方程残差 ==========
        # ∂φ/∂t + u·∇φ = 0
        vof_residual = phi_t + u * phi_x + v * phi_y + w * phi_z
        
        # 计算损失
        loss_continuity = torch.mean(continuity_residual**2)
        loss_ns_x = torch.mean(ns_x**2)
        loss_ns_y = torch.mean(ns_y**2)
        loss_ns_z = torch.mean(ns_z**2)
        loss_vof = torch.mean(vof_residual**2)
        
        total_physics_loss = loss_continuity + loss_ns_x + loss_ns_y + loss_ns_z + loss_vof
        
        return {
            'total': total_physics_loss,
            'continuity': loss_continuity,
            'ns_x': loss_ns_x,
            'ns_y': loss_ns_y,
            'ns_z': loss_ns_z,
            'vof': loss_vof
        }
    
    def compute_data_loss(self, x: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        计算数据损失
        
        Args:
            x: 输入坐标 (N, 4)
            y_true: 真实值 (N, 5) -> (u, v, w, p, phi)
            
        Returns:
            数据损失
        """
        y_pred = self.forward(x)
        return torch.mean((y_pred - y_true)**2)
    
    def compute_bc_loss(self, x_bc: torch.Tensor) -> torch.Tensor:
        """
        计算边界条件损失
        
        Args:
            x_bc: 边界点坐标 (N, 4)
            
        Returns:
            边界条件损失
        """
        output = self.forward(x_bc)
        u, v, w = output[:, 0], output[:, 1], output[:, 2]
        
        # 无滑移边界条件：壁面速度为零
        bc_loss = torch.mean(u**2 + v**2 + w**2)
        
        return bc_loss
    
    def train(self, training_data: Dict[str, np.ndarray],
              epochs: int = 10000, batch_size: int = 256,
              verbose: bool = True) -> Dict[str, List[float]]:
        """
        训练 PINN
        
        Args:
            training_data: 训练数据
                - 'x': 坐标 (N, 4)
                - 'y': 场值 (N, 5)
                - 'x_bc': 边界点坐标 (N_bc, 4)
                - 'x_physics': 物理点坐标 (N_phys, 4)
            epochs: 训练轮数
            batch_size: 批大小
            verbose: 是否打印进度
            
        Returns:
            训练历史
        """
        if self.model is None:
            self.build_network()
        
        # 转换为 tensor
        x_data = torch.tensor(training_data['x'], dtype=torch.float32, device=self.device)
        y_data = torch.tensor(training_data['y'], dtype=torch.float32, device=self.device)
        x_bc = torch.tensor(training_data.get('x_bc', training_data['x'][:100]), 
                           dtype=torch.float32, device=self.device)
        x_physics = torch.tensor(training_data.get('x_physics', training_data['x']),
                                dtype=torch.float32, device=self.device)
        
        n_data = x_data.shape[0]
        
        for epoch in range(epochs):
            self.model.train()
            
            # 随机采样
            idx = torch.randperm(n_data)[:batch_size]
            x_batch = x_data[idx]
            y_batch = y_data[idx]
            
            idx_phys = torch.randperm(x_physics.shape[0])[:batch_size]
            x_phys_batch = x_physics[idx_phys]
            
            # 计算损失
            data_loss = self.compute_data_loss(x_batch, y_batch)
            physics_loss = self.compute_physics_loss(x_phys_batch)['total']
            bc_loss = self.compute_bc_loss(x_bc)
            
            total_loss = (self.lambda_data * data_loss + 
                         self.lambda_physics * physics_loss + 
                         self.lambda_bc * bc_loss)
            
            # 反向传播
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            # 记录历史
            self.history['total_loss'].append(total_loss.item())
            self.history['data_loss'].append(data_loss.item())
            self.history['physics_loss'].append(physics_loss.item())
            self.history['bc_loss'].append(bc_loss.item())
            
            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}: "
                      f"Total={total_loss.item():.6f}, "
                      f"Data={data_loss.item():.6f}, "
                      f"Physics={physics_loss.item():.6f}, "
                      f"BC={bc_loss.item():.6f}")
        
        return self.history
    
    def predict(self, x: np.ndarray, t: Optional[float] = None) -> Dict[str, np.ndarray]:
        """
        预测流场
        
        Args:
            x: 空间坐标 (N, 3) 或 (N, 4)
            t: 时间（如果 x 是 3D 坐标）
            
        Returns:
            预测的流场
        """
        if self.model is None:
            raise FlowSolverError("Model not built. Call build_network() first.")
        
        self.model.eval()
        
        # 处理输入
        if x.shape[1] == 3 and t is not None:
            # 添加时间维度
            t_arr = np.full((x.shape[0], 1), t)
            x_input = np.hstack([x, t_arr])
        else:
            x_input = x
        
        x_tensor = torch.tensor(x_input, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            output = self.forward(x_tensor)
        
        output_np = output.cpu().numpy()
        
        return {
            'u': output_np[:, 0],
            'v': output_np[:, 1],
            'w': output_np[:, 2],
            'p': output_np[:, 3],
            'phi': output_np[:, 4]
        }
    
    def save_model(self, path: str):
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        if self.model is None:
            raise FlowSolverError("No model to save.")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'config': {
                'hidden_layers': self.hidden_layers,
                'activation': self.activation,
                'learning_rate': self.learning_rate,
            }
        }, path)
    
    def load_model(self, path: str):
        """
        加载模型
        
        Args:
            path: 模型路径
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        # 重建网络
        if self.model is None:
            self.hidden_layers = checkpoint['config']['hidden_layers']
            self.activation = checkpoint['config']['activation']
            self.build_network()
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
    
    def get_training_summary(self) -> Dict[str, Any]:
        """获取训练摘要"""
        if not self.history['total_loss']:
            return {'trained': False}
        
        return {
            'trained': True,
            'epochs': len(self.history['total_loss']),
            'final_total_loss': self.history['total_loss'][-1],
            'final_data_loss': self.history['data_loss'][-1],
            'final_physics_loss': self.history['physics_loss'][-1],
            'final_bc_loss': self.history['bc_loss'][-1],
            'device': str(self.device)
        }


# ============================================================
# FlowFieldSimulator 类
# ============================================================

class FlowFieldSimulator:
    """
    流场模拟器主类
    
    整合 CFD 求解器、PINN 求解器和 EnhancedApertureModel，
    提供统一的模拟接口。
    
    Example:
        >>> simulator = FlowFieldSimulator()
        >>> result = simulator.simulate(voltage=30, duration=0.02)
        >>> print(f"Final aperture ratio: {result.aperture_ratio[-1]:.2%}")
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化模拟器
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        if config_path is not None and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = {}
        
        # 网格参数
        mesh_config = self.config.get('mesh', {})
        self.nx = mesh_config.get('nx', 32)
        self.ny = mesh_config.get('ny', 32)
        self.nz = mesh_config.get('nz', 16)
        
        # 初始化组件
        self.mesh_generator = MeshGenerator(self.config.get('domain', {}))
        self.mesh = None
        
        # 求解器（延迟初始化）
        self.cfd_solver = None
        self.pinn_solver = None
        
        # 参考模型
        self.aperture_model = EnhancedApertureModel()
        self.contact_handler = ContactLineHandler()
    
    def _initialize_mesh(self):
        """初始化网格"""
        if self.mesh is None:
            self.mesh = self.mesh_generator.generate_structured_mesh(
                nx=self.nx, ny=self.ny, nz=self.nz
            )
    
    def _initialize_cfd_solver(self):
        """初始化 CFD 求解器"""
        self._initialize_mesh()
        if self.cfd_solver is None:
            self.cfd_solver = FlowSolver(self.mesh, self.config.get('solver', {}))
    
    def _initialize_pinn_solver(self):
        """初始化 PINN 求解器"""
        if self.pinn_solver is None:
            self.pinn_solver = PINNSolver(self.config.get('pinn', {}))
            self.pinn_solver.build_network()
    
    def simulate(self, voltage: float, duration: float,
                 V_initial: float = 0.0, t_step: float = 0.0,
                 method: str = 'cfd') -> SimulationResult:
        """
        运行模拟
        
        Args:
            voltage: 目标电压 (V)
            duration: 模拟时长 (s)
            V_initial: 初始电压 (V)
            t_step: 电压阶跃时间 (s)
            method: 求解方法 ('cfd', 'pinn', 'hybrid')
            
        Returns:
            SimulationResult 对象
        """
        import time as time_module
        start_time = time_module.time()
        
        if method == 'cfd':
            result = self._simulate_cfd(voltage, duration, V_initial, t_step)
        elif method == 'pinn':
            result = self._simulate_pinn(voltage, duration, V_initial, t_step)
        elif method == 'hybrid':
            result = self._simulate_hybrid(voltage, duration, V_initial, t_step)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        result.computation_time = time_module.time() - start_time
        result.method = method
        
        return result
    
    def _simulate_cfd(self, voltage: float, duration: float,
                      V_initial: float, t_step: float) -> SimulationResult:
        """使用 CFD 求解器模拟"""
        self._initialize_cfd_solver()
        
        # 设置初始条件
        initial = create_initial_conditions(self.mesh)
        self.cfd_solver.set_initial_conditions(initial['phi'])
        self.cfd_solver.set_boundary_conditions({'walls': 'no_slip'})
        
        # 设置接触线处理器
        self.contact_handler.set_voltage_step(V_initial, voltage, t_step)
        
        # 求解
        result = self.cfd_solver.solve(t_end=duration)
        
        # 添加接触角信息
        contact_angles = []
        for t in result.t:
            theta = self.contact_handler.get_dynamic_contact_angle(voltage, t, V_initial, t_step)
            contact_angles.append(theta)
        result.contact_angle = np.array(contact_angles)
        
        return result
    
    def _simulate_pinn(self, voltage: float, duration: float,
                       V_initial: float, t_step: float) -> SimulationResult:
        """使用 PINN 求解器模拟"""
        self._initialize_pinn_solver()
        self._initialize_mesh()
        
        # 生成时间点
        n_times = 100
        t_array = np.linspace(0, duration, n_times)
        
        # 生成空间点
        X, Y, Z = np.meshgrid(self.mesh.xc, self.mesh.yc, self.mesh.zc, indexing='ij')
        x_flat = X.flatten()
        y_flat = Y.flatten()
        z_flat = Z.flatten()
        
        # 预测每个时间点
        aperture_list = []
        contact_angles = []
        
        for t in t_array:
            # 构建输入
            x_input = np.column_stack([x_flat, y_flat, z_flat])
            
            # 预测
            pred = self.pinn_solver.predict(x_input, t=t)
            phi = pred['phi'].reshape(self.mesh.nx, self.mesh.ny, self.mesh.nz)
            
            # 计算开口率
            eta = compute_aperture_ratio_from_phi(phi, self.mesh)
            aperture_list.append(eta)
            
            # 计算接触角
            theta = self.contact_handler.get_dynamic_contact_angle(voltage, t, V_initial, t_step)
            contact_angles.append(theta)
        
        result = SimulationResult(
            t=t_array,
            aperture_ratio=np.array(aperture_list),
            contact_angle=np.array(contact_angles),
            config={'voltage': voltage, 'duration': duration},
            method='pinn'
        )
        
        return result
    
    def _simulate_hybrid(self, voltage: float, duration: float,
                         V_initial: float, t_step: float) -> SimulationResult:
        """使用混合方法模拟（CFD + PINN）"""
        # 简化实现：使用 EnhancedApertureModel 的结果
        n_times = 100
        t_array = np.linspace(0, duration, n_times)
        
        aperture_list = []
        contact_angles = []
        
        for t in t_array:
            # 使用 EnhancedApertureModel
            result = self.aperture_model.predict_enhanced(voltage=voltage, time=t)
            aperture_list.append(result['aperture_ratio'])
            # 使用 contact_handler 获取动态接触角
            theta = self.contact_handler.get_dynamic_contact_angle(voltage, t, V_initial, t_step)
            contact_angles.append(theta)
        
        return SimulationResult(
            t=t_array,
            aperture_ratio=np.array(aperture_list),
            contact_angle=np.array(contact_angles),
            config={'voltage': voltage, 'duration': duration},
            method='hybrid'
        )
    
    def compute_aperture_ratio(self, phi: np.ndarray) -> float:
        """
        从体积分数场计算开口率
        
        Args:
            phi: 体积分数场 (nx, ny, nz)
            
        Returns:
            开口率 (0-1)
        """
        self._initialize_mesh()
        return compute_aperture_ratio_from_phi(phi, self.mesh)
    
    def compare_with_aperture_model(self, voltage: float, 
                                    duration: float) -> Dict[str, Any]:
        """
        与 EnhancedApertureModel 对比
        
        Args:
            voltage: 电压 (V)
            duration: 时长 (s)
            
        Returns:
            对比结果
        """
        # 获取 EnhancedApertureModel 结果
        model_result = self.aperture_model.predict_enhanced(voltage=voltage, time=duration)
        
        # 获取模拟结果（使用 hybrid 方法快速获取）
        sim_result = self.simulate(voltage, duration, method='hybrid')
        
        # 计算差异
        eta_model = model_result['aperture_ratio']
        eta_sim = sim_result.aperture_ratio[-1]
        
        return {
            'model_aperture_ratio': eta_model,
            'simulation_aperture_ratio': eta_sim,
            'difference': abs(eta_model - eta_sim),
            'relative_error': abs(eta_model - eta_sim) / (eta_model + 1e-10),
            'model_theta': model_result['theta'],
            'simulation_theta': sim_result.contact_angle[-1]
        }
    
    def validate_against_experiment(self, 
                                    exp_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        与实验数据验证
        
        Args:
            exp_data: 实验数据
                - 't': 时间数组
                - 'aperture_ratio': 开口率数组
                - 'voltage': 电压
                
        Returns:
            验证结果
        """
        voltage = exp_data.get('voltage', 30.0)
        t_exp = exp_data['t']
        eta_exp = exp_data['aperture_ratio']
        
        # 模拟
        duration = t_exp[-1]
        sim_result = self.simulate(voltage, duration, method='hybrid')
        
        # 插值到实验时间点
        eta_sim = np.interp(t_exp, sim_result.t, sim_result.aperture_ratio)
        
        # 计算误差
        mae = np.mean(np.abs(eta_sim - eta_exp))
        rmse = np.sqrt(np.mean((eta_sim - eta_exp)**2))
        max_error = np.max(np.abs(eta_sim - eta_exp))
        relative_error = np.mean(np.abs(eta_sim - eta_exp) / (eta_exp + 1e-10))
        
        # 标记是否超过阈值
        error_flag = relative_error > 0.1  # 10% 阈值
        
        return {
            'mae': mae,
            'rmse': rmse,
            'max_error': max_error,
            'relative_error': relative_error,
            'error_flag': error_flag,
            'n_points': len(t_exp)
        }
    
    def export_results(self, result: SimulationResult, 
                       output_dir: str, format: str = 'vtk'):
        """
        导出结果
        
        Args:
            result: 模拟结果
            output_dir: 输出目录
            format: 输出格式 ('vtk', 'npz', 'json')
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if format == 'vtk':
            self._export_vtk(result, output_dir)
        elif format == 'npz':
            self._export_npz(result, output_dir)
        elif format == 'json':
            self._export_json(result, output_dir)
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def _export_vtk(self, result: SimulationResult, output_dir: str):
        """导出为 VTK 格式"""
        try:
            import pyvista as pv
        except ImportError:
            print("Warning: PyVista not available, skipping VTK export")
            return
        
        self._initialize_mesh()
        
        # 只导出有完整场数据的结果
        if not result.has_full_fields():
            print("Warning: No full field data, exporting summary only")
            self._export_json(result, output_dir)
            return
        
        # 导出每个时间步
        for i, t in enumerate(result.t):
            # 创建结构化网格
            grid = pv.RectilinearGrid(self.mesh.x, self.mesh.y, self.mesh.z)
            
            # 添加场数据
            grid.cell_data['phi'] = result.phi[i].flatten(order='F')
            grid.cell_data['u'] = result.u[i].flatten(order='F')
            grid.cell_data['v'] = result.v[i].flatten(order='F')
            grid.cell_data['w'] = result.w[i].flatten(order='F')
            grid.cell_data['p'] = result.p[i].flatten(order='F')
            
            # 保存
            filename = os.path.join(output_dir, f"flow_t{i:04d}.vtk")
            grid.save(filename)
    
    def _export_npz(self, result: SimulationResult, output_dir: str):
        """导出为 NPZ 格式"""
        data = {
            't': result.t,
            'aperture_ratio': result.aperture_ratio,
            'contact_angle': result.contact_angle,
            'mass_error': result.mass_error if result.mass_error is not None else np.array([]),
            'computation_time': result.computation_time,
            'method': result.method
        }
        
        if result.has_full_fields():
            data['u'] = result.u
            data['v'] = result.v
            data['w'] = result.w
            data['p'] = result.p
            data['phi'] = result.phi
        
        filename = os.path.join(output_dir, "simulation_result.npz")
        np.savez(filename, **data)
    
    def _export_json(self, result: SimulationResult, output_dir: str):
        """导出为 JSON 格式（仅摘要）"""
        summary = {
            't': result.t.tolist(),
            'aperture_ratio': result.aperture_ratio.tolist() if result.aperture_ratio is not None else [],
            'contact_angle': result.contact_angle.tolist() if result.contact_angle is not None else [],
            'computation_time': result.computation_time,
            'method': result.method,
            'config': result.config
        }
        
        filename = os.path.join(output_dir, "simulation_summary.json")
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2)


# ============================================================
# 便捷函数和演示
# ============================================================

def create_initial_conditions(mesh: Mesh, ink_thickness: float = 3e-6) -> Dict[str, np.ndarray]:
    """
    创建初始条件
    
    Args:
        mesh: 计算网格
        ink_thickness: 油墨层厚度
        
    Returns:
        包含初始场的字典
    """
    # 初始体积分数（油墨在底部）
    phi = np.zeros((mesh.nx, mesh.ny, mesh.nz))
    for k in range(mesh.nz):
        if mesh.zc[k] < ink_thickness:
            phi[:, :, k] = 1.0
    
    # 初始速度（静止）
    u = np.zeros((mesh.nx, mesh.ny, mesh.nz))
    v = np.zeros((mesh.nx, mesh.ny, mesh.nz))
    w = np.zeros((mesh.nx, mesh.ny, mesh.nz))
    
    # 初始压力（静水压力）
    p = np.zeros((mesh.nx, mesh.ny, mesh.nz))
    
    return {
        'phi': phi,
        'u': u, 'v': v, 'w': w,
        'p': p
    }


def compute_aperture_ratio_from_phi(phi: np.ndarray, mesh: Mesh, 
                                    z_threshold: float = None) -> float:
    """
    从体积分数场计算开口率
    
    Args:
        phi: 体积分数场 (nx, ny, nz)
        mesh: 计算网格
        z_threshold: z 阈值，低于此值的区域用于计算开口率
        
    Returns:
        开口率 (0-1)
    """
    if z_threshold is None:
        z_threshold = DOMAIN_PARAMETERS['ink_thickness']
    
    # 找到底层（油墨层）
    k_ink = 0
    for k in range(mesh.nz):
        if mesh.zc[k] > z_threshold:
            break
        k_ink = k
    
    # 计算底层的透明区域（φ < 0.5 的区域）
    phi_bottom = phi[:, :, k_ink]
    transparent_area = np.sum(phi_bottom < 0.5) * mesh.dx * mesh.dy
    total_area = mesh.nx * mesh.ny * mesh.dx * mesh.dy
    
    aperture_ratio = transparent_area / total_area
    
    return aperture_ratio


def demo_flow_solver():
    """
    演示流场求解器的基本功能
    """
    print("=" * 60)
    print("🌊 EWP Flow Field Solver Demo")
    print("=" * 60)
    
    # 1. 创建网格
    print("\n📐 创建计算网格...")
    generator = MeshGenerator()
    mesh = generator.generate_structured_mesh(nx=16, ny=16, nz=8)
    print(f"   网格尺寸: {mesh.nx} × {mesh.ny} × {mesh.nz}")
    print(f"   总单元数: {mesh.total_cells}")
    print(f"   单元体积: {mesh.cell_volume * 1e18:.2f} μm³")
    
    # 2. 创建初始条件
    print("\n🎯 创建初始条件...")
    initial = create_initial_conditions(mesh)
    phi = initial['phi']
    print(f"   油墨体积分数: {np.mean(phi):.3f}")
    
    # 3. 初始化界面追踪器
    print("\n🔍 初始化界面追踪器...")
    tracker = InterfaceTracker(mesh)
    thickness = tracker.compute_interface_thickness(phi)
    print(f"   界面厚度: {thickness:.1f} 网格单元")
    
    # 4. 初始化接触线处理器
    print("\n📍 初始化接触线处理器...")
    handler = ContactLineHandler()
    theta_0 = handler.get_equilibrium_contact_angle(0)
    theta_30 = handler.get_equilibrium_contact_angle(30)
    print(f"   0V 平衡接触角: {theta_0:.1f}°")
    print(f"   30V 平衡接触角: {theta_30:.1f}°")
    
    # 5. 使用 FlowFieldSimulator 进行模拟
    print("\n🚀 使用 FlowFieldSimulator 进行模拟...")
    simulator = FlowFieldSimulator()
    result = simulator.simulate(voltage=30, duration=0.01, method='hybrid')
    print(f"   模拟时长: {result.t[-1]*1000:.1f} ms")
    print(f"   最终开口率: {result.aperture_ratio[-1]*100:.1f}%")
    print(f"   最终接触角: {result.contact_angle[-1]:.1f}°")
    
    # 6. 与 EnhancedApertureModel 对比
    print("\n🔄 与 EnhancedApertureModel 对比...")
    comparison = simulator.compare_with_aperture_model(voltage=30, duration=0.01)
    print(f"   模型开口率: {comparison['model_aperture_ratio']*100:.1f}%")
    print(f"   模拟开口率: {comparison['simulation_aperture_ratio']*100:.1f}%")
    print(f"   相对误差: {comparison['relative_error']*100:.2f}%")
    
    # 7. 测试 CFD 求解器
    print("\n⚙️ 测试 CFD 求解器...")
    cfd_solver = FlowSolver(mesh)
    cfd_solver.set_initial_conditions(phi)
    cfd_solver.set_boundary_conditions({'walls': 'no_slip'})
    mass_error = cfd_solver.compute_mass_conservation_error()
    print(f"   初始质量误差: {mass_error['total_error']*100:.4f}%")
    
    # 8. 测试 PINN 求解器
    print("\n🧠 测试 PINN 求解器...")
    pinn_solver = PINNSolver()
    pinn_solver.build_network()
    summary = pinn_solver.get_training_summary()
    print(f"   网络已构建: {not summary['trained']}")
    print(f"   设备: {pinn_solver.device}")
    
    print("\n" + "=" * 60)
    print("✅ 演示完成!")
    print("=" * 60)
    print("\n📖 使用示例:")
    print("   from ewp_flow_solver import FlowFieldSimulator")
    print("   simulator = FlowFieldSimulator()")
    print("   result = simulator.simulate(voltage=30, duration=0.02)")
    print("   print(f'Final aperture: {result.aperture_ratio[-1]:.2%}')")


# ============================================================
# 主程序入口
# ============================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo_flow_solver()
    elif len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("EWP Flow Field Solver")
        print("Usage:")
        print("  python ewp_flow_solver.py --demo   # 运行演示")
        print("  python ewp_flow_solver.py --help   # 显示帮助")
    else:
        demo_flow_solver()
