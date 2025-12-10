"""
EWP PINN Data Generator - PINN 训练数据生成模块

为流场求解器的 PINN 组件提供训练数据生成功能：
- 合成数据生成（基于 FlowFieldSimulator）
- 物理约束点采样
- 实验数据加载和验证
- 数据增强
- 数据预处理和导出
- 数据可视化

Author: EFD-PINNs Team
Date: 2025-12-03
"""

import numpy as np
import torch
import json
import csv
import hashlib
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Tuple, Dict, Any, Union
from datetime import datetime

# 导入现有模块
from src.solvers.flow_solver import (
    FlowFieldSimulator, Mesh, MeshGenerator, SimulationResult,
    DOMAIN_PARAMETERS, FLUID_PROPERTIES
)
from src.models.aperture_model import EnhancedApertureModel
from src.predictors.hybrid_predictor import HybridPredictor

# 设置日志
logger = logging.getLogger(__name__)


# ============================================================
# 异常类定义
# ============================================================

class ConfigurationError(Exception):
    """配置错误 - 配置文件缺少必需参数"""
    def __init__(self, message: str, missing_params: Optional[List[str]] = None):
        super().__init__(message)
        self.missing_params = missing_params or []


class ValidationError(Exception):
    """验证错误 - 数据格式不正确"""
    def __init__(self, message: str, failed_checks: Optional[List[str]] = None):
        super().__init__(message)
        self.failed_checks = failed_checks or []


class DataIntegrityError(Exception):
    """数据完整性错误 - 校验和不匹配"""
    def __init__(self, message: str, expected: str = "", actual: str = ""):
        super().__init__(message)
        self.expected = expected
        self.actual = actual


class MassConservationError(Exception):
    """质量守恒错误 - 质量误差超过阈值"""
    def __init__(self, message: str, error_value: float = 0.0, threshold: float = 0.001):
        super().__init__(message)
        self.error_value = error_value
        self.threshold = threshold


class FileFormatError(Exception):
    """文件格式错误 - 不支持的文件格式"""
    def __init__(self, message: str, supported_formats: Optional[List[str]] = None):
        super().__init__(message)
        self.supported_formats = supported_formats or []


# ============================================================
# 数据类定义
# ============================================================

@dataclass
class NormParams:
    """归一化参数"""
    method: str  # 'standard' 或 'minmax'
    mean: Optional[np.ndarray] = None
    std: Optional[np.ndarray] = None
    min_val: Optional[np.ndarray] = None
    max_val: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        result = {'method': self.method}
        if self.mean is not None:
            result['mean'] = self.mean.tolist()
        if self.std is not None:
            result['std'] = self.std.tolist()
        if self.min_val is not None:
            result['min_val'] = self.min_val.tolist()
        if self.max_val is not None:
            result['max_val'] = self.max_val.tolist()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NormParams':
        """从字典反序列化"""
        return cls(
            method=data['method'],
            mean=np.array(data['mean']) if 'mean' in data else None,
            std=np.array(data['std']) if 'std' in data else None,
            min_val=np.array(data['min_val']) if 'min_val' in data else None,
            max_val=np.array(data['max_val']) if 'max_val' in data else None
        )


@dataclass
class SyntheticDataset:
    """合成数据集"""
    # 网格信息
    mesh_info: Dict[str, Any]  # 网格参数（不存储完整 Mesh 对象）
    
    # 时间和电压
    t: np.ndarray           # (n_time,) 时间点 [s]
    voltages: np.ndarray    # (n_voltage,) 电压值 [V]
    
    # 流场数据 (n_voltage, n_time, nx, ny, nz)
    u: np.ndarray           # x 方向速度
    v: np.ndarray           # y 方向速度
    w: np.ndarray           # z 方向速度
    p: np.ndarray           # 压力
    phi: np.ndarray         # 体积分数
    
    # 积分量 (n_voltage, n_time)
    aperture_ratio: np.ndarray
    contact_angle: np.ndarray
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def has_all_fields(self) -> bool:
        """检查是否包含所有必需字段"""
        required = ['u', 'v', 'w', 'p', 'phi', 'aperture_ratio', 'contact_angle']
        for field_name in required:
            if getattr(self, field_name, None) is None:
                return False
        return True
    
    def get_dimensions(self) -> Dict[str, int]:
        """获取数据维度"""
        return {
            'n_voltage': len(self.voltages),
            'n_time': len(self.t),
            'nx': self.mesh_info.get('nx', 0),
            'ny': self.mesh_info.get('ny', 0),
            'nz': self.mesh_info.get('nz', 0)
        }


@dataclass
class PhysicsConstraintPoints:
    """物理约束点集"""
    # 坐标 (n_points, 4) - [x, y, z, t]
    domain_points: np.ndarray
    boundary_points: np.ndarray
    interface_points: np.ndarray
    contact_line_points: np.ndarray
    
    # 物理参数
    voltage: float
    contact_angle: float
    
    # 边界类型标记 (n_boundary,) - 0=bottom, 1=top, 2=front, 3=back, 4=left, 5=right
    boundary_types: np.ndarray = field(default_factory=lambda: np.array([]))
    
    def get_all_points(self) -> np.ndarray:
        """获取所有点的坐标"""
        all_points = []
        for pts in [self.domain_points, self.boundary_points, 
                    self.interface_points, self.contact_line_points]:
            if pts is not None and len(pts) > 0:
                all_points.append(pts)
        return np.vstack(all_points) if all_points else np.array([]).reshape(0, 4)
    
    def has_required_fields(self) -> bool:
        """检查是否包含所有必需字段"""
        # 检查坐标
        for pts in [self.domain_points, self.boundary_points]:
            if pts is None or len(pts) == 0:
                return False
            if pts.shape[1] != 4:  # [x, y, z, t]
                return False
        # 检查物理参数
        if self.voltage is None or self.contact_angle is None:
            return False
        return True


@dataclass
class ExperimentalDataset:
    """实验数据集"""
    # 时间序列
    t: np.ndarray               # (n_time,) 时间点 [s]
    aperture_ratio: np.ndarray  # (n_time,) 开口率
    contact_angle: Optional[np.ndarray] = None  # (n_time,) 接触角 [度]
    
    # 元数据
    voltage: float = 0.0              # 施加电压 [V]
    temperature: float = 25.0         # 温度 [°C]
    pixel_width: float = 174.0        # 像素宽度 [μm]
    pixel_height: float = 174.0       # 像素高度 [μm]
    
    # 可选字段
    experiment_id: str = ""
    date: str = ""
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {
            'metadata': {
                'experiment_id': self.experiment_id,
                'date': self.date,
                'voltage': self.voltage,
                'temperature': self.temperature,
                'pixel_width_um': self.pixel_width,
                'pixel_height_um': self.pixel_height,
                'notes': self.notes
            },
            'time_series': {
                'time_ms': (self.t * 1000).tolist(),  # 转换为 ms
                'aperture_ratio': self.aperture_ratio.tolist()
            }
        }
        if self.contact_angle is not None:
            result['time_series']['contact_angle_deg'] = self.contact_angle.tolist()
        return result


@dataclass
class ValidationResult:
    """验证结果"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def add_error(self, error: str):
        """添加错误"""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str):
        """添加警告"""
        self.warnings.append(warning)


# ============================================================
# 默认配置
# ============================================================

DEFAULT_CONFIG = {
    "synthetic_data": {
        "voltage_range": [0, 30],
        "voltage_resolution": 7,
        "time_range": [0, 0.02],
        "time_resolution": 100,
        "mesh": {
            "nx": 32,
            "ny": 32,
            "nz": 16
        }
    },
    "physics_points": {
        "n_domain": 10000,
        "n_boundary": 2000,
        "n_interface": 1000,
        "n_contact": 500
    },
    "augmentation": {
        "rotations": [90, 180, 270],
        "reflections": ["x", "y"],
        "noise_magnitude": 0.01,
        "enabled": True
    },
    "preprocessing": {
        "normalization": "standard",
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        "test_ratio": 0.15
    },
    "export": {
        "format": "pytorch",
        "output_dir": "data/pinn_training"
    },
    "mass_conservation_threshold": 0.001  # 0.1%
}

# 必需的配置参数
REQUIRED_CONFIG_PARAMS = [
    "synthetic_data.voltage_range",
    "synthetic_data.time_range",
    "preprocessing.train_ratio"
]


# ============================================================
# 配置管理函数
# ============================================================

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径，None 则使用默认配置
        
    Returns:
        配置字典
        
    Raises:
        ConfigurationError: 缺少必需参数
        FileNotFoundError: 配置文件不存在
    """
    if config_path is None:
        logger.info("使用默认配置")
        return DEFAULT_CONFIG.copy()
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 验证必需参数
    missing = validate_config(config)
    if missing:
        raise ConfigurationError(
            f"配置文件缺少必需参数: {missing}",
            missing_params=missing
        )
    
    # 填充默认值
    config = fill_defaults(config)
    
    # 记录配置
    logger.info(f"加载配置: {config_path}")
    log_config(config)
    
    return config


def validate_config(config: Dict[str, Any]) -> List[str]:
    """
    验证配置是否包含所有必需参数
    
    Args:
        config: 配置字典
        
    Returns:
        缺失参数列表
    """
    missing = []
    
    for param_path in REQUIRED_CONFIG_PARAMS:
        parts = param_path.split('.')
        current = config
        found = True
        
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                found = False
                break
        
        if not found:
            missing.append(param_path)
    
    return missing


def fill_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    用默认值填充缺失的可选参数
    
    Args:
        config: 配置字典
        
    Returns:
        填充后的配置字典
    """
    def deep_merge(base: Dict, override: Dict) -> Dict:
        """深度合并字典"""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    return deep_merge(DEFAULT_CONFIG, config)


def log_config(config: Dict[str, Any], prefix: str = ""):
    """记录配置参数"""
    for key, value in config.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            log_config(value, full_key)
        else:
            logger.debug(f"  {full_key}: {value}")


def save_config(config: Dict[str, Any], output_path: str):
    """
    保存配置到文件
    
    Args:
        config: 配置字典
        output_path: 输出路径
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    logger.info(f"配置已保存: {output_path}")


# ============================================================
# SyntheticDataGenerator 类
# ============================================================

class SyntheticDataGenerator:
    """
    合成数据生成器
    
    使用 FlowFieldSimulator 生成 PINN 训练数据。
    
    Example:
        >>> simulator = FlowFieldSimulator()
        >>> generator = SyntheticDataGenerator(simulator)
        >>> dataset = generator.generate(voltages=[0, 15, 30], 
        ...                              time_points=np.linspace(0, 0.02, 100))
    """
    
    def __init__(self, simulator: Optional[FlowFieldSimulator] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        初始化生成器
        
        Args:
            simulator: FlowFieldSimulator 实例，None 则创建默认实例
            config: 生成配置
        """
        self.simulator = simulator or FlowFieldSimulator()
        self.config = config or DEFAULT_CONFIG.get('synthetic_data', {})
        
        # 质量守恒阈值
        self.mass_threshold = config.get('mass_conservation_threshold', 0.001) if config else 0.001
    
    def generate(self, voltages: Optional[List[float]] = None,
                 time_points: Optional[np.ndarray] = None) -> SyntheticDataset:
        """
        生成合成数据集
        
        Args:
            voltages: 电压列表 [V]，None 则使用配置
            time_points: 时间点数组 [s]，None 则使用配置
            
        Returns:
            SyntheticDataset 包含流场快照
        """
        # 使用配置或默认值
        if voltages is None:
            v_range = self.config.get('voltage_range', [0, 30])
            v_res = self.config.get('voltage_resolution', 7)
            voltages = np.linspace(v_range[0], v_range[1], v_res).tolist()
        
        if time_points is None:
            t_range = self.config.get('time_range', [0, 0.02])
            t_res = self.config.get('time_resolution', 100)
            time_points = np.linspace(t_range[0], t_range[1], t_res)
        
        logger.info(f"生成合成数据: {len(voltages)} 个电压, {len(time_points)} 个时间点")
        
        # 获取网格信息
        mesh_config = self.config.get('mesh', {'nx': 32, 'ny': 32, 'nz': 16})
        nx, ny, nz = mesh_config['nx'], mesh_config['ny'], mesh_config['nz']
        
        # 初始化数据数组
        n_voltage = len(voltages)
        n_time = len(time_points)
        
        u_data = np.zeros((n_voltage, n_time, nx, ny, nz))
        v_data = np.zeros((n_voltage, n_time, nx, ny, nz))
        w_data = np.zeros((n_voltage, n_time, nx, ny, nz))
        p_data = np.zeros((n_voltage, n_time, nx, ny, nz))
        phi_data = np.zeros((n_voltage, n_time, nx, ny, nz))
        aperture_data = np.zeros((n_voltage, n_time))
        angle_data = np.zeros((n_voltage, n_time))
        
        # 对每个电压进行模拟
        for i, voltage in enumerate(voltages):
            logger.debug(f"模拟电压 {voltage}V...")
            
            # 运行模拟
            result = self.simulator.simulate(
                voltage=voltage,
                duration=time_points[-1],
                method='hybrid'
            )
            
            # 提取数据（插值到指定时间点）
            for j, t in enumerate(time_points):
                # 找到最近的时间索引
                t_idx = np.argmin(np.abs(result.t - t))
                
                # 提取流场数据
                if result.has_full_fields():
                    u_data[i, j] = result.u[t_idx]
                    v_data[i, j] = result.v[t_idx]
                    w_data[i, j] = result.w[t_idx]
                    p_data[i, j] = result.p[t_idx]
                    phi_data[i, j] = result.phi[t_idx]
                
                # 提取积分量
                aperture_data[i, j] = result.aperture_ratio[t_idx]
                angle_data[i, j] = result.contact_angle[t_idx]
        
        # 创建数据集
        dataset = SyntheticDataset(
            mesh_info=mesh_config,
            t=time_points,
            voltages=np.array(voltages),
            u=u_data,
            v=v_data,
            w=w_data,
            p=p_data,
            phi=phi_data,
            aperture_ratio=aperture_data,
            contact_angle=angle_data,
            metadata={
                'generation_time': datetime.now().isoformat(),
                'config': self.config
            }
        )
        
        logger.info(f"合成数据生成完成: {dataset.get_dimensions()}")
        
        return dataset
    
    def validate_mass_conservation(self, dataset: SyntheticDataset) -> float:
        """
        验证质量守恒
        
        Args:
            dataset: 合成数据集
            
        Returns:
            最大质量误差
            
        Raises:
            MassConservationError: 误差超过阈值
        """
        max_error = 0.0
        
        # 计算每个快照的质量
        for i in range(len(dataset.voltages)):
            for j in range(len(dataset.t)):
                phi = dataset.phi[i, j]
                
                # 计算油墨总体积（φ=1 表示油墨）
                oil_volume = np.sum(phi)
                
                # 与初始体积比较
                if j == 0:
                    initial_volume = oil_volume
                else:
                    if initial_volume > 0:
                        error = abs(oil_volume - initial_volume) / initial_volume
                        max_error = max(max_error, error)
        
        if max_error > self.mass_threshold:
            raise MassConservationError(
                f"质量守恒误差 {max_error*100:.3f}% 超过阈值 {self.mass_threshold*100:.1f}%",
                error_value=max_error,
                threshold=self.mass_threshold
            )
        
        logger.info(f"质量守恒验证通过: 最大误差 {max_error*100:.4f}%")
        return max_error



# ============================================================
# PhysicsPointsSampler 类
# ============================================================

class PhysicsPointsSampler:
    """
    物理约束点采样器
    
    为 PINN 训练生成域内、边界、界面和接触线采样点。
    
    Example:
        >>> generator = MeshGenerator()
        >>> mesh = generator.generate_structured_mesh(nx=32, ny=32, nz=16)
        >>> sampler = PhysicsPointsSampler(mesh)
        >>> points = sampler.generate_constraint_points(
        ...     n_domain=1000, n_boundary=200, n_interface=100, n_contact=50,
        ...     phi=phi_field, voltage=30, theta=85
        ... )
    """
    
    def __init__(self, mesh: Optional[Mesh] = None, config: Optional[Dict[str, Any]] = None):
        """
        初始化采样器
        
        Args:
            mesh: 计算网格，None 则创建默认网格
            config: 采样配置
        """
        if mesh is None:
            generator = MeshGenerator()
            mesh = generator.generate_structured_mesh()
        
        self.mesh = mesh
        self.config = config or DEFAULT_CONFIG.get('physics_points', {})
        
        # 域边界
        self.Lx = mesh.x[-1] - mesh.x[0]
        self.Ly = mesh.y[-1] - mesh.y[0]
        self.Lz = mesh.z[-1] - mesh.z[0]
        
        # 域原点（可能是负值，如果以中心为原点）
        self.x_min = mesh.x[0]
        self.y_min = mesh.y[0]
        self.z_min = mesh.z[0]
    
    def sample_domain_points(self, n_points: int, 
                             t_range: Tuple[float, float] = (0, 0.02)) -> np.ndarray:
        """
        采样域内点
        
        Args:
            n_points: 采样点数
            t_range: 时间范围 [s]
            
        Returns:
            点坐标数组 (n_points, 4) - [x, y, z, t]
        """
        # 均匀随机采样
        x = np.random.uniform(self.x_min, self.x_min + self.Lx, n_points)
        y = np.random.uniform(self.y_min, self.y_min + self.Ly, n_points)
        z = np.random.uniform(self.z_min, self.z_min + self.Lz, n_points)
        t = np.random.uniform(t_range[0], t_range[1], n_points)
        
        return np.column_stack([x, y, z, t])
    
    def sample_boundary_points(self, n_points: int,
                               t_range: Tuple[float, float] = (0, 0.02)) -> Tuple[np.ndarray, np.ndarray]:
        """
        采样边界点
        
        Args:
            n_points: 采样点数
            t_range: 时间范围 [s]
            
        Returns:
            (点坐标数组 (n_points, 4), 边界类型数组 (n_points,))
            边界类型: 0=bottom, 1=top, 2=front, 3=back, 4=left, 5=right
        """
        # 每个边界分配的点数
        n_per_boundary = n_points // 6
        remainder = n_points % 6
        
        points = []
        types = []
        
        # 0: bottom (z = z_min)
        n = n_per_boundary + (1 if remainder > 0 else 0)
        x = np.random.uniform(self.x_min, self.x_min + self.Lx, n)
        y = np.random.uniform(self.y_min, self.y_min + self.Ly, n)
        z = np.full(n, self.z_min)
        t = np.random.uniform(t_range[0], t_range[1], n)
        points.append(np.column_stack([x, y, z, t]))
        types.append(np.zeros(n, dtype=int))
        
        # 1: top (z = z_min + Lz)
        n = n_per_boundary + (1 if remainder > 1 else 0)
        x = np.random.uniform(self.x_min, self.x_min + self.Lx, n)
        y = np.random.uniform(self.y_min, self.y_min + self.Ly, n)
        z = np.full(n, self.z_min + self.Lz)
        t = np.random.uniform(t_range[0], t_range[1], n)
        points.append(np.column_stack([x, y, z, t]))
        types.append(np.ones(n, dtype=int))
        
        # 2: front (y = y_min)
        n = n_per_boundary + (1 if remainder > 2 else 0)
        x = np.random.uniform(self.x_min, self.x_min + self.Lx, n)
        y = np.full(n, self.y_min)
        z = np.random.uniform(self.z_min, self.z_min + self.Lz, n)
        t = np.random.uniform(t_range[0], t_range[1], n)
        points.append(np.column_stack([x, y, z, t]))
        types.append(np.full(n, 2, dtype=int))
        
        # 3: back (y = y_min + Ly)
        n = n_per_boundary + (1 if remainder > 3 else 0)
        x = np.random.uniform(self.x_min, self.x_min + self.Lx, n)
        y = np.full(n, self.y_min + self.Ly)
        z = np.random.uniform(self.z_min, self.z_min + self.Lz, n)
        t = np.random.uniform(t_range[0], t_range[1], n)
        points.append(np.column_stack([x, y, z, t]))
        types.append(np.full(n, 3, dtype=int))
        
        # 4: left (x = x_min)
        n = n_per_boundary + (1 if remainder > 4 else 0)
        x = np.full(n, self.x_min)
        y = np.random.uniform(self.y_min, self.y_min + self.Ly, n)
        z = np.random.uniform(self.z_min, self.z_min + self.Lz, n)
        t = np.random.uniform(t_range[0], t_range[1], n)
        points.append(np.column_stack([x, y, z, t]))
        types.append(np.full(n, 4, dtype=int))
        
        # 5: right (x = x_min + Lx)
        n = n_per_boundary
        x = np.full(n, self.x_min + self.Lx)
        y = np.random.uniform(self.y_min, self.y_min + self.Ly, n)
        z = np.random.uniform(self.z_min, self.z_min + self.Lz, n)
        t = np.random.uniform(t_range[0], t_range[1], n)
        points.append(np.column_stack([x, y, z, t]))
        types.append(np.full(n, 5, dtype=int))
        
        return np.vstack(points), np.concatenate(types)
    
    def sample_interface_points(self, phi: np.ndarray, n_points: int,
                                t_range: Tuple[float, float] = (0, 0.02)) -> np.ndarray:
        """
        采样界面点 (0.01 < φ < 0.99)
        
        Args:
            phi: 体积分数场 (nx, ny, nz)
            n_points: 采样点数
            t_range: 时间范围 [s]
            
        Returns:
            点坐标数组 (n_points, 4) - [x, y, z, t]
        """
        # 找到界面单元
        interface_mask = (phi > 0.01) & (phi < 0.99)
        interface_indices = np.argwhere(interface_mask)
        
        if len(interface_indices) == 0:
            logger.warning("未找到界面单元，返回空数组")
            return np.array([]).reshape(0, 4)
        
        # 随机选择界面单元
        n_available = len(interface_indices)
        n_sample = min(n_points, n_available)
        
        if n_sample < n_points:
            # 重复采样
            selected_indices = np.random.choice(n_available, n_points, replace=True)
        else:
            selected_indices = np.random.choice(n_available, n_points, replace=False)
        
        selected_cells = interface_indices[selected_indices]
        
        # 转换为物理坐标（单元中心 + 随机偏移）
        points = []
        for i, j, k in selected_cells:
            x = self.mesh.xc[i] + np.random.uniform(-self.mesh.dx/2, self.mesh.dx/2)
            y = self.mesh.yc[j] + np.random.uniform(-self.mesh.dy/2, self.mesh.dy/2)
            z = self.mesh.zc[k] + np.random.uniform(-self.mesh.dz/2, self.mesh.dz/2)
            t = np.random.uniform(t_range[0], t_range[1])
            points.append([x, y, z, t])
        
        return np.array(points)
    
    def sample_contact_line_points(self, phi: np.ndarray, n_points: int,
                                   t_range: Tuple[float, float] = (0, 0.02)) -> np.ndarray:
        """
        采样接触线点 (z=0 且在界面附近)
        
        Args:
            phi: 体积分数场 (nx, ny, nz)
            n_points: 采样点数
            t_range: 时间范围 [s]
            
        Returns:
            点坐标数组 (n_points, 4) - [x, y, z, t]
        """
        # 在底面 (k=0) 找到界面单元
        phi_bottom = phi[:, :, 0]
        contact_mask = (phi_bottom > 0.01) & (phi_bottom < 0.99)
        contact_indices = np.argwhere(contact_mask)
        
        if len(contact_indices) == 0:
            logger.warning("未找到接触线单元，返回空数组")
            return np.array([]).reshape(0, 4)
        
        # 随机选择接触线单元
        n_available = len(contact_indices)
        n_sample = min(n_points, n_available)
        
        if n_sample < n_points:
            selected_indices = np.random.choice(n_available, n_points, replace=True)
        else:
            selected_indices = np.random.choice(n_available, n_points, replace=False)
        
        selected_cells = contact_indices[selected_indices]
        
        # 转换为物理坐标
        points = []
        for i, j in selected_cells:
            x = self.mesh.xc[i] + np.random.uniform(-self.mesh.dx/2, self.mesh.dx/2)
            y = self.mesh.yc[j] + np.random.uniform(-self.mesh.dy/2, self.mesh.dy/2)
            z = self.z_min  # 底面
            t = np.random.uniform(t_range[0], t_range[1])
            points.append([x, y, z, t])
        
        return np.array(points)
    
    def generate_constraint_points(self, n_domain: int, n_boundary: int,
                                   n_interface: int, n_contact: int,
                                   phi: np.ndarray, voltage: float,
                                   theta: float,
                                   t_range: Tuple[float, float] = (0, 0.02)) -> PhysicsConstraintPoints:
        """
        生成完整的物理约束点集
        
        Args:
            n_domain: 域内点数
            n_boundary: 边界点数
            n_interface: 界面点数
            n_contact: 接触线点数
            phi: 体积分数场 (nx, ny, nz)
            voltage: 电压 [V]
            theta: 接触角 [度]
            t_range: 时间范围 [s]
            
        Returns:
            PhysicsConstraintPoints 对象
        """
        domain_points = self.sample_domain_points(n_domain, t_range)
        boundary_points, boundary_types = self.sample_boundary_points(n_boundary, t_range)
        interface_points = self.sample_interface_points(phi, n_interface, t_range)
        contact_points = self.sample_contact_line_points(phi, n_contact, t_range)
        
        return PhysicsConstraintPoints(
            domain_points=domain_points,
            boundary_points=boundary_points,
            interface_points=interface_points,
            contact_line_points=contact_points,
            voltage=voltage,
            contact_angle=theta,
            boundary_types=boundary_types
        )



# ============================================================
# ExperimentalDataLoader 类
# ============================================================

class ExperimentalDataLoader:
    """
    实验数据加载器
    
    加载和验证实验数据，支持 JSON 和 CSV 格式。
    
    Example:
        >>> loader = ExperimentalDataLoader()
        >>> data = loader.load_json('experiment_data.json')
        >>> result = loader.validate(data)
        >>> if result.is_valid:
        ...     print("数据有效")
    """
    
    # 必需字段
    REQUIRED_METADATA = ['voltage']
    REQUIRED_TIME_SERIES = ['time_ms', 'aperture_ratio']
    
    def __init__(self):
        """初始化加载器"""
        pass
    
    def load_json(self, path: str) -> ExperimentalDataset:
        """
        从 JSON 文件加载实验数据
        
        Args:
            path: JSON 文件路径
            
        Returns:
            ExperimentalDataset 对象
            
        Raises:
            FileNotFoundError: 文件不存在
            ValidationError: 数据格式错误
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return self._parse_json_data(data)
    
    def _parse_json_data(self, data: Dict[str, Any]) -> ExperimentalDataset:
        """解析 JSON 数据"""
        metadata = data.get('metadata', {})
        time_series = data.get('time_series', {})
        
        # 提取时间序列
        time_ms = np.array(time_series.get('time_ms', []))
        aperture_ratio = np.array(time_series.get('aperture_ratio', []))
        contact_angle = time_series.get('contact_angle_deg')
        if contact_angle is not None:
            contact_angle = np.array(contact_angle)
        
        # 转换时间单位 (ms -> s)
        t = time_ms / 1000.0
        
        return ExperimentalDataset(
            t=t,
            aperture_ratio=aperture_ratio,
            contact_angle=contact_angle,
            voltage=metadata.get('voltage', 0.0),
            temperature=metadata.get('temperature', 25.0),
            pixel_width=metadata.get('pixel_width_um', 174.0),
            pixel_height=metadata.get('pixel_height_um', 174.0),
            experiment_id=metadata.get('experiment_id', ''),
            date=metadata.get('date', ''),
            notes=metadata.get('notes', '')
        )
    
    def load_csv(self, path: str) -> ExperimentalDataset:
        """
        从 CSV 文件加载实验数据
        
        Args:
            path: CSV 文件路径
            
        Returns:
            ExperimentalDataset 对象
            
        Raises:
            FileNotFoundError: 文件不存在
            ValidationError: 数据格式错误
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {path}")
        
        # 读取元数据（注释行）
        metadata = {}
        data_lines = []
        
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#'):
                    # 解析元数据注释
                    if ':' in line:
                        key_value = line[1:].strip()
                        if ':' in key_value:
                            key, value = key_value.split(':', 1)
                            key = key.strip().lower().replace(' ', '_')
                            value = value.strip()
                            # 尝试转换为数值
                            try:
                                if '.' in value:
                                    value = float(value.split()[0])  # 去掉单位
                                else:
                                    value = int(value.split()[0])
                            except ValueError:
                                pass
                            metadata[key] = value
                elif line:
                    data_lines.append(line)
        
        # 解析数据行
        if not data_lines:
            raise ValidationError("CSV 文件没有数据行", ['no_data'])
        
        # 第一行是表头
        headers = [h.strip().lower() for h in data_lines[0].split(',')]
        
        # 解析数据
        time_ms = []
        aperture_ratio = []
        contact_angle = []
        
        for line in data_lines[1:]:
            values = line.split(',')
            row = {headers[i]: values[i].strip() for i in range(min(len(headers), len(values)))}
            
            if 'time_ms' in row:
                time_ms.append(float(row['time_ms']))
            if 'aperture_ratio' in row:
                aperture_ratio.append(float(row['aperture_ratio']))
            if 'contact_angle_deg' in row:
                contact_angle.append(float(row['contact_angle_deg']))
        
        # 转换为数组
        t = np.array(time_ms) / 1000.0  # ms -> s
        aperture_ratio = np.array(aperture_ratio)
        contact_angle = np.array(contact_angle) if contact_angle else None
        
        return ExperimentalDataset(
            t=t,
            aperture_ratio=aperture_ratio,
            contact_angle=contact_angle,
            voltage=metadata.get('voltage', 0.0),
            temperature=metadata.get('temperature', 25.0),
            pixel_width=metadata.get('pixel_width_um', 174.0),
            pixel_height=metadata.get('pixel_height_um', 174.0),
            experiment_id=metadata.get('experiment_id', ''),
            date=metadata.get('date', ''),
            notes=metadata.get('notes', '')
        )
    
    def validate(self, data: ExperimentalDataset) -> ValidationResult:
        """
        验证实验数据完整性
        
        Args:
            data: 实验数据集
            
        Returns:
            ValidationResult 对象
        """
        result = ValidationResult(is_valid=True)
        
        # 检查时间序列
        if data.t is None or len(data.t) == 0:
            result.add_error("缺少时间序列 (time_ms)")
        
        if data.aperture_ratio is None or len(data.aperture_ratio) == 0:
            result.add_error("缺少开口率数据 (aperture_ratio)")
        
        # 检查数据长度一致性
        if data.t is not None and data.aperture_ratio is not None:
            if len(data.t) != len(data.aperture_ratio):
                result.add_error(f"时间和开口率数据长度不一致: {len(data.t)} vs {len(data.aperture_ratio)}")
        
        if data.contact_angle is not None and data.t is not None:
            if len(data.contact_angle) != len(data.t):
                result.add_error(f"时间和接触角数据长度不一致: {len(data.t)} vs {len(data.contact_angle)}")
        
        # 检查数据范围
        if data.aperture_ratio is not None:
            if np.any(data.aperture_ratio < 0) or np.any(data.aperture_ratio > 1):
                result.add_warning("开口率数据超出 [0, 1] 范围")
        
        if data.contact_angle is not None:
            if np.any(data.contact_angle < 0) or np.any(data.contact_angle > 180):
                result.add_warning("接触角数据超出 [0, 180] 范围")
        
        # 检查元数据
        if data.voltage <= 0:
            result.add_warning("电压值未设置或为零")
        
        return result
    
    def generate_template_json(self, output_path: str):
        """
        生成 JSON 模板文件
        
        Args:
            output_path: 输出路径
        """
        template = {
            "_comment": "EWP 实验数据模板 - 请按此格式整理您的实验数据",
            "metadata": {
                "experiment_id": "EXP_001",
                "date": datetime.now().strftime("%Y-%m-%d"),
                "voltage": 30.0,
                "temperature": 25.0,
                "pixel_width_um": 174.0,
                "pixel_height_um": 174.0,
                "notes": "实验备注 - 请填写实验条件和观察"
            },
            "time_series": {
                "_comment_time": "时间单位: 毫秒 (ms)",
                "time_ms": [0, 1, 2, 3, 4, 5, 10, 15, 20],
                "_comment_aperture": "开口率: 0-1 之间的小数",
                "aperture_ratio": [0.0, 0.15, 0.28, 0.36, 0.41, 0.44, 0.47, 0.48, 0.48],
                "_comment_angle": "接触角: 度数 (可选)",
                "contact_angle_deg": [115, 105, 98, 93, 90, 88, 86, 85, 85]
            }
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(template, f, indent=2, ensure_ascii=False)
        
        logger.info(f"JSON 模板已生成: {output_path}")
    
    def generate_template_csv(self, output_path: str):
        """
        生成 CSV 模板文件
        
        Args:
            output_path: 输出路径
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8', newline='') as f:
            # 写入元数据注释
            f.write("# EWP 实验数据模板\n")
            f.write("# 请按此格式整理您的实验数据\n")
            f.write("#\n")
            f.write("# experiment_id: EXP_001\n")
            f.write(f"# date: {datetime.now().strftime('%Y-%m-%d')}\n")
            f.write("# voltage: 30.0 V\n")
            f.write("# temperature: 25.0 C\n")
            f.write("# pixel_width_um: 174.0\n")
            f.write("# pixel_height_um: 174.0\n")
            f.write("# notes: 实验备注\n")
            f.write("#\n")
            
            # 写入数据
            writer = csv.writer(f)
            writer.writerow(['time_ms', 'aperture_ratio', 'contact_angle_deg'])
            
            # 示例数据
            example_data = [
                [0, 0.0, 115],
                [1, 0.15, 105],
                [2, 0.28, 98],
                [3, 0.36, 93],
                [4, 0.41, 90],
                [5, 0.44, 88],
                [10, 0.47, 86],
                [15, 0.48, 85],
                [20, 0.48, 85]
            ]
            writer.writerows(example_data)
        
        logger.info(f"CSV 模板已生成: {output_path}")



# ============================================================
# DataAugmenter 类
# ============================================================

class DataAugmenter:
    """
    数据增强器
    
    利用像素对称性进行数据增强：旋转、翻转、噪声注入。
    
    Example:
        >>> augmenter = DataAugmenter()
        >>> augmented = augmenter.rotate(data, angle=90)
        >>> augmented = augmenter.reflect(data, axis='x')
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化增强器
        
        Args:
            config: 增强配置
        """
        self.config = config or DEFAULT_CONFIG.get('augmentation', {})
        self.enabled = self.config.get('enabled', True)
    
    def rotate(self, data: np.ndarray, angle: int) -> np.ndarray:
        """
        旋转 3D 数据
        
        Args:
            data: 3D 数组 (nx, ny, nz) 或 5D 数组 (n_v, n_t, nx, ny, nz)
            angle: 旋转角度 (90, 180, 270)
            
        Returns:
            旋转后的数组（维度不变）
        """
        if angle not in [90, 180, 270]:
            raise ValueError(f"不支持的旋转角度: {angle}，支持 90, 180, 270")
        
        k = angle // 90  # 旋转次数
        
        if data.ndim == 3:
            # 3D: 在 xy 平面旋转
            return np.rot90(data, k=k, axes=(0, 1))
        elif data.ndim == 5:
            # 5D: 对每个 (voltage, time) 切片旋转
            result = np.zeros_like(data)
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    result[i, j] = np.rot90(data[i, j], k=k, axes=(0, 1))
            return result
        else:
            raise ValueError(f"不支持的数据维度: {data.ndim}")
    
    def reflect(self, data: np.ndarray, axis: str) -> np.ndarray:
        """
        反射 3D 数据
        
        Args:
            data: 3D 数组 (nx, ny, nz) 或 5D 数组 (n_v, n_t, nx, ny, nz)
            axis: 反射轴 ('x' 或 'y')
            
        Returns:
            反射后的数组（维度不变）
        """
        if axis not in ['x', 'y']:
            raise ValueError(f"不支持的反射轴: {axis}，支持 'x', 'y'")
        
        flip_axis = 0 if axis == 'x' else 1
        
        if data.ndim == 3:
            return np.flip(data, axis=flip_axis)
        elif data.ndim == 5:
            return np.flip(data, axis=flip_axis + 2)  # 偏移 2 (跳过 voltage, time)
        else:
            raise ValueError(f"不支持的数据维度: {data.ndim}")
    
    def add_noise(self, data: np.ndarray, magnitude: float) -> np.ndarray:
        """
        添加高斯噪声
        
        Args:
            data: 数据数组
            magnitude: 噪声幅度（相对于数据标准差）
            
        Returns:
            添加噪声后的数组
        """
        std = np.std(data)
        noise = np.random.normal(0, magnitude * std, data.shape)
        return data + noise
    
    def augment(self, dataset: SyntheticDataset,
                augmentations: Optional[List[str]] = None) -> Tuple[SyntheticDataset, int]:
        """
        应用增强并返回扩充后的数据集
        
        Args:
            dataset: 原始数据集
            augmentations: 增强列表，如 ['rotate_90', 'reflect_x', 'noise']
                          None 则使用配置
            
        Returns:
            (扩充后的数据集, 增强因子)
        """
        if not self.enabled:
            return dataset, 1
        
        if augmentations is None:
            # 从配置构建增强列表
            augmentations = []
            for angle in self.config.get('rotations', []):
                augmentations.append(f'rotate_{angle}')
            for axis in self.config.get('reflections', []):
                augmentations.append(f'reflect_{axis}')
            if self.config.get('noise_magnitude', 0) > 0:
                augmentations.append('noise')
        
        # 收集所有增强版本
        all_u = [dataset.u]
        all_v = [dataset.v]
        all_w = [dataset.w]
        all_p = [dataset.p]
        all_phi = [dataset.phi]
        
        for aug in augmentations:
            if aug.startswith('rotate_'):
                angle = int(aug.split('_')[1])
                all_u.append(self.rotate(dataset.u, angle))
                all_v.append(self.rotate(dataset.v, angle))
                all_w.append(self.rotate(dataset.w, angle))
                all_p.append(self.rotate(dataset.p, angle))
                all_phi.append(self.rotate(dataset.phi, angle))
            elif aug.startswith('reflect_'):
                axis = aug.split('_')[1]
                all_u.append(self.reflect(dataset.u, axis))
                all_v.append(self.reflect(dataset.v, axis))
                all_w.append(self.reflect(dataset.w, axis))
                all_p.append(self.reflect(dataset.p, axis))
                all_phi.append(self.reflect(dataset.phi, axis))
            elif aug == 'noise':
                magnitude = self.config.get('noise_magnitude', 0.01)
                all_u.append(self.add_noise(dataset.u, magnitude))
                all_v.append(self.add_noise(dataset.v, magnitude))
                all_w.append(self.add_noise(dataset.w, magnitude))
                all_p.append(self.add_noise(dataset.p, magnitude))
                # phi 不加噪声（保持体积分数约束）
                all_phi.append(dataset.phi.copy())
        
        augmentation_factor = len(all_u)
        
        # 合并数据
        augmented = SyntheticDataset(
            mesh_info=dataset.mesh_info,
            t=dataset.t,
            voltages=dataset.voltages,
            u=np.concatenate(all_u, axis=0),
            v=np.concatenate(all_v, axis=0),
            w=np.concatenate(all_w, axis=0),
            p=np.concatenate(all_p, axis=0),
            phi=np.concatenate(all_phi, axis=0),
            aperture_ratio=np.tile(dataset.aperture_ratio, (augmentation_factor, 1)),
            contact_angle=np.tile(dataset.contact_angle, (augmentation_factor, 1)),
            metadata={
                **dataset.metadata,
                'augmentation_factor': augmentation_factor,
                'augmentations': augmentations
            }
        )
        
        logger.info(f"数据增强完成: 增强因子 {augmentation_factor}")
        
        return augmented, augmentation_factor
    
    def verify_physical_constraints(self, original: np.ndarray, 
                                    augmented: np.ndarray,
                                    threshold: float = 0.001) -> bool:
        """
        验证增强后数据仍满足物理约束（质量守恒）
        
        Args:
            original: 原始体积分数场
            augmented: 增强后体积分数场
            threshold: 误差阈值
            
        Returns:
            是否满足约束
        """
        original_mass = np.sum(original)
        augmented_mass = np.sum(augmented)
        
        if original_mass > 0:
            error = abs(augmented_mass - original_mass) / original_mass
            return error < threshold
        
        return True


# ============================================================
# DataPreprocessor 类
# ============================================================

class DataPreprocessor:
    """
    数据预处理器
    
    提供归一化、分割和导出功能。
    
    Example:
        >>> preprocessor = DataPreprocessor()
        >>> normalized, params = preprocessor.normalize(data, method='standard')
        >>> train, val, test = preprocessor.split(dataset, train_ratio=0.7)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化预处理器
        
        Args:
            config: 预处理配置
        """
        self.config = config or DEFAULT_CONFIG.get('preprocessing', {})
    
    def normalize(self, data: np.ndarray, 
                  method: str = 'standard') -> Tuple[np.ndarray, NormParams]:
        """
        归一化数据
        
        Args:
            data: 输入数据
            method: 'standard' (零均值单位方差) 或 'minmax' ([0,1] 范围)
            
        Returns:
            (归一化后的数据, 归一化参数)
        """
        if method == 'standard':
            mean = np.mean(data, axis=0, keepdims=True)
            std = np.std(data, axis=0, keepdims=True)
            std = np.where(std == 0, 1, std)  # 避免除零
            normalized = (data - mean) / std
            params = NormParams(method='standard', mean=mean.flatten(), std=std.flatten())
        elif method == 'minmax':
            min_val = np.min(data, axis=0, keepdims=True)
            max_val = np.max(data, axis=0, keepdims=True)
            range_val = max_val - min_val
            range_val = np.where(range_val == 0, 1, range_val)
            normalized = (data - min_val) / range_val
            params = NormParams(method='minmax', min_val=min_val.flatten(), max_val=max_val.flatten())
        else:
            raise ValueError(f"不支持的归一化方法: {method}")
        
        return normalized, params
    
    def denormalize(self, data: np.ndarray, params: NormParams) -> np.ndarray:
        """
        反归一化
        
        Args:
            data: 归一化后的数据
            params: 归一化参数
            
        Returns:
            原始数据
        """
        if params.method == 'standard':
            return data * params.std + params.mean
        elif params.method == 'minmax':
            return data * (params.max_val - params.min_val) + params.min_val
        else:
            raise ValueError(f"不支持的归一化方法: {params.method}")
    
    def split(self, dataset: SyntheticDataset, 
              train_ratio: float = 0.7,
              val_ratio: float = 0.15) -> Tuple[SyntheticDataset, SyntheticDataset, SyntheticDataset]:
        """
        分割数据集，保持时间连续性
        
        Args:
            dataset: 原始数据集
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            
        Returns:
            (训练集, 验证集, 测试集)
        """
        n_voltage = len(dataset.voltages)
        
        # 按电压分割（保持每个电压的时间序列完整）
        n_train = int(n_voltage * train_ratio)
        n_val = int(n_voltage * val_ratio)
        n_test = n_voltage - n_train - n_val
        
        # 确保至少有一个样本
        n_train = max(1, n_train)
        n_val = max(1, n_val) if n_voltage > 1 else 0
        n_test = max(0, n_voltage - n_train - n_val)
        
        # 随机打乱电压索引
        indices = np.random.permutation(n_voltage)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]
        
        def create_subset(idx):
            if len(idx) == 0:
                return None
            return SyntheticDataset(
                mesh_info=dataset.mesh_info,
                t=dataset.t,
                voltages=dataset.voltages[idx],
                u=dataset.u[idx],
                v=dataset.v[idx],
                w=dataset.w[idx],
                p=dataset.p[idx],
                phi=dataset.phi[idx],
                aperture_ratio=dataset.aperture_ratio[idx],
                contact_angle=dataset.contact_angle[idx],
                metadata=dataset.metadata
            )
        
        return create_subset(train_idx), create_subset(val_idx), create_subset(test_idx)
    
    def export_pytorch(self, dataset: SyntheticDataset, output_dir: str):
        """
        导出为 PyTorch 格式
        
        Args:
            dataset: 数据集
            output_dir: 输出目录
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 转换为张量并保存
        data = {
            't': torch.tensor(dataset.t, dtype=torch.float32),
            'voltages': torch.tensor(dataset.voltages, dtype=torch.float32),
            'u': torch.tensor(dataset.u, dtype=torch.float32),
            'v': torch.tensor(dataset.v, dtype=torch.float32),
            'w': torch.tensor(dataset.w, dtype=torch.float32),
            'p': torch.tensor(dataset.p, dtype=torch.float32),
            'phi': torch.tensor(dataset.phi, dtype=torch.float32),
            'aperture_ratio': torch.tensor(dataset.aperture_ratio, dtype=torch.float32),
            'contact_angle': torch.tensor(dataset.contact_angle, dtype=torch.float32)
        }
        
        torch.save(data, output_dir / 'data.pt')
        
        # 保存元数据
        self._save_metadata(dataset, output_dir)
        
        # 计算并保存校验和
        checksum = self.compute_checksum(dataset.u)
        with open(output_dir / 'checksum.txt', 'w') as f:
            f.write(checksum)
        
        logger.info(f"PyTorch 数据已导出: {output_dir}")
    
    def export_numpy(self, dataset: SyntheticDataset, output_dir: str):
        """
        导出为 NumPy 格式
        
        Args:
            dataset: 数据集
            output_dir: 输出目录
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        np.savez(
            output_dir / 'data.npz',
            t=dataset.t,
            voltages=dataset.voltages,
            u=dataset.u,
            v=dataset.v,
            w=dataset.w,
            p=dataset.p,
            phi=dataset.phi,
            aperture_ratio=dataset.aperture_ratio,
            contact_angle=dataset.contact_angle
        )
        
        # 保存元数据
        self._save_metadata(dataset, output_dir)
        
        # 计算并保存校验和
        checksum = self.compute_checksum(dataset.u)
        with open(output_dir / 'checksum.txt', 'w') as f:
            f.write(checksum)
        
        logger.info(f"NumPy 数据已导出: {output_dir}")
    
    def _save_metadata(self, dataset: SyntheticDataset, output_dir: Path):
        """保存元数据"""
        metadata = {
            'dimensions': dataset.get_dimensions(),
            'mesh_info': dataset.mesh_info,
            'generation_metadata': dataset.metadata,
            'export_time': datetime.now().isoformat()
        }
        
        with open(output_dir / 'metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    def compute_checksum(self, data: np.ndarray) -> str:
        """
        计算数据校验和
        
        Args:
            data: 数据数组
            
        Returns:
            MD5 校验和字符串
        """
        return hashlib.md5(data.tobytes()).hexdigest()



# ============================================================
# DataVisualizer 类
# ============================================================

class DataVisualizer:
    """
    数据可视化器
    
    生成数据质量验证图表。
    
    Example:
        >>> visualizer = DataVisualizer('outputs/plots')
        >>> visualizer.plot_voltage_aperture(dataset, 'voltage_aperture.png')
    """
    
    def __init__(self, output_dir: str = 'outputs/pinn_data'):
        """
        初始化可视化器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_voltage_aperture(self, dataset: SyntheticDataset, save_path: str):
        """
        绘制电压-开口率曲线
        
        Args:
            dataset: 数据集
            save_path: 保存路径
        """
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 取稳态值（最后时间点）
        steady_aperture = dataset.aperture_ratio[:, -1]
        
        ax.plot(dataset.voltages, steady_aperture * 100, 'bo-', linewidth=2, markersize=8)
        ax.set_xlabel('Voltage (V)', fontsize=12)
        ax.set_ylabel('Aperture Ratio (%)', fontsize=12)
        ax.set_title('Voltage vs Steady-State Aperture Ratio', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, max(dataset.voltages) * 1.1)
        ax.set_ylim(0, 100)
        
        plt.tight_layout()
        
        full_path = self.output_dir / save_path
        plt.savefig(full_path, dpi=150)
        plt.close()
        
        logger.info(f"电压-开口率图已保存: {full_path}")
    
    def plot_time_response(self, dataset: SyntheticDataset, voltage: float, save_path: str):
        """
        绘制时间响应曲线
        
        Args:
            dataset: 数据集
            voltage: 目标电压
            save_path: 保存路径
        """
        import matplotlib.pyplot as plt
        
        # 找到最接近的电压索引
        v_idx = np.argmin(np.abs(dataset.voltages - voltage))
        actual_voltage = dataset.voltages[v_idx]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        t_ms = dataset.t * 1000  # 转换为 ms
        
        # 开口率
        ax1.plot(t_ms, dataset.aperture_ratio[v_idx] * 100, 'b-', linewidth=2)
        ax1.set_ylabel('Aperture Ratio (%)', fontsize=12)
        ax1.set_title(f'Dynamic Response at {actual_voltage:.1f}V', fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 100)
        
        # 接触角
        ax2.plot(t_ms, dataset.contact_angle[v_idx], 'r-', linewidth=2)
        ax2.set_xlabel('Time (ms)', fontsize=12)
        ax2.set_ylabel('Contact Angle (°)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        full_path = self.output_dir / save_path
        plt.savefig(full_path, dpi=150)
        plt.close()
        
        logger.info(f"时间响应图已保存: {full_path}")
    
    def plot_spatial_distribution(self, points: np.ndarray, save_path: str):
        """
        绘制采样点空间分布
        
        Args:
            points: 点坐标数组 (n_points, 4) - [x, y, z, t]
            save_path: 保存路径
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(12, 5))
        
        # 3D 散点图
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(points[:, 0] * 1e6, points[:, 1] * 1e6, points[:, 2] * 1e6, 
                   c=points[:, 3] * 1000, cmap='viridis', s=1, alpha=0.5)
        ax1.set_xlabel('X (μm)')
        ax1.set_ylabel('Y (μm)')
        ax1.set_zlabel('Z (μm)')
        ax1.set_title('Spatial Distribution')
        
        # XY 投影
        ax2 = fig.add_subplot(122)
        scatter = ax2.scatter(points[:, 0] * 1e6, points[:, 1] * 1e6, 
                             c=points[:, 2] * 1e6, cmap='viridis', s=1, alpha=0.5)
        ax2.set_xlabel('X (μm)')
        ax2.set_ylabel('Y (μm)')
        ax2.set_title('XY Projection (color = Z)')
        ax2.set_aspect('equal')
        plt.colorbar(scatter, ax=ax2, label='Z (μm)')
        
        plt.tight_layout()
        
        full_path = self.output_dir / save_path
        plt.savefig(full_path, dpi=150)
        plt.close()
        
        logger.info(f"空间分布图已保存: {full_path}")
    
    def plot_histograms(self, dataset: SyntheticDataset, save_path: str):
        """
        绘制数据分布直方图
        
        Args:
            dataset: 数据集
            save_path: 保存路径
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 电压分布
        axes[0, 0].hist(dataset.voltages, bins=20, edgecolor='black')
        axes[0, 0].set_xlabel('Voltage (V)')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Voltage Distribution')
        
        # 开口率分布
        axes[0, 1].hist(dataset.aperture_ratio.flatten(), bins=50, edgecolor='black')
        axes[0, 1].set_xlabel('Aperture Ratio')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Aperture Ratio Distribution')
        
        # 接触角分布
        axes[0, 2].hist(dataset.contact_angle.flatten(), bins=50, edgecolor='black')
        axes[0, 2].set_xlabel('Contact Angle (°)')
        axes[0, 2].set_ylabel('Count')
        axes[0, 2].set_title('Contact Angle Distribution')
        
        # 速度分布
        axes[1, 0].hist(dataset.u.flatten(), bins=50, edgecolor='black', alpha=0.7, label='u')
        axes[1, 0].hist(dataset.v.flatten(), bins=50, edgecolor='black', alpha=0.7, label='v')
        axes[1, 0].hist(dataset.w.flatten(), bins=50, edgecolor='black', alpha=0.7, label='w')
        axes[1, 0].set_xlabel('Velocity')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Velocity Distribution')
        axes[1, 0].legend()
        
        # 压力分布
        axes[1, 1].hist(dataset.p.flatten(), bins=50, edgecolor='black')
        axes[1, 1].set_xlabel('Pressure')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Pressure Distribution')
        
        # 体积分数分布
        axes[1, 2].hist(dataset.phi.flatten(), bins=50, edgecolor='black')
        axes[1, 2].set_xlabel('Volume Fraction (φ)')
        axes[1, 2].set_ylabel('Count')
        axes[1, 2].set_title('Volume Fraction Distribution')
        
        plt.tight_layout()
        
        full_path = self.output_dir / save_path
        plt.savefig(full_path, dpi=150)
        plt.close()
        
        logger.info(f"直方图已保存: {full_path}")


# ============================================================
# PINNDataGenerator 主类
# ============================================================

class PINNDataGenerator:
    """
    PINN 数据生成器主类
    
    整合所有组件，提供统一的数据生成接口。
    
    Example:
        >>> generator = PINNDataGenerator()
        >>> generator.generate_training_data()
        >>> generator.generate_physics_points()
        >>> generator.export_all('outputs/pinn_data')
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化生成器
        
        Args:
            config_path: 配置文件路径，None 则使用默认配置
        """
        self.config = load_config(config_path)
        
        # 初始化组件
        self.simulator = FlowFieldSimulator()
        self.synthetic_generator = SyntheticDataGenerator(self.simulator, self.config)
        
        mesh_config = self.config.get('synthetic_data', {}).get('mesh', {})
        mesh_generator = MeshGenerator()
        self.mesh = mesh_generator.generate_structured_mesh(
            nx=mesh_config.get('nx', 32),
            ny=mesh_config.get('ny', 32),
            nz=mesh_config.get('nz', 16)
        )
        self.physics_sampler = PhysicsPointsSampler(self.mesh, self.config)
        
        self.exp_loader = ExperimentalDataLoader()
        self.augmenter = DataAugmenter(self.config.get('augmentation', {}))
        self.preprocessor = DataPreprocessor(self.config.get('preprocessing', {}))
        self.visualizer = DataVisualizer(self.config.get('export', {}).get('output_dir', 'outputs/pinn_data'))
        
        # 数据存储
        self.synthetic_data: Optional[SyntheticDataset] = None
        self.physics_points: Optional[PhysicsConstraintPoints] = None
        self.experimental_data: List[ExperimentalDataset] = []
    
    def generate_training_data(self, voltages: Optional[List[float]] = None,
                               time_points: Optional[np.ndarray] = None) -> SyntheticDataset:
        """
        生成合成训练数据
        
        Args:
            voltages: 电压列表
            time_points: 时间点数组
            
        Returns:
            SyntheticDataset
        """
        self.synthetic_data = self.synthetic_generator.generate(voltages, time_points)
        return self.synthetic_data
    
    def generate_physics_points(self, phi: Optional[np.ndarray] = None,
                                voltage: float = 30.0,
                                theta: float = 85.0) -> PhysicsConstraintPoints:
        """
        生成物理约束点
        
        Args:
            phi: 体积分数场，None 则使用默认初始条件
            voltage: 电压
            theta: 接触角
            
        Returns:
            PhysicsConstraintPoints
        """
        if phi is None:
            # 使用默认初始条件
            mesh_generator = MeshGenerator()
            phi = mesh_generator.get_initial_phi(self.mesh)
        
        points_config = self.config.get('physics_points', {})
        
        self.physics_points = self.physics_sampler.generate_constraint_points(
            n_domain=points_config.get('n_domain', 10000),
            n_boundary=points_config.get('n_boundary', 2000),
            n_interface=points_config.get('n_interface', 1000),
            n_contact=points_config.get('n_contact', 500),
            phi=phi,
            voltage=voltage,
            theta=theta
        )
        
        return self.physics_points
    
    def load_experimental_data(self, path: str) -> ExperimentalDataset:
        """
        加载实验数据
        
        Args:
            path: 数据文件路径
            
        Returns:
            ExperimentalDataset
        """
        path = Path(path)
        
        if path.suffix.lower() == '.json':
            data = self.exp_loader.load_json(str(path))
        elif path.suffix.lower() == '.csv':
            data = self.exp_loader.load_csv(str(path))
        else:
            raise FileFormatError(
                f"不支持的文件格式: {path.suffix}",
                supported_formats=['.json', '.csv']
            )
        
        # 验证数据
        result = self.exp_loader.validate(data)
        if not result.is_valid:
            raise ValidationError(
                f"实验数据验证失败: {result.errors}",
                failed_checks=result.errors
            )
        
        self.experimental_data.append(data)
        return data
    
    def export_all(self, output_dir: str, format: str = 'pytorch'):
        """
        导出所有数据
        
        Args:
            output_dir: 输出目录
            format: 导出格式 ('pytorch' 或 'numpy')
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 导出合成数据
        if self.synthetic_data is not None:
            if format == 'pytorch':
                self.preprocessor.export_pytorch(self.synthetic_data, str(output_dir / 'synthetic'))
            else:
                self.preprocessor.export_numpy(self.synthetic_data, str(output_dir / 'synthetic'))
        
        # 导出物理约束点
        if self.physics_points is not None:
            points_dir = output_dir / 'physics_points'
            points_dir.mkdir(exist_ok=True)
            
            np.savez(
                points_dir / 'points.npz',
                domain=self.physics_points.domain_points,
                boundary=self.physics_points.boundary_points,
                interface=self.physics_points.interface_points,
                contact_line=self.physics_points.contact_line_points,
                boundary_types=self.physics_points.boundary_types
            )
            
            with open(points_dir / 'params.json', 'w') as f:
                json.dump({
                    'voltage': self.physics_points.voltage,
                    'contact_angle': self.physics_points.contact_angle
                }, f, indent=2)
        
        # 保存配置
        save_config(self.config, str(output_dir / 'config.json'))
        
        logger.info(f"所有数据已导出: {output_dir}")



# ============================================================
# 便捷函数
# ============================================================

def generate_synthetic_data(voltages: Optional[List[float]] = None,
                           time_points: Optional[np.ndarray] = None,
                           config_path: Optional[str] = None) -> SyntheticDataset:
    """
    生成合成训练数据（便捷函数）
    
    Args:
        voltages: 电压列表
        time_points: 时间点数组
        config_path: 配置文件路径
        
    Returns:
        SyntheticDataset
    """
    generator = PINNDataGenerator(config_path)
    return generator.generate_training_data(voltages, time_points)


def generate_physics_points(phi: Optional[np.ndarray] = None,
                           voltage: float = 30.0,
                           theta: float = 85.0,
                           config_path: Optional[str] = None) -> PhysicsConstraintPoints:
    """
    生成物理约束点（便捷函数）
    
    Args:
        phi: 体积分数场
        voltage: 电压
        theta: 接触角
        config_path: 配置文件路径
        
    Returns:
        PhysicsConstraintPoints
    """
    generator = PINNDataGenerator(config_path)
    return generator.generate_physics_points(phi, voltage, theta)


def create_experimental_template(output_dir: str = 'data/templates'):
    """
    创建实验数据模板（便捷函数）
    
    Args:
        output_dir: 输出目录
    """
    loader = ExperimentalDataLoader()
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    loader.generate_template_json(str(output_dir / 'experiment_template.json'))
    loader.generate_template_csv(str(output_dir / 'experiment_template.csv'))
    
    print(f"✅ 实验数据模板已创建: {output_dir}")
    print(f"   - experiment_template.json")
    print(f"   - experiment_template.csv")


def visualize_data(dataset: SyntheticDataset, output_dir: str = 'outputs/pinn_data'):
    """
    可视化数据（便捷函数）
    
    Args:
        dataset: 数据集
        output_dir: 输出目录
    """
    visualizer = DataVisualizer(output_dir)
    
    visualizer.plot_voltage_aperture(dataset, 'voltage_aperture.png')
    visualizer.plot_time_response(dataset, 30.0, 'time_response_30V.png')
    visualizer.plot_histograms(dataset, 'histograms.png')
    
    print(f"✅ 可视化图表已保存: {output_dir}")


# ============================================================
# 演示函数
# ============================================================

def demo_pinn_data_generation():
    """
    演示 PINN 数据生成功能
    """
    print("=" * 60)
    print("EWP PINN 数据生成演示")
    print("=" * 60)
    
    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    # 1. 创建实验数据模板
    print("\n📝 1. 创建实验数据模板...")
    create_experimental_template('data/templates')
    
    # 2. 生成物理约束点
    print("\n🎯 2. 生成物理约束点...")
    generator = PINNDataGenerator()
    
    # 创建初始体积分数场
    mesh_generator = MeshGenerator()
    mesh = mesh_generator.generate_structured_mesh(nx=16, ny=16, nz=8)
    phi = mesh_generator.get_initial_phi(mesh)
    
    # 使用较小的采样数进行演示
    sampler = PhysicsPointsSampler(mesh)
    points = sampler.generate_constraint_points(
        n_domain=1000,
        n_boundary=200,
        n_interface=100,
        n_contact=50,
        phi=phi,
        voltage=30.0,
        theta=85.0
    )
    
    print(f"   域内点: {len(points.domain_points)}")
    print(f"   边界点: {len(points.boundary_points)}")
    print(f"   界面点: {len(points.interface_points)}")
    print(f"   接触线点: {len(points.contact_line_points)}")
    print(f"   电压: {points.voltage}V")
    print(f"   接触角: {points.contact_angle}°")
    
    # 3. 可视化采样点
    print("\n📊 3. 可视化采样点...")
    visualizer = DataVisualizer('outputs/pinn_data_demo')
    all_points = points.get_all_points()
    if len(all_points) > 0:
        visualizer.plot_spatial_distribution(all_points, 'sampling_points.png')
    
    # 4. 演示数据增强
    print("\n🔄 4. 演示数据增强...")
    augmenter = DataAugmenter()
    
    # 创建示例 3D 数据
    sample_data = np.random.rand(8, 8, 4)
    
    rotated = augmenter.rotate(sample_data, 90)
    reflected = augmenter.reflect(sample_data, 'x')
    noisy = augmenter.add_noise(sample_data, 0.01)
    
    print(f"   原始数据形状: {sample_data.shape}")
    print(f"   旋转后形状: {rotated.shape}")
    print(f"   翻转后形状: {reflected.shape}")
    print(f"   加噪后形状: {noisy.shape}")
    
    # 5. 演示归一化
    print("\n📐 5. 演示归一化...")
    preprocessor = DataPreprocessor()
    
    sample_1d = np.random.rand(100) * 10 + 5
    normalized, params = preprocessor.normalize(sample_1d, method='standard')
    recovered = preprocessor.denormalize(normalized, params)
    
    print(f"   原始数据: mean={np.mean(sample_1d):.2f}, std={np.std(sample_1d):.2f}")
    print(f"   归一化后: mean={np.mean(normalized):.2f}, std={np.std(normalized):.2f}")
    print(f"   恢复误差: {np.max(np.abs(recovered - sample_1d)):.2e}")
    
    # 6. 打印使用说明
    print("\n" + "=" * 60)
    print("使用说明")
    print("=" * 60)
    print("""
# 生成合成训练数据
from ewp_pinn_data_generator import PINNDataGenerator

generator = PINNDataGenerator()
dataset = generator.generate_training_data(
    voltages=[0, 10, 20, 30],
    time_points=np.linspace(0, 0.02, 50)
)

# 生成物理约束点
points = generator.generate_physics_points(voltage=30, theta=85)

# 加载实验数据
exp_data = generator.load_experimental_data('data/experiment.json')

# 导出所有数据
generator.export_all('outputs/pinn_training', format='pytorch')

# 创建实验数据模板
from ewp_pinn_data_generator import create_experimental_template
create_experimental_template('data/templates')
""")
    
    print("\n✅ 演示完成!")
    print(f"   输出目录: outputs/pinn_data_demo/")
    print(f"   模板目录: data/templates/")


# ============================================================
# 主程序入口
# ============================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='EWP PINN 数据生成器')
    parser.add_argument('--demo', action='store_true', help='运行演示')
    parser.add_argument('--template', action='store_true', help='生成实验数据模板')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--output', type=str, default='outputs/pinn_data', help='输出目录')
    
    args = parser.parse_args()
    
    if args.demo:
        demo_pinn_data_generation()
    elif args.template:
        create_experimental_template(args.output)
    else:
        # 默认运行演示
        demo_pinn_data_generation()
