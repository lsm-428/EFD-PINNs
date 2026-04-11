#!/usr/bin/env python3
"""
EWP 开口率模型
==============

职责：接触角 → 油墨分布 → 开口率 → 可视化

与 HybridPredictor 的分工：
- HybridPredictor: 电压 → 接触角（Young-Lippmann + 动力学）
- ApertureModel: 接触角 → 开口率 + 油墨分布 + 电容计算 + 可视化

物理原理：
- 无电压时，油墨平铺在像素底部（开口率=0）
- 施加电压后，电润湿驱动极性液体铺展（水推油走）
- 油墨被推到像素边缘，中心形成透明区域
- 开口率 = 透明区域面积 / 像素面积

使用方法：
    from src.models.aperture_model import EnhancedApertureModel
    
    model = EnhancedApertureModel()
    result = model.predict_enhanced(voltage=30)
    print(f"开口率: {result['aperture_percent']:.1f}%")

作者: EFD-PINNs Team
日期: 2025-12-02
"""

import numpy as np
from typing import Tuple, Dict, Optional, Any, List
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import json
import logging
from pathlib import Path

# 导入统一物理配置
try:
    from src.config import PHYSICS, get_physics_config
    _HAS_CONFIG = True
except ImportError:
    _HAS_CONFIG = False
    PHYSICS = {}

logger = logging.getLogger(__name__)


class ApertureModel:
    """
    开口率模型：接触角 → 油墨分布 → 开口率
    """
    
    def __init__(self):
        """初始化开口率模型"""
        # 从统一配置获取参数
        if _HAS_CONFIG:
            config = get_physics_config()
            Lx = config.Lx
            h_ink = config.h_ink
            theta0 = config.theta0
            theta_min = config.theta_min
            epsilon_0 = config.epsilon_0
            epsilon_h = config.epsilon_h
            d_dielectric = config.d_dielectric
            d_hydrophobic = config.d_hydrophobic
            Lz = config.Lz
        else:
            Lx = 174e-6
            h_ink = 3e-6
            theta0 = 120.0
            theta_min = 60.0
            epsilon_0 = 8.854e-12
            epsilon_h = 1.9
            d_dielectric = 4e-7
            d_hydrophobic = 4e-7
            Lz = 20e-6
        
        # 像素几何参数
        self.pixel_size = Lx  # 内沿尺寸 (m)
        self.pixel_area = self.pixel_size ** 2  # 像素面积 (m²)
        self.R_pixel = self.pixel_size / 2  # 等效半径 (m)
        
        # 油墨参数
        self.ink_thickness = h_ink  # 油墨厚度 (m)
        self.ink_volume = self.ink_thickness * self.pixel_area  # 油墨体积 (m³)
        
        # 接触角参数
        self.theta_0 = theta0  # 初始接触角 (度)
        self.theta_min = theta_min  # 最小接触角 (度)，物理下限
        
        # 表面张力参数
        self.gamma_water = 0.072  # 水表面张力 (N/m)
        self.gamma_ink = 0.030   # 油墨表面张力 (N/m)
        self.gamma_ink_water = 0.020  # 油墨-水界面张力 (N/m)，估计值
        
        # 介电参数（用于电容计算）
        # EWP 电容结构：介电层 + 疏水层 + 流体层（三层串联）
        self.epsilon_0 = epsilon_0  # 真空介电常数 (F/m)
        
        # 介电层 (SU-8 光刻胶)
        self.epsilon_dielectric = 3.0  # SU-8 相对介电常数
        self.d_dielectric = d_dielectric  # 介电层厚度 (m) = 400nm
        
        # 疏水层 (Teflon AF)
        self.epsilon_hydrophobic = epsilon_h  # Teflon AF 相对介电常数
        self.d_hydrophobic = d_hydrophobic  # 疏水层厚度 (m) = 400nm
        
        # 流体层
        self.epsilon_ink = 3.0  # 油墨相对介电常数（非极性）
        self.epsilon_polar = 80.0  # 极性液体相对介电常数（水基）
        self.d_fluid = Lz  # 流体层厚度 (m) = 20μm
        
        logger.info(f"ApertureModel 初始化完成")
        logger.info(f"   像素尺寸: {self.pixel_size*1e6:.0f}×{self.pixel_size*1e6:.0f} μm")
        logger.info(f"   油墨体积: {self.ink_volume*1e18:.0f} μm³")
    
    def calculate_capacitance(self, aperture_ratio: float) -> float:
        """
        计算像素电容（随开口率变化）
        
        关键物理：极性液体是导电的！
        
        未开口区域（油墨覆盖）：
        - 电容 = 油墨 + SU-8 + Teflon 串联
        - 1/C_ink_region = 1/C_ink + 1/C_SU8 + 1/C_Teflon
        
        开口区域（极性液体覆盖）：
        - 极性液体导电，直接接触 Teflon 表面
        - 电容 = SU-8 + Teflon 串联（没有流体层！）
        - 1/C_open_region = 1/C_SU8 + 1/C_Teflon
        
        总电容 = 两个区域并联
        C_total = (1-η) × C_ink_region + η × C_open_region
        
        Args:
            aperture_ratio: 开口率 (0-1)
        
        Returns:
            像素电容 (F)
        """
        # 介电层电容密度 (F/m²) - SU-8
        C_d_density = self.epsilon_0 * self.epsilon_dielectric / self.d_dielectric
        
        # 疏水层电容密度 (F/m²) - Teflon
        C_h_density = self.epsilon_0 * self.epsilon_hydrophobic / self.d_hydrophobic
        
        # 油墨层电容密度 (F/m²)
        C_ink_density = self.epsilon_0 * self.epsilon_ink / self.d_fluid
        
        # 未开口区域：油墨 + SU-8 + Teflon 串联
        C_ink_region = 1.0 / (1.0 / C_d_density + 1.0 / C_h_density + 1.0 / C_ink_density)
        
        # 开口区域：SU-8 + Teflon 串联（极性液体导电，不参与电容）
        C_open_region = 1.0 / (1.0 / C_d_density + 1.0 / C_h_density)
        
        # 两个区域并联（面积加权）
        C_total_density = (1 - aperture_ratio) * C_ink_region + aperture_ratio * C_open_region
        
        # 总电容
        C_total = C_total_density * self.pixel_area
        
        return C_total
    
    def capacitance_ratio(self, aperture_ratio: float) -> float:
        """
        计算电容比（相对于初始状态）
        
        C_ratio = C(η) / C(0)
        
        这个比值反映了电容变化对电润湿力的增强效应
        
        Args:
            aperture_ratio: 开口率 (0-1)
        
        Returns:
            电容比 (≥1)
        """
        C_0 = self.calculate_capacitance(0.0)
        C_eta = self.calculate_capacitance(aperture_ratio)
        return C_eta / C_0
    
    def contact_angle_to_aperture_ratio(self, theta: float) -> float:
        """
        接触角 → 开口率（考虑电容变化的正反馈效应）
        
        物理机制（电润湿驱动）：
        1. 电润湿作用在极性液体上，使其润湿疏水层（Teflon）
        2. 极性液体铺展，将油墨从像素中心挤向边缘/角落
        3. 油墨亲疏水层（底部Teflon），不亲围堰壁（相对亲水）
        4. 油墨被动收缩，贴底形成液滴，不会主动爬墙
        5. 开口率 = 极性液体覆盖的透明区域面积 / 像素面积
        
        电容变化效应（正反馈）：
        - 开口率增加 → 极性液体（导电）覆盖面积增加 → 电容增加
        - 电容增加 → 电润湿力增强 → 开口率进一步增加
        - C_ratio 可从 1 增加到 ~14（η=0→67%）
        - 但 α=0.05 限制了正反馈强度，避免不稳定
        
        实验校准（SU-8 400nm + Teflon 400nm，乙二醇混合液）：
        - 6V 开始有开口 (V_T ≈ 3V)
        - 20V 时开口率 ≈ 67%（Δθ ≈ 4.8°）
        - 20V 以上可能翻墙（油墨被挤压到极限）
        
        Args:
            theta: 接触角 (度)
        
        Returns:
            开口率 (0-1)
        """
        # 直接使用接触角变化量作为驱动力
        theta_change = self.theta_0 - theta  # 角度变化量 (度)
        theta_change = max(0, theta_change)  # 只有正向变化
        
        # 最大开口率（受围堰高度限制）
        # 油墨收缩后堆高，最高不超过围堰
        # η_max = 1 - V_ink / (h_wall * A_pixel)
        wall_height = self.d_fluid  # 使用流体层高度作为围堰高度 (20μm)
        A_ink_min = self.ink_volume / wall_height  # 油墨最小占地面积
        aperture_max = 1 - A_ink_min / self.pixel_area  # 最大开口率 (~85%)
        
        # 使用迭代求解考虑电容正反馈的开口率
        # 
        # 物理模型：
        # η = η_max * tanh(k * Δθ * (1 + α*(C_ratio(η)-1)) / θ_scale)
        # 
        # 其中：
        # - C_ratio(η) = C(η) / C(0) 是电容比
        # - α 是电容反馈强度系数（0-1），控制正反馈的影响程度
        # 
        # 这是一个隐式方程，需要迭代求解
        
        # 从实例属性读取参数（如果有），否则使用默认值
        # 参数校准：6V开始有开口，20V时开口率≈67%（实验值）
        k = getattr(self, 'aperture_k', 0.8)  # 陡度参数
        theta_scale = getattr(self, 'aperture_theta_scale', 6.0)  # 角度缩放因子（度）
        alpha = getattr(self, 'aperture_alpha', 0.05)  # 电容反馈强度
        
        # 迭代求解（简单不动点迭代）
        eta = 0.0
        for _ in range(10):  # 通常 5-9 次迭代即可收敛
            C_ratio = self.capacitance_ratio(eta)
            # 电容增强因子：1 + α*(C_ratio - 1)
            # 当 η=0 时，C_ratio=1，增强因子=1
            # 当 η=67% 时，C_ratio≈14，增强因子≈1.65
            # α=0.05 限制正反馈强度，避免发散
            enhancement = 1.0 + alpha * (C_ratio - 1.0)
            x = k * theta_change * enhancement / theta_scale
            eta_new = aperture_max * np.tanh(x)
            if abs(eta_new - eta) < 1e-6:
                break
            eta = eta_new
        
        return eta
    
    def aperture_ratio_to_open_radius(self, aperture_ratio: float) -> float:
        """
        开口率 → 透明区域半径
        
        假设透明区域为圆形：
        aperture_ratio = π * r_open² / pixel_area
        
        Args:
            aperture_ratio: 开口率 (0-1)
        
        Returns:
            透明区域半径 (m)
        """
        if aperture_ratio <= 0:
            return 0.0
        
        # r_open = sqrt(aperture_ratio * pixel_area / π)
        r_open = np.sqrt(aperture_ratio * self.pixel_area / np.pi)
        
        # 不能超过像素半径
        r_open = min(r_open, self.R_pixel * 0.95)
        
        return r_open
    
    def calculate_ink_distribution(
        self, 
        theta: float, 
        num_points: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算油墨高度分布 h(r)
        
        假设油墨呈环形分布，高度在径向上变化
        
        Args:
            theta: 接触角 (度)
            num_points: 径向采样点数
        
        Returns:
            (r, h): 半径数组和对应的油墨高度数组
        """
        aperture_ratio = self.contact_angle_to_aperture_ratio(theta)
        r_open = self.aperture_ratio_to_open_radius(aperture_ratio)
        
        # 径向坐标
        r = np.linspace(0, self.R_pixel, num_points)
        h = np.zeros_like(r)
        
        if aperture_ratio <= 0.01:
            # 几乎无开口，油墨均匀分布
            h[:] = self.ink_thickness
        else:
            # 油墨被推到边缘
            # 简化模型：油墨高度在 r_open 到 R_pixel 之间线性增加
            
            # 透明区域内无油墨
            mask_open = r < r_open
            h[mask_open] = 0
            
            # 油墨区域
            mask_ink = r >= r_open
            
            # 计算油墨区域的平均高度（体积守恒）
            ink_area = self.pixel_area - np.pi * r_open**2
            if ink_area > 0:
                h_avg = self.ink_volume / ink_area
                
                # 简化：油墨高度从内到外线性增加
                # h(r) = h_min + (h_max - h_min) * (r - r_open) / (R - r_open)
                # 体积守恒确定 h_min 和 h_max
                
                # 更简单的模型：均匀高度
                h[mask_ink] = h_avg
        
        return r, h
    
    def predict(self, voltage: float, theta: float = None) -> Dict:
        """
        预测给定电压下的开口率和油墨分布
        
        Args:
            voltage: 电压 (V)
            theta: 接触角 (度)，如果为 None 则使用 Young-Lippmann 计算
        
        Returns:
            预测结果字典
        """
        if theta is None:
            # 使用 Young-Lippmann 方程计算接触角
            from src.predictors.hybrid_predictor import HybridPredictor
            from src.config import CONFIG_PATH
            predictor = HybridPredictor(config_path=str(CONFIG_PATH))
            theta = predictor.young_lippmann(voltage)
        
        aperture_ratio = self.contact_angle_to_aperture_ratio(theta)
        r_open = self.aperture_ratio_to_open_radius(aperture_ratio)
        r, h = self.calculate_ink_distribution(theta)
        
        return {
            'voltage': voltage,
            'theta': theta,
            'aperture_ratio': aperture_ratio,
            'aperture_percent': aperture_ratio * 100,
            'r_open': r_open,
            'r_open_um': r_open * 1e6,
            'r': r,
            'h': h,
        }


class EnhancedApertureModel(ApertureModel):
    """
    增强版开口率模型
    
    职责（与 HybridPredictor 分工）：
    - 接触角 → 开口率映射（含电容正反馈）
    - 油墨分布计算（体积守恒）
    - 电容计算（随开口率变化）
    - 可视化方法（油墨剖面、像素俯视图等）
    
    注意：接触角计算委托给 HybridPredictor，本模块不重复实现
    
    在 ApertureModel 基础上添加：
    - 电容器充电动力学 (τ_RC)
    - 从配置文件加载参数
    - 改进的油墨分布计算（体积守恒）
    - 可视化和验证方法
    """
    
    def __init__(
        self,
        config_path: str = None,
        predictor: Optional['HybridPredictor'] = None,
        tau_rc: float = 0.1e-3  # RC 时间常数，默认 0.1ms
    ):
        """
        初始化增强模型
        
        Args:
            config_path: 配置文件路径（默认使用 CONFIG_PATH）
            predictor: HybridPredictor 实例（可选）
            tau_rc: 电容器充电时间常数 (s)
        """
        super().__init__()
        
        # 使用统一配置路径
        if config_path is None:
            from src.config import CONFIG_PATH
            config_path = str(CONFIG_PATH)
        
        self.config_path = config_path
        self.predictor = predictor
        self.tau_rc = tau_rc
        
        # 从配置文件加载参数
        self._load_config(config_path)
        
        # 如果没有提供 predictor，创建一个内部使用的
        self._internal_predictor = None

        logger.info(f"✅ EnhancedApertureModel 初始化完成")
        logger.info(f"   τ_RC: {self.tau_rc*1e3:.2f} ms")
        logger.info(f"   配置文件: {config_path}")
    
    def _load_config(self, config_path: str) -> None:
        """
        从 JSON 配置文件加载参数
        
        Args:
            config_path: 配置文件路径
        """
        # 尝试多个可能的路径
        possible_paths = [
            config_path,
            f"config/{config_path}",
            Path(config_path).name,
            f"config/{Path(config_path).name}",
        ]
        
        found_path = None
        for p in possible_paths:
            if Path(p).exists():
                found_path = p
                break

        if found_path is None:
            logger.warning(f"⚠️ 配置文件不存在: {config_path}，使用默认参数")
            logger.warning(f"   {Path(config_path).absolute()}")
            self.config = {}
            return
        
        config_path = found_path
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # 从配置更新材料参数
        materials = self.config.get('materials', {})
        self.theta_0 = materials.get('theta0', self.theta_0)
        self.theta_wall = materials.get('theta_wall', 70.0)
        self.epsilon_r = materials.get('epsilon_r', 4.0)
        self.gamma = materials.get('gamma', 0.072)
        self.d = materials.get('dielectric_thickness', 4e-7)
        
        # 从配置更新动力学参数
        data_config = self.config.get('data', {})
        dynamics = data_config.get('dynamics_params', {})
        self.tau = dynamics.get('tau', 0.005)
        self.zeta = dynamics.get('zeta', 0.8)
        
        # 从配置更新几何参数
        geometry = self.config.get('geometry', {})
        if 'Lx' in geometry:
            self.pixel_size = geometry['Lx']
            self.pixel_area = self.pixel_size ** 2
            self.R_pixel = self.pixel_size / 2
            self.ink_volume = self.ink_thickness * self.pixel_area
        
        # 从配置更新开口率映射参数
        aperture_mapping = self.config.get('aperture_mapping', {})
        self.aperture_k = aperture_mapping.get('k', 0.8)
        self.aperture_theta_scale = aperture_mapping.get('theta_scale', 6.0)
        self.aperture_alpha = aperture_mapping.get('alpha', 0.05)
        self.aperture_max = aperture_mapping.get('aperture_max', 0.85)
    
    def calibrate_with_experimental_data(
        self,
        data: List[Dict[str, float]],
        k_range: tuple = (0.3, 1.5),
        theta_scale_range: tuple = (3.0, 10.0),
        num_k: int = 25,
        num_theta_scale: int = 25,
    ) -> Dict[str, float]:
        """
        使用稳态开口率实验数据校正开口率映射参数
        
        Args:
            data: 包含实验点的列表，每个元素形如
                  {"voltage": 20.0, "aperture_percent": 67.0}
                  或 {"voltage": 20.0, "aperture_ratio": 0.67}
            k_range: aperture_k 搜索范围
            theta_scale_range: aperture_theta_scale 搜索范围
            num_k: k 方向离散个数
            num_theta_scale: theta_scale 方向离散个数
        
        Returns:
            包含最优参数和误差的字典
        """
        measurements = []
        for item in data:
            if "voltage" not in item:
                continue
            v = float(item["voltage"])
            if "aperture_ratio" in item:
                target = float(item["aperture_ratio"])
            elif "aperture_percent" in item:
                target = float(item["aperture_percent"]) / 100.0
            else:
                continue
            target = max(0.0, min(0.99, target))
            measurements.append((v, target))
        
        if not measurements:
            raise ValueError("实验数据为空或格式不正确")
        
        k_values = np.linspace(k_range[0], k_range[1], num_k)
        theta_scale_values = np.linspace(theta_scale_range[0], theta_scale_range[1], num_theta_scale)
        
        original_k = getattr(self, "aperture_k", 0.8)
        original_theta_scale = getattr(self, "aperture_theta_scale", 6.0)
        
        best_k = original_k
        best_theta_scale = original_theta_scale
        best_error = float("inf")
        
        for k in k_values:
            for theta_scale in theta_scale_values:
                self.aperture_k = float(k)
                self.aperture_theta_scale = float(theta_scale)
                error = 0.0
                for voltage, target_eta in measurements:
                    theta = self.get_contact_angle(voltage)
                    eta = self.contact_angle_to_aperture_ratio(theta)
                    diff = eta - target_eta
                    error += diff * diff
                if error < best_error:
                    best_error = error
                    best_k = float(k)
                    best_theta_scale = float(theta_scale)
        
        self.aperture_k = best_k
        self.aperture_theta_scale = best_theta_scale
        
        if hasattr(self, "config"):
            aperture_mapping = self.config.setdefault("aperture_mapping", {})
            aperture_mapping["k"] = best_k
            aperture_mapping["theta_scale"] = best_theta_scale
        
        return {
            "aperture_k": best_k,
            "aperture_theta_scale": best_theta_scale,
            "mse": best_error / len(measurements),
        }
    
    def theta_eta_from_triad(
        self,
        V_from: float,
        V_to: float,
        t_since: float,
    ) -> Tuple[float, float]:
        predictor = self._get_predictor()
        V_from_val = float(V_from)
        V_to_val = float(V_to)
        t_val = max(0.0, float(t_since))
        if V_to_val >= V_from_val:
            theta_start = predictor.young_lippmann(V_from_val)
            theta_eq = predictor.young_lippmann(V_to_val)
            if t_val <= 0.0:
                theta = theta_start
            else:
                theta = predictor.dynamic_response(t_val, theta_start, theta_eq)
        else:
            theta_start = predictor.young_lippmann(V_from_val)
            theta_eq = predictor.young_lippmann(V_to_val)
            if t_val <= 0.0:
                theta = theta_start
            else:
                theta = predictor.surface_tension_recovery(t_val, theta_start, theta_eq)
        eta = self.contact_angle_to_aperture_ratio(theta)
        return theta, eta
    
    def _validate_inputs(self, voltage: float, time: float = None) -> None:
        """
        验证输入参数
        
        Args:
            voltage: 电压 (V)
            time: 时间 (s)，可选
        
        Raises:
            ValueError: 如果输入参数无效
        """
        if voltage < 0:
            raise ValueError(f"电压不能为负: {voltage}")
        if voltage > 50:
            raise ValueError(f"电压超出范围: {voltage} > 50V")
        if time is not None and time < 0:
            raise ValueError(f"时间不能为负: {time}")
    
    def _get_predictor(self) -> 'HybridPredictor':
        """
        获取 HybridPredictor 实例
        
        Returns:
            HybridPredictor 实例
        """
        if self.predictor is not None:
            return self.predictor
        
        if self._internal_predictor is None:
            from src.predictors.hybrid_predictor import HybridPredictor
            # 确保使用正确的配置路径
            config_path = self.config_path
            if not Path(config_path).exists() and not config_path.startswith('config/'):
                config_path = f'config/{config_path}'
            self._internal_predictor = HybridPredictor(
                config_path=config_path,
                use_model_for_steady_state=False
            )
        
        return self._internal_predictor
    
    # === 电容器充电模型 ===
    
    def effective_voltage(self, V_target: float, t: float) -> float:
        """
        计算有效电压（电容器充电模型）
        
        V_eff(t) = V_target × (1 - exp(-t/τ_RC))
        
        物理背景：
        - EWP 是平板电容器结构
        - 电压施加后，电容器需要时间充电
        - τ_RC = R_liquid × C_dielectric ≈ 0.1ms
        
        Args:
            V_target: 目标电压 (V)
            t: 时间 (s)
        
        Returns:
            有效电压 (V)
        """
        if t <= 0:
            return 0.0
        return V_target * (1 - np.exp(-t / self.tau_rc))
    
    def charging_progress(self, t: float) -> float:
        """
        计算充电进度百分比
        
        progress = (1 - exp(-t/τ_RC)) × 100%
        
        Args:
            t: 时间 (s)
        
        Returns:
            充电进度 (0-100%)
        """
        if t <= 0:
            return 0.0
        progress = (1 - np.exp(-t / self.tau_rc)) * 100
        return min(progress, 100.0)
    
    # === 接触角计算（委托给 HybridPredictor）===
    
    def get_contact_angle(
        self, 
        voltage: float, 
        time: float = None,
        V_initial: float = 0.0,
        t_step: float = 0.0
    ) -> float:
        """
        获取接触角（委托给 HybridPredictor）
        
        本方法是对 HybridPredictor 的简单封装，避免重复实现。
        如需更多控制，请直接使用 HybridPredictor。
        
        Args:
            voltage: 电压 (V)
            time: 时间 (s)，可选。如果提供则计算动态响应
            V_initial: 初始电压 (V)，用于动态响应
            t_step: 电压阶跃时间 (s)，用于动态响应
        
        Returns:
            接触角 (度)
        """
        predictor = self._get_predictor()
        
        if time is None:
            # 稳态计算
            return predictor.young_lippmann(voltage)
        else:
            # 动态响应计算
            return predictor.predict(
                voltage=voltage,
                time=time,
                V_initial=V_initial,
                t_step=t_step
            )
    
    # === 油墨分布计算 ===
    
    def calculate_ink_distribution_enhanced(
        self, 
        theta: float, 
        num_points: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算满足体积守恒的油墨分布
        
        物理模型：
        1. 透明区域 (r < r_open): h = 0
        2. 油墨区域 (r_open ≤ r ≤ R_pixel): h = h(r)
        3. 体积守恒: ∫∫ h(r) dA = V_ink
        
        简化假设：
        - 轴对称分布
        - 油墨区域高度均匀 (一阶近似)
        
        Args:
            theta: 接触角 (度)
            num_points: 径向采样点数
        
        Returns:
            (r, h): 半径数组和对应的油墨高度数组
        """
        aperture_ratio = self.contact_angle_to_aperture_ratio(theta)
        r_open = self.aperture_ratio_to_open_radius(aperture_ratio)
        
        # 径向坐标 - 确保 r_open 是采样点之一
        if aperture_ratio > 0.01 and r_open > 0:
            # 在透明区域和油墨区域分别采样
            n_open = max(1, int(num_points * r_open / self.R_pixel))
            n_ink = num_points - n_open
            
            r_open_part = np.linspace(0, r_open, n_open, endpoint=False)
            r_ink_part = np.linspace(r_open, self.R_pixel, n_ink)
            r = np.concatenate([r_open_part, r_ink_part])
        else:
            r = np.linspace(0, self.R_pixel, num_points)
        
        h = np.zeros_like(r)
        
        if aperture_ratio <= 0.01:
            # 无开口，油墨均匀分布
            h[:] = self.ink_thickness
        else:
            # 油墨被推到边缘
            # 透明区域内无油墨 (使用 <= 以确保边界正确)
            mask_open = r < r_open
            h[mask_open] = 0
            
            # 油墨区域 (方形像素减去圆形透明区域)
            mask_ink = r >= r_open
            ink_area = self.pixel_area - np.pi * r_open**2
            
            if ink_area > 0:
                # 体积守恒计算平均高度
                h_avg = self.ink_volume / ink_area
                h[mask_ink] = h_avg
        
        return r, h
    
    def verify_volume_conservation(
        self, 
        r: np.ndarray, 
        h: np.ndarray
    ) -> float:
        """
        验证体积守恒，返回误差百分比
        
        对于方形像素，使用面积加权计算：
        - 透明区域是圆形 (面积 = π * r_open²)
        - 油墨区域是方形减去圆形 (面积 = pixel_area - π * r_open²)
        
        Args:
            r: 半径数组 (m)
            h: 油墨高度数组 (m)
        
        Returns:
            体积误差百分比 (%)
        """
        # 找到透明区域边界
        mask_ink = h > 0
        if not np.any(mask_ink):
            # 全透明，体积为0
            return 100.0
        
        # 找到 r_open (透明区域半径)
        r_open = 0.0
        for i in range(len(r)):
            if h[i] > 0:
                r_open = r[i]
                break
        
        # 计算油墨区域面积（方形像素减去圆形透明区域）
        ink_area = self.pixel_area - np.pi * r_open**2
        
        # 计算平均油墨高度
        h_avg = np.mean(h[mask_ink])
        
        # 计算实际体积
        V_actual = h_avg * ink_area
        
        # 计算误差百分比
        if self.ink_volume > 0:
            error_percent = abs(V_actual - self.ink_volume) / self.ink_volume * 100
        else:
            error_percent = 0.0
        
        return error_percent
    
    # === 动态响应 ===
    
    def aperture_step_response(
        self,
        V_start: float = 0.0,
        V_end: float = 30.0,
        duration: float = 0.10,
        t_step: float = 0.002,
        num_points: int = 500
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算开口率阶跃响应
        
        流程：
        1. 计算有效电压 V_eff(t)
        2. 获取接触角 θ(V_eff, t)
        3. 计算开口率 η(θ)
        
        Args:
            V_start: 初始电压 (V)
            V_end: 最终电压 (V)
            duration: 总时长 (s)
            t_step: 阶跃时间 (s)
            num_points: 采样点数
        
        Returns:
            (t, eta): 时间数组和开口率数组
        """
        t = np.linspace(0, duration, num_points)
        eta = np.zeros(num_points)
        
        for i, ti in enumerate(t):
            if ti < t_step:
                # 阶跃前
                theta = self.get_contact_angle(V_start)
            else:
                # 阶跃后，考虑电容器充电
                t_since = ti - t_step
                V_eff = self.effective_voltage(V_end - V_start, t_since) + V_start
                theta = self.get_contact_angle(V_eff, ti, V_initial=V_start, t_step=t_step)
            
            eta[i] = self.contact_angle_to_aperture_ratio(theta)
        
        return t, eta
    
    def aperture_cycle_response(
        self,
        V_target: float,
        duration: float = 0.05,
        t_rise: float = 0.002,
        t_fall: float = 0.03,
        num_points: int = 500
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        predictor = self._get_predictor()
        t, V, theta, eta = predictor.square_wave_response(
            V_low=0.0,
            V_high=V_target,
            duration=duration,
            t_rise=t_rise,
            t_fall=t_fall,
            num_points=num_points,
        )
        return t, V, eta
    
    def get_aperture_metrics(
        self, 
        t: np.ndarray, 
        eta: np.ndarray, 
        t_step: float = 0.002
    ) -> Dict[str, float]:
        """
        计算响应指标 (t_90, t_10, overshoot)
        
        Args:
            t: 时间数组 (s)
            eta: 开口率数组 (0-1)
            t_step: 阶跃时间 (s)
        
        Returns:
            指标字典，包含:
            - t_90: 达到 90% 稳态的时间 (ms)
            - t_10: 达到 10% 稳态的时间 (ms)
            - overshoot_percent: 超调百分比 (%)
            - eta_initial: 初始开口率
            - eta_final: 最终开口率
        """
        # 找到阶跃点
        step_idx = np.searchsorted(t, t_step)
        
        eta_initial = eta[step_idx] if step_idx < len(eta) else eta[0]
        eta_final = eta[-1]
        eta_change = eta_final - eta_initial
        
        # 计算 t_10 和 t_90
        t_10 = np.nan
        t_90 = np.nan
        
        if abs(eta_change) > 0.001:
            eta_10 = eta_initial + 0.1 * eta_change
            eta_90 = eta_initial + 0.9 * eta_change
            
            # 找到达到 10% 的时间
            for i in range(step_idx, len(eta)):
                if (eta_change > 0 and eta[i] >= eta_10) or \
                   (eta_change < 0 and eta[i] <= eta_10):
                    t_10 = (t[i] - t_step) * 1000  # 转换为 ms
                    break
            
            # 找到达到 90% 的时间
            for i in range(step_idx, len(eta)):
                if (eta_change > 0 and eta[i] >= eta_90) or \
                   (eta_change < 0 and eta[i] <= eta_90):
                    t_90 = (t[i] - t_step) * 1000  # 转换为 ms
                    break
        
        # 计算超调
        overshoot_percent = 0.0
        if abs(eta_change) > 0.001:
            eta_after_step = eta[step_idx:]
            if eta_change > 0:
                # 开口率增加，检查是否超过最终值
                eta_max = np.max(eta_after_step)
                if eta_max > eta_final:
                    overshoot_percent = (eta_max - eta_final) / abs(eta_change) * 100
            else:
                # 开口率减少，检查是否低于最终值
                eta_min = np.min(eta_after_step)
                if eta_min < eta_final:
                    overshoot_percent = (eta_final - eta_min) / abs(eta_change) * 100
        
        return {
            't_10_ms': t_10,
            't_90_ms': t_90,
            'overshoot_percent': overshoot_percent,
            'eta_initial': eta_initial,
            'eta_final': eta_final,
            'eta_change': eta_change,
        }
    
    # === 增强预测接口 ===
    
    def predict_enhanced(
        self, 
        voltage: float, 
        time: float = None
    ) -> Dict:
        """
        增强预测，包含电容器充电效应
        
        Args:
            voltage: 电压 (V)
            time: 时间 (s)，可选。如果提供则计算动态响应
        
        Returns:
            预测结果字典，包含:
            - voltage: 输入电压 (V)
            - time: 时间 (s)
            - effective_voltage: 有效电压 (V)
            - charging_progress: 充电进度 (%)
            - theta: 接触角 (度)
            - aperture_ratio: 开口率 (0-1)
            - aperture_percent: 开口率 (%)
            - r_open: 透明区域半径 (m)
            - r_open_um: 透明区域半径 (μm)
            - r: 径向坐标数组 (m)
            - h: 油墨高度数组 (m)
            - volume_error: 体积误差 (%)
        """
        self._validate_inputs(voltage, time)
        
        # 计算有效电压和充电进度
        if time is not None and time > 0:
            V_eff = self.effective_voltage(voltage, time)
            progress = self.charging_progress(time)
        else:
            V_eff = voltage
            progress = 100.0 if voltage > 0 else 0.0
        
        # 获取接触角
        theta = self.get_contact_angle(voltage, time)
        
        # 计算开口率
        aperture_ratio = self.contact_angle_to_aperture_ratio(theta)
        r_open = self.aperture_ratio_to_open_radius(aperture_ratio)
        
        # 计算油墨分布
        r, h = self.calculate_ink_distribution_enhanced(theta)
        volume_error = self.verify_volume_conservation(r, h)
        
        return {
            'voltage': voltage,
            'time': time,
            'effective_voltage': V_eff,
            'charging_progress': progress,
            'theta': theta,
            'aperture_ratio': aperture_ratio,
            'aperture_percent': aperture_ratio * 100,
            'r_open': r_open,
            'r_open_um': r_open * 1e6,
            'r': r,
            'h': h,
            'volume_error': volume_error,
        }
    
    # === 可视化方法 ===
    
    def plot_comparison(
        self, 
        V_list: List[float] = None, 
        save_path: str = None
    ) -> None:
        """
        生成电压-开口率对比图（原模型 vs 增强模型）
        
        Args:
            V_list: 电压列表，默认 [0, 5, 10, 15, 20, 25, 30]
            save_path: 保存路径，如果为 None 则显示图表
        """
        if V_list is None:
            V_list = [0, 5, 10, 15, 20, 25, 30]
        
        # 原模型结果
        base_model = ApertureModel()
        base_apertures = []
        for V in V_list:
            theta = self.get_contact_angle(V)
            eta = base_model.contact_angle_to_aperture_ratio(theta)
            base_apertures.append(eta * 100)
        
        # 增强模型结果
        enhanced_apertures = []
        for V in V_list:
            result = self.predict_enhanced(V)
            enhanced_apertures.append(result['aperture_percent'])
        
        # 绘图
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(V_list, base_apertures, 'b-o', linewidth=2, markersize=8, label='原模型')
        ax.plot(V_list, enhanced_apertures, 'r-s', linewidth=2, markersize=8, label='增强模型')
        ax.set_xlabel('Voltage (V)', fontsize=12)
        ax.set_ylabel('Aperture Ratio (%)', fontsize=12)
        ax.set_title('Voltage vs Aperture Ratio Comparison', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 30)
        ax.set_ylim(0, 80)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            logger.info(f"📊 图表已保存: {save_path}")
        else:
            plt.show()
        plt.close()
    
    def plot_ink_profile(
        self, 
        theta: float, 
        save_path: str = None
    ) -> None:
        """
        生成油墨高度剖面图 (r vs h)
        
        Args:
            theta: 接触角 (度)
            save_path: 保存路径，如果为 None 则显示图表
        """
        r, h = self.calculate_ink_distribution_enhanced(theta)
        aperture = self.contact_angle_to_aperture_ratio(theta)
        r_open = self.aperture_ratio_to_open_radius(aperture)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(r * 1e6, h * 1e6, 'b-', linewidth=2)
        ax.fill_between(r * 1e6, 0, h * 1e6, alpha=0.3, color='blue')
        
        if r_open > 0:
            ax.axvline(r_open * 1e6, color='r', linestyle='--', alpha=0.7, 
                      label=f'透明区域边界 r={r_open*1e6:.1f}μm')
        
        ax.set_xlabel('Radius (μm)', fontsize=12)
        ax.set_ylabel('Ink Height (μm)', fontsize=12)
        ax.set_title(f'Ink Profile at θ={theta:.1f}° (Aperture={aperture*100:.1f}%)', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, self.R_pixel * 1e6)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            logger.info(f"📊 图表已保存: {save_path}")
        else:
            plt.show()
        plt.close()
    
    def plot_aperture_dynamics(
        self, 
        V_start: float = 0.0, 
        V_end: float = 30.0,
        duration: float = 0.10,
        save_path: str = None
    ) -> None:
        """
        生成开口率动态响应图
        
        Args:
            V_start: 初始电压 (V)
            V_end: 最终电压 (V)
            duration: 总时长 (s)
            save_path: 保存路径，如果为 None 则显示图表
        """
        t, eta = self.aperture_step_response(V_start, V_end, duration, t_step=0.002)
        metrics = self.get_aperture_metrics(t, eta, t_step=0.002)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(t * 1000, eta * 100, 'b-', linewidth=2)
        
        # 标记关键点
        ax.axhline(metrics['eta_final'] * 100, color='g', linestyle='--', alpha=0.5, 
                  label=f'稳态 η={metrics["eta_final"]*100:.1f}%')
        ax.axvline(2, color='r', linestyle='--', alpha=0.5, label='阶跃时刻')
        
        if not np.isnan(metrics['t_90_ms']):
            ax.axvline(2 + metrics['t_90_ms'], color='orange', linestyle=':', alpha=0.7,
                      label=f't_90={metrics["t_90_ms"]:.1f}ms')
        
        ax.set_xlabel('Time (ms)', fontsize=12)
        ax.set_ylabel('Aperture Ratio (%)', fontsize=12)
        ax.set_title(f'Aperture Dynamic Response ({V_start}V → {V_end}V)', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            logger.info(f"📊 图表已保存: {save_path}")
        else:
            plt.show()
        plt.close()
    
    def plot_pixel_view(
        self, 
        theta: float, 
        save_path: str = None
    ) -> None:
        """
        生成像素俯视图
        
        Args:
            theta: 接触角 (度)
            save_path: 保存路径，如果为 None 则显示图表
        """
        aperture = self.contact_angle_to_aperture_ratio(theta)
        r_open = self.aperture_ratio_to_open_radius(aperture) * 1e6  # 转换为 μm
        pixel_half = self.pixel_size * 1e6 / 2  # 转换为 μm
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect('equal')
        
        # 像素边界（方形）
        pixel_rect = plt.Rectangle(
            (-pixel_half, -pixel_half), 
            pixel_half * 2, pixel_half * 2, 
            fill=True, facecolor='lightcoral', edgecolor='black', linewidth=2,
            label='油墨区域'
        )
        ax.add_patch(pixel_rect)
        
        # 透明区域（圆形）
        if r_open > 0:
            circle_open = plt.Circle(
                (0, 0), r_open, 
                fill=True, facecolor='white', edgecolor='blue', linewidth=2,
                label='透明区域'
            )
            ax.add_patch(circle_open)
        
        ax.set_xlim(-pixel_half * 1.1, pixel_half * 1.1)
        ax.set_ylim(-pixel_half * 1.1, pixel_half * 1.1)
        ax.set_xlabel('X (μm)', fontsize=12)
        ax.set_ylabel('Y (μm)', fontsize=12)
        ax.set_title(f'Pixel Top View at θ={theta:.1f}° (Aperture={aperture*100:.1f}%)', fontsize=14)
        ax.legend(loc='upper right', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            logger.info(f"📊 图表已保存: {save_path}")
        else:
            plt.show()
        plt.close()

    def visualize_3d(
        self,
        voltage: float,
        time: float = None,
        save_path: str = None,
    ):
        from scripts.visualization.generate_pyvista_3d import create_aperture_model_visualization
        return create_aperture_model_visualization(voltage=voltage, time=time, save_path=save_path)

    # === 验证方法 ===
    
    def validate(self) -> Dict[str, Any]:
        """
        验证模型物理一致性
        
        检查：
        1. 体积守恒误差 < 0.1%
        2. 开口率范围 [0, 1]
        3. 开口率随电压单调递增
        
        Returns:
            验证结果字典，包含:
            - valid: 是否通过所有验证
            - volume_conservation: 体积守恒检查结果
            - aperture_range: 开口率范围检查结果
            - monotonicity: 单调性检查结果
            - errors: 错误列表
        """
        errors = []
        
        # 1. 体积守恒检查
        volume_results = {'passed': True, 'max_error': 0.0, 'failed_angles': []}
        test_angles = [60, 70, 80, 90, 100, 110, 120]
        for theta in test_angles:
            r, h = self.calculate_ink_distribution_enhanced(theta)
            error = self.verify_volume_conservation(r, h)
            volume_results['max_error'] = max(volume_results['max_error'], error)
            if error >= 0.1:
                volume_results['passed'] = False
                volume_results['failed_angles'].append((theta, error))
        
        if not volume_results['passed']:
            errors.append(f"体积守恒失败: 最大误差 {volume_results['max_error']:.4f}%")
        
        # 2. 开口率范围检查
        range_results = {'passed': True, 'min': 1.0, 'max': 0.0, 'violations': []}
        test_voltages = [0, 5, 10, 15, 20, 25, 30]
        for V in test_voltages:
            theta = self.get_contact_angle(V)
            eta = self.contact_angle_to_aperture_ratio(theta)
            range_results['min'] = min(range_results['min'], eta)
            range_results['max'] = max(range_results['max'], eta)
            if eta < 0 or eta > 1:
                range_results['passed'] = False
                range_results['violations'].append((V, eta))
        
        if not range_results['passed']:
            errors.append(f"开口率范围违规: {range_results['violations']}")
        
        # 3. 单调性检查
        mono_results = {'passed': True, 'violations': []}
        prev_eta = -1
        for V in test_voltages:
            theta = self.get_contact_angle(V)
            eta = self.contact_angle_to_aperture_ratio(theta)
            if eta < prev_eta - 1e-10:  # 允许小的数值误差
                mono_results['passed'] = False
                mono_results['violations'].append((V, eta, prev_eta))
            prev_eta = eta
        
        if not mono_results['passed']:
            errors.append(f"单调性违规: {mono_results['violations']}")
        
        return {
            'valid': len(errors) == 0,
            'volume_conservation': volume_results,
            'aperture_range': range_results,
            'monotonicity': mono_results,
            'errors': errors,
        }
    
    # === 配置管理 ===
    
    def get_config(self) -> Dict:
        """
        返回所有参数的字典
        
        Returns:
            配置字典
        """
        return {
            'config_path': self.config_path,
            'tau_rc': self.tau_rc,
            'pixel_size': self.pixel_size,
            'ink_thickness': self.ink_thickness,
            'theta_0': self.theta_0,
            'theta_min': self.theta_min,
            'theta_wall': getattr(self, 'theta_wall', 70.0),
            'epsilon_r': getattr(self, 'epsilon_r', 4.0),
            'gamma': getattr(self, 'gamma', 0.072),
            'd': getattr(self, 'd', 4e-7),
            'tau': getattr(self, 'tau', 0.005),
            'zeta': getattr(self, 'zeta', 0.8),
        }
    
    def save_config(self, path: str) -> None:
        """
        保存配置为 JSON 文件
        
        Args:
            path: 保存路径
        """
        config = self.get_config()
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"📁 配置已保存: {path}")
    
    @classmethod
    def from_config(cls, path: str) -> 'EnhancedApertureModel':
        """
        从 JSON 文件创建实例
        
        Args:
            path: 配置文件路径
        
        Returns:
            EnhancedApertureModel 实例
        """
        with open(path, 'r') as f:
            config = json.load(f)
        
        # 创建实例（config_path 默认为 None，会自动使用 CONFIG_PATH）
        instance = cls(
            config_path=config.get('config_path', None),
            tau_rc=config.get('tau_rc', 0.1e-3)
        )
        
        # 更新参数
        instance.pixel_size = config.get('pixel_size', instance.pixel_size)
        instance.pixel_area = instance.pixel_size ** 2
        instance.R_pixel = instance.pixel_size / 2
        instance.ink_thickness = config.get('ink_thickness', instance.ink_thickness)
        instance.ink_volume = instance.ink_thickness * instance.pixel_area
        instance.theta_0 = config.get('theta_0', instance.theta_0)
        instance.theta_min = config.get('theta_min', instance.theta_min)
        instance.theta_wall = config.get('theta_wall', 70.0)
        instance.epsilon_r = config.get('epsilon_r', 4.0)
        instance.gamma = config.get('gamma', 0.072)
        instance.d = config.get('d', 4e-7)
        instance.tau = config.get('tau', 0.005)
        instance.zeta = config.get('zeta', 0.8)
        
        return instance


def demo():
    """演示增强版开口率模型"""
    print("=" * 60)
    print("🔬 EWP 增强版开口率模型演示")
    print("=" * 60)
    
    # 创建增强模型
    model = EnhancedApertureModel()
    
    # 创建原模型用于对比
    base_model = ApertureModel()
    
    print("\n📊 电压-接触角-开口率关系:")
    print("-" * 70)
    print(f"{'电压(V)':<8} {'接触角(°)':<12} {'开口率(%)':<12} {'透明半径(μm)':<15} {'体积误差(%)':<12}")
    print("-" * 70)
    
    results = []
    for V in range(0, 31, 5):
        result = model.predict_enhanced(V)
        results.append(result)
        print(f"{V:<8} {result['theta']:<12.1f} {result['aperture_percent']:<12.1f} {result['r_open_um']:<15.1f} {result['volume_error']:<12.6f}")
    
    print("-" * 70)
    
    # 动态响应
    print("\n📈 动态响应 (0V → 30V, 0-100ms):")
    t, eta = model.aperture_step_response(V_start=0, V_end=30, duration=0.10, t_step=0.002)
    metrics = model.get_aperture_metrics(t, eta, t_step=0.002)
    
    print(f"   初始开口率: {metrics['eta_initial']*100:.2f}%")
    print(f"   最终开口率: {metrics['eta_final']*100:.2f}%")
    print(f"   t_10: {metrics['t_10_ms']:.2f} ms")
    print(f"   t_90: {metrics['t_90_ms']:.2f} ms")
    print(f"   超调: {metrics['overshoot_percent']:.2f}%")
    
    # 验证
    print("\n🔍 模型验证:")
    validation = model.validate()
    print(f"   体积守恒: {'✅' if validation['volume_conservation']['passed'] else '❌'}")
    print(f"   开口率范围: {'✅' if validation['aperture_range']['passed'] else '❌'}")
    print(f"   单调性: {'✅' if validation['monotonicity']['passed'] else '❌'}")
    
    # 绘图 - 4个子图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 图1: 电压-开口率对比曲线
    ax1 = axes[0, 0]
    voltages = [r['voltage'] for r in results]
    apertures = [r['aperture_percent'] for r in results]
    
    # 原模型
    base_apertures = []
    for V in voltages:
        theta = model.get_contact_angle(V)
        eta_base = base_model.contact_angle_to_aperture_ratio(theta)
        base_apertures.append(eta_base * 100)
    
    ax1.plot(voltages, base_apertures, 'b--o', linewidth=2, markersize=6, label='Base Model', alpha=0.7)
    ax1.plot(voltages, apertures, 'r-s', linewidth=2, markersize=8, label='Enhanced Model')
    ax1.set_xlabel('Voltage (V)')
    ax1.set_ylabel('Aperture Ratio (%)')
    ax1.set_title('Voltage vs Aperture Ratio Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 30)
    ax1.set_ylim(0, 100)
    
    # 图2: 油墨分布 (0V vs 30V)
    ax2 = axes[0, 1]
    result_0V = model.predict_enhanced(0)
    result_30V = model.predict_enhanced(30)
    
    ax2.plot(result_0V['r']*1e6, result_0V['h']*1e6, 'b-', linewidth=2, label='0V (OFF)')
    ax2.plot(result_30V['r']*1e6, result_30V['h']*1e6, 'r-', linewidth=2, label='30V (ON)')
    ax2.axvline(result_30V['r_open_um'], color='r', linestyle='--', alpha=0.5, 
               label=f'Open region r={result_30V["r_open_um"]:.1f}um')
    ax2.set_xlabel('Radius (um)')
    ax2.set_ylabel('Ink Height (um)')
    ax2.set_title('Ink Distribution (0V vs 30V)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 95)
    
    # 图3: 动态响应
    ax3 = axes[1, 0]
    ax3.plot(t * 1000, eta * 100, 'b-', linewidth=2)
    ax3.axhline(metrics['eta_final'] * 100, color='g', linestyle='--', alpha=0.5, 
               label=f'Steady state {metrics["eta_final"]*100:.1f}%')
    ax3.axvline(2, color='r', linestyle='--', alpha=0.5, label='Step time')
    if not np.isnan(metrics['t_90_ms']):
        ax3.axvline(2 + metrics['t_90_ms'], color='orange', linestyle=':', alpha=0.7,
                   label=f't_90={metrics["t_90_ms"]:.1f}ms')
    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('Aperture Ratio (%)')
    ax3.set_title('Aperture Dynamic Response (0V -> 30V)')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # 图4: 像素俯视图
    ax4 = axes[1, 1]
    ax4.set_aspect('equal')
    
    pixel_half = model.pixel_size * 1e6 / 2
    r_open = result_30V['r_open_um']
    
    # 像素边界（方形）- 油墨区域
    pixel_rect = plt.Rectangle(
        (-pixel_half, -pixel_half), pixel_half * 2, pixel_half * 2, 
        fill=True, facecolor='lightcoral', edgecolor='black', linewidth=2,
        label='Ink region'
    )
    ax4.add_patch(pixel_rect)
    
    # 透明区域（圆形）
    circle_open = plt.Circle(
        (0, 0), r_open, 
        fill=True, facecolor='white', edgecolor='blue', linewidth=2,
        label='Open region'
    )
    ax4.add_patch(circle_open)
    
    ax4.set_xlim(-pixel_half * 1.1, pixel_half * 1.1)
    ax4.set_ylim(-pixel_half * 1.1, pixel_half * 1.1)
    ax4.set_xlabel('X (um)')
    ax4.set_ylabel('Y (um)')
    ax4.set_title(f'Pixel Top View @ 30V (Aperture={result_30V["aperture_percent"]:.1f}%)')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('enhanced_aperture_demo.png', dpi=150)
    print(f"\n📊 图表已保存: enhanced_aperture_demo.png")
    
    print("\n📈 生成全电压范围完整开口率循环图...")
    V_start = 1.0
    V_end = 30.0
    V_step = 1.0
    voltages_cycle = np.arange(V_start, V_end + 0.1, V_step)
    duration = 0.100
    t_rise = 0.002
    t_fall = 0.040
    num_points = 1000
    fig2, ax = plt.subplots(figsize=(14, 9), dpi=150)
    norm = plt.Normalize(V_start, V_end)
    cmap = cm.turbo
    for V in voltages_cycle:
        t_cycle, V_t_cycle, eta_cycle = model.aperture_cycle_response(
            V_target=V,
            duration=duration,
            t_rise=t_rise,
            t_fall=t_fall,
            num_points=num_points
        )
        color = cmap(norm(V))
        ax.plot(t_cycle * 1000, eta_cycle * 100, color=color, linewidth=1.5, alpha=0.9)
        if V in [10, 20, 30]:
            idx_label = np.searchsorted(t_cycle, t_fall) - 20
            if 0 <= idx_label < len(t_cycle):
                ax.text(
                    t_cycle[idx_label] * 1000,
                    eta_cycle[idx_label] * 100 + 2,
                    f'{int(V)}V',
                    color=color,
                    fontweight='bold',
                    fontsize=10,
                    ha='center'
                )
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('Applied Voltage (V)', fontsize=12, fontweight='bold')
    cbar.set_ticks(np.arange(0, 31, 5))
    ax.axvspan(0, t_rise * 1000, color='gray', alpha=0.1)
    ax.text(t_rise * 500, 95, "OFF", ha='center', color='gray', fontweight='bold')
    ax.text((t_rise + t_fall) * 500, 95, "ON (Steady State Reached)", ha='center', color='black', fontweight='bold')
    ax.axvspan(t_fall * 1000, duration * 1000, color='gray', alpha=0.1)
    ax.text((t_fall + duration) * 500, 95, "OFF (Full Recovery)", ha='center', color='gray', fontweight='bold')
    ax.axvline(x=t_rise * 1000, color='k', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(x=t_fall * 1000, color='k', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_xlabel('Time (ms)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Aperture Ratio (%)', fontsize=12, fontweight='bold')
    ax.set_title('Stage 1: Full Cycle Aperture Response (Steady State $\\to$ Zero Recovery)\n0V $\\to$ (1-30V) $\\to$ 0V', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlim(0, 100)
    ax.set_ylim(-2, 105)
    ax.grid(True, which='major', linestyle='-', alpha=0.3)
    ax.grid(True, which='minor', linestyle=':', alpha=0.1)
    ax.minorticks_on()
    textstr = '\n'.join((
        'Physics Parameters:',
        '• Rise Time (t_ON): 38 ms',
        '  Enough for >5τ (τ ≈ 5ms)',
        '• Fall Time (t_OFF): 60 ms',
        '  Enough for >5τ_rec (τ_rec ≈ 7.5ms)',
        '• Result: Full Return to 0%'
    ))
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.02, 0.85, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)
    plt.tight_layout()
    plt.savefig('stage1_aperture_cycle_full_recovery.png')
    print(f"\n📊 图表已保存: stage1_aperture_cycle_full_recovery.png")
    
    print("\n✅ 演示完成!")
    
    return results


if __name__ == '__main__':
    demo()
