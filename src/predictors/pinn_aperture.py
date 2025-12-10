#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PINN 开口率预测器
================

基于训练好的两相流 PINN 模型预测开口率。

核心功能：
1. 加载 TwoPhasePINN 模型
2. 预测 φ(x,y,z,t,V) 场
3. 从 φ 场积分计算开口率

作者: EFD-PINNs Team
日期: 2024-12
"""

import glob
import logging
import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


class PINNAperturePredictor:
    """
    基于 PINN 的开口率预测器
    
    使用训练好的 TwoPhasePINN 模型预测体积分数场 φ，
    然后通过积分计算开口率。
    
    开口率定义：η = ∫(1-φ)dA / A_pixel
    其中 φ=1 表示油墨，φ=0 表示极性液体（透明）
    
    Example:
        >>> predictor = PINNAperturePredictor()
        >>> eta = predictor.predict(voltage=30, time=0.01)
        >>> print(f"开口率: {eta:.3f}")
    """
    
    def __init__(self, checkpoint_path: Optional[str] = None, device: Optional[str] = None):
        """
        初始化预测器
        
        Args:
            checkpoint_path: 模型检查点路径，None 则自动查找最新的
            device: 计算设备 ('cuda', 'cpu', None=自动)
        """
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        
        self.model = None
        self.config = None
        self.physics = None
        self._model_loaded = False
        
        # 尝试加载模型
        if checkpoint_path:
            self._load_model(checkpoint_path)
        else:
            self._auto_load_model()
        
        # 物理参数（从模型配置或默认值）
        self._init_physics()
    
    def _init_physics(self):
        """初始化物理参数"""
        # 默认物理参数
        self.physics = {
            "Lx": 174e-6,       # 像素宽度 (m)
            "Ly": 174e-6,       # 像素高度 (m)
            "Lz": 20e-6,        # 围堰高度 (m)
            "t_max": 0.02,      # 最大时间 (s)
        }
    
    def _auto_load_model(self):
        """自动查找并加载最新的模型"""
        # 查找输出目录
        output_dirs = sorted(glob.glob("outputs_pinn_*"))
        
        if not output_dirs:
            logger.warning("未找到 PINN 训练输出目录，将使用解析模型作为备选")
            return
        
        latest_dir = output_dirs[-1]
        checkpoint_path = os.path.join(latest_dir, "best_model.pth")
        
        if os.path.exists(checkpoint_path):
            self._load_model(checkpoint_path)
        else:
            logger.warning(f"未找到模型文件: {checkpoint_path}")
    
    def _load_model(self, checkpoint_path: str):
        """加载模型"""
        try:
            # 延迟导入，避免循环依赖
            from src.models.pinn_two_phase import TwoPhasePINN, DEFAULT_CONFIG
            
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.config = checkpoint.get("config", DEFAULT_CONFIG)
            self.model = TwoPhasePINN(self.config).to(self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()
            
            self._model_loaded = True
            
            best_loss = checkpoint.get("best_loss", "N/A")
            epoch = checkpoint.get("epoch", "N/A")
            logger.info(f"✅ PINN 模型加载成功: {checkpoint_path}")
            logger.info(f"   最佳损失: {best_loss:.4e}, 训练轮数: {epoch}")
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            self._model_loaded = False
    
    @property
    def is_available(self) -> bool:
        """检查 PINN 模型是否可用"""
        return self._model_loaded and self.model is not None
    
    def predict(self, voltage: float, time: float, n_points: int = 100) -> float:
        """
        预测开口率
        
        Args:
            voltage: 电压 (V), 范围 [0, 30]
            time: 时间 (s), 范围 [0, 0.02]
            n_points: 采样点数（每个方向）
        
        Returns:
            开口率 η ∈ [0, 1]
        
        Raises:
            RuntimeError: 如果模型不可用
        """
        if not self.is_available:
            raise RuntimeError("PINN 模型不可用，请先加载模型或使用解析方法")
        
        phi_field = self.predict_phi_field(voltage, time, n_points)
        aperture = self._integrate_aperture(phi_field)
        
        return aperture
    
    def predict_phi_field(
        self, 
        voltage: float, 
        time: float, 
        n_points: int = 100,
        z: float = 0.0
    ) -> np.ndarray:
        """
        预测 φ 场
        
        Args:
            voltage: 电压 (V)
            time: 时间 (s)
            n_points: 采样点数（每个方向）
            z: z 坐标 (m)，默认 0（底面）
        
        Returns:
            φ 场，shape (n_points, n_points)
        """
        if not self.is_available:
            raise RuntimeError("PINN 模型不可用")
        
        Lx, Ly = self.physics["Lx"], self.physics["Ly"]
        
        # 在指定 z 平面均匀采样
        x = np.linspace(0, Lx, n_points)
        y = np.linspace(0, Ly, n_points)
        X, Y = np.meshgrid(x, y)
        x_flat = X.flatten()
        y_flat = Y.flatten()
        z_flat = np.full_like(x_flat, z)
        t_flat = np.full_like(x_flat, time)
        V_flat = np.full_like(x_flat, voltage)
        
        inputs = np.stack([x_flat, y_flat, z_flat, t_flat, V_flat], axis=1).astype(np.float32)
        
        with torch.no_grad():
            outputs = self.model(torch.tensor(inputs, device=self.device))
            phi = outputs[:, 4].cpu().numpy()
        
        return phi.reshape(n_points, n_points)
    
    def predict_full_field(
        self, 
        voltage: float, 
        time: float, 
        n_points: Tuple[int, int, int] = (50, 50, 20)
    ) -> Dict[str, np.ndarray]:
        """
        预测完整的 3D 场
        
        Args:
            voltage: 电压 (V)
            time: 时间 (s)
            n_points: (nx, ny, nz) 采样点数
        
        Returns:
            {
                'phi': φ 场,
                'u': x 方向速度,
                'v': y 方向速度,
                'w': z 方向速度,
                'p': 压力,
                'X': x 坐标网格,
                'Y': y 坐标网格,
                'Z': z 坐标网格
            }
        """
        if not self.is_available:
            raise RuntimeError("PINN 模型不可用")
        
        nx, ny, nz = n_points
        Lx, Ly, Lz = self.physics["Lx"], self.physics["Ly"], self.physics["Lz"]
        
        x = np.linspace(0, Lx, nx)
        y = np.linspace(0, Ly, ny)
        z = np.linspace(0, Lz, nz)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        x_flat = X.flatten()
        y_flat = Y.flatten()
        z_flat = Z.flatten()
        t_flat = np.full_like(x_flat, time)
        V_flat = np.full_like(x_flat, voltage)
        
        inputs = np.stack([x_flat, y_flat, z_flat, t_flat, V_flat], axis=1).astype(np.float32)
        
        with torch.no_grad():
            outputs = self.model(torch.tensor(inputs, device=self.device)).cpu().numpy()
        
        return {
            'u': outputs[:, 0].reshape(nx, ny, nz),
            'v': outputs[:, 1].reshape(nx, ny, nz),
            'w': outputs[:, 2].reshape(nx, ny, nz),
            'p': outputs[:, 3].reshape(nx, ny, nz),
            'phi': outputs[:, 4].reshape(nx, ny, nz),
            'X': X, 'Y': Y, 'Z': Z
        }
    
    def _integrate_aperture(self, phi_field: np.ndarray, method: str = "contour") -> float:
        """
        从 φ 场计算开口率
        
        物理定义：
        - φ=0: 完全透明（极性液体，开口区域）
        - φ=0.5: 油墨平铺（初始状态，无开口）
        - φ=0.8: 油墨堆高（边缘）
        
        开口率 η = 透明区域面积 / 像素面积
        透明区域定义：φ < 0.3 的区域（明显透明）
        
        Args:
            phi_field: 底面 φ 场
            method: 计算方法
                - "contour": 使用接触线轮廓计算面积（更准确）
                - "threshold": 简单阈值法
        
        Returns:
            开口率 η
        """
        # 透明阈值：φ < 0.3 表示透明区域
        # 这样 φ≈0.5（油墨平铺）不会被误判为开口
        TRANSPARENT_THRESHOLD = 0.3
        
        if method == "contour":
            # 方法1：接触线轮廓法 - 计算 φ=TRANSPARENT_THRESHOLD 等值线围成的面积
            try:
                from skimage import measure
                
                # 找到 φ=TRANSPARENT_THRESHOLD 的等值线
                contours = measure.find_contours(phi_field, level=TRANSPARENT_THRESHOLD)
                
                if len(contours) == 0:
                    # 没有接触线，检查整体状态
                    if np.mean(phi_field) < TRANSPARENT_THRESHOLD:
                        return 1.0  # 全透明
                    else:
                        return 0.0  # 无开口
                
                # 计算所有闭合轮廓围成的面积
                total_area = 0.0
                n_pixels = phi_field.shape[0] * phi_field.shape[1]
                
                for contour in contours:
                    # 使用 Shoelace 公式计算多边形面积
                    # 注意：contour 是 (row, col) 格式
                    n = len(contour)
                    if n < 3:
                        continue
                    
                    # Shoelace formula
                    area = 0.0
                    for i in range(n):
                        j = (i + 1) % n
                        area += contour[i, 0] * contour[j, 1]
                        area -= contour[j, 0] * contour[i, 1]
                    area = abs(area) / 2.0
                    
                    # 判断轮廓内部是透明还是油墨
                    # 取轮廓中心点判断
                    center_row = int(np.mean(contour[:, 0]))
                    center_col = int(np.mean(contour[:, 1]))
                    center_row = np.clip(center_row, 0, phi_field.shape[0] - 1)
                    center_col = np.clip(center_col, 0, phi_field.shape[1] - 1)
                    
                    if phi_field[center_row, center_col] < TRANSPARENT_THRESHOLD:
                        # 轮廓内是透明区域
                        total_area += area
                
                aperture = total_area / n_pixels
                return float(np.clip(aperture, 0, 1))
                
            except ImportError:
                # 如果没有 skimage，回退到阈值法
                method = "threshold"
        
        if method == "threshold":
            # 方法2：简单阈值法 - 统计 φ < TRANSPARENT_THRESHOLD 的像素比例
            transparent_ratio = np.mean(phi_field < TRANSPARENT_THRESHOLD)
            return float(np.clip(transparent_ratio, 0, 1))
        
        return float(np.clip(np.mean(phi_field < TRANSPARENT_THRESHOLD), 0, 1))
    
    def compute_aperture_contour(
        self, 
        voltage: float, 
        time: float, 
        n_points: int = 100,
        method: str = "contour"
    ) -> dict:
        """
        计算开口率和三相接触线轮廓
        
        开口率定义：接触线内面积 / 像素面积
        
        Args:
            voltage: 电压 (V)
            time: 时间 (s)
            n_points: 采样点数
            method: 计算方法 ("contour" 或 "threshold")
        
        Returns:
            {
                'aperture': 开口率 (接触线内面积/像素面积),
                'contour_x': 接触线 x 坐标 (m),
                'contour_y': 接触线 y 坐标 (m),
                'contour_r': 等效半径 (m),
                'contour_area': 接触线内面积 (m²),
                'pixel_area': 像素面积 (m²),
                'phi_field': φ 场
            }
        """
        phi_field = self.predict_phi_field(voltage, time, n_points)
        
        Lx, Ly = self.physics["Lx"], self.physics["Ly"]
        pixel_area = Lx * Ly
        
        # 提取 φ=0.5 等值线并计算面积
        contour_x, contour_y = [], []
        contour_area = 0.0
        
        try:
            from skimage import measure
            
            # 找到 φ=0.5 的等值线
            contours = measure.find_contours(phi_field, level=0.5)
            
            # 网格间距
            dx = Lx / (n_points - 1)
            dy = Ly / (n_points - 1)
            
            for contour in contours:
                # 转换为物理坐标 (contour 是 row, col 格式)
                x_coords = contour[:, 1] * dx  # col -> x
                y_coords = contour[:, 0] * dy  # row -> y
                
                contour_x.extend(x_coords)
                contour_y.extend(y_coords)
                
                # 计算该轮廓围成的面积 (Shoelace formula)
                n = len(contour)
                if n < 3:
                    continue
                
                area = 0.0
                for i in range(n):
                    j = (i + 1) % n
                    area += x_coords[i] * y_coords[j]
                    area -= x_coords[j] * y_coords[i]
                area = abs(area) / 2.0
                
                # 判断轮廓内部是透明还是油墨
                center_row = int(np.mean(contour[:, 0]))
                center_col = int(np.mean(contour[:, 1]))
                center_row = np.clip(center_row, 0, phi_field.shape[0] - 1)
                center_col = np.clip(center_col, 0, phi_field.shape[1] - 1)
                
                if phi_field[center_row, center_col] < 0.5:
                    # 轮廓内是透明区域（极性液体）
                    contour_area += area
            
            aperture = contour_area / pixel_area
            
        except ImportError:
            # 回退到阈值法
            aperture = np.mean(phi_field < 0.5)
            contour_area = aperture * pixel_area
        
        contour_x = np.array(contour_x)
        contour_y = np.array(contour_y)
        
        # 计算等效半径（假设圆形）
        equiv_radius = np.sqrt(contour_area / np.pi) if contour_area > 0 else 0.0
        
        return {
            'aperture': float(np.clip(aperture, 0, 1)),
            'contour_x': contour_x,
            'contour_y': contour_y,
            'contour_r': equiv_radius,
            'contour_area': contour_area,
            'pixel_area': pixel_area,
            'phi_field': phi_field
        }
    
    def get_interface_contour(
        self, 
        voltage: float, 
        time: float, 
        n_points: int = 100,
        level: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取油水界面轮廓 (φ=0.5)
        
        Args:
            voltage: 电压 (V)
            time: 时间 (s)
            n_points: 采样点数
            level: 界面等值线级别
        
        Returns:
            (X, Y) 界面坐标网格
        """
        import matplotlib.pyplot as plt
        
        phi_field = self.predict_phi_field(voltage, time, n_points)
        
        Lx, Ly = self.physics["Lx"], self.physics["Ly"]
        x = np.linspace(0, Lx, n_points)
        y = np.linspace(0, Ly, n_points)
        X, Y = np.meshgrid(x, y)
        
        # 使用 matplotlib 提取等高线
        fig, ax = plt.subplots()
        cs = ax.contour(X, Y, phi_field, levels=[level])
        plt.close(fig)
        
        # 提取轮廓点
        contour_points = []
        for path in cs.collections[0].get_paths():
            contour_points.append(path.vertices)
        
        if contour_points:
            all_points = np.vstack(contour_points)
            return all_points[:, 0], all_points[:, 1]
        else:
            return np.array([]), np.array([])


# 便捷函数
def predict_aperture_pinn(voltage: float, time: float, checkpoint_path: str = None) -> float:
    """
    便捷函数：使用 PINN 预测开口率
    
    Args:
        voltage: 电压 (V)
        time: 时间 (s)
        checkpoint_path: 模型路径（可选）
    
    Returns:
        开口率 η
    """
    predictor = PINNAperturePredictor(checkpoint_path)
    return predictor.predict(voltage, time)
