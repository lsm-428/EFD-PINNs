"""
EWP 3D Visualizer - 电润湿像素 3D 可视化模块

基于 EnhancedApertureModel 的高质量 3D 可视化系统，支持：
- 静态 3D 渲染
- 多电压对比图
- 油墨高度剖面
- 动画帧生成
- 交互式可视化
- 数据导出

Author: EFD-PINNs Team
Date: 2025-12-03
"""

import numpy as np
import pyvista as pv
import json
import os
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Tuple, Dict, Any
from datetime import datetime

# 导入 EnhancedApertureModel
from src.models.aperture_model import EnhancedApertureModel

# ============================================================
# 常量和配置
# ============================================================

# 材料颜色映射
MATERIAL_COLORS = {
    "底层ITO玻璃": "#00CC00",   # 绿色
    "围堰": "#FF9900",          # 橙色
    "介电层": "#FF9900",        # 橙色
    "疏水层": "#9900CC",        # 紫色
    "油墨层": "#FF0000",        # 红色
    "极性液体层": "#00FFFF",    # 青色
    "顶层ITO层": "#00CC00",     # 绿色
    "透明区域": "#FFFFFF",      # 白色
}

# 材料属性（用于渲染）
MATERIAL_PROPERTIES = {
    "底层ITO玻璃": {"opacity": 0.9, "specular": 0.4, "diffuse": 0.6, "ambient": 0.2},
    "围堰": {"opacity": 0.95, "specular": 0.3, "diffuse": 0.7, "ambient": 0.2},
    "介电层": {"opacity": 0.85, "specular": 0.5, "diffuse": 0.5, "ambient": 0.1},
    "疏水层": {"opacity": 0.8, "specular": 0.6, "diffuse": 0.4, "ambient": 0.1},
    "油墨层": {"opacity": 0.9, "specular": 0.3, "diffuse": 0.7, "ambient": 0.2},
    "极性液体层": {"opacity": 0.5, "specular": 0.8, "diffuse": 0.3, "ambient": 0.1},
    "顶层ITO层": {"opacity": 0.6, "specular": 0.5, "diffuse": 0.5, "ambient": 0.2},
    "透明区域": {"opacity": 0.3, "specular": 0.2, "diffuse": 0.8, "ambient": 0.1},
}

# 像素结构参数（从 generate_pyvista_3d.py 提取）
PIXEL_STRUCTURE = {
    "pixel_width": 184e-6,      # 像素宽度 184μm
    "pixel_height": 184e-6,     # 像素高度 184μm
    "inner_width": 174e-6,      # 内沿宽度 174μm
    "inner_height": 174e-6,     # 内沿高度 174μm
    "wall_thickness": 5e-6,     # 围堰厚度 5μm
    "ito_thickness": 27.5e-9,   # ITO 厚度 27.5nm
    "dielectric_thickness": 0.4e-6,  # 介电层厚度 0.4μm
    "hydrophobic_thickness": 0.4e-6, # 疏水层厚度 0.4μm
    "ink_thickness": 3e-6,      # 油墨层厚度 3μm
    "polar_thickness": 17e-6,   # 极性液体层厚度 17μm
    "weir_height": 20e-6,       # 围堰高度 20μm
}


@dataclass
class VisualizationConfig:
    """可视化配置"""
    resolution: Tuple[int, int] = (1920, 1080)
    background_color: str = 'black'
    background_top: str = 'gray'
    transparent_bg: bool = False
    show_edges: bool = True
    edge_color: str = 'black'
    edge_width: float = 0.5
    camera_position: str = 'iso'
    camera_zoom: float = 1.1
    font_size: int = 14
    title_font_size: int = 16
    scale_factor: float = 1e6  # 转换为微米
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VisualizationConfig':
        """从字典创建"""
        return cls(**data)


# ============================================================
# 异常类
# ============================================================

class VisualizationError(Exception):
    """可视化错误基类"""
    pass


class ModelNotFoundError(VisualizationError):
    """模型未找到错误"""
    pass


class InvalidVoltageError(VisualizationError):
    """无效电压错误"""
    pass


class RenderingError(VisualizationError):
    """渲染错误"""
    pass


# ============================================================
# PixelVisualizer 类
# ============================================================

class PixelVisualizer:
    """
    基于 EnhancedApertureModel 的 3D 像素可视化器
    
    提供高质量的电润湿像素 3D 渲染，支持：
    - 单电压状态渲染
    - 多电压对比图
    - 油墨高度剖面
    - 交互式可视化
    
    Example:
        >>> visualizer = PixelVisualizer()
        >>> visualizer.render(voltage=30, save_path='pixel_30V.png')
    """
    
    def __init__(self, model: Optional[EnhancedApertureModel] = None,
                 config: Optional[VisualizationConfig] = None):
        """
        初始化可视化器
        
        Args:
            model: EnhancedApertureModel 实例，如果为 None 则创建默认实例
            config: 可视化配置，如果为 None 则使用默认配置
        """
        # 初始化模型
        if model is None:
            self.model = EnhancedApertureModel()
        else:
            self.model = model
        
        # 初始化配置
        if config is None:
            self.config = VisualizationConfig()
        else:
            self.config = config
        
        # 加载像素结构参数
        self.pixel_structure = PIXEL_STRUCTURE.copy()
        self.scale_factor = self.config.scale_factor
        
        # 计算层边界（缩放后）
        self._calculate_layer_boundaries()
    
    def _calculate_layer_boundaries(self):
        """计算各层的 Z 坐标边界（已缩放）"""
        sf = self.scale_factor
        ps = self.pixel_structure
        
        # 从底部开始计算
        z = 0.0
        
        # 底层 ITO
        self.z_ito_bottom_start = z
        z += ps["ito_thickness"] * sf
        self.z_ito_bottom_end = z
        
        # 介电层
        self.z_dielectric_start = z
        z += ps["dielectric_thickness"] * sf
        self.z_dielectric_end = z
        
        # 疏水层
        self.z_hydrophobic_start = z
        z += ps["hydrophobic_thickness"] * sf
        self.z_hydrophobic_end = z
        
        # 围堰起始（与疏水层顶部对齐）
        self.z_weir_start = self.z_hydrophobic_end
        self.z_weir_end = self.z_weir_start + ps["weir_height"] * sf
        
        # 油墨层（在围堰内部）
        self.z_ink_start = self.z_hydrophobic_end
        self.z_ink_end = self.z_ink_start + ps["ink_thickness"] * sf
        
        # 极性液体层（在油墨层之上）
        self.z_polar_start = self.z_ink_end
        self.z_polar_end = self.z_weir_end
        
        # 顶层 ITO
        self.z_ito_top_start = self.z_weir_end
        self.z_ito_top_end = self.z_ito_top_start + ps["ito_thickness"] * sf
    
    def _get_scaled_dimensions(self) -> Dict[str, float]:
        """获取缩放后的尺寸"""
        sf = self.scale_factor
        ps = self.pixel_structure
        
        return {
            "pixel_width": ps["pixel_width"] * sf,
            "pixel_height": ps["pixel_height"] * sf,
            "inner_width": ps["inner_width"] * sf,
            "inner_height": ps["inner_height"] * sf,
            "wall_thickness": ps["wall_thickness"] * sf,
        }
    
    def _validate_voltage(self, voltage: float):
        """验证电压范围"""
        if voltage < 0 or voltage > 40:
            raise InvalidVoltageError(f"电压必须在 [0, 40] V 范围内，当前值: {voltage}")
    
    def _validate_time(self, time: Optional[float]):
        """验证时间范围"""
        if time is not None and time < 0:
            raise InvalidVoltageError(f"时间必须 >= 0，当前值: {time}")

    
    def _create_base_structure(self, plotter: pv.Plotter) -> None:
        """
        创建像素基础结构（ITO、介电层、疏水层、围堰）
        
        Args:
            plotter: PyVista Plotter 对象
        """
        dims = self._get_scaled_dimensions()
        pw = dims["pixel_width"]
        ph = dims["pixel_height"]
        iw = dims["inner_width"]
        ih = dims["inner_height"]
        wt = dims["wall_thickness"]
        
        # 1. 底层 ITO
        ito_bottom = pv.RectilinearGrid(
            np.linspace(-pw/2, pw/2, 20),
            np.linspace(-ph/2, ph/2, 20),
            np.linspace(self.z_ito_bottom_start, self.z_ito_bottom_end, 3)
        )
        props = MATERIAL_PROPERTIES["底层ITO玻璃"]
        plotter.add_mesh(
            ito_bottom,
            color=MATERIAL_COLORS["底层ITO玻璃"],
            opacity=props["opacity"],
            specular=props["specular"],
            show_edges=self.config.show_edges,
            edge_color=self.config.edge_color,
            line_width=self.config.edge_width,
            label="Bottom ITO"
        )
        
        # 2. 介电层
        dielectric = pv.RectilinearGrid(
            np.linspace(-pw/2, pw/2, 20),
            np.linspace(-ph/2, ph/2, 20),
            np.linspace(self.z_dielectric_start, self.z_dielectric_end, 3)
        )
        props = MATERIAL_PROPERTIES["介电层"]
        plotter.add_mesh(
            dielectric,
            color=MATERIAL_COLORS["介电层"],
            opacity=props["opacity"],
            specular=props["specular"],
            show_edges=self.config.show_edges,
            edge_color=self.config.edge_color,
            line_width=self.config.edge_width,
            label="Dielectric"
        )
        
        # 3. 疏水层
        hydrophobic = pv.RectilinearGrid(
            np.linspace(-pw/2, pw/2, 20),
            np.linspace(-ph/2, ph/2, 20),
            np.linspace(self.z_hydrophobic_start, self.z_hydrophobic_end, 3)
        )
        props = MATERIAL_PROPERTIES["疏水层"]
        plotter.add_mesh(
            hydrophobic,
            color=MATERIAL_COLORS["疏水层"],
            opacity=props["opacity"],
            specular=props["specular"],
            show_edges=self.config.show_edges,
            edge_color=self.config.edge_color,
            line_width=self.config.edge_width,
            label="Hydrophobic"
        )
        
        # 4. 围堰（四面墙）
        outer_x = pw / 2
        outer_y = ph / 2
        inner_x = iw / 2
        inner_y = ih / 2
        
        walls = [
            # 前墙
            pv.Box(bounds=[-outer_x, outer_x, -outer_y, -outer_y + wt,
                          self.z_weir_start, self.z_weir_end]),
            # 后墙
            pv.Box(bounds=[-outer_x, outer_x, outer_y - wt, outer_y,
                          self.z_weir_start, self.z_weir_end]),
            # 左墙
            pv.Box(bounds=[-outer_x, -outer_x + wt, -outer_y + wt, outer_y - wt,
                          self.z_weir_start, self.z_weir_end]),
            # 右墙
            pv.Box(bounds=[outer_x - wt, outer_x, -outer_y + wt, outer_y - wt,
                          self.z_weir_start, self.z_weir_end]),
        ]
        
        props = MATERIAL_PROPERTIES["围堰"]
        for i, wall in enumerate(walls):
            plotter.add_mesh(
                wall,
                color=MATERIAL_COLORS["围堰"],
                opacity=props["opacity"],
                specular=props["specular"],
                show_edges=self.config.show_edges,
                edge_color=self.config.edge_color,
                label="Weir" if i == 0 else None  # 只给第一个添加标签
            )
    
    def _create_ink_geometry(self, plotter: pv.Plotter, 
                            prediction: Dict[str, Any]) -> Dict[str, Any]:
        """
        创建油墨层几何（基于模型预测）
        
        Args:
            plotter: PyVista Plotter 对象
            prediction: EnhancedApertureModel.predict_enhanced() 的返回结果
            
        Returns:
            包含几何信息的字典
        """
        dims = self._get_scaled_dimensions()
        iw = dims["inner_width"]
        ih = dims["inner_height"]
        sf = self.scale_factor
        
        aperture_ratio = prediction['aperture_ratio']
        r_open = prediction['r_open'] * sf  # 缩放到可视化单位
        r_array = prediction['r'] * sf
        h_array = prediction['h'] * sf
        
        geometry_info = {
            'aperture_ratio': aperture_ratio,
            'r_open': r_open,
            'has_opening': aperture_ratio > 0.01,
        }
        
        if aperture_ratio > 0.01 and r_open > 0:
            # 有开口的情况
            
            # 1. 创建透明区域（白色圆柱）
            ink_height = self.z_ink_end - self.z_ink_start
            open_cylinder = pv.Cylinder(
                center=(0, 0, (self.z_ink_start + self.z_ink_end) / 2),
                direction=(0, 0, 1),
                radius=r_open,
                height=ink_height
            )
            props = MATERIAL_PROPERTIES["透明区域"]
            plotter.add_mesh(
                open_cylinder,
                color=MATERIAL_COLORS["透明区域"],
                opacity=props["opacity"],
                specular=props["specular"],
                label=f"Open (r={r_open/sf*1e6:.1f}μm)"
            )
            
            # 2. 创建环形油墨区域
            outer_radius = min(iw, ih) / 2
            
            # 计算平均油墨高度（从模型数据）
            h_ink = h_array[h_array > 0]
            if len(h_ink) > 0:
                h_avg = np.mean(h_ink)
            else:
                h_avg = 0
            
            if h_avg > 0:
                # 创建环形底面
                ink_disc = pv.Disc(
                    center=(0, 0, self.z_ink_start),
                    inner=r_open,
                    outer=outer_radius,
                    normal=(0, 0, 1),
                    r_res=30,
                    c_res=60
                )
                
                # 挤出成 3D
                ink_3d = ink_disc.extrude([0, 0, h_avg], capping=True)
                
                props = MATERIAL_PROPERTIES["油墨层"]
                plotter.add_mesh(
                    ink_3d,
                    color=MATERIAL_COLORS["油墨层"],
                    opacity=props["opacity"],
                    specular=props["specular"],
                    show_edges=False,
                    label=f"Ink (h={h_avg/sf*1e6:.2f}μm)"
                )
                
                geometry_info['h_avg'] = h_avg
                geometry_info['ink_volume_3d'] = np.pi * (outer_radius**2 - r_open**2) * h_avg
        else:
            # 无开口，油墨均匀分布
            ink_box = pv.Box(
                bounds=[-iw/2, iw/2, -ih/2, ih/2,
                       self.z_ink_start, self.z_ink_end]
            )
            props = MATERIAL_PROPERTIES["油墨层"]
            plotter.add_mesh(
                ink_box,
                color=MATERIAL_COLORS["油墨层"],
                opacity=props["opacity"],
                specular=props["specular"],
                show_edges=self.config.show_edges,
                edge_color=self.config.edge_color,
                label="Ink (uniform)"
            )
            
            geometry_info['h_avg'] = self.z_ink_end - self.z_ink_start
            geometry_info['ink_volume_3d'] = iw * ih * geometry_info['h_avg']
        
        return geometry_info

    
    def _add_polar_liquid_and_top_ito(self, plotter: pv.Plotter) -> None:
        """添加极性液体层和顶层 ITO"""
        dims = self._get_scaled_dimensions()
        iw = dims["inner_width"]
        ih = dims["inner_height"]
        pw = dims["pixel_width"]
        ph = dims["pixel_height"]
        
        # 极性液体层
        polar_box = pv.Box(
            bounds=[-iw/2, iw/2, -ih/2, ih/2,
                   self.z_polar_start, self.z_polar_end]
        )
        props = MATERIAL_PROPERTIES["极性液体层"]
        plotter.add_mesh(
            polar_box,
            color=MATERIAL_COLORS["极性液体层"],
            opacity=props["opacity"],
            specular=props["specular"],
            show_edges=False,
            label="Polar liquid"
        )
        
        # 顶层 ITO
        ito_top = pv.RectilinearGrid(
            np.linspace(-pw/2, pw/2, 20),
            np.linspace(-ph/2, ph/2, 20),
            np.linspace(self.z_ito_top_start, self.z_ito_top_end, 3)
        )
        props = MATERIAL_PROPERTIES["顶层ITO层"]
        plotter.add_mesh(
            ito_top,
            color=MATERIAL_COLORS["顶层ITO层"],
            opacity=props["opacity"],
            specular=props["specular"],
            show_edges=self.config.show_edges,
            edge_color=self.config.edge_color,
            label="Top ITO"
        )
    
    def _setup_plotter(self, off_screen: bool = False) -> pv.Plotter:
        """创建并配置 Plotter"""
        plotter = pv.Plotter(
            window_size=self.config.resolution,
            off_screen=off_screen
        )
        
        # 设置背景
        if self.config.transparent_bg:
            plotter.set_background('white')
        else:
            plotter.set_background(
                self.config.background_color, 
                top=self.config.background_top
            )
        
        # 添加光源
        light1 = pv.Light(position=(1, 1, 1), focal_point=(0, 0, 0), intensity=0.8)
        light2 = pv.Light(position=(-1, -1, 1), focal_point=(0, 0, 0), intensity=0.6)
        plotter.add_light(light1)
        plotter.add_light(light2)
        
        return plotter
    
    def render(self, voltage: float, time: Optional[float] = None,
               save_path: Optional[str] = None,
               resolution: Optional[Tuple[int, int]] = None,
               transparent_bg: bool = False) -> pv.Plotter:
        """
        渲染单个电压状态的 3D 可视化
        
        Args:
            voltage: 电压 (V)
            time: 时间 (s)，可选
            save_path: 保存路径，如果为 None 则返回 plotter
            resolution: 图像分辨率，覆盖配置
            transparent_bg: 是否使用透明背景
            
        Returns:
            PyVista Plotter 对象
        """
        # 验证输入
        self._validate_voltage(voltage)
        self._validate_time(time)
        
        # 临时更新配置
        if resolution is not None:
            self.config.resolution = resolution
        if transparent_bg:
            self.config.transparent_bg = transparent_bg
        
        # 获取模型预测
        prediction = self.model.predict_enhanced(voltage, time)
        
        # 创建 plotter
        plotter = self._setup_plotter(off_screen=(save_path is not None))
        
        # 创建基础结构
        self._create_base_structure(plotter)
        
        # 创建油墨几何
        geometry_info = self._create_ink_geometry(plotter, prediction)
        
        # 添加极性液体和顶层 ITO
        self._add_polar_liquid_and_top_ito(plotter)
        
        # 设置标题
        theta = prediction['theta']
        eta = prediction['aperture_ratio'] * 100
        title = f"EWP Pixel @ {voltage}V"
        if time is not None:
            title += f", t={time*1000:.1f}ms"
        title += f"\nθ={theta:.1f}°, η={eta:.1f}%"
        
        plotter.add_title(
            title,
            font_size=self.config.title_font_size,
            color='white' if not self.config.transparent_bg else 'black'
        )
        
        # 添加坐标轴
        plotter.show_bounds(
            grid='front',
            location='outer',
            xtitle='X (μm)',
            ytitle='Y (μm)',
            ztitle='Z (μm)',
            font_size=self.config.font_size,
            color='white' if not self.config.transparent_bg else 'black'
        )
        
        # 添加图例
        plotter.add_legend(loc='upper right', bcolor='white', border=True)
        
        # 设置相机
        plotter.camera_position = self.config.camera_position
        plotter.camera.zoom(self.config.camera_zoom)
        
        # 保存或返回
        if save_path:
            # 确保目录存在
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plotter.screenshot(save_path)
            print(f"📊 3D 可视化已保存: {save_path}")
            plotter.close()
        
        return plotter
    
    def get_title_info(self, voltage: float, time: Optional[float] = None) -> Dict[str, Any]:
        """
        获取可视化标题信息（用于测试）
        
        Args:
            voltage: 电压 (V)
            time: 时间 (s)
            
        Returns:
            包含 theta 和 aperture_ratio 的字典
        """
        prediction = self.model.predict_enhanced(voltage, time)
        return {
            'voltage': voltage,
            'time': time,
            'theta': prediction['theta'],
            'aperture_ratio': prediction['aperture_ratio'],
            'aperture_percent': prediction['aperture_ratio'] * 100,
        }

    
    def render_comparison(self, voltages: List[float] = [0, 15, 30],
                         save_path: Optional[str] = None,
                         layout: Optional[Tuple[int, int]] = None) -> pv.Plotter:
        """
        渲染多电压对比图
        
        Args:
            voltages: 电压列表
            save_path: 保存路径
            layout: 布局 (rows, cols)，如果为 None 则自动计算
            
        Returns:
            PyVista Plotter 对象
        """
        n = len(voltages)
        
        # 自动计算布局
        if layout is None:
            if n <= 3:
                layout = (1, n)
            elif n <= 6:
                layout = (2, (n + 1) // 2)
            else:
                cols = int(np.ceil(np.sqrt(n)))
                rows = int(np.ceil(n / cols))
                layout = (rows, cols)
        
        rows, cols = layout
        
        # 创建多面板 plotter
        plotter = pv.Plotter(
            shape=(rows, cols),
            window_size=(self.config.resolution[0], self.config.resolution[1]),
            off_screen=(save_path is not None)
        )
        
        # 设置背景
        plotter.set_background(
            self.config.background_color,
            top=self.config.background_top
        )
        
        # 为每个电压创建子图
        for i, voltage in enumerate(voltages):
            row = i // cols
            col = i % cols
            
            plotter.subplot(row, col)
            
            # 获取预测
            prediction = self.model.predict_enhanced(voltage)
            
            # 创建结构
            self._create_base_structure(plotter)
            self._create_ink_geometry(plotter, prediction)
            self._add_polar_liquid_and_top_ito(plotter)
            
            # 设置标题
            theta = prediction['theta']
            eta = prediction['aperture_ratio'] * 100
            plotter.add_title(
                f"{voltage}V: θ={theta:.1f}°, η={eta:.1f}%",
                font_size=12,
                color='white'
            )
            
            # 设置相机（一致的角度）
            plotter.camera_position = self.config.camera_position
            plotter.camera.zoom(self.config.camera_zoom)
        
        # 保存或返回
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plotter.screenshot(save_path)
            print(f"📊 对比图已保存: {save_path}")
            plotter.close()
        
        return plotter
    
    def render_ink_profile(self, voltage: float,
                          save_path: Optional[str] = None) -> pv.Plotter:
        """
        渲染油墨高度剖面 3D 表面图
        
        Args:
            voltage: 电压 (V)
            save_path: 保存路径
            
        Returns:
            PyVista Plotter 对象
        """
        self._validate_voltage(voltage)
        
        # 获取预测
        prediction = self.model.predict_enhanced(voltage)
        
        sf = self.scale_factor
        r_array = prediction['r'] * sf
        h_array = prediction['h'] * sf
        
        # 创建 2D 网格
        n_theta = 60
        theta_angles = np.linspace(0, 2 * np.pi, n_theta)
        
        # 创建网格点
        R, Theta = np.meshgrid(r_array, theta_angles)
        X = R * np.cos(Theta)
        Y = R * np.sin(Theta)
        
        # 高度数组（广播到 2D）
        H = np.tile(h_array, (n_theta, 1))
        
        # 创建结构化网格
        grid = pv.StructuredGrid(X, Y, H + self.z_ink_start)
        
        # 添加高度作为标量数据（用于颜色映射）
        grid['height'] = H.flatten()
        
        # 创建 plotter
        plotter = pv.Plotter(
            window_size=self.config.resolution,
            off_screen=(save_path is not None)
        )
        
        plotter.set_background(
            self.config.background_color,
            top=self.config.background_top
        )
        
        # 添加光源
        light1 = pv.Light(position=(1, 1, 1), focal_point=(0, 0, 0), intensity=0.8)
        plotter.add_light(light1)
        
        # 添加表面
        plotter.add_mesh(
            grid,
            scalars='height',
            cmap='hot',
            show_edges=False,
            smooth_shading=True,
            scalar_bar_args={
                'title': 'Ink Height (μm)',
                'vertical': True,
                'position_x': 0.85,
                'position_y': 0.1,
                'width': 0.1,
                'height': 0.8,
            }
        )
        
        # 设置标题
        theta = prediction['theta']
        eta = prediction['aperture_ratio'] * 100
        r_open = prediction['r_open'] * 1e6  # 转换为 μm
        
        plotter.add_title(
            f"Ink Profile @ {voltage}V\nθ={theta:.1f}°, η={eta:.1f}%, r_open={r_open:.1f}μm",
            font_size=self.config.title_font_size,
            color='white'
        )
        
        # 添加坐标轴
        plotter.show_bounds(
            grid='front',
            location='outer',
            xtitle='X (μm)',
            ytitle='Y (μm)',
            ztitle='Z (μm)',
            font_size=self.config.font_size,
            color='white'
        )
        
        # 设置相机
        plotter.camera_position = 'iso'
        plotter.camera.zoom(1.0)
        
        # 保存或返回
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plotter.screenshot(save_path)
            print(f"📊 油墨剖面图已保存: {save_path}")
            plotter.close()
        
        return plotter
    
    def get_ink_profile_data(self, voltage: float) -> Dict[str, np.ndarray]:
        """
        获取油墨剖面数据（用于测试）
        
        Args:
            voltage: 电压 (V)
            
        Returns:
            包含 r, h, r_open 的字典
        """
        prediction = self.model.predict_enhanced(voltage)
        return {
            'r': prediction['r'],
            'h': prediction['h'],
            'r_open': prediction['r_open'],
            'aperture_ratio': prediction['aperture_ratio'],
        }

    
    def interactive(self, initial_voltage: float = 0.0):
        """
        启动交互式可视化
        
        Args:
            initial_voltage: 初始电压
        """
        # 创建 plotter（非离屏模式）
        plotter = pv.Plotter(
            window_size=self.config.resolution,
            off_screen=False
        )
        
        plotter.set_background(
            self.config.background_color,
            top=self.config.background_top
        )
        
        # 添加光源
        light1 = pv.Light(position=(1, 1, 1), focal_point=(0, 0, 0), intensity=0.8)
        light2 = pv.Light(position=(-1, -1, 1), focal_point=(0, 0, 0), intensity=0.6)
        plotter.add_light(light1)
        plotter.add_light(light2)
        
        # 存储当前 actor 引用
        self._interactive_actors = []
        self._title_actor = None
        
        def update_visualization(voltage):
            """更新可视化的回调函数"""
            # 移除旧的 actors
            for actor in self._interactive_actors:
                try:
                    plotter.remove_actor(actor)
                except:
                    pass
            self._interactive_actors.clear()
            
            # 获取预测
            prediction = self.model.predict_enhanced(voltage)
            
            # 重新创建几何
            dims = self._get_scaled_dimensions()
            iw = dims["inner_width"]
            ih = dims["inner_height"]
            sf = self.scale_factor
            
            aperture_ratio = prediction['aperture_ratio']
            r_open = prediction['r_open'] * sf
            h_array = prediction['h'] * sf
            
            # 创建油墨几何
            if aperture_ratio > 0.01 and r_open > 0:
                # 透明区域
                ink_height = self.z_ink_end - self.z_ink_start
                open_cylinder = pv.Cylinder(
                    center=(0, 0, (self.z_ink_start + self.z_ink_end) / 2),
                    direction=(0, 0, 1),
                    radius=r_open,
                    height=ink_height
                )
                actor = plotter.add_mesh(
                    open_cylinder,
                    color=MATERIAL_COLORS["透明区域"],
                    opacity=0.3
                )
                self._interactive_actors.append(actor)
                
                # 油墨环
                outer_radius = min(iw, ih) / 2
                h_ink = h_array[h_array > 0]
                h_avg = np.mean(h_ink) if len(h_ink) > 0 else 0
                
                if h_avg > 0:
                    ink_disc = pv.Disc(
                        center=(0, 0, self.z_ink_start),
                        inner=r_open,
                        outer=outer_radius,
                        normal=(0, 0, 1),
                        r_res=30,
                        c_res=60
                    )
                    ink_3d = ink_disc.extrude([0, 0, h_avg], capping=True)
                    actor = plotter.add_mesh(
                        ink_3d,
                        color=MATERIAL_COLORS["油墨层"],
                        opacity=0.9
                    )
                    self._interactive_actors.append(actor)
            else:
                # 均匀油墨
                ink_box = pv.Box(
                    bounds=[-iw/2, iw/2, -ih/2, ih/2,
                           self.z_ink_start, self.z_ink_end]
                )
                actor = plotter.add_mesh(
                    ink_box,
                    color=MATERIAL_COLORS["油墨层"],
                    opacity=0.9
                )
                self._interactive_actors.append(actor)
            
            # 更新标题
            theta = prediction['theta']
            eta = prediction['aperture_ratio'] * 100
            
            if self._title_actor is not None:
                try:
                    plotter.remove_actor(self._title_actor)
                except:
                    pass
            
            self._title_actor = plotter.add_title(
                f"Interactive: V={voltage:.1f}V, θ={theta:.1f}°, η={eta:.1f}%",
                font_size=14,
                color='white'
            )
        
        # 添加静态结构
        self._create_base_structure(plotter)
        self._add_polar_liquid_and_top_ito(plotter)
        
        # 初始化油墨
        update_visualization(initial_voltage)
        
        # 添加滑块
        plotter.add_slider_widget(
            update_visualization,
            [0, 40],
            title="Voltage (V)",
            pointa=(0.1, 0.1),
            pointb=(0.9, 0.1),
            value=initial_voltage,
            style='modern'
        )
        
        # 设置相机
        plotter.camera_position = self.config.camera_position
        plotter.camera.zoom(self.config.camera_zoom)
        
        # 显示
        plotter.show()


# ============================================================
# AnimationEngine 类
# ============================================================

class AnimationEngine:
    """
    开口率动态响应动画生成器
    
    生成开口率随时间变化的动画帧序列。
    
    Example:
        >>> engine = AnimationEngine()
        >>> frames = engine.generate_frames(V_start=0, V_end=30, num_frames=30)
        >>> print(engine.get_ffmpeg_command('./outputs/animation'))
    """
    
    def __init__(self, model: Optional[EnhancedApertureModel] = None,
                 config: Optional[VisualizationConfig] = None):
        """
        初始化动画引擎
        
        Args:
            model: EnhancedApertureModel 实例
            config: 可视化配置
        """
        if model is None:
            self.model = EnhancedApertureModel()
        else:
            self.model = model
        
        if config is None:
            self.config = VisualizationConfig()
        else:
            self.config = config
        
        self.visualizer = PixelVisualizer(model=self.model, config=self.config)
    
    def generate_frames(self, V_start: float = 0, V_end: float = 30,
                       duration: float = 0.02, num_frames: int = 30,
                       output_dir: str = "./outputs/animation",
                       resolution: Optional[Tuple[int, int]] = None,
                       t_step: float = 0.002) -> List[str]:
        """
        生成动画帧序列
        
        Args:
            V_start: 初始电压 (V)
            V_end: 最终电压 (V)
            duration: 总时长 (s)
            num_frames: 帧数
            output_dir: 输出目录
            resolution: 分辨率
            t_step: 阶跃时间 (s)
            
        Returns:
            生成的帧文件路径列表
        """
        # 确保输出目录存在
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # 计算动态响应
        t_array, eta_array = self.model.aperture_step_response(
            V_start=V_start, V_end=V_end, 
            duration=duration, t_step=t_step, 
            num_points=num_frames
        )
        
        print(f"🎬 生成 {num_frames} 帧动画...")
        
        frame_paths = []
        
        for i, (t, eta) in enumerate(zip(t_array, eta_array)):
            # 计算当前电压
            if t < t_step:
                V_current = V_start
            else:
                V_current = V_end
            
            # 生成帧
            frame_path = os.path.join(output_dir, f"frame_{i:04d}.png")
            
            self.visualizer.render(
                voltage=V_current,
                time=t,
                save_path=frame_path,
                resolution=resolution
            )
            
            frame_paths.append(frame_path)
            print(f"  帧 {i+1}/{num_frames}: t={t*1000:.1f}ms, η={eta*100:.1f}%")
        
        print(f"✅ 动画帧已保存到 {output_dir}")
        print(f"   {self.get_ffmpeg_command(output_dir)}")
        
        return frame_paths
    
    def get_ffmpeg_command(self, output_dir: str,
                          output_file: str = "aperture_animation.mp4",
                          framerate: int = 10) -> str:
        """
        获取 ffmpeg 合成命令
        
        Args:
            output_dir: 帧文件目录
            output_file: 输出视频文件名
            framerate: 帧率
            
        Returns:
            ffmpeg 命令字符串
        """
        return f"ffmpeg -framerate {framerate} -i {output_dir}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p {output_file}"


# ============================================================
# DataExporter 类
# ============================================================

class DataExporter:
    """
    可视化数据导出器
    
    导出模型预测结果为 JSON 格式。
    
    Example:
        >>> exporter = DataExporter()
        >>> data = exporter.export_prediction(voltage=30, output_path='prediction.json')
    """
    
    def __init__(self, model: Optional[EnhancedApertureModel] = None):
        """
        初始化导出器
        
        Args:
            model: EnhancedApertureModel 实例
        """
        if model is None:
            self.model = EnhancedApertureModel()
        else:
            self.model = model
    
    def export_prediction(self, voltage: float, time: Optional[float] = None,
                         output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        导出模型预测结果为 JSON
        
        Args:
            voltage: 电压 (V)
            time: 时间 (s)
            output_path: 输出路径，如果为 None 则只返回字典
            
        Returns:
            预测结果字典
        """
        # 获取预测
        prediction = self.model.predict_enhanced(voltage, time)
        
        # 构建导出数据
        export_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "model_version": "EnhancedApertureModel v1.0",
                "units": {
                    "voltage": "V",
                    "time": "s",
                    "theta": "degrees",
                    "r_open": "m",
                    "h": "m"
                }
            },
            "prediction": {
                "voltage": voltage,
                "time": time,
                "effective_voltage": prediction.get('effective_voltage', voltage),
                "charging_progress": prediction.get('charging_progress', 100.0),
                "theta": prediction['theta'],
                "aperture_ratio": prediction['aperture_ratio'],
                "aperture_percent": prediction['aperture_ratio'] * 100,
                "r_open": float(prediction['r_open']),
                "volume_error": prediction['volume_error']
            },
            "ink_distribution": {
                "r": prediction['r'].tolist(),
                "h": prediction['h'].tolist()
            }
        }
        
        # 保存到文件
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            print(f"📄 数据已导出: {output_path}")
        
        return export_data
    
    def export_animation_data(self, V_start: float, V_end: float,
                             duration: float = 0.02, num_points: int = 100,
                             t_step: float = 0.002,
                             output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        导出动画数据为 JSON
        
        Args:
            V_start: 初始电压
            V_end: 最终电压
            duration: 时长
            num_points: 数据点数
            t_step: 阶跃时间
            output_path: 输出路径
            
        Returns:
            动画数据字典
        """
        # 计算动态响应
        t_array, eta_array = self.model.aperture_step_response(
            V_start=V_start, V_end=V_end,
            duration=duration, t_step=t_step,
            num_points=num_points
        )
        
        # 获取每个时间点的详细预测
        frames_data = []
        for t, eta in zip(t_array, eta_array):
            V_current = V_start if t < t_step else V_end
            prediction = self.model.predict_enhanced(V_current, t)
            
            frames_data.append({
                "time": float(t),
                "voltage": V_current,
                "theta": prediction['theta'],
                "aperture_ratio": float(eta),
                "r_open": float(prediction['r_open'])
            })
        
        # 构建导出数据
        export_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "model_version": "EnhancedApertureModel v1.0",
                "animation_params": {
                    "V_start": V_start,
                    "V_end": V_end,
                    "duration": duration,
                    "t_step": t_step,
                    "num_points": num_points
                }
            },
            "time_series": {
                "t": t_array.tolist(),
                "eta": eta_array.tolist()
            },
            "frames": frames_data
        }
        
        # 保存到文件
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            print(f"📄 动画数据已导出: {output_path}")
        
        return export_data
    
    @staticmethod
    def load_prediction(input_path: str) -> Dict[str, Any]:
        """
        从 JSON 文件加载预测数据
        
        Args:
            input_path: 输入文件路径
            
        Returns:
            预测数据字典
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 将列表转换回 numpy 数组
        if 'ink_distribution' in data:
            data['ink_distribution']['r'] = np.array(data['ink_distribution']['r'])
            data['ink_distribution']['h'] = np.array(data['ink_distribution']['h'])
        
        return data


# ============================================================
# 便捷函数接口
# ============================================================

def visualize_pixel(voltage: float, time: Optional[float] = None,
                   save_path: Optional[str] = None, **kwargs) -> pv.Plotter:
    """
    快速生成单个电压状态的 3D 可视化
    
    Args:
        voltage: 电压 (V)
        time: 时间 (s)
        save_path: 保存路径
        **kwargs: 传递给 PixelVisualizer.render() 的其他参数
        
    Returns:
        PyVista Plotter 对象
    """
    visualizer = PixelVisualizer()
    return visualizer.render(voltage, time, save_path, **kwargs)


def visualize_comparison(voltages: List[float] = [0, 15, 30],
                        save_path: Optional[str] = None, **kwargs) -> pv.Plotter:
    """
    快速生成多电压对比图
    
    Args:
        voltages: 电压列表
        save_path: 保存路径
        **kwargs: 传递给 PixelVisualizer.render_comparison() 的其他参数
        
    Returns:
        PyVista Plotter 对象
    """
    visualizer = PixelVisualizer()
    return visualizer.render_comparison(voltages, save_path, **kwargs)


def visualize_ink_profile(voltage: float,
                         save_path: Optional[str] = None) -> pv.Plotter:
    """
    快速生成油墨高度剖面图
    
    Args:
        voltage: 电压 (V)
        save_path: 保存路径
        
    Returns:
        PyVista Plotter 对象
    """
    visualizer = PixelVisualizer()
    return visualizer.render_ink_profile(voltage, save_path)


def generate_animation(V_start: float = 0, V_end: float = 30,
                      num_frames: int = 30,
                      output_dir: str = "./outputs/animation") -> List[str]:
    """
    快速生成动画帧
    
    Args:
        V_start: 初始电压 (V)
        V_end: 最终电压 (V)
        num_frames: 帧数
        output_dir: 输出目录
        
    Returns:
        生成的帧文件路径列表
    """
    engine = AnimationEngine()
    return engine.generate_frames(V_start, V_end, num_frames=num_frames, output_dir=output_dir)


def export_data(voltage: float, time: Optional[float] = None,
               output_path: Optional[str] = None) -> Dict[str, Any]:
    """
    快速导出预测数据
    
    Args:
        voltage: 电压 (V)
        time: 时间 (s)
        output_path: 输出路径
        
    Returns:
        预测数据字典
    """
    exporter = DataExporter()
    return exporter.export_prediction(voltage, time, output_path)


def interactive_visualization(initial_voltage: float = 0.0):
    """
    启动交互式可视化
    
    Args:
        initial_voltage: 初始电压
    """
    visualizer = PixelVisualizer()
    visualizer.interactive(initial_voltage)


# ============================================================
# 演示函数
# ============================================================

def demo_3d_visualization():
    """
    演示 3D 可视化功能
    """
    print("=" * 60)
    print("🔬 EWP 3D Visualization Demo")
    print("=" * 60)
    
    # 确保输出目录存在
    output_dir = "./outputs"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. 生成不同电压下的 3D 可视化
    print("\n📊 生成单电压 3D 可视化...")
    voltages = [0, 15, 30]
    for V in voltages:
        save_path = f"{output_dir}/pixel_3d_{V}V.png"
        visualize_pixel(V, save_path=save_path)
    
    # 2. 生成对比图
    print("\n📊 生成多电压对比图...")
    visualize_comparison(
        voltages=[0, 10, 20, 30],
        save_path=f"{output_dir}/pixel_comparison.png"
    )
    
    # 3. 生成油墨剖面图
    print("\n📊 生成油墨剖面图...")
    for V in [0, 30]:
        save_path = f"{output_dir}/ink_profile_{V}V.png"
        visualize_ink_profile(V, save_path=save_path)
    
    # 4. 导出数据
    print("\n📄 导出预测数据...")
    export_data(30, output_path=f"{output_dir}/prediction_30V.json")
    
    # 5. 打印使用说明
    print("\n" + "=" * 60)
    print("✅ 3D 可视化演示完成!")
    print("=" * 60)
    print("\n📁 输出文件:")
    print(f"   - {output_dir}/pixel_3d_*.png - 单电压 3D 可视化")
    print(f"   - {output_dir}/pixel_comparison.png - 多电压对比图")
    print(f"   - {output_dir}/ink_profile_*.png - 油墨剖面图")
    print(f"   - {output_dir}/prediction_30V.json - 预测数据")
    print("\n📖 使用示例:")
    print("   from ewp_3d_visualizer import visualize_pixel, visualize_comparison")
    print("   visualize_pixel(30, save_path='my_pixel.png')")
    print("   visualize_comparison([0, 15, 30], save_path='comparison.png')")
    print("\n🎮 交互式可视化:")
    print("   from ewp_3d_visualizer import interactive_visualization")
    print("   interactive_visualization()")


# ============================================================
# 主程序入口
# ============================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--demo":
            demo_3d_visualization()
        elif sys.argv[1] == "--interactive":
            interactive_visualization()
        elif sys.argv[1] == "--help":
            print("EWP 3D Visualizer")
            print("Usage:")
            print("  python ewp_3d_visualizer.py --demo        # 运行演示")
            print("  python ewp_3d_visualizer.py --interactive # 交互式可视化")
            print("  python ewp_3d_visualizer.py --help        # 显示帮助")
        else:
            print(f"未知参数: {sys.argv[1]}")
            print("使用 --help 查看帮助")
    else:
        # 默认运行演示
        demo_3d_visualization()
