import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 从1.py导入材料结构和相关函数
try:
    from one import material_structure, dimensions, get_layer_boundaries
except ImportError:
    # 如果导入失败，我们在这里定义必要的数据结构和函数
    # 从1.py复制的材料结构定义
    material_structure = {
        "底层ITO玻璃": {
            "厚度": 27.5e-9,  # 25-30nm，取中间值27.5nm
            "相对介电常数": 9.0,
            "表面张力": 0.072,
            "电导率": 1e-6,
            "层索引": 0,
            "角色": "基底",
            "extends_to_boundary": True,
            "方阻": "60-90Ω/□",
            "界面类型": "刚性"
        },
        "围堰": {
            "厚度": 20e-6,  # 20μm
            "外沿尺寸": [184e-6, 184e-6],  # 外沿184×184μm
            "内沿尺寸": [174e-6, 174e-6],  # 内沿174×174μm
            "宽度": 5e-6,  # 宽度5μm
            "材料": "SU-8光刻胶",
            "相对介电常数": 4.0,  # SU-8典型值
            "表面张力": 0.045,
            "电导率": 1e-14,
            "层索引": 1,
            "角色": "结构支持/像素墙",
            "parent_layer": "FP表面",  # 修改为FP表面
            "vertical_offset": 0.1e-6
        },
        "介电层": {
            "厚度": 0.4e-6,  # 0.4μm
            "材料": "SU-8光刻胶",
            "相对介电常数": 4.0,  # SU-8典型值 (修正：原10.0错误)
            "表面张力": 0.040,
            "电导率": 1e-14,  # SU-8是绝缘体 (修正：原1e-12)
            "层索引": 2,
            "角色": "电场隔离",
            "parent_layer": "底层ITO玻璃"
        },
        "疏水层": {
            "厚度": 0.4e-6,  # 0.4μm
            "材料": "Teflon AF 1600X",
            "相对介电常数": 1.93,  # Teflon AF 1600X典型值
            "表面张力": 0.016,  # 超疏水特性
            "电导率": 1e-16,  # 极佳绝缘体
            "接触角": 120.0,  # 对水的接触角
            "层索引": 3,
            "角色": "控制润湿性/超疏水层",
            "parent_layer": "介电层"
        },
        "油墨层": {
            "厚度": 3e-6,  # 3μm
            "填充尺寸": [174e-6, 174e-6],  # 174×174μm
            "相对介电常数": 3.0,
            "表面张力": 0.030,
            "电导率": 1e-12,
            "层索引": 4,
            "角色": "显示介质",
            "parent_layers": ["疏水层", "围堰"],  # 修改为多个父层
            "parent_boundary_type": "围堰内四个侧壁靠底面一部分",
            "vertical_offset": 0.1e-6
        },
        "极性液体层": {
            "厚度": 17e-6,  # 17μm
            "填充尺寸": [174e-6, 174e-6],  # 174×174μm
            "相对介电常数": 78.0,
            "表面张力": 0.072,
            "电导率": 5e-5,
            "层索引": 5,
            "角色": "电润湿驱动",
            "parent_layers": ["油墨层", "围堰"],  # 修改为多个父层
            "parent_boundary_type": "围堰内四个侧壁靠顶的部分",
            "adheres_to": "顶层ITO层",  # 与顶面ITO紧密贴合
            "vertical_offset": 0.1e-6
        },
        "顶层ITO层": {
            "厚度": 27.5e-9,  # 25-30nm，取中间值27.5nm
            "相对介电常数": 9.0,
            "表面张力": 0.072,
            "电导率": 1e-6,
            "层索引": 6,
            "角色": "透明电极",
            "extends_to_boundary": True,
            "方阻": "60-90Ω/□",
            "界面类型": "刚性"
        }
    }
    
    # 像素尺寸和其他关键参数
    # 计算实际总厚度：底层ITO(27.5nm) + 介电层(0.4μm) + 疏水层(0.4μm) + 围堰高度(20μm，包含油墨层和极性液体层) + 顶层ITO(27.5nm)
    # 注意：油墨层(3μm)和极性液体层(17μm)都在围堰内部，它们的总厚度(3+17=20μm)等于围堰高度
    total_thickness = 27.5e-9 + 0.4e-6 + 0.4e-6 + 20e-6 + 27.5e-9
    
    dimensions = {
        "像素尺寸": {
            "宽度": 184e-6,  # 184μm
            "高度": 184e-6,  # 184μm
            "总厚度": total_thickness  # 实际总厚度
        },
        "界面特性": {
            "接触角": 120.0,  # 与疏水层一致 (Teflon AF 1600X)
            "接触角滞后": 5.0,
            "界面厚度": 1e-9  # 1nm
        },
        "电学参数": {
            "驱动电压": "0-30.0",  # 0~30V
            "最大工作电压": 30.0,  # 40V    
            "频率范围": [10, 1000]  # Hz
        }   
    }
    
    def get_layer_boundaries():
        """
        获取各层的边界位置 - 正确处理图层关系和父层结构
        支持多层父层、边界类型和特殊贴合关系
        """
        boundaries = {}
        
        # 确保FP表面作为一个虚拟层存在
        boundaries["FP表面"] = {
            "start": 0.0,
            "end": 0.0,
            "thickness": 0.0,
            "vertical_offset": 0.0,
            "extends_to_boundary": True
        }
        
        # 按层索引排序，确保从底层开始构建
        sorted_layers = sorted(material_structure.items(), key=lambda x: x[1]["层索引"])
        
        for layer_name, layer_props in sorted_layers:
            layer_thickness = layer_props["厚度"]
            vertical_offset = layer_props.get("vertical_offset", 0.0)
            
            # 确定图层起始位置
            if layer_name == "底层ITO玻璃":
                # 底层从0开始
                layer_start = 0
            elif "parent_layer" in layer_props and layer_props["parent_layer"] in boundaries:
                # 单个父层的情况
                parent_end = boundaries[layer_props["parent_layer"]]["end"]
                layer_start = parent_end + vertical_offset
            elif "parent_layers" in layer_props:  # 处理多个父层的情况
                layer_start = 0.0
                # 对于多个父层，考虑所有父层的边界情况
                for parent in layer_props["parent_layers"]:
                    if parent in boundaries:
                        if parent == "疏水层":
                            # 对于疏水层，使用其结束位置作为起始参考
                            parent_end = boundaries[parent]["end"]
                            layer_start = max(layer_start, parent_end + vertical_offset)
                        elif parent == "围堰" and layer_name == "油墨层":
                            # 油墨层在围堰内靠底面部分
                            # 确保油墨层底部与疏水层接触，顶部到指定高度
                            if "疏水层" in boundaries:
                                parent_end = boundaries["疏水层"]["end"]
                                layer_start = parent_end + vertical_offset
                        elif parent == "围堰" and layer_name == "极性液体层":
                            # 极性液体层在围堰内靠顶部分
                            pass
                        elif parent == "油墨层":
                            # 极性液体层在油墨层之上
                            parent_end = boundaries[parent]["end"]
                            layer_start = max(layer_start, parent_end + vertical_offset)
            elif layer_name == "顶层ITO层":
                # 顶层应该在所有其他图层之上
                max_end = max(b["end"] for b in boundaries.values())
                layer_start = max_end
                # 如果有层需要与顶层ITO紧密贴合，调整该层的结束位置
                for existing_layer, existing_props in boundaries.items():
                    if existing_props.get("adheres_to") == "顶层ITO层":
                        # 确保极性液体层与顶层ITO层紧密贴合
                        existing_props["end"] = layer_start
                        boundaries[existing_layer] = existing_props
            else:
                # 默认起始位置
                layer_start = vertical_offset
            
            # 计算图层结束位置
            layer_end = layer_start + layer_thickness
            
            # 特殊处理：确保油墨层和极性液体层在围堰内部，且它们的总厚度等于围堰高度
            if layer_name == "围堰":
                # 围堰的垂直位置从FP表面开始
                if "FP表面" in boundaries:
                    layer_start = boundaries["FP表面"]["end"] + vertical_offset
                    layer_end = layer_start + layer_thickness
            elif layer_name == "极性液体层":
                # 确保极性液体层在围堰内部，并且顶部与顶层ITO层接触
                if "围堰" in boundaries and "油墨层" in boundaries:
                    # 极性液体层顶部受限于围堰顶部
                    weir_end = boundaries["围堰"]["end"]
                    # 极性液体层底部从油墨层顶部开始
                    ink_end = boundaries["油墨层"]["end"]
                    layer_start = ink_end + vertical_offset
                    # 确保极性液体层在围堰内部
                    layer_end = weir_end  # 极性液体层顶部与围堰顶部齐平
            elif layer_name == "油墨层":
                # 确保油墨层在围堰内部
                if "围堰" in boundaries:
                    weir_start = boundaries["围堰"]["start"]
                    weir_end = boundaries["围堰"]["end"]
                    # 确保油墨层底部与围堰底部对齐
                    layer_start = weir_start + vertical_offset
                    # 油墨层厚度保持不变，但不能超过围堰高度
                    layer_end = min(layer_start + layer_thickness, weir_end)
            
            # 为层添加边界信息
            boundary_info = {
                "start": layer_start,
                "end": layer_end,
                "thickness": layer_thickness,
                "vertical_offset": vertical_offset,
                "extends_to_boundary": layer_props.get("extends_to_boundary", False)
            }
            
            # 添加特殊属性信息
            if "parent_boundary_type" in layer_props:
                boundary_info["parent_boundary_type"] = layer_props["parent_boundary_type"]
            
            if "adheres_to" in layer_props:
                boundary_info["adheres_to"] = layer_props["adheres_to"]
            
            boundaries[layer_name] = boundary_info
        
        return boundaries

def create_material_color_map():
    """
    创建材料层的颜色映射 - 改进版：更鲜明的颜色方案
    """
    # 使用更鲜明、对比更强烈的颜色方案
    colors = {
        "底层ITO玻璃": "#00CC00",  # 绿色
        "围堰": "#FF9900",      # 橙色
        "介电层": "#FF9900",     # 橙色
        "疏水层": "#9900CC",      # 紫色
        "油墨层": "#FF0000",      # 红色
        "极性液体层": "#00FFFF",   # 亮青色
        "顶层ITO层": "#00CC00"     # 绿色
    }
    return colors

# 定义材质属性
def get_material_properties(layer_name):
    """
    获取各层的材质属性，以增强3D效果
    """
    properties = {
        "底层ITO玻璃": {"opacity": 0.9, "specular": 0.4, "diffuse": 0.6, "ambient": 0.2},
        "围堰": {"opacity": 0.95, "specular": 0.3, "diffuse": 0.7, "ambient": 0.2},
        "介电层": {"opacity": 0.85, "specular": 0.5, "diffuse": 0.5, "ambient": 0.1},
        "疏水层": {"opacity": 0.8, "specular": 0.6, "diffuse": 0.4, "ambient": 0.1},
        "油墨层": {"opacity": 0.9, "specular": 0.3, "diffuse": 0.7, "ambient": 0.2},
        "极性液体层": {"opacity": 0.7, "specular": 0.8, "diffuse": 0.3, "ambient": 0.1},
        "顶层ITO层": {"opacity": 0.8, "specular": 0.5, "diffuse": 0.5, "ambient": 0.2}
    }
    return properties.get(layer_name, {"opacity": 0.8, "specular": 0.4, "diffuse": 0.6, "ambient": 0.2})

def create_3d_pixel_structure():
    """
    创建电润湿像素的3D结构 - 改进版：更高质量的可视化
    """
    # 获取像素尺寸
    pixel_width = dimensions["像素尺寸"]["宽度"]
    pixel_height = dimensions["像素尺寸"]["高度"]
    
    # 获取各层边界
    boundaries = get_layer_boundaries()
    
    # 创建颜色映射
    colors = create_material_color_map()
    
    # 创建一个Plotter对象 - 更高分辨率
    plotter = pv.Plotter(window_size=[1920, 1080], off_screen=True)
    
    # 添加高级光照
    # 设置环境光
    plotter.set_background('black', top='gray')
    # 添加光源
    light1 = pv.Light(position=(1, 1, 1), focal_point=(0, 0, 0), intensity=0.8)
    light2 = pv.Light(position=(-1, -1, 1), focal_point=(0, 0, 0), intensity=0.6)
    plotter.add_light(light1)
    plotter.add_light(light2)
    
    # 创建各层的3D模型 - 更高精度
    models = []
    
    # 为了更好的可视化，我们需要调整层的尺寸比例，使层厚度在视觉上更明显
    # 计算最大层厚度用于缩放
    max_thickness = max(layer["end"] - layer["start"] for layer in boundaries.values())
    scale_factor = 1e6  # 缩放因子，使纳米级厚度可见
    
    # 先创建背景网格作为参考
    bounds = [-pixel_width/2*scale_factor, pixel_width/2*scale_factor, 
              -pixel_height/2*scale_factor, pixel_height/2*scale_factor, 
              0, max(boundary["end"] for boundary in boundaries.values())*scale_factor*5]
    
    # 调整层的位置和厚度，使视觉效果更好
    adjusted_boundaries = {}
    for layer_name, props in boundaries.items():
        adjusted_boundaries[layer_name] = {
            "start": props["start"] * scale_factor,
            "end": props["end"] * scale_factor,
            "extends_to_boundary": props["extends_to_boundary"]
        }
    
    # 为了显示内部结构，我们将创建一个半透明的外部外壳
    # 先创建底层ITO玻璃
    bottom_layer = adjusted_boundaries["底层ITO玻璃"]
    if bottom_layer["extends_to_boundary"]:
        x_min, x_max = -pixel_width/2*scale_factor, pixel_width/2*scale_factor
        y_min, y_max = -pixel_height/2*scale_factor, pixel_height/2*scale_factor
    else:
        x_min, x_max = -pixel_width/3*scale_factor, pixel_width/3*scale_factor
        y_min, y_max = -pixel_height/3*scale_factor, pixel_height/3*scale_factor
    
    # 更高分辨率的网格
    grid = pv.RectilinearGrid(
        np.linspace(x_min, x_max, 40),
        np.linspace(y_min, y_max, 40),
        np.linspace(bottom_layer["start"], bottom_layer["end"], 10)
    )
    models.append((grid, "底层ITO玻璃"))
    
    # 创建围堰 - 作为框架结构
    weir_layer = adjusted_boundaries["围堰"]
    # 使用材料结构中定义的围堰尺寸参数
    wall_thickness = material_structure["围堰"]["宽度"] * scale_factor
    inner_width, inner_height = material_structure["围堰"]["内沿尺寸"]
    outer_width, outer_height = material_structure["围堰"]["外沿尺寸"]
    
    # 计算内沿和外沿的坐标
    inner_x_min, inner_x_max = -inner_width/2*scale_factor, inner_width/2*scale_factor
    inner_y_min, inner_y_max = -inner_height/2*scale_factor, inner_height/2*scale_factor
    outer_x_min, outer_x_max = -outer_width/2*scale_factor, outer_width/2*scale_factor
    outer_y_min, outer_y_max = -outer_height/2*scale_factor, outer_height/2*scale_factor
    
    # 创建围堰的四个侧面（外沿）
    # 前墙（外沿）
    wall1 = pv.Box(
        bounds=[outer_x_min, outer_x_max, 
                outer_y_min, outer_y_min + wall_thickness, 
                weir_layer["start"], weir_layer["end"]]
    )
    models.append((wall1, "围堰"))
    
    # 后墙（外沿）
    wall2 = pv.Box(
        bounds=[outer_x_min, outer_x_max, 
                outer_y_max - wall_thickness, outer_y_max, 
                weir_layer["start"], weir_layer["end"]]
    )
    models.append((wall2, "围堰"))
    
    # 左墙（外沿）
    wall3 = pv.Box(
        bounds=[outer_x_min, outer_x_min + wall_thickness, 
                outer_y_min + wall_thickness, outer_y_max - wall_thickness, 
                weir_layer["start"], weir_layer["end"]]
    )
    models.append((wall3, "围堰"))
    
    # 右墙（外沿）
    wall4 = pv.Box(
        bounds=[outer_x_max - wall_thickness, outer_x_max, 
                outer_y_min + wall_thickness, outer_y_max - wall_thickness, 
                weir_layer["start"], weir_layer["end"]]
    )
    models.append((wall4, "围堰"))
    
    # 创建介电层 - 184×184×0.4μm
    if "介电层" in adjusted_boundaries:
        dielectric_layer = adjusted_boundaries["介电层"]
        # 介电层应延伸到边界，覆盖整个像素区域
        x_min_dielectric, x_max_dielectric = -pixel_width/2*scale_factor, pixel_width/2*scale_factor
        y_min_dielectric, y_max_dielectric = -pixel_height/2*scale_factor, pixel_height/2*scale_factor
        
        dielectric_grid = pv.RectilinearGrid(
            np.linspace(x_min_dielectric, x_max_dielectric, 40),
            np.linspace(y_min_dielectric, y_max_dielectric, 40),
            np.linspace(dielectric_layer["start"], dielectric_layer["end"], 10)
        )
        models.append((dielectric_grid, "介电层"))
    
    # 创建疏水层 - 184×184×0.4μm
    if "疏水层" in adjusted_boundaries:
        hydrophobic_layer = adjusted_boundaries["疏水层"]
        # 疏水层应延伸到边界，覆盖整个像素区域
        x_min_hydrophobic, x_max_hydrophobic = -pixel_width/2*scale_factor, pixel_width/2*scale_factor
        y_min_hydrophobic, y_max_hydrophobic = -pixel_height/2*scale_factor, pixel_height/2*scale_factor
        
        hydrophobic_grid = pv.RectilinearGrid(
            np.linspace(x_min_hydrophobic, x_max_hydrophobic, 40),
            np.linspace(y_min_hydrophobic, y_max_hydrophobic, 40),
            np.linspace(hydrophobic_layer["start"], hydrophobic_layer["end"], 10)
        )
        models.append((hydrophobic_grid, "疏水层"))
    
    # 创建油墨层 - 174×174×3μm，位于围堰内部，底部与疏水层接触，四周与围堰底部接触
    if "油墨层" in adjusted_boundaries:
        ink_layer = adjusted_boundaries["油墨层"]
        # 使用材料结构中定义的填充尺寸
        ink_width, ink_height = material_structure["油墨层"]["填充尺寸"]
        x_min_ink, x_max_ink = -ink_width/2*scale_factor, ink_width/2*scale_factor
        y_min_ink, y_max_ink = -ink_height/2*scale_factor, ink_height/2*scale_factor
        
        # 确保油墨层底部与疏水层接触，顶部到指定高度
        ink_z_start = ink_layer["start"]
        ink_z_end = ink_layer["end"]
        
        # 更精确的网格划分，特别是在z方向
        ink_grid = pv.RectilinearGrid(
            np.linspace(x_min_ink, x_max_ink, 40),
            np.linspace(y_min_ink, y_max_ink, 40),
            np.linspace(ink_z_start, ink_z_end, 20)  # 更细的z轴划分
        )
        models.append((ink_grid, "油墨层"))
    
    # 创建极性液体层 - 174×174×17μm，位于围堰内部，底部与油墨层接触，四周与围堰上部接触，顶部与ITO层紧密贴合
    if "极性液体层" in adjusted_boundaries:
        polar_layer = adjusted_boundaries["极性液体层"]
        top_layer = adjusted_boundaries["顶层ITO层"]
        
        # 使用材料结构中定义的填充尺寸
        polar_width, polar_height = material_structure["极性液体层"]["填充尺寸"]
        x_min_polar, x_max_polar = -polar_width/2*scale_factor, polar_width/2*scale_factor
        y_min_polar, y_max_polar = -polar_height/2*scale_factor, polar_height/2*scale_factor
        
        # 确保极性液体层底部与油墨层接触，顶部与顶层ITO层紧密贴合
        polar_z_start = polar_layer["start"]
        polar_z_end = top_layer["start"]  # 极性液体层顶部与ITO层底部接触
        
        # 更精确的网格划分，特别是在z方向
        polar_grid = pv.RectilinearGrid(
            np.linspace(x_min_polar, x_max_polar, 40),
            np.linspace(y_min_polar, y_max_polar, 40),
            np.linspace(polar_z_start, polar_z_end, 20)  # 更细的z轴划分
        )
        models.append((polar_grid, "极性液体层"))
    
    # 创建顶层ITO层 - 184×184μm
    top_layer = adjusted_boundaries["顶层ITO层"]
    # 顶层ITO应延伸到边界，覆盖整个像素区域
    x_min_top, x_max_top = -pixel_width/2*scale_factor, pixel_width/2*scale_factor
    y_min_top, y_max_top = -pixel_height/2*scale_factor, pixel_height/2*scale_factor
    
    top_grid = pv.RectilinearGrid(
        np.linspace(x_min_top, x_max_top, 40),
        np.linspace(y_min_top, y_max_top, 40),
        np.linspace(top_layer["start"], top_layer["end"], 10)
    )
    models.append((top_grid, "顶层ITO层"))
    
    # 添加所有模型到plotter - 使用材质属性
    for grid, layer_name in models:
        material_props = get_material_properties(layer_name)
        plotter.add_mesh(
            grid,
            color=colors[layer_name],
            opacity=material_props["opacity"],
            specular=material_props["specular"],
            diffuse=material_props["diffuse"],
            ambient=material_props["ambient"],
            show_edges=True,
            edge_color='black',
            line_width=1.0,
            label=layer_name
        )
    
    # 添加高质量的坐标轴和网格
    plotter.show_bounds(
        grid='front', 
        location='outer', 
        all_edges=True,
        xlabel='X (μm)', 
        ylabel='Y (μm)', 
        ztitle='Z (μm)',  # 使用ztitle代替zlabel
        font_size=12,
        color='white'
    )
    
    # 添加图例
    plotter.add_legend(
        loc='upper right',
        bcolor='white',
        border=True
    )
    
    # 设置高质量标题
    plotter.add_title(
        "电润湿显示像素三维结构 - 高分辨率可视化",
        font_size=16,
        color='white',
        shadow=True,
        font='times'
    )
    
    # 优化相机位置和角度
    plotter.camera_position = 'iso'  # 使用等距视角
    plotter.camera.zoom(1.2)  # 调整缩放
    
    # 设置背景
    plotter.set_background('black', top='gray')
    
    return plotter

def create_cross_sectional_view():
    """
    创建像素的横截面视图 - 改进版：更高质量的可视化
    """
    # 获取像素尺寸
    pixel_width = dimensions["像素尺寸"]["宽度"]
    
    # 获取各层边界
    boundaries = get_layer_boundaries()
    
    # 创建颜色映射
    colors = create_material_color_map()
    
    # 创建一个Plotter对象 - 更高分辨率
    plotter = pv.Plotter(window_size=[1920, 1080], off_screen=True)
    
    # 添加高级光照
    # 设置环境光
    plotter.set_background('black', top='gray')
    # 添加光源
    light1 = pv.Light(position=(1, 1, 1), focal_point=(0, 0, 0), intensity=0.8)
    light2 = pv.Light(position=(-1, -1, 1), focal_point=(0, 0, 0), intensity=0.6)
    plotter.add_light(light1)
    plotter.add_light(light2)
    
    # 缩放因子，使纳米级厚度可见
    scale_factor = 1e6 * 2  # 放大倍数更大，使层结构更明显
    
    # 调整层的位置和厚度
    adjusted_boundaries = {}
    for layer_name, props in boundaries.items():
        adjusted_boundaries[layer_name] = {
            "start": props["start"] * scale_factor,
            "end": props["end"] * scale_factor,
            "extends_to_boundary": props["extends_to_boundary"]
        }
    
    # 创建横截面模型
    for layer_name, layer_props in adjusted_boundaries.items():
        z_start = layer_props["start"]
        z_end = layer_props["end"]
        
        # 创建矩形表示层
        if layer_props["extends_to_boundary"] or layer_name == "底层ITO玻璃" or layer_name == "顶层ITO层":
            # 完整层
            x_min, x_max = -pixel_width/2*scale_factor, pixel_width/2*scale_factor
        else:
            # 围堰内的层
            x_min, x_max = -pixel_width/3*scale_factor, pixel_width/3*scale_factor
        
        # 创建高质量的2D矩形（在y=0平面）
        # 使用更高分辨率的网格
        points = [
            [x_min, 0, z_start],
            [x_max, 0, z_start],
            [x_max, 0, z_end],
            [x_min, 0, z_end]
        ]
        faces = [4, 0, 1, 2, 3]
        
        layer = pv.PolyData(points, faces)
        
        # 添加材质属性
        material_props = get_material_properties(layer_name)
        
        # 添加到plotter - 使用材质属性
        plotter.add_mesh(
            layer,
            color=colors[layer_name],
            opacity=material_props["opacity"],
            specular=material_props["specular"],
            diffuse=material_props["diffuse"],
            ambient=material_props["ambient"],
            show_edges=True,
            edge_color='black',
            line_width=1.5,
            label=layer_name
        )
    
    # 添加高质量的坐标轴和网格
    plotter.show_bounds(
        grid='front', 
        location='outer', 
        all_edges=True,
        xlabel='X (μm)', 
        ylabel='Y (μm)', 
        ztitle='Z (μm)',  # 使用ztitle代替zlabel
        font_size=12,
        color='white'
    )
    
    # 添加图例
    plotter.add_legend(
        loc='upper right',
        bcolor='white',
        border=True
    )
    
    # 设置高质量标题
    plotter.add_title(
        "电润湿显示像素横截面视图 - 高分辨率可视化",
        font_size=16,
        color='white',
        shadow=True,
        font='times'
    )
    
    # 优化相机位置
    plotter.camera_position = 'xy'
    plotter.camera.zoom(1.1)
    
    # 设置背景
    plotter.set_background('black', top='gray')
    
    return plotter

def create_electrowetting_effect_visualization(voltage=0.0):
    """
    创建电润湿效应的可视化 - 改进版：高分辨率、高质量渲染
    
    参数:
    - voltage: 施加的电压
    """
    # 获取像素尺寸
    pixel_width = dimensions["像素尺寸"]["宽度"]
    
    # 获取材料属性
    epsilon_0 = 8.854e-12  # 真空介电常数
    epsilon_r = material_structure["介电层"]["相对介电常数"]
    gamma = material_structure["极性液体层"]["表面张力"]
    d = material_structure["介电层"]["厚度"]
    
    # 计算电润湿接触角变化
    # Young-Lippmann方程：cosθ = cosθ0 + (ε0εrV²)/(2γd)
    # 注意：电润湿使接触角减小，cos(θ)增大，所以是 + 号
    theta0 = material_structure["疏水层"]["接触角"]  # 初始接触角来自疏水层 (120°)
    cos_theta = np.cos(np.radians(theta0)) + (epsilon_0 * epsilon_r * voltage**2) / (2 * gamma * d)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # 确保在有效范围内
    theta = np.degrees(np.arccos(cos_theta))
    
    # 创建一个Plotter对象 - 更高分辨率
    plotter = pv.Plotter(window_size=[1920, 1080], off_screen=True)
    
    # 添加高级光照
    # 设置环境光
    plotter.set_background('black', top='gray')
    # 添加光源
    light1 = pv.Light(position=(1, 1, 1), focal_point=(0, 0, 0), intensity=0.8)
    light2 = pv.Light(position=(-1, -1, 1), focal_point=(0, 0, 0), intensity=0.6)
    plotter.add_light(light1)
    plotter.add_light(light2)
    
    # 缩放因子，使纳米级厚度可见
    scale_factor = 1e6 * 3  # 更大的缩放倍数
    
    # 获取各层边界
    boundaries = get_layer_boundaries()
    
    # 创建颜色映射
    colors = create_material_color_map()
    
    # 调整层的位置和厚度
    adjusted_boundaries = {}
    for layer_name, props in boundaries.items():
        adjusted_boundaries[layer_name] = {
            "start": props["start"] * scale_factor,
            "end": props["end"] * scale_factor,
            "extends_to_boundary": props["extends_to_boundary"]
        }
    
    # 添加底层结构
    for layer_name in ["底层ITO玻璃", "围堰", "介电层", "疏水层", "油墨层"]:
        if layer_name in adjusted_boundaries:
            layer_props = adjusted_boundaries[layer_name]
            z_start = layer_props["start"]
            z_end = layer_props["end"]
            
            if layer_props["extends_to_boundary"] or layer_name == "底层ITO玻璃":
                x_min, x_max = -pixel_width/2*scale_factor, pixel_width/2*scale_factor
                y_min, y_max = -pixel_width/2*scale_factor, pixel_width/2*scale_factor
            elif layer_name == "油墨层":
                # 油墨层使用填充尺寸
                ink_width, ink_height = material_structure["油墨层"]["填充尺寸"]
                x_min, x_max = -ink_width/2*scale_factor, ink_width/2*scale_factor
                y_min, y_max = -ink_height/2*scale_factor, ink_height/2*scale_factor
            else:
                x_min, x_max = -pixel_width/3*scale_factor, pixel_width/3*scale_factor
                y_min, y_max = -pixel_width/3*scale_factor, pixel_width/3*scale_factor
            
            # 创建高分辨率矩形网格
            grid = pv.RectilinearGrid(
                np.linspace(x_min, x_max, 40),
                np.linspace(y_min, y_max, 40),
                np.linspace(z_start, z_end, 10)
            )
            
            # 添加材质属性
            material_props = get_material_properties(layer_name)
            
            # 添加到plotter
            plotter.add_mesh(
                grid,
                color=colors[layer_name],
                opacity=material_props["opacity"],
                specular=material_props["specular"],
                diffuse=material_props["diffuse"],
                ambient=material_props["ambient"],
                show_edges=True,
                edge_color='black',
                line_width=1.0,
                label=layer_name
            )
    
    # 创建弯曲的极性液体表面 - 更高质量
    # 获取极性液体层的起始位置（调整后）
    polar_start = adjusted_boundaries["极性液体层"]["start"]
    polar_end = adjusted_boundaries["极性液体层"]["end"]
    
    # 创建更高分辨率的x和y网格
    x = np.linspace(-pixel_width/3*scale_factor, pixel_width/3*scale_factor, 150)
    y = np.linspace(-pixel_width/3*scale_factor, pixel_width/3*scale_factor, 150)
    x_grid, y_grid = np.meshgrid(x, y)
    
    # 计算弯曲表面的z坐标（基于接触角）
    # 使用更准确的球形表面近似
    r = pixel_width/3*scale_factor  # 近似半径
    h = r * np.sin(np.radians(theta))  # 表面高度变化
    
    # 创建一个更平滑的半球形表面
    # 添加更真实的曲率变化
    center_z = polar_start + (polar_end - polar_start) - h
    
    # 计算球心
    sphere_center = (0, 0, center_z - np.sqrt(r**2 - h**2))
    
    # 计算球面上的点
    dist_from_center = np.sqrt((x_grid - sphere_center[0])**2 + 
                              (y_grid - sphere_center[1])**2 + 
                              (center_z - sphere_center[2])**2)
    
    # 计算z坐标
    z_grid = center_z - np.sqrt(r**2 - x_grid**2 - y_grid**2) + (dist_from_center - r) * 0.1
    
    # 仅保留有效区域内的点并平滑边缘
    mask = x_grid**2 + y_grid**2 <= r**2
    z_grid[~mask] = np.nan
    
    # 创建高质量的结构化网格
    surf = pv.StructuredGrid(x_grid, y_grid, z_grid)
    
    # 添加极性液体表面 - 使用高质量设置
    polar_props = get_material_properties("极性液体层")
    plotter.add_mesh(
        surf,
        color=colors["极性液体层"],
        opacity=polar_props["opacity"],
        specular=min(polar_props["specular"] + 0.1, 1.0),  # 液体更闪亮，但不超过1.0
        diffuse=polar_props["diffuse"],
        ambient=polar_props["ambient"],
        smooth_shading=True,
        show_edges=False,
        label=f"极性液体层 (θ={theta:.1f}°)"
    )
    
    # 添加接触角指示器
    # 计算接触点位置
    contact_radius = r * np.sin(np.radians(theta))
    contact_point_x = contact_radius
    contact_point_z = adjusted_boundaries["疏水层"]["end"]
    
    # 绘制接触角弧线
    angle_points = []
    for angle in np.linspace(0, theta, 50):
        r_angle = contact_radius * 0.3  # 弧线半径
        x_angle = contact_point_x - r_angle * np.sin(np.radians(angle))
        z_angle = contact_point_z + r_angle * (1 - np.cos(np.radians(angle)))
        angle_points.append([x_angle, 0, z_angle])
    
    # 创建角度弧线
    if len(angle_points) > 2:
        angle_line = pv.PolyData(angle_points)
        angle_line.lines = np.array([len(angle_points)] + list(range(len(angle_points))))
        
        plotter.add_mesh(
            angle_line,
            color='white',
            line_width=2,
            render_lines_as_tubes=True
        )
        
        # 添加接触角文本
        text_pos = [contact_point_x - contact_radius*0.2, 0, contact_point_z + contact_radius*0.15]
        plotter.add_point_labels(
            [text_pos],
            [f"{theta:.1f}°"],
            font_size=14,
            text_color='white',
            shape_color='black',
            shape_opacity=0.7,
            point_color=None,
            point_size=0
        )
    
    # 添加高质量的坐标轴和网格
    plotter.show_bounds(
        grid='front', 
        location='outer', 
        all_edges=True,
        xlabel='X (μm)', 
        ylabel='Y (μm)', 
        zlabel='Z (μm)',
        font_size=12,
        color='white'
    )
    
    # 添加图例
    plotter.add_legend(
        loc='upper right',
        bcolor='white',
        border=True
    )
    
    # 设置高质量标题
    plotter.add_title(
        f"电润湿效应可视化 - 电压: {voltage}V, 接触角: {theta:.1f}°",
        font_size=16,
        color='white',
        shadow=True,
        font='times'
    )
    
    # 优化相机位置和角度
    plotter.camera_position = 'iso'
    plotter.camera.zoom(1.2)
    
    # 设置背景
    plotter.set_background('black', top='gray')
    
    return plotter

def generate_all_visualizations(save=False, output_dir="./outputs"):
    """
    生成所有可视化并显示或保存
    
    参数:
    - save: 是否保存图像而非显示
    - output_dir: 保存图像的目录
    """
    # 确保输出目录存在
    if save and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1. 3D像素结构
    plotter_3d = create_3d_pixel_structure()
    if save:
        plotter_3d.screenshot(os.path.join(output_dir, "3d_pixel_structure.png"))
        plotter_3d.close()
    else:
        plotter_3d.show()
    
    # 2. 横截面视图
    plotter_cross = create_cross_sectional_view()
    if save:
        plotter_cross.screenshot(os.path.join(output_dir, "cross_sectional_view.png"))
        plotter_cross.close()
    else:
        plotter_cross.show()
    
    # 3. 电润湿效应可视化（不同电压）
    voltages = [0.0, 15.0, 30.0]
    for voltage in voltages:
        plotter_ew = create_electrowetting_effect_visualization(voltage)
        if save:
            plotter_ew.screenshot(os.path.join(output_dir, f"electrowetting_effect_{voltage}V.png"))
            plotter_ew.close()
        else:
            plotter_ew.show()

def create_interactive_visualization():
    """
    创建交互式可视化，允许用户调整电压
    """
    # 创建一个Plotter对象
    plotter = pv.Plotter(window_size=[1024, 768], notebook=False, off_screen=False)
    
    # 获取像素尺寸
    pixel_width = dimensions["像素尺寸"]["宽度"]
    
    # 获取各层边界
    boundaries = get_layer_boundaries()
    
    # 创建颜色映射
    colors = create_material_color_map()
    
    # 添加底层结构
    for layer_name, layer_props in boundaries.items():
        if layer_name in ["底层ITO玻璃", "围堰", "介电层", "疏水层", "顶层ITO层"]:
            z_start = layer_props["start"]
            z_end = layer_props["end"]
            
            if layer_props["extends_to_boundary"] or layer_name == "底层ITO玻璃" or layer_name == "顶层TITO层":
                x_min, x_max = -pixel_width/2, pixel_width/2
                y_min, y_max = -pixel_width/2, pixel_width/2
            else:
                x_min, x_max = -pixel_width/3, pixel_width/3
                y_min, y_max = -pixel_width/3, pixel_width/3
            
            # 创建矩形网格
            grid = pv.RectilinearGrid(
                np.linspace(x_min, x_max, 20),
                np.linspace(y_min, y_max, 20),
                np.linspace(z_start, z_end, 5)
            )
            
            plotter.add_mesh(
                grid,
                color=colors[layer_name],
                opacity=0.8,
                show_edges=True,
                label=layer_name
            )
    
    # 定义更新函数
    def update_voltage(value):
        # 移除之前的极性液体表面
        if hasattr(update_voltage, 'polar_surface'):
            plotter.remove_actor(update_voltage.polar_surface)
        
        voltage = value
        
        # 获取材料属性
        epsilon_0 = 8.854e-12  # 真空介电常数
        epsilon_r = material_structure["介电层"]["相对介电常数"]
        gamma = material_structure["极性液体层"]["表面张力"]
        d = material_structure["介电层"]["厚度"]
        
        # 计算电润湿接触角变化
        # Young-Lippmann方程：cosθ = cosθ0 + (ε0εrV²)/(2γd)
        theta0 = material_structure["疏水层"]["接触角"]  # 初始接触角来自疏水层 (120°)
        cos_theta = np.cos(np.radians(theta0)) + (epsilon_0 * epsilon_r * voltage**2) / (2 * gamma * d)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)  # 确保在有效范围内
        theta = np.degrees(np.arccos(cos_theta))
        
        # 创建弯曲的极性液体表面
        polar_start = boundaries["极性液体层"]["start"]
        polar_end = boundaries["极性液体层"]["end"]
        
        # 创建x和y网格
        x = np.linspace(-pixel_width/3, pixel_width/3, 50)
        y = np.linspace(-pixel_width/3, pixel_width/3, 50)
        x_grid, y_grid = np.meshgrid(x, y)
        
        # 计算弯曲表面的z坐标
        r = pixel_width/3  # 近似半径
        h = r * np.sin(np.radians(theta))  # 表面高度变化
        
        # 创建一个近似的表面
        z_grid = polar_start + (polar_end - polar_start) - h * (1 - np.sqrt(np.clip(1 - (x_grid**2 + y_grid**2)/r**2, 0, 1)))
        
        # 创建表面
        surf = pv.StructuredGrid(x_grid, y_grid, z_grid)
        
        # 添加极性液体表面
        update_voltage.polar_surface = plotter.add_mesh(
            surf,
            color=colors["极性液体层"],
            opacity=0.9,
            show_edges=True,
            label=f"极性液体层 (θ={theta:.1f}°)"
        )
        
        # 更新标题
        # 移除旧标题（简单实现，可能不完美）
        if hasattr(update_voltage, 'title_id'):
            plotter.remove_actor(update_voltage.title_id)
        # 添加新标题
        update_voltage.title_id = plotter.add_title(f"电润湿效应交互式可视化 (电压={voltage}V)")
    
    # 添加滑块控件
    plotter.add_slider_widget(
        update_voltage,
        [0, 40],  # 电压范围 (0-40V)
        title="电压 (V)",
        pointa=(0.05, 0.1),
        pointb=(0.95, 0.1),
        value=0.0
    )
    
    # 添加坐标轴
    plotter.show_bounds(all_edges=True, grid='front', location='outer')
    
    # 添加图例
    plotter.add_legend()
    
    # 设置标题
    update_voltage.title_id = plotter.add_title("电润湿效应交互式可视化")
    
    # 显示交互界面
    plotter.show()

if __name__ == "__main__":
    print("电润湿显示像素三维结构验证器")
    print("=" * 50)
    
    # 自动创建outputs目录
    if not os.path.exists("./outputs"):
        os.makedirs("./outputs")
    
    # 只生成3D结构验证
    print("正在生成并保存3D结构验证图像到outputs目录...")
    # 创建3D像素结构并保存
    plotter = create_3d_pixel_structure()
    plotter.show(interactive=False, screenshot="./outputs/3d_structure_validation.png")
    print("3D结构验证图像已保存到outputs目录")


# ============================================================
# 集成 EnhancedApertureModel 的 3D 可视化
# ============================================================

def create_aperture_model_visualization(voltage=30.0, time=None, save_path=None):
    """
    使用 EnhancedApertureModel 创建基于物理的 3D 可视化
    
    参数:
    - voltage: 施加的电压 (V)
    - time: 时间 (s)，如果提供则使用动态响应
    - save_path: 保存路径，如果为 None 则显示
    
    返回:
    - plotter: PyVista Plotter 对象
    """
    from src.models.aperture_model import EnhancedApertureModel
    
    # 创建增强模型
    model = EnhancedApertureModel()
    
    # 获取预测结果
    result = model.predict_enhanced(voltage, time)
    
    theta = result['theta']
    aperture_ratio = result['aperture_ratio']
    r_open = result['r_open']
    r_array = result['r']
    h_array = result['h']
    
    # 获取像素尺寸
    pixel_width = dimensions["像素尺寸"]["宽度"]
    pixel_height = dimensions["像素尺寸"]["高度"]
    
    # 获取各层边界
    boundaries = get_layer_boundaries()
    
    # 创建颜色映射
    colors = create_material_color_map()
    
    # 创建 Plotter
    plotter = pv.Plotter(window_size=[1920, 1080], off_screen=(save_path is not None))
    
    # 设置背景和光照
    plotter.set_background('black', top='gray')
    light1 = pv.Light(position=(1, 1, 1), focal_point=(0, 0, 0), intensity=0.8)
    light2 = pv.Light(position=(-1, -1, 1), focal_point=(0, 0, 0), intensity=0.6)
    plotter.add_light(light1)
    plotter.add_light(light2)
    
    # 缩放因子
    scale_factor = 1e6
    
    # 调整层边界
    adjusted_boundaries = {}
    for layer_name, props in boundaries.items():
        adjusted_boundaries[layer_name] = {
            "start": props["start"] * scale_factor,
            "end": props["end"] * scale_factor,
            "extends_to_boundary": props.get("extends_to_boundary", False)
        }
    
    # 添加底层结构（ITO、介电层、疏水层、围堰）
    for layer_name in ["底层ITO玻璃", "介电层", "疏水层"]:
        if layer_name in adjusted_boundaries:
            layer_props = adjusted_boundaries[layer_name]
            z_start = layer_props["start"]
            z_end = layer_props["end"]
            
            x_min, x_max = -pixel_width/2*scale_factor, pixel_width/2*scale_factor
            y_min, y_max = -pixel_height/2*scale_factor, pixel_height/2*scale_factor
            
            grid = pv.RectilinearGrid(
                np.linspace(x_min, x_max, 30),
                np.linspace(y_min, y_max, 30),
                np.linspace(z_start, z_end, 5)
            )
            
            material_props = get_material_properties(layer_name)
            plotter.add_mesh(
                grid,
                color=colors[layer_name],
                opacity=material_props["opacity"],
                specular=material_props["specular"],
                show_edges=True,
                edge_color='black',
                line_width=0.5,
                label=layer_name
            )
    
    # 添加围堰
    if "围堰" in adjusted_boundaries:
        weir_layer = adjusted_boundaries["围堰"]
        wall_thickness = material_structure["围堰"]["宽度"] * scale_factor
        inner_width, inner_height = material_structure["围堰"]["内沿尺寸"]
        outer_width, outer_height = material_structure["围堰"]["外沿尺寸"]
        
        inner_x = inner_width/2*scale_factor
        inner_y = inner_height/2*scale_factor
        outer_x = outer_width/2*scale_factor
        outer_y = outer_height/2*scale_factor
        
        # 四面围堰
        walls = [
            pv.Box(bounds=[-outer_x, outer_x, -outer_y, -outer_y + wall_thickness, 
                          weir_layer["start"], weir_layer["end"]]),
            pv.Box(bounds=[-outer_x, outer_x, outer_y - wall_thickness, outer_y, 
                          weir_layer["start"], weir_layer["end"]]),
            pv.Box(bounds=[-outer_x, -outer_x + wall_thickness, -outer_y + wall_thickness, outer_y - wall_thickness, 
                          weir_layer["start"], weir_layer["end"]]),
            pv.Box(bounds=[outer_x - wall_thickness, outer_x, -outer_y + wall_thickness, outer_y - wall_thickness, 
                          weir_layer["start"], weir_layer["end"]]),
        ]
        
        for wall in walls:
            plotter.add_mesh(
                wall,
                color=colors["围堰"],
                opacity=0.9,
                show_edges=True,
                edge_color='black'
            )
    
    # 使用 EnhancedApertureModel 的油墨分布创建 3D 油墨层
    if "油墨层" in adjusted_boundaries:
        ink_layer = adjusted_boundaries["油墨层"]
        ink_width, ink_height = material_structure["油墨层"]["填充尺寸"]
        
        # 创建油墨分布的 3D 表示
        # 使用模型计算的 r 和 h 数组
        r_scaled = r_array * scale_factor
        h_scaled = h_array * scale_factor
        
        # 创建圆柱坐标网格
        n_theta = 60
        n_r = len(r_array)
        
        theta_angles = np.linspace(0, 2*np.pi, n_theta)
        
        # 创建 3D 点
        points = []
        for i, r_val in enumerate(r_scaled):
            for t in theta_angles:
                x = r_val * np.cos(t)
                y = r_val * np.sin(t)
                z_base = ink_layer["start"]
                z_top = z_base + h_scaled[i]
                points.append([x, y, z_base])
                points.append([x, y, z_top])
        
        # 如果有开口（透明区域），创建环形油墨层
        if aperture_ratio > 0.01:
            r_open_scaled = r_open * scale_factor
            
            # 创建透明区域（白色圆柱）
            if r_open_scaled > 0:
                open_cylinder = pv.Cylinder(
                    center=(0, 0, (ink_layer["start"] + ink_layer["end"])/2),
                    direction=(0, 0, 1),
                    radius=r_open_scaled,
                    height=ink_layer["end"] - ink_layer["start"]
                )
                plotter.add_mesh(
                    open_cylinder,
                    color='white',
                    opacity=0.3,
                    label=f'Open region (r={r_open*1e6:.1f}um)'
                )
            
            # 创建油墨环形区域
            outer_radius = ink_width/2 * scale_factor
            ink_ring = pv.Disc(
                center=(0, 0, ink_layer["start"]),
                inner=r_open_scaled,
                outer=outer_radius,
                normal=(0, 0, 1),
                r_res=30,
                c_res=60
            )
            
            # 挤出成 3D
            h_avg = np.mean(h_scaled[h_scaled > 0]) if np.any(h_scaled > 0) else 0
            if h_avg > 0:
                ink_3d = ink_ring.extrude([0, 0, h_avg])
                plotter.add_mesh(
                    ink_3d,
                    color=colors["油墨层"],
                    opacity=0.85,
                    specular=0.3,
                    show_edges=False,
                    label=f'Ink layer (h={h_avg/scale_factor*1e6:.2f}um)'
                )
        else:
            # 无开口，油墨均匀分布
            ink_box = pv.Box(
                bounds=[-ink_width/2*scale_factor, ink_width/2*scale_factor,
                       -ink_height/2*scale_factor, ink_height/2*scale_factor,
                       ink_layer["start"], ink_layer["end"]]
            )
            plotter.add_mesh(
                ink_box,
                color=colors["油墨层"],
                opacity=0.85,
                show_edges=True,
                label='Ink layer (uniform)'
            )
    
    # 添加极性液体层
    if "极性液体层" in adjusted_boundaries:
        polar_layer = adjusted_boundaries["极性液体层"]
        polar_width, polar_height = material_structure["极性液体层"]["填充尺寸"]
        
        polar_box = pv.Box(
            bounds=[-polar_width/2*scale_factor, polar_width/2*scale_factor,
                   -polar_height/2*scale_factor, polar_height/2*scale_factor,
                   polar_layer["start"], polar_layer["end"]]
        )
        plotter.add_mesh(
            polar_box,
            color=colors["极性液体层"],
            opacity=0.5,
            specular=0.8,
            show_edges=False,
            label='Polar liquid'
        )
    
    # 添加顶层 ITO
    if "顶层ITO层" in adjusted_boundaries:
        top_layer = adjusted_boundaries["顶层ITO层"]
        top_grid = pv.RectilinearGrid(
            np.linspace(-pixel_width/2*scale_factor, pixel_width/2*scale_factor, 30),
            np.linspace(-pixel_height/2*scale_factor, pixel_height/2*scale_factor, 30),
            np.linspace(top_layer["start"], top_layer["end"], 3)
        )
        plotter.add_mesh(
            top_grid,
            color=colors["顶层ITO层"],
            opacity=0.6,
            show_edges=True,
            label='Top ITO'
        )
    
    # 添加标题和信息
    title = f"EWP Pixel @ {voltage}V"
    if time is not None:
        title += f", t={time*1000:.1f}ms"
    title += f"\nθ={theta:.1f}°, η={aperture_ratio*100:.1f}%"
    
    plotter.add_title(title, font_size=14, color='white')
    
    # 添加图例
    plotter.add_legend(loc='upper right', bcolor='white', border=True)
    
    # 设置相机
    plotter.camera_position = 'iso'
    plotter.camera.zoom(1.1)
    
    # 保存或显示
    if save_path:
        plotter.screenshot(save_path)
        print(f"📊 3D 可视化已保存: {save_path}")
        plotter.close()
    
    return plotter


def create_aperture_animation(V_start=0, V_end=30, num_frames=30, output_dir="./outputs"):
    """
    创建开口率动态响应的动画帧
    
    参数:
    - V_start: 初始电压 (V)
    - V_end: 最终电压 (V)
    - num_frames: 帧数
    - output_dir: 输出目录
    """
    from src.models.aperture_model import EnhancedApertureModel
    
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 创建模型
    model = EnhancedApertureModel()
    
    # 计算动态响应
    duration = 0.02  # 20ms
    t_step = 0.002   # 2ms 阶跃时间
    
    t_array, eta_array = model.aperture_step_response(
        V_start=V_start, V_end=V_end, duration=duration, t_step=t_step, num_points=num_frames
    )
    
    print(f"🎬 生成 {num_frames} 帧动画...")
    
    for i, (t, eta) in enumerate(zip(t_array, eta_array)):
        # 计算当前电压
        if t < t_step:
            V_current = V_start
        else:
            V_current = V_end
        
        # 生成帧
        save_path = os.path.join(output_dir, f"frame_{i:04d}.png")
        create_aperture_model_visualization(
            voltage=V_current, 
            time=t, 
            save_path=save_path
        )
        
        print(f"  帧 {i+1}/{num_frames}: t={t*1000:.1f}ms, η={eta*100:.1f}%")
    
    print(f"✅ 动画帧已保存到 {output_dir}")
    print(f"   使用 ffmpeg 合成视频: ffmpeg -framerate 10 -i {output_dir}/frame_%04d.png -c:v libx264 aperture_animation.mp4")


def demo_aperture_3d():
    """
    演示 EnhancedApertureModel 的 3D 可视化
    
    注意: 推荐使用新的 ewp_3d_visualizer.py 模块，提供更完整的功能。
    """
    # 尝试使用新的可视化模块
    try:
        from src.visualization.visualizer_3d import demo_3d_visualization
        print("使用新的 visualizer_3d 模块...")
        demo_3d_visualization()
        return
    except ImportError:
        pass
    
    # 回退到原有实现
    print("=" * 60)
    print("🔬 EnhancedApertureModel 3D 可视化演示")
    print("=" * 60)
    
    # 确保输出目录存在
    if not os.path.exists("./outputs"):
        os.makedirs("./outputs")
    
    # 生成不同电压下的 3D 可视化
    voltages = [0, 15, 30]
    
    for V in voltages:
        print(f"\n📊 生成 {V}V 3D 可视化...")
        save_path = f"./outputs/aperture_3d_{V}V.png"
        create_aperture_model_visualization(voltage=V, save_path=save_path)
    
    print("\n✅ 3D 可视化演示完成!")
    print("   图像已保存到 ./outputs/ 目录")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--aperture":
        # 运行 EnhancedApertureModel 3D 可视化演示
        demo_aperture_3d()
    elif len(sys.argv) > 1 and sys.argv[1] == "--new":
        # 使用新的可视化模块
        try:
            from src.visualization.visualizer_3d import demo_3d_visualization
            demo_3d_visualization()
        except ImportError as e:
            print(f"无法导入 visualizer_3d: {e}")
            print("请确保 src/visualization/visualizer_3d.py 存在")
    else:
        # 原有的验证代码
        print("电润湿显示像素三维结构验证器")
        print("=" * 50)
        
        if not os.path.exists("./outputs"):
            os.makedirs("./outputs")
        
        print("正在生成并保存3D结构验证图像到outputs目录...")
        plotter = create_3d_pixel_structure()
        plotter.show(interactive=False, screenshot="./outputs/3d_structure_validation.png")
        print("3D结构验证图像已保存到outputs目录")
        
        print("\n提示:")
        print("  - 运行 'python generate_pyvista_3d.py --aperture' 使用 EnhancedApertureModel 生成 3D 可视化")
        print("  - 运行 'python generate_pyvista_3d.py --new' 使用新的 ewp_3d_visualizer 模块")
        print("  - 推荐: 'python ewp_3d_visualizer.py --demo' 获取完整功能")
