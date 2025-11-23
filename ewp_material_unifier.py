"""
EWP材料属性统一器

此模块负责统一3D结构中的材料属性定义与输入层特征之间的映射关系，
确保电润湿显示像素模型的材料参数在不同组件间保持一致。

主要功能：
- 定义标准化的材料属性命名规范
- 提供材料属性的统一访问接口
- 确保3D结构与输入层间的材料参数一致性
- 支持材料属性的单位转换和标准化处理
"""

import numpy as np
import os
import sys

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class EWPMaterialUnifier:
    """
    电润湿材料属性统一器类
    负责统一材料属性定义并提供一致的访问接口
    """
    
    # 标准材料属性命名映射
    STANDARD_PROPERTY_NAMES = {
        "厚度": "thickness",
        "相对介电常数": "relative_permittivity",
        "表面张力": "surface_tension",
        "电导率": "conductivity",
        "层索引": "layer_index",
        "角色": "role",
        "外沿尺寸": "outer_dimensions",
        "内沿尺寸": "inner_dimensions",
        "宽度": "width",
        "parent_layer": "parent_layer",
        "parent_enclosure": "parent_enclosure",
        "vertical_offset": "vertical_offset",
        "extends_to_boundary": "extends_to_boundary",
        "填充尺寸": "fill_dimensions"
    }
    
    # 反向命名映射（用于输出时使用中文名称）
    REVERSE_PROPERTY_NAMES = {v: k for k, v in STANDARD_PROPERTY_NAMES.items()}
    
    # 材料属性标准范围
    PROPERTY_STANDARD_RANGES = {
        "thickness": (0.0, 50.0e-6),  # [m] 0到50μm
        "relative_permittivity": (1.0, 100.0),  # 1到100
        "surface_tension": (0.01, 0.1),  # [N/m] 0.01到0.1
        "conductivity": (1e-16, 1e-3),  # [S/m] 1e-16到0.001
        "contact_angle": (0.0, 180.0),  # [度]
        "voltage": (0.0, 50.0)  # [V] 0到50V
    }
    
    # 材料属性默认值
    PROPERTY_DEFAULT_VALUES = {
        "thickness": 1.0e-6,
        "relative_permittivity": 1.0,
        "surface_tension": 0.05,
        "conductivity": 1e-12,
        "layer_index": 0,
        "role": "unknown",
        "extends_to_boundary": False,
        "vertical_offset": 0.0
    }
    
    # 输入层特征与材料属性的映射关系
    FEATURE_TO_PROPERTY_MAPPING = {
        # 材料与界面特性映射
        "layer_position": "layer_index",
        "surface_energy": "surface_tension",
        "wettability": "contact_angle",
        "dielectric_factor": "relative_permittivity",
        "capacitance_density": "relative_permittivity",
        
        # 电润湿特异性参数映射
        "electrowetting_number": "relative_permittivity",
        "voltage_saturation": "voltage",
        "dielectric_variation": "relative_permittivity",
        "relaxation_time": "conductivity",
        
        # 流体动力学参数映射
        "surface_tension": "surface_tension",
        "marangoni_effect": "surface_tension"
    }
    
    def __init__(self):
        # 存储标准化的材料结构
        self.standardized_materials = {}
        # 存储维度信息
        self.dimensions = {}
    
    def standardize_material_structure(self, material_structure):
        """
        标准化材料结构，将中文属性名转换为标准英文名
        
        参数:
        - material_structure: 原始材料结构字典
        
        返回:
        - 标准化后的材料结构字典
        """
        standardized = {}
        
        for layer_name, properties in material_structure.items():
            standardized_layer = {}
            
            for prop_name, value in properties.items():
                # 转换属性名到标准英文名
                if prop_name in self.STANDARD_PROPERTY_NAMES:
                    std_prop_name = self.STANDARD_PROPERTY_NAMES[prop_name]
                    standardized_layer[std_prop_name] = value
                else:
                    # 保留未知属性
                    standardized_layer[prop_name] = value
            
            # 添加层名
            standardized_layer["layer_name"] = layer_name
            
            # 确保所有必需属性都有值
            for prop, default in self.PROPERTY_DEFAULT_VALUES.items():
                if prop not in standardized_layer:
                    standardized_layer[prop] = default
            
            standardized[layer_name] = standardized_layer
        
        self.standardized_materials = standardized
        return standardized
    
    def standardize_dimensions(self, dimensions):
        """
        标准化维度信息
        
        参数:
        - dimensions: 原始维度信息字典
        
        返回:
        - 标准化后的维度信息字典
        """
        standardized = {}
        
        # 处理像素尺寸
        if "像素尺寸" in dimensions:
            pixel_dim = dimensions["像素尺寸"]
            standardized["pixel_dimensions"] = {
                "width": pixel_dim.get("宽度", 0.0),
                "height": pixel_dim.get("高度", 0.0),
                "total_thickness": pixel_dim.get("总厚度", 0.0)
            }
        
        # 处理界面特性
        if "界面特性" in dimensions:
            interface_prop = dimensions["界面特性"]
            standardized["interface_properties"] = {
                "contact_angle": interface_prop.get("接触角", 90.0),
                "contact_angle_hysteresis": interface_prop.get("接触角滞后", 0.0),
                "interface_thickness": interface_prop.get("界面厚度", 0.0)
            }
        
        # 处理电学参数
        if "电学参数" in dimensions:
            electric_param = dimensions["电学参数"]
            # 处理驱动电压字符串范围
            drive_voltage = electric_param.get("驱动电压", "0-0")
            if isinstance(drive_voltage, str) and "-" in drive_voltage:
                min_v, max_v = map(float, drive_voltage.split("-"))
            else:
                min_v, max_v = 0.0, float(drive_voltage)
            
            standardized["electrical_parameters"] = {
                "drive_voltage_min": min_v,
                "drive_voltage_max": max_v,
                "max_operating_voltage": float(electric_param.get("最大工作电压", max_v)),
                "frequency_range": electric_param.get("频率范围", [0, 0])
            }
        
        self.dimensions = standardized
        return standardized
    
    def get_material_property(self, layer_name, property_name, use_standard_names=True):
        """
        获取材料属性值，支持中英文属性名
        
        参数:
        - layer_name: 图层名称
        - property_name: 属性名称
        - use_standard_names: 是否使用标准英文名返回
        
        返回:
        - 属性值
        """
        if layer_name not in self.standardized_materials:
            raise ValueError(f"未知的图层名称: {layer_name}")
        
        # 确保使用标准英文名查找
        if property_name in self.STANDARD_PROPERTY_NAMES:
            std_property_name = self.STANDARD_PROPERTY_NAMES[property_name]
        else:
            std_property_name = property_name
        
        layer = self.standardized_materials[layer_name]
        
        # 检查属性是否存在
        if std_property_name in layer:
            return layer[std_property_name]
        
        # 如果在材料中找不到，尝试从维度信息中获取
        if std_property_name in self.dimensions:
            return self.dimensions[std_property_name]
        
        # 返回默认值
        return self.PROPERTY_DEFAULT_VALUES.get(std_property_name, None)
    
    def set_material_property(self, layer_name, property_name, value, use_standard_names=True):
        """
        设置材料属性值，支持中英文属性名
        
        参数:
        - layer_name: 图层名称
        - property_name: 属性名称
        - value: 新的属性值
        - use_standard_names: 是否使用标准英文名设置
        """
        # 确保图层存在
        if layer_name not in self.standardized_materials:
            self.standardized_materials[layer_name] = {"layer_name": layer_name}
        
        # 确保使用标准英文名
        if property_name in self.STANDARD_PROPERTY_NAMES:
            std_property_name = self.STANDARD_PROPERTY_NAMES[property_name]
        else:
            std_property_name = property_name
        
        # 验证属性值范围
        if std_property_name in self.PROPERTY_STANDARD_RANGES:
            min_val, max_val = self.PROPERTY_STANDARD_RANGES[std_property_name]
            if value < min_val or value > max_val:
                print(f"警告: 属性 {property_name} 的值 {value} 超出标准范围 [{min_val}, {max_val}]")
        
        # 设置属性值
        self.standardized_materials[layer_name][std_property_name] = value
    
    def get_input_layer_mapping(self, feature_name):
        """
        获取输入层特征对应的材料属性
        
        参数:
        - feature_name: 输入层特征名称
        
        返回:
        - 对应的材料属性名称和映射方法
        """
        if feature_name in self.FEATURE_TO_PROPERTY_MAPPING:
            property_name = self.FEATURE_TO_PROPERTY_MAPPING[feature_name]
            
            # 根据特征类型返回不同的映射方法
            if feature_name in ["wettability", "electrowetting_number"]:
                return property_name, "specialized"
            elif feature_name in ["layer_position", "dielectric_factor"]:
                return property_name, "normalized"
            else:
                return property_name, "linear"
        
        # 如果没有明确映射，尝试基于名称相似度查找
        for prop_name in self.STANDARD_PROPERTY_NAMES.values():
            if prop_name.lower() in feature_name.lower():
                return prop_name, "linear"
        
        return None, None
    
    def get_unified_material_structure(self, use_chinese_names=False):
        """
        获取统一的材料结构，可选择使用中文或英文属性名
        
        参数:
        - use_chinese_names: 是否使用中文属性名
        
        返回:
        - 统一的材料结构字典
        """
        unified = {}
        
        for layer_name, properties in self.standardized_materials.items():
            unified_layer = {}
            
            for prop_name, value in properties.items():
                # 如果需要中文名称
                if use_chinese_names and prop_name in self.REVERSE_PROPERTY_NAMES:
                    unified_prop_name = self.REVERSE_PROPERTY_NAMES[prop_name]
                else:
                    unified_prop_name = prop_name
                
                unified_layer[unified_prop_name] = value
            
            unified[layer_name] = unified_layer
        
        return unified
    
    def get_unified_dimensions(self, use_chinese_names=False):
        """
        获取统一的维度信息，可选择使用中文或英文属性名
        
        参数:
        - use_chinese_names: 是否使用中文属性名
        
        返回:
        - 统一的维度信息字典
        """
        if use_chinese_names:
            # 转换为中文格式
            chinese_dimensions = {}
            
            if "pixel_dimensions" in self.dimensions:
                pd = self.dimensions["pixel_dimensions"]
                chinese_dimensions["像素尺寸"] = {
                    "宽度": pd.get("width", 0.0),
                    "高度": pd.get("height", 0.0),
                    "总厚度": pd.get("total_thickness", 0.0)
                }
            
            if "interface_properties" in self.dimensions:
                ip = self.dimensions["interface_properties"]
                chinese_dimensions["界面特性"] = {
                    "接触角": ip.get("contact_angle", 90.0),
                    "接触角滞后": ip.get("contact_angle_hysteresis", 0.0),
                    "界面厚度": ip.get("interface_thickness", 0.0)
                }
            
            if "electrical_parameters" in self.dimensions:
                ep = self.dimensions["electrical_parameters"]
                chinese_dimensions["电学参数"] = {
                    "驱动电压": f"{ep.get('drive_voltage_min', 0.0)}-{ep.get('drive_voltage_max', 0.0)}",
                    "最大工作电压": ep.get("max_operating_voltage", 0.0),
                    "频率范围": ep.get("frequency_range", [0, 0])
                }
            
            return chinese_dimensions
        else:
            return self.dimensions.copy()
    
    def validate_material_consistency(self):
        """
        验证材料属性的一致性
        
        返回:
        - 验证结果字典，包含一致性问题
        """
        results = {"consistent": True, "issues": []}
        
        # 检查层索引的连续性
        layer_indices = [props.get("layer_index", 0) for props in self.standardized_materials.values()]
        if sorted(layer_indices) != list(range(len(layer_indices))):
            results["consistent"] = False
            results["issues"].append({
                "type": "layer_index_gap",
                "message": "图层索引不连续",
                "indices": layer_indices
            })
        
        # 检查材料属性是否在合理范围内
        for layer_name, properties in self.standardized_materials.items():
            for prop_name, value in properties.items():
                if prop_name in self.PROPERTY_STANDARD_RANGES:
                    min_val, max_val = self.PROPERTY_STANDARD_RANGES[prop_name]
                    if isinstance(value, (int, float)) and (value < min_val or value > max_val):
                        results["consistent"] = False
                        results["issues"].append({
                            "type": "property_out_of_range",
                            "layer": layer_name,
                            "property": prop_name,
                            "value": value,
                            "range": [min_val, max_val]
                        })
        
        # 检查关键界面参数
        if "interface_properties" in self.dimensions:
            contact_angle = self.dimensions["interface_properties"].get("contact_angle", 90.0)
            if contact_angle < 0 or contact_angle > 180:
                results["consistent"] = False
                results["issues"].append({
                    "type": "invalid_contact_angle",
                    "value": contact_angle,
                    "message": "接触角必须在0-180度范围内"
                })
        
        return results
    
    def export_unified_parameters(self, output_file=None):
        """
        导出统一的参数定义到文件
        
        参数:
        - output_file: 输出文件路径，None表示返回字典
        
        返回:
        - 如果指定了output_file，返回True表示成功；否则返回参数字典
        """
        unified_params = {
            "materials": self.get_unified_material_structure(use_chinese_names=True),
            "dimensions": self.get_unified_dimensions(use_chinese_names=True),
            "property_mappings": self.FEATURE_TO_PROPERTY_MAPPING,
            "standard_ranges": {self.REVERSE_PROPERTY_NAMES.get(k, k): v 
                               for k, v in self.PROPERTY_STANDARD_RANGES.items()}
        }
        
        if output_file:
            import json
            # 处理numpy类型以确保JSON可序列化
            def convert_numpy_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(unified_params, f, ensure_ascii=False, indent=2, default=convert_numpy_types)
            return True
        
        return unified_params

# 示例用法
if __name__ == "__main__":
    # 尝试从generate_pyvista_3d.py导入材料结构和尺寸
    try:
        from generate_pyvista_3d import material_structure, dimensions
        print("成功导入材料结构和尺寸信息")
    except ImportError:
        print("无法导入generate_pyvista_3d模块，使用示例数据")
        
        # 示例材料结构
        material_structure = {
            "底层ITO玻璃": {
                "厚度": 0.5e-6,
                "相对介电常数": 9.0,
                "表面张力": 0.072,
                "电导率": 1e-6,
                "层索引": 0,
                "角色": "基底",
                "extends_to_boundary": True
            },
            "疏水层": {
                "厚度": 0.4e-6,
                "相对介电常数": 2.1,
                "表面张力": 0.015,
                "电导率": 1e-14,
                "层索引": 3,
                "角色": "控制润湿性"
            }
        }
        
        # 示例尺寸信息
        dimensions = {
            "像素尺寸": {
                "宽度": 184e-6,
                "高度": 184e-6,
                "总厚度": 41.8e-6
            },
            "界面特性": {
                "接触角": 110.0,
                "接触角滞后": 5.0
            },
            "电学参数": {
                "驱动电压": "0-30.0",
                "最大工作电压": 30.0
            }
        }
    
    # 创建材料统一器实例
    unifier = EWPMaterialUnifier()
    
    print("=== EWP材料属性统一器示例 ===")
    
    # 标准化材料结构
    std_materials = unifier.standardize_material_structure(material_structure)
    print(f"标准化后的材料数量: {len(std_materials)}")
    
    # 标准化维度信息
    std_dimensions = unifier.standardize_dimensions(dimensions)
    print("维度信息已标准化")
    
    # 获取材料属性示例
    if "底层ITO玻璃" in std_materials:
        thickness = unifier.get_material_property("底层ITO玻璃", "厚度")
        epsilon = unifier.get_material_property("底层ITO玻璃", "相对介电常数")
        print(f"底层ITO玻璃厚度: {thickness} m, 相对介电常数: {epsilon}")
    
    # 输入层特征映射示例
    feature_mapping = unifier.get_input_layer_mapping("wettability")
    print(f"润湿性(wettability)特征映射到材料属性: {feature_mapping}")
    
    # 验证材料一致性
    consistency = unifier.validate_material_consistency()
    print(f"\n材料一致性验证: {'一致' if consistency['consistent'] else '不一致'}")
    if not consistency['consistent']:
        print("一致性问题:")
        for issue in consistency['issues']:
            print(f"  - {issue['message'] if 'message' in issue else str(issue)}")
    
    # 导出统一参数
    output_file = "./ewp_unified_parameters.json"
    if unifier.export_unified_parameters(output_file):
        print(f"\n统一参数已导出到: {output_file}")
    
    print("\n材料属性统一器功能演示完成！")