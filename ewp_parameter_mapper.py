"""
EWP参数映射器

此模块提供3D结构物理参数与PINN输入层归一化特征之间的转换函数，
确保电润湿显示像素模型的物理参数在不同表示方式间保持一致。

主要功能：
- 3D物理参数到输入层归一化特征的映射
- 输入层归一化特征到3D物理参数的反向映射
- 材料属性的统一转换
- 几何参数的标准化处理
"""

import numpy as np
import os
import sys

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class EWPParameterMapper:
    """
    电润湿参数映射器类
    负责3D结构物理参数与输入层归一化特征之间的相互转换
    """
    
    def __init__(self):
        # 3D结构参数范围（从generate_pyvista_3d.py中提取）
        self.physical_ranges = {
            # 几何参数
            'pixel_size_x': (0.0, 184e-6),  # 像素X尺寸 [m]
            'pixel_size_y': (0.0, 184e-6),  # 像素Y尺寸 [m]
            'total_thickness': (0.0, 41.8e-6),  # 总厚度 [m]
            
            # 图层厚度
            'ito_bottom_thickness': (0.0, 0.5e-6),
            'ito_top_thickness': (0.0, 0.5e-6),
            'dielectric_thickness': (0.0, 0.4e-6),
            'hydrophobic_thickness': (0.0, 0.4e-6),
            'ink_thickness': (0.0, 3e-6),
            'polar_liquid_thickness': (0.0, 17e-6),
            'dam_thickness': (0.0, 20e-6),
            
            # 填充尺寸
            'ink_fill_size': (0.0, 174e-6),
            'polar_liquid_fill_size': (0.0, 174e-6),
            
            # 电学参数
            'voltage': (0.0, 30.0),  # 电压范围 [V]
            
            # 材料属性范围
            'relative_permittivity': (1.0, 100.0),
            'surface_tension': (0.01, 0.1),  # [N/m]
            'conductivity': (1e-14, 1e-5),  # [S/m]
            
            # 界面参数
            'contact_angle': (0.0, 180.0),  # [度]
        }
        
        # 输入层归一化范围
        self.normalization_ranges = {
            # 基础时空电压参数
            'X_norm': (0.0, 1.0),
            'Y_norm': (0.0, 1.0),
            'Z_norm': (0.0, 1.0),
            'T_norm': (0.0, 1.0),
            'T_phase': (-1.0, 1.0),
            'V_norm': (0.0, 1.0),
            
            # 几何结构特征
            'dist_wall_x': (0.0, 1.0),
            'dist_wall_y': (0.0, 1.0),
            'dist_wall_min': (0.0, 1.0),
            'corner_effect': (0.0, 1.0),
            'curvature_mean': (-1.0, 1.0),
            'curvature_gaussian': (-1.0, 1.0),
            'curvature_gradient': (0.0, 1.0),
            'symmetry_index': (0.0, 1.0),
            'radial_position': (0.0, 1.0),
            'polar_thickness_norm': (0.0, 1.0),
            'ink_thickness_norm': (0.0, 1.0),
            'center_proximity': (0.0, 1.0),
            
            # 材料与界面特性
            'layer_position': (0.0, 1.0),
            'interface_zone': (0.0, 1.0),
            'material_gradient': (0.0, 1.0),
            'normal_z': (-1.0, 1.0),
            'surface_energy': (0.0, 1.0),
            'hysteresis_factor': (0.0, 1.0),
            'roughness_effect': (0.0, 1.0),
            'wettability': (0.0, 1.0),
            'triple_line_proximity': (0.0, 1.0),
            'pinning_strength': (0.0, 1.0),
            
            # 电场与介质响应
            'E_z': (0.0, 1.0),
            'E_magnitude': (0.0, 1.0),
            'field_gradient': (0.0, 1.0),
            'field_uniformity': (0.0, 1.0),
            'V_effective': (0.0, 1.0),
            'dV_dt': (-1.0, 1.0),
            'charge_relaxation_norm': (0.0, 1.0),
            'ink_permittivity_norm': (0.0, 1.0),
        }
        
        # 关键参数映射配置
        self.parameter_mappings = {
            # 几何参数映射
            'X_norm': {'physical_param': 'pixel_size_x', 'method': 'linear', 'scale_factor': 1.0},
            'Y_norm': {'physical_param': 'pixel_size_y', 'method': 'linear', 'scale_factor': 1.0},
            'Z_norm': {'physical_param': 'total_thickness', 'method': 'linear', 'scale_factor': 1.0},
            
            # 电压参数映射
            'V_norm': {'physical_param': 'voltage', 'method': 'linear', 'scale_factor': 1.0},
            'V_effective': {'physical_param': 'voltage', 'method': 'linear', 'scale_factor': 0.95},
            
            # 材料参数映射
            'surface_energy': {'physical_param': 'surface_tension', 'method': 'linear', 'scale_factor': 10.0},
            
            # 界面参数映射
            'wettability': {'physical_param': 'contact_angle', 'method': 'inverse_cosine', 'scale_factor': 1.0},
        }

        self.EPS0 = 8.854e-12

    def _normalize_clamp(self, x):
        return float(max(0.0, min(1.0, x)))

    def compute_charge_relaxation_norm(self, material):
        epsilon = float(material.get('相对介电常数', 10.0))
        sigma = float(material.get('电导率', 1e-12))
        tau = epsilon * self.EPS0 / max(sigma, 1e-20)
        return self._normalize_clamp(tau / 1e-3)

    def compute_conductivity_norm(self, sigma):
        smin, smax = 1e-14, 1e-3
        import math
        val = (math.log10(max(sigma, smin)) - math.log10(smin)) / (math.log10(smax) - math.log10(smin))
        return self._normalize_clamp(val)

    def create_feature_dict_from_3d(self, material_structure, dimensions, voltage=None):
        vmax = float(dimensions.get('电学参数', {}).get('最大工作电压', 30.0))
        if voltage is None:
            voltage = 0.5 * vmax
        V_norm = self._normalize_clamp(voltage / vmax if vmax > 0 else 0.0)
        ink_t = float(material_structure.get('油墨层', {}).get('厚度', 3e-6))
        polar_t = float(material_structure.get('极性液体层', {}).get('厚度', 17e-6))
        total_t = ink_t + polar_t
        ink_frac = self._normalize_clamp(ink_t / total_t if total_t > 0 else 0.0)
        polar_frac = self._normalize_clamp(polar_t / total_t if total_t > 0 else 0.0)
        ink_eps = float(material_structure.get('油墨层', {}).get('相对介电常数', 3.0))
        ink_perm_norm = self._normalize_clamp(ink_eps / 100.0)
        polar_sigma = float(material_structure.get('极性液体层', {}).get('电导率', 5e-5))
        polar_cond_norm = self.compute_conductivity_norm(polar_sigma)
        charge_relax = self.compute_charge_relaxation_norm(material_structure.get('极性液体层', {}))
        return {
            'V_norm': V_norm,
            'polar_thickness_norm': polar_frac,
            'ink_thickness_norm': ink_frac,
            'ink_permittivity_norm': ink_perm_norm,
            'polar_conductivity_norm': polar_cond_norm,
            'ink_volume_fraction': ink_frac,
            'charge_relaxation_norm': charge_relax,
        }

    def create_stage3_batch_from_3d(self, n, material_structure, dimensions, voltage=None):
        base = self.create_feature_dict_from_3d(material_structure, dimensions, voltage)
        return [base.copy() for _ in range(n)]
    
    def physical_to_normalized(self, physical_value, physical_param):
        """
        将物理参数值转换为归一化值
        
        参数:
        - physical_value: 物理参数值
        - physical_param: 物理参数名称
        
        返回:
        - 归一化后的值（在0-1或-1-1范围内）
        """
        if physical_param not in self.physical_ranges:
            raise ValueError(f"未知的物理参数: {physical_param}")
        
        min_val, max_val = self.physical_ranges[physical_param]
        
        # 处理除零情况
        if max_val == min_val:
            return 0.0
        
        # 线性归一化到0-1范围
        normalized = (physical_value - min_val) / (max_val - min_val)
        
        # 确保值在0-1范围内
        return np.clip(normalized, 0.0, 1.0)
    
    def normalized_to_physical(self, normalized_value, physical_param):
        """
        将归一化值转换为物理参数值
        
        参数:
        - normalized_value: 归一化值（应在0-1范围内）
        - physical_param: 物理参数名称
        
        返回:
        - 物理参数值
        """
        if physical_param not in self.physical_ranges:
            raise ValueError(f"未知的物理参数: {physical_param}")
        
        min_val, max_val = self.physical_ranges[physical_param]
        
        # 确保归一化值在0-1范围内
        normalized_value = np.clip(normalized_value, 0.0, 1.0)
        
        # 从0-1范围映射回物理范围
        physical_value = min_val + normalized_value * (max_val - min_val)
        
        return physical_value
    
    def map_3d_to_input(self, feature_name, physical_value=None, layer_name=None, material_structure=None):
        """
        将3D结构参数映射到输入层特征
        
        参数:
        - feature_name: 输入层特征名称
        - physical_value: 可选，直接提供的物理值
        - layer_name: 可选，3D结构中的图层名称
        - material_structure: 可选，材料结构字典
        
        返回:
        - 归一化的输入层特征值
        """
        # 检查是否有直接映射关系
        if feature_name in self.parameter_mappings and physical_value is not None:
            mapping = self.parameter_mappings[feature_name]
            phys_param = mapping['physical_param']
            method = mapping['method']
            
            # 基本归一化
            base_norm = self.physical_to_normalized(physical_value, phys_param)
            
            # 应用特定的映射方法
            if method == 'linear':
                return base_norm * mapping.get('scale_factor', 1.0)
            elif method == 'logarithmic':
                ref_value = mapping.get('reference', 1.0)
                # 对相对值进行对数映射
                rel_value = physical_value / ref_value
                if rel_value > 0:
                    return np.log(rel_value + 1) / np.log(self.physical_ranges[phys_param][1] / ref_value + 1)
                return 0.0
            elif method == 'inverse_cosine':
                # 接触角到润湿性的转换
                # 接触角(0-180度) -> 余弦值(-1到1) -> 归一化到0-1
                angle_rad = np.radians(physical_value)
                cos_angle = np.cos(angle_rad)
                return (cos_angle + 1) / 2
            
            return base_norm
        
        # 处理基于图层的映射
        if layer_name and material_structure:
            if layer_name not in material_structure:
                raise ValueError(f"材料结构中找不到图层: {layer_name}")
            
            layer_props = material_structure[layer_name]
            
            # 处理位置相关特征
            if feature_name == 'layer_position':
                # 根据层索引计算位置
                layer_idx = layer_props.get('层索引', 0)
                max_layers = 7  # 总共有7层
                return layer_idx / (max_layers - 1)
            
            # 处理材料相关特征
            if feature_name == 'dielectric_factor':
                epsilon = layer_props.get('相对介电常数', 1.0)
                return self.physical_to_normalized(epsilon, 'relative_permittivity')
            
            if feature_name == 'surface_energy':
                surface_tension = layer_props.get('表面张力', 0.0)
                return self.physical_to_normalized(surface_tension, 'surface_tension')
        
        # 默认处理
        if feature_name in self.normalization_ranges:
            # 对于没有明确映射的特征，使用基本归一化
            if physical_value is not None:
                # 尝试找到合适的物理范围
                for phys_param, (min_val, max_val) in self.physical_ranges.items():
                    if phys_param.lower() in feature_name.lower():
                        return self.physical_to_normalized(physical_value, phys_param)
            
            # 如果无法映射，返回默认值0.5
            return 0.5
        
        raise ValueError(f"无法映射特征: {feature_name}")
    
    def map_input_to_3d(self, feature_name, normalized_value, layer_name=None, material_structure=None):
        """
        将输入层特征映射回3D结构参数
        
        参数:
        - feature_name: 输入层特征名称
        - normalized_value: 归一化特征值
        - layer_name: 可选，3D结构中的图层名称
        - material_structure: 可选，材料结构字典
        
        返回:
        - 对应的物理参数值
        """
        # 确保归一化值在有效范围内
        if feature_name in self.normalization_ranges:
            min_norm, max_norm = self.normalization_ranges[feature_name]
            normalized_value = np.clip(normalized_value, min_norm, max_norm)
            
            # 如果是[-1,1]范围，先转换到[0,1]
            if min_norm == -1.0 and max_norm == 1.0:
                normalized_value = (normalized_value + 1) / 2
        
        # 检查是否有直接映射关系
        if feature_name in self.parameter_mappings:
            mapping = self.parameter_mappings[feature_name]
            phys_param = mapping['physical_param']
            method = mapping['method']
            
            # 应用特定的反向映射方法
            if method == 'linear':
                adjusted_norm = normalized_value / mapping.get('scale_factor', 1.0)
                return self.normalized_to_physical(adjusted_norm, phys_param)
            elif method == 'logarithmic':
                ref_value = mapping.get('reference', 1.0)
                max_rel_value = self.physical_ranges[phys_param][1] / ref_value
                # 对数映射的反向计算
                rel_value = np.exp(normalized_value * np.log(max_rel_value + 1)) - 1
                return rel_value * ref_value
            elif method == 'inverse_cosine':
                # 润湿性到接触角的转换
                # 归一化值(0-1) -> 余弦值(-1到1) -> 角度(0-180度)
                cos_angle = 2 * normalized_value - 1
                angle_rad = np.arccos(cos_angle)
                return np.degrees(angle_rad)
        
        # 处理基于图层的反向映射
        if layer_name and material_structure:
            if layer_name not in material_structure:
                raise ValueError(f"材料结构中找不到图层: {layer_name}")
            
            # 处理材料相关特征的反向映射
            if feature_name == 'dielectric_factor':
                return self.normalized_to_physical(normalized_value, 'relative_permittivity')
            
            if feature_name == 'surface_energy':
                return self.normalized_to_physical(normalized_value, 'surface_tension')
        
        # 对于空间坐标的反向映射
        if feature_name == 'X_norm':
            return self.normalized_to_physical(normalized_value, 'pixel_size_x')
        elif feature_name == 'Y_norm':
            return self.normalized_to_physical(normalized_value, 'pixel_size_y')
        elif feature_name == 'Z_norm':
            return self.normalized_to_physical(normalized_value, 'total_thickness')
        elif feature_name == 'V_norm' or feature_name == 'V_effective':
            return self.normalized_to_physical(normalized_value, 'voltage')
        
        # 默认返回
        return normalized_value
    
    def create_physical_to_feature_mapping(self, material_structure, dimensions):
        """
        从3D结构创建完整的特征映射字典
        
        参数:
        - material_structure: 3D结构材料字典
        - dimensions: 3D结构尺寸字典
        
        返回:
        - 特征映射字典
        """
        mapping_dict = {}
        
        # 映射几何尺寸
        pixel_size = dimensions['像素尺寸']['宽度']
        mapping_dict['X_norm'] = self.map_3d_to_input('X_norm', physical_value=pixel_size/2)  # 中心位置
        mapping_dict['Y_norm'] = self.map_3d_to_input('Y_norm', physical_value=pixel_size/2)
        mapping_dict['Z_norm'] = self.map_3d_to_input('Z_norm', physical_value=dimensions['像素尺寸']['总厚度']/2)
        
        # 映射电压参数
        max_voltage = float(dimensions['电学参数']['最大工作电压'])
        mapping_dict['V_norm'] = self.map_3d_to_input('V_norm', physical_value=max_voltage/2)  # 中等电压
        mapping_dict['V_effective'] = self.map_3d_to_input('V_effective', physical_value=max_voltage/2)
        
        # 映射接触角
        contact_angle = dimensions['界面特性']['接触角']
        mapping_dict['wettability'] = self.map_3d_to_input('wettability', physical_value=contact_angle)
        
        # 为每个图层映射材料属性
        for layer_name, layer_props in material_structure.items():
            if '相对介电常数' in layer_props:
                key = f"{layer_name}_dielectric_factor"
                mapping_dict[key] = self.map_3d_to_input('dielectric_factor', 
                                                        physical_value=layer_props['相对介电常数'])
            
            if '表面张力' in layer_props:
                key = f"{layer_name}_surface_energy"
                mapping_dict[key] = self.map_3d_to_input('surface_energy', 
                                                        physical_value=layer_props['表面张力'])
        
        return mapping_dict
    
    def validate_mapping_consistency(self, material_structure, dimensions):
        """
        验证映射的一致性
        
        参数:
        - material_structure: 3D结构材料字典
        - dimensions: 3D结构尺寸字典
        
        返回:
        - 验证结果字典，包含一致性信息
        """
        results = {'consistent': True, 'issues': []}
        
        # 创建映射
        mappings = self.create_physical_to_feature_mapping(material_structure, dimensions)
        
        # 验证关键参数的双向映射
        test_parameters = [
            ('V_norm', dimensions['电学参数']['最大工作电压'] / 2),
            ('wettability', dimensions['界面特性']['接触角']),
        ]
        
        for feature_name, physical_value in test_parameters:
            # 正向映射
            norm_value = self.map_3d_to_input(feature_name, physical_value)
            # 反向映射
            reversed_physical = self.map_input_to_3d(feature_name, norm_value)
            
            # 计算误差
            relative_error = abs(physical_value - reversed_physical) / (physical_value if physical_value != 0 else 1)
            
            # 检查一致性
            if relative_error > 0.01:  # 1%误差阈值
                results['consistent'] = False
                results['issues'].append({
                    'feature': feature_name,
                    'original': physical_value,
                    'reversed': reversed_physical,
                    'error': relative_error
                })
        
        return results

# 示例用法
if __name__ == "__main__":
    # 从generate_pyvista_3d.py导入材料结构和尺寸
    try:
        from generate_pyvista_3d import material_structure, dimensions
    except ImportError:
        print("无法导入generate_pyvista_3d模块，使用默认参数进行演示")
        
        # 模拟材料结构和尺寸
        material_structure = {
            "底层ITO玻璃": {
                "厚度": 0.5e-6,
                "相对介电常数": 9.0,
                "表面张力": 0.072,
                "层索引": 0
            },
            "疏水层": {
                "厚度": 0.4e-6,
                "相对介电常数": 2.1,
                "表面张力": 0.015,
                "层索引": 3
            }
        }
        
        dimensions = {
            "像素尺寸": {
                "宽度": 184e-6,
                "总厚度": 41.8e-6
            },
            "电学参数": {
                "最大工作电压": 30.0
            },
            "界面特性": {
                "接触角": 110.0
            }
        }
    
    # 创建映射器实例
    mapper = EWPParameterMapper()
    
    print("=== EWP参数映射器示例 ===")
    
    # 测试电压映射
    voltage = 15.0  # 15V
    v_norm = mapper.map_3d_to_input('V_norm', physical_value=voltage)
    reversed_voltage = mapper.map_input_to_3d('V_norm', v_norm)
    print(f"电压映射: {voltage}V -> 归一化值: {v_norm:.4f} -> 反向映射: {reversed_voltage:.4f}V")
    
    # 测试接触角映射
    contact_angle = 110.0  # 110度
    wettability = mapper.map_3d_to_input('wettability', physical_value=contact_angle)
    reversed_angle = mapper.map_input_to_3d('wettability', wettability)
    print(f"接触角映射: {contact_angle}° -> 润湿性: {wettability:.4f} -> 反向映射: {reversed_angle:.4f}°")
    
    # 测试材料参数映射
    if "疏水层" in material_structure:
        epsilon = material_structure["疏水层"]["相对介电常数"]
        dielectric_factor = mapper.map_3d_to_input('dielectric_factor', 
                                                 physical_value=epsilon)
        print(f"介电常数映射: {epsilon} -> 介电因子: {dielectric_factor:.4f}")
    
    # 验证映射一致性
    consistency = mapper.validate_mapping_consistency(material_structure, dimensions)
    print(f"\n映射一致性: {'一致' if consistency['consistent'] else '不一致'}")
    if not consistency['consistent']:
        print("不一致问题:")
        for issue in consistency['issues']:
            print(f"  特征: {issue['feature']}, 误差: {issue['error']:.4%}")
    
    print("\n参数映射器功能演示完成！")
