import numpy as np
import torch
import torch.nn as nn

class EWPINNInputLayer:
    """
    电润湿显示像素PINN输入层
    实现62维核心物理特征的构建和归一化
    """
    
    def __init__(self, device='cpu'):
        self.device = device
        self.feature_groups = {
            'basic_spacetime_voltage': 6,
            'geometry': 12,
            'material_interface': 10,
            'electric_field': 8,
            'fluid_dynamics': 10,
            'time_dynamics': 6,
            'electrowetting_specific': 10
        }
        self.total_features = sum(self.feature_groups.values())  # 62维
        
        # 阶段化实现标记
        self.stage = 1  # 1: 基础(32维), 2: 扩展(48维), 3: 完整(62维)
        
        # 特征归一化范围配置
        self.normalization_ranges = self._init_normalization_ranges()
        
        # 特征重要性分组
        self.core_features = 20
        self.important_features = 24
        self.enhanced_features = 18
    
    def _init_normalization_ranges(self):
        """
        初始化各特征的归一化范围
        """
        ranges = {}
        
        # 基础时空电压参数
        ranges.update({
            'X_norm': (0.0, 1.0),
            'Y_norm': (0.0, 1.0),
            'Z_norm': (0.0, 1.0),
            'T_norm': (0.0, 1.0),
            'T_phase': (-1.0, 1.0),
            'V_norm': (0.0, 1.0)
        })
        
        # 几何结构特征
        ranges.update({
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
            'center_proximity': (0.0, 1.0)
        })
        
        # 材料与界面特性
        ranges.update({
            'layer_position': (0.0, 1.0),
            'interface_zone': (0.0, 1.0),
            'material_gradient': (0.0, 1.0),
            'normal_z': (-1.0, 1.0),
            'surface_energy': (0.0, 1.0),
            'hysteresis_factor': (0.0, 1.0),
            'roughness_effect': (0.0, 1.0),
            'wettability': (0.0, 1.0),
            'triple_line_proximity': (0.0, 1.0),
            'pinning_strength': (0.0, 1.0)
        })
        
        # 电场与介质响应（双相）
        ranges.update({
            'E_z': (0.0, 1.0),
            'E_magnitude': (0.0, 1.0),
            'field_gradient': (0.0, 1.0),
            'field_uniformity': (0.0, 1.0),
            'V_effective': (0.0, 1.0),
            'dV_dt': (-1.0, 1.0),
            'charge_relaxation_norm': (0.0, 1.0),
            'ink_permittivity_norm': (0.0, 1.0)
        })
        
        # 流体动力学参数
        ranges.update({
            'reynolds_local': (0.0, 1.0),
            'capillary_number': (0.0, 1.0),
            'weber_number': (0.0, 1.0),
            'flow_regime': (0.0, 1.0),
            'viscosity_ratio': (0.0, 1.0),
            'density_ratio': (0.0, 1.0),
            'surface_tension': (0.0, 1.0),
            'marangoni_effect': (0.0, 1.0),
            'interfacial_stress': (0.0, 1.0),
            'curvature_pressure': (0.0, 1.0)
        })
        
        # 时间动态特征
        ranges.update({
            'time_fourier': (-1.0, 1.0),
            'time_decay': (0.0, 1.0),
            'response_phase': (0.0, 1.0),
            'interface_history': (-1.0, 1.0),
            'velocity_trend': (-1.0, 1.0),
            'angle_evolution': (-1.0, 1.0)
        })
        
        # 电润湿特异性参数（双相特征）
        ranges.update({
            'electrowetting_number': (0.0, 1.0),
            'voltage_saturation': (0.0, 1.0),
            'dielectric_variation': (0.0, 1.0),
            'young_lippmann_dev': (0.0, 1.0),
            'cox_voinov_param': (0.0, 1.0),
            'line_mobility': (0.0, 1.0),
            'dynamic_hysteresis': (0.0, 1.0),
            'relaxation_time': (0.0, 1.0),
            'polar_conductivity_norm': (0.0, 1.0),
            'ink_volume_fraction': (0.0, 1.0)
        })
        
        return ranges
    
    def set_implementation_stage(self, stage):
        """
        设置输入层实现阶段
        stage: 1(基础32维), 2(扩展48维), 3(完整62维)
        """
        if stage in [1, 2, 3]:
            self.stage = stage
            print(f"已设置为阶段{stage}实现，当前特征维度: {self.get_current_dim()}")
        else:
            raise ValueError("阶段必须为1、2或3")
    
    def get_current_dim(self):
        """
        获取当前阶段的特征维度
        """
        if self.stage == 1:
            return 32
        elif self.stage == 2:
            return 48
        elif self.stage == 3:
            return 62
    
    def normalize_feature(self, feature_name, value):
        """
        归一化单个特征值
        """
        if feature_name not in self.normalization_ranges:
            raise ValueError(f"未知特征名称: {feature_name}")
        
        min_val, max_val = self.normalization_ranges[feature_name]
        
        # 处理除零情况
        if max_val == min_val:
            return 0.0
        
        # 线性归一化到目标范围
        normalized = (value - min_val) / (max_val - min_val)
        
        # 如果目标范围是[-1,1]，需要进一步转换
        if min_val == -1.0 and max_val == 1.0:
            normalized = 2 * normalized - 1
        
        # 确保值在目标范围内
        return np.clip(normalized, min_val, max_val)
    
    def create_input_vector(self, feature_dict):
        """
        根据特征字典创建输入向量
        返回numpy数组
        """
        # 根据当前阶段确定需要的特征
        required_features = []
        
        if self.stage >= 1:
            # 阶段1: 基础输入层 (32维)
            # 基础时空电压: 6维
            required_features.extend(['X_norm', 'Y_norm', 'Z_norm', 'T_norm', 'T_phase', 'V_norm'])
            # 几何结构特征: 8维
            required_features.extend(['dist_wall_x', 'dist_wall_y', 'dist_wall_min', 'corner_effect'])
            required_features.extend(['curvature_mean', 'curvature_gaussian', 'symmetry_index', 'radial_position'])
            # 材料界面特性: 6维
            required_features.extend(['layer_position', 'interface_zone', 'material_gradient', 'normal_z'])
            required_features.extend(['surface_energy', 'wettability'])
            # 电场/介质特性: 6维（含双相）
            required_features.extend(['E_z', 'E_magnitude', 'field_gradient'])
            required_features.extend(['V_effective', 'charge_relaxation_norm', 'ink_permittivity_norm'])
            # 流体参数: 6维
            required_features.extend(['reynolds_local', 'capillary_number'])
            required_features.extend(['viscosity_ratio', 'density_ratio', 'surface_tension'])
            required_features.extend(['interfacial_stress'])
        
        if self.stage >= 2:
            # 阶段2: 扩展输入层 (48维) - 增加16维
            # 几何结构特征: +4维
            required_features.extend(['curvature_gradient', 'polar_thickness_norm', 'ink_thickness_norm', 'center_proximity'])
            # 材料界面特性: +4维
            required_features.extend(['hysteresis_factor', 'roughness_effect', 'triple_line_proximity', 'pinning_strength'])
            # 时间动态特征: +4维
            required_features.extend(['time_fourier', 'time_decay', 'response_phase', 'interface_history'])
            # 电润湿特异性: +4维
            required_features.extend(['electrowetting_number', 'voltage_saturation', 'dielectric_variation', 'young_lippmann_dev'])
        
        if self.stage >= 3:
            # 阶段3: 完整输入层 (62维) - 增加14维
            # 流体动力学: +4维
            required_features.extend(['weber_number', 'flow_regime', 'marangoni_effect', 'curvature_pressure'])
            # 时间动态特征: +2维
            required_features.extend(['velocity_trend', 'angle_evolution'])
            # 电润湿特异性: +6维 (确保总维度正确)
            required_features.extend(['cox_voinov_param', 'line_mobility', 'dynamic_hysteresis', 'relaxation_time'])
            required_features.extend(['polar_conductivity_norm', 'ink_volume_fraction'])
            # 添加缺失的2个特征
            required_features.extend(['field_uniformity', 'dV_dt'])
        
        # 调试信息：打印特征数量
        # print(f"阶段{self.stage}特征数量: {len(required_features)}")
        
        # 构建输入向量
        input_vector = []
        for feature_name in required_features:
            if feature_name in feature_dict:
                normalized_value = self.normalize_feature(feature_name, feature_dict[feature_name])
                input_vector.append(normalized_value)
            else:
                # 对于缺失的特征，使用默认值（中间值）
                min_val, max_val = self.normalization_ranges[feature_name]
                default_value = (min_val + max_val) / 2
                input_vector.append(default_value)
        
        return np.array(input_vector)
    
    def create_batch_input(self, feature_dicts_list):
        """
        创建批量输入
        feature_dicts_list: 特征字典列表
        返回形状为 [batch_size, feature_dim] 的numpy数组
        """
        batch_size = len(feature_dicts_list)
        feature_dim = self.get_current_dim()
        batch_input = np.zeros((batch_size, feature_dim))
        
        for i, feature_dict in enumerate(feature_dicts_list):
            batch_input[i] = self.create_input_vector(feature_dict)
        
        return batch_input
    
    def to_tensor(self, input_array):
        """
        将numpy数组转换为PyTorch张量
        """
        return torch.tensor(input_array, dtype=torch.float32, device=self.device)
    
    def validate_input(self, input_vector):
        """
        验证输入向量的物理一致性
        返回验证结果和错误信息
        """
        errors = []
        
        # 检查维度
        expected_dim = self.get_current_dim()
        if len(input_vector) != expected_dim:
            errors.append(f"维度不匹配: 预期{expected_dim}，实际{len(input_vector)}")
        
        # 检查数值范围
        if not np.all((input_vector >= -1.0) & (input_vector <= 1.0)):
            errors.append("输入值超出 [-1.0, 1.0] 范围")
        
        # 检查NaN和无穷大
        if np.any(np.isnan(input_vector)) or np.any(np.isinf(input_vector)):
            errors.append("输入包含NaN或无穷大值")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def generate_example_input(self):
        """
        生成示例输入特征字典
        用于测试和演示
        """
        example = {
            # 基础时空电压参数
            'X_norm': 0.5,
            'Y_norm': 0.5,
            'Z_norm': 0.5,
            'T_norm': 0.3,
            'T_phase': np.sin(2 * np.pi * 0.3),
            'V_norm': 0.7,
            
            # 几何结构特征
            'dist_wall_x': 0.4,
            'dist_wall_y': 0.4,
            'dist_wall_min': 0.4,
            'corner_effect': 0.1,
            'curvature_mean': 0.2,
            'curvature_gaussian': -0.1,
            'curvature_gradient': 0.3,
            'symmetry_index': 0.8,
            'radial_position': 0.5,
            'polar_thickness_norm': 0.6,
            'ink_thickness_norm': 0.4,
            'center_proximity': 0.5,
            
            # 材料与界面特性
            'layer_position': 0.5,
            'interface_zone': 1.0,
            'material_gradient': 0.2,
            'normal_z': 1.0,
            'surface_energy': 0.6,
            'hysteresis_factor': 0.3,
            'roughness_effect': 0.2,
            'wettability': 0.4,
            'triple_line_proximity': 0.8,
            'pinning_strength': 0.3,
            
            # 电场与介电特性
            'E_z': 0.7,
            'E_magnitude': 0.7,
            'field_gradient': 0.2,
            'field_uniformity': 0.9,
            'V_effective': 0.65,
            'dV_dt': 0.1,
            'charge_relaxation_norm': 0.5,
            'ink_permittivity_norm': 0.2,
            
            # 流体动力学参数
            'reynolds_local': 0.1,
            'capillary_number': 0.05,
            'weber_number': 0.02,
            'flow_regime': 0.1,
            'viscosity_ratio': 2.0,  # 将在归一化中处理
            'density_ratio': 1.5,    # 将在归一化中处理
            'surface_tension': 0.8,
            'marangoni_effect': 0.1,
            'interfacial_stress': 0.3,
            'curvature_pressure': 0.4,
            
            # 时间动态特征
            'time_fourier': np.cos(2 * np.pi * 0.3),
            'time_decay': 0.9,
            'response_phase': 0.2,
            'interface_history': 0.1,
            'velocity_trend': -0.1,
            'angle_evolution': 0.2,
            
            # 电润湿特异性参数
            'electrowetting_number': 0.5,
            'voltage_saturation': 0.3,
            'dielectric_variation': 0.05,
            'young_lippmann_dev': 0.1,
            'cox_voinov_param': 0.2,
            'line_mobility': 0.7,
            'dynamic_hysteresis': 0.4,
            'relaxation_time': 0.5,
            'polar_conductivity_norm': 0.3,
            'ink_volume_fraction': 0.5
        }
        
        return example
    
    def get_feature_importance_group(self, feature_index):
        """
        获取特征的重要性分组
        返回: 'core', 'important', 或 'enhanced'
        """
        if feature_index < self.core_features:
            return 'core'
        elif feature_index < self.core_features + self.important_features:
            return 'important'
        else:
            return 'enhanced'

# 演示代码
if __name__ == "__main__":
    # 创建输入层实例
    input_layer = EWPINNInputLayer()
    
    # 生成示例输入
    example_input = input_layer.generate_example_input()
    
    # 测试不同阶段
    print("=== 阶段1: 基础输入层 (32维) ===")
    input_layer.set_implementation_stage(1)
    input_vec1 = input_layer.create_input_vector(example_input)
    print(f"输入向量形状: {input_vec1.shape}")
    is_valid, errors = input_layer.validate_input(input_vec1)
    print(f"验证结果: {'有效' if is_valid else '无效'}")
    if errors:
        print(f"错误: {errors}")
    
    print("\n=== 阶段2: 扩展输入层 (48维) ===")
    input_layer.set_implementation_stage(2)
    input_vec2 = input_layer.create_input_vector(example_input)
    print(f"输入向量形状: {input_vec2.shape}")
    
    print("\n=== 阶段3: 完整输入层 (62维) ===")
    input_layer.set_implementation_stage(3)
    input_vec3 = input_layer.create_input_vector(example_input)
    print(f"输入向量形状: {input_vec3.shape}")
    
    # 测试批量输入
    batch_size = 3
    batch_input = input_layer.create_batch_input([example_input] * batch_size)
    print(f"\n批量输入形状: {batch_input.shape}")
    
    # 转换为PyTorch张量
    tensor_input = input_layer.to_tensor(batch_input)
    print(f"PyTorch张量形状: {tensor_input.shape}")
    print(f"PyTorch张量设备: {tensor_input.device}")
