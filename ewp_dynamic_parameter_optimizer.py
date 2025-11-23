"""
EWP动态参数优化器

此模块负责增强3D结构物理模型与输入层动态特征之间的关联，
并为输入层中的动态特征提供更准确的归一化范围，确保电润湿显示
像素模型的时间相关参数与实际物理过程匹配。

主要功能：
- 动态参数的物理过程建模
- 时间相关特征的准确归一化范围计算
- 松弛时间和响应时间的物理约束
- 动态参数与静态参数的关联处理
"""

import numpy as np
import os
import sys

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class EWPDynamicParameterOptimizer:
    """
    电润湿动态参数优化器类
    负责增强动态参数关联和优化归一化范围
    """
    
    # 动态参数标准名称映射
    DYNAMIC_PARAMETER_NAMES = {
        # 时间相关参数
        "T_norm": "normalized_time",
        "T_phase": "time_phase",
        "T_prev": "previous_time",
        "T_next": "next_time",
        "delta_T": "time_step",
        
        # 电压动态参数
        "dV_dt": "voltage_rate",
        "V_prev": "previous_voltage",
        "V_next": "next_voltage",
        "V_effective": "effective_voltage",
        "voltage_saturation": "voltage_saturation",
        
        # 松弛和响应参数
        "relaxation_time": "relaxation_time",
        "response_time": "response_time",
        "characteristic_time": "characteristic_time",
        
        # 电润湿动态参数
        "electrowetting_number": "electrowetting_number",
        "dynamic_hysteresis": "dynamic_hysteresis",
        "young_lippmann_dev": "young_lippmann_deviation",
        "line_mobility": "contact_line_mobility",
        
        # 界面动态参数
        "interface_history": "interface_history",
        "velocity_trend": "velocity_trend",
        "angle_evolution": "contact_angle_evolution",
        "instability_index": "instability_index",
        "fluctuation_level": "fluctuation_level",
        
        # 流体动态参数
        "reynolds_local": "local_reynolds_number",
        "capillary_number": "capillary_number",
        "weber_number": "weber_number",
        "flow_regime": "flow_regime",
        "marangoni_effect": "marangoni_effect"
    }
    
    # 物理过程时间尺度 [秒]
    PHYSICAL_TIME_SCALES = {
        "electrical_relaxation": 1e-9,  # 电气松弛时间 (ns)
        "dielectric_relaxation": 1e-6,  # 介电松弛时间 (μs)
        "thermal_relaxation": 1e-3,     # 热松弛时间 (ms)
        "capillary_response": 1e-2,     # 毛细管响应时间 (10ms)
        "contact_line_motion": 5e-3,    # 接触线运动时间 (5ms)
        "interface_stabilization": 1e-1, # 界面稳定时间 (100ms)
        "complete_equilibration": 1.0   # 完全平衡时间 (1s)
    }
    
    # 动态参数与物理参数的关联配置
    PARAMETER_CORRELATIONS = {
        # 松弛时间与材料参数的关联
        "relaxation_time": {
            "dependencies": ["relative_permittivity", "conductivity", "thickness"],
            "formula": "epsilon * epsilon0 / sigma",  # 介电松弛时间公式
            "range_scale": 1e-6,  # 归一化参考尺度 (μs)
            "unit": "秒"
        },
        
        # 响应时间与几何参数的关联
        "response_time": {
            "dependencies": ["thickness", "surface_tension", "viscosity"],
            "formula": "mu * h^2 / gamma",  # 特征响应时间公式
            "range_scale": 1e-3,  # 归一化参考尺度 (ms)
            "unit": "秒"
        },
        
        # 电润湿数与电压和材料参数的关联
        "electrowetting_number": {
            "dependencies": ["voltage", "relative_permittivity", "thickness", "surface_tension"],
            "formula": "epsilon * epsilon0 * V^2 / (2 * gamma * d^2)",  # 电润湿数公式
            "range_scale": 1.0,
            "unit": "无量纲"
        },
        
        # 毛细管数与动态参数的关联
        "capillary_number": {
            "dependencies": ["velocity", "viscosity", "surface_tension"],
            "formula": "mu * v / gamma",  # 毛细管数公式
            "range_scale": 1e-3,
            "unit": "无量纲"
        }
    }
    
    # 物理常数
    PHYSICAL_CONSTANTS = {
        "epsilon0": 8.854e-12,  # 真空介电常数 [F/m]
        "mu_water": 8.90e-4,   # 水的粘度 [Pa·s] at 25°C
        "mu_oil": 1.0e-3,      # 典型油的粘度 [Pa·s]
        "gamma_water_air": 0.072, # 水-空气表面张力 [N/m]
        "gamma_oil_water": 0.020  # 油-水界面张力 [N/m]
    }
    
    def __init__(self):
        # 存储动态参数的优化范围
        self.optimized_ranges = {}
        # 存储时间尺度参数
        self.time_scales = {}
        # 存储材料和几何参数引用
        self.materials = {}
        self.dimensions = {}
        # 初始化默认值
        self._init_default_ranges()
    
    def _init_default_ranges(self):
        """
        初始化默认的动态参数范围
        """
        # 时间相关特征的默认范围
        self.optimized_ranges = {
            # 基础时间参数
            "T_norm": (0.0, 1.0),
            "T_phase": (-1.0, 1.0),
            
            # 电压动态参数
            "dV_dt": (-1.0, 1.0),  # 归一化的电压变化率
            "V_effective": (0.0, 1.0),
            "voltage_saturation": (0.0, 1.0),
            
            # 松弛和响应时间参数
            "relaxation_time": (0.0, 1.0),  # 归一化的松弛时间
            "response_time": (0.0, 1.0),    # 归一化的响应时间
            
            # 电润湿动态参数
            "electrowetting_number": (0.0, 1.0),
            "dynamic_hysteresis": (0.0, 1.0),
            "young_lippmann_dev": (0.0, 1.0),
            "line_mobility": (0.0, 1.0),
            
            # 界面动态参数
            "interface_history": (-1.0, 1.0),
            "velocity_trend": (-1.0, 1.0),
            "angle_evolution": (-1.0, 1.0),
            "instability_index": (0.0, 1.0),
            "fluctuation_level": (0.0, 1.0),
            
            # 流体动态参数
            "reynolds_local": (0.0, 1.0),
            "capillary_number": (0.0, 1.0),
            "weber_number": (0.0, 1.0),
            "flow_regime": (0.0, 1.0),
            "marangoni_effect": (0.0, 1.0)
        }
        
        # 默认时间尺度
        self.time_scales = self.PHYSICAL_TIME_SCALES.copy()
    
    def set_material_parameters(self, material_structure, dimensions):
        """
        设置材料和维度参数，用于后续的动态参数计算
        
        参数:
        - material_structure: 材料结构字典
        - dimensions: 维度信息字典
        """
        self.materials = material_structure
        self.dimensions = dimensions
        
        # 根据材料参数优化时间尺度
        self._optimize_time_scales()
    
    def _optimize_time_scales(self):
        """
        根据材料参数优化时间尺度
        """
        # 计算介电层的松弛时间
        if "介电层" in self.materials:
            dielectric = self.materials["介电层"]
            epsilon = dielectric.get("相对介电常数", 10.0)
            sigma = dielectric.get("电导率", 1e-12)
            thickness = dielectric.get("厚度", 0.4e-6)
            
            # 计算介电松弛时间
            relaxation_time = epsilon * self.PHYSICAL_CONSTANTS["epsilon0"] / sigma
            self.time_scales["dielectric_relaxation"] = relaxation_time
        
        # 计算接触线运动时间
        if "疏水层" in self.materials and "极性液体层" in self.materials:
            hydrophobic = self.materials["疏水层"]
            polar_liquid = self.materials["极性液体层"]
            
            surface_tension = hydrophobic.get("表面张力", 0.015)
            viscosity = self.PHYSICAL_CONSTANTS["mu_water"]  # 假设极性液体是水
            thickness = polar_liquid.get("厚度", 17e-6)
            
            # 估算接触线运动时间
            # 基于Capillary数和特征速度
            char_velocity = surface_tension / viscosity
            if char_velocity > 0:
                contact_time = thickness / char_velocity * 10  # 调整系数
                self.time_scales["contact_line_motion"] = contact_time
        
        # 更新总的时间尺度范围
        min_time = min(self.time_scales.values())
        max_time = max(self.time_scales.values())
        
        # 确保T_norm的范围能够覆盖所有物理过程
        self.time_scales["min_physical_time"] = min_time
        self.time_scales["max_physical_time"] = max_time
        self.time_scales["total_time_range"] = max_time / min_time
    
    def calculate_relaxation_time(self, layer_name=None):
        """
        计算指定层的松弛时间
        
        参数:
        - layer_name: 图层名称，默认为介电层
        
        返回:
        - 松弛时间 [秒]
        """
        if layer_name is None:
            layer_name = "介电层"
        
        if layer_name not in self.materials:
            raise ValueError(f"找不到图层: {layer_name}")
        
        layer = self.materials[layer_name]
        epsilon = layer.get("相对介电常数", 1.0)
        sigma = layer.get("电导率", 1e-12)
        
        # 介电松弛时间公式: τ = εε₀/σ
        relaxation_time = epsilon * self.PHYSICAL_CONSTANTS["epsilon0"] / sigma
        
        return relaxation_time
    
    def calculate_electrowetting_number(self, voltage, layer_name=None):
        """
        计算电润湿数
        
        参数:
        - voltage: 施加电压 [V]
        - layer_name: 介电层名称，默认为介电层
        
        返回:
        - 电润湿数 (无量纲)
        """
        if layer_name is None:
            layer_name = "介电层"
        
        if layer_name not in self.materials:
            raise ValueError(f"找不到图层: {layer_name}")
        
        layer = self.materials[layer_name]
        epsilon = layer.get("相对介电常数", 1.0)
        thickness = layer.get("厚度", 1.0e-6)
        
        # 获取极性液体和疏水层的表面张力
        if "疏水层" in self.materials:
            surface_tension = self.materials["疏水层"].get("表面张力", 0.015)
        else:
            surface_tension = self.PHYSICAL_CONSTANTS["gamma_oil_water"]
        
        # 电润湿数公式: We = εε₀V²/(2γd²)
        electrowetting_number = (epsilon * self.PHYSICAL_CONSTANTS["epsilon0"] * voltage**2) / \
                               (2 * surface_tension * thickness**2)
        
        # 限制最大值为1（Young-Lippmann方程的理论极限）
        return min(electrowetting_number, 1.0)
    
    def normalize_time(self, time_value, reference_time=None):
        """
        归一化时间值
        
        参数:
        - time_value: 时间值 [秒]
        - reference_time: 参考时间，默认为最大物理时间
        
        返回:
        - 归一化的时间值 (0-1范围)
        """
        if reference_time is None:
            reference_time = self.time_scales.get("max_physical_time", 1.0)
        
        # 确保参考时间不为零
        if reference_time <= 0:
            reference_time = 1.0
        
        # 归一化到0-1范围
        normalized = min(time_value / reference_time, 1.0)
        return max(normalized, 0.0)  # 确保非负
    
    def denormalize_time(self, normalized_value, reference_time=None):
        """
        将归一化的时间值转换回物理时间
        
        参数:
        - normalized_value: 归一化的时间值 (0-1范围)
        - reference_time: 参考时间，默认为最大物理时间
        
        返回:
        - 物理时间值 [秒]
        """
        if reference_time is None:
            reference_time = self.time_scales.get("max_physical_time", 1.0)
        
        # 确保归一化值在0-1范围内
        normalized_value = max(0.0, min(1.0, normalized_value))
        
        return normalized_value * reference_time
    
    def get_optimized_range(self, feature_name):
        """
        获取优化后的特征范围
        
        参数:
        - feature_name: 特征名称
        
        返回:
        - (最小值, 最大值) 的范围元组
        """
        if feature_name in self.optimized_ranges:
            return self.optimized_ranges[feature_name]
        
        # 返回默认范围
        return (0.0, 1.0)
    
    def optimize_dynamic_ranges(self):
        """
        根据材料参数优化所有动态参数的归一化范围
        
        返回:
        - 优化后的范围字典
        """
        # 优化时间相关参数范围
        if "min_physical_time" in self.time_scales and "max_physical_time" in self.time_scales:
            # T_norm范围保持0-1，但内部映射会基于物理时间范围
            pass
        
        # 优化电压变化率范围
        if "电学参数" in self.dimensions:
            voltage_range = self.dimensions["电学参数"].get("频率范围", [0, 0])
            if len(voltage_range) == 2 and voltage_range[1] > voltage_range[0]:
                # 基于频率范围估算电压变化率的合理范围
                max_freq = voltage_range[1]
                # dV/dt的物理范围约为 V_max * 2πf
                # 这里我们只需要更新内部计算逻辑，保持归一化范围为[-1, 1]
                pass
        
        # 优化松弛时间范围
        if "介电层" in self.materials:
            relaxation_time = self.calculate_relaxation_time("介电层")
            # 松弛时间的归一化范围应该能够覆盖从最小到最大的物理松弛时间
            # 这里我们保持范围为[0, 1]，但内部映射会使用实际的松弛时间值
            pass
        
        # 优化电润湿数范围
        if "电学参数" in self.dimensions:
            max_voltage = float(self.dimensions["电学参数"].get("最大工作电压", 30.0))
            max_electrowetting = self.calculate_electrowetting_number(max_voltage)
            # 电润湿数的理论最大值为1，所以保持范围为[0, 1]
            pass
        
        return self.optimized_ranges.copy()
    
    def get_time_phase(self, time_value, frequency=None):
        """
        计算时间相位（用于周期性过程）
        
        参数:
        - time_value: 时间值 [秒]
        - frequency: 频率 [Hz]，默认为1Hz
        
        返回:
        - 时间相位 (-1到1范围，对应-π到π)
        """
        if frequency is None:
            # 基于最小物理时间计算特征频率
            min_time = self.time_scales.get("min_physical_time", 1.0)
            frequency = 1.0 / min_time
        
        # 计算相位 (cos(ωt))，范围为[-1, 1]
        phase = np.cos(2 * np.pi * frequency * time_value)
        return phase
    
    def calculate_voltage_rate(self, voltage_prev, voltage_next, time_prev, time_next):
        """
        计算电压变化率
        
        参数:
        - voltage_prev: 前一时刻的电压 [V]
        - voltage_next: 下一时刻的电压 [V]
        - time_prev: 前一时刻的时间 [秒]
        - time_next: 下一时刻的时间 [秒]
        
        返回:
        - 归一化的电压变化率
        """
        # 计算实际电压变化率
        delta_time = time_next - time_prev
        if delta_time == 0:
            return 0.0
        
        actual_rate = (voltage_next - voltage_prev) / delta_time
        
        # 归一化到[-1, 1]范围
        # 假设最大合理变化率为 1000 V/s
        max_rate = 1000.0
        normalized_rate = actual_rate / max_rate
        
        # 限制在[-1, 1]范围内
        return max(-1.0, min(1.0, normalized_rate))
    
    def create_dynamic_feature_map(self, current_time, voltage, frequency=None):
        """
        创建动态特征映射
        
        参数:
        - current_time: 当前时间 [秒]
        - voltage: 当前电压 [V]
        - frequency: 驱动频率 [Hz]
        
        返回:
        - 动态特征字典
        """
        feature_map = {}
        
        # 基础时间参数
        feature_map["T_norm"] = self.normalize_time(current_time)
        feature_map["T_phase"] = self.get_time_phase(current_time, frequency)
        
        # 松弛时间参数
        if "介电层" in self.materials:
            relaxation_time = self.calculate_relaxation_time("介电层")
            feature_map["relaxation_time"] = self.normalize_time(relaxation_time, 
                                                               self.time_scales.get("max_physical_time", 1.0))
        
        # 电润湿参数
        feature_map["electrowetting_number"] = self.calculate_electrowetting_number(voltage)
        
        # 归一化电压
        max_voltage = float(self.dimensions.get("电学参数", {}).get("最大工作电压", 30.0))
        if max_voltage > 0:
            feature_map["V_effective"] = min(voltage / max_voltage, 1.0)
        
        # 电压饱和参数
        # 当电压接近最大值时，电润湿效应趋于饱和
        if max_voltage > 0:
            voltage_ratio = voltage / max_voltage
            # 非线性映射，模拟饱和效应
            feature_map["voltage_saturation"] = voltage_ratio**2 / (0.5 + voltage_ratio**2)
        
        # 动态滞后参数
        # 简化模型，基于电压和频率
        if frequency is not None and frequency > 0:
            # 高频下滞后增加
            feature_map["dynamic_hysteresis"] = min(frequency / 1000.0, 1.0)
        else:
            feature_map["dynamic_hysteresis"] = 0.1  # 默认值
        
        # Young-Lippmann方程偏差
        # 随着电润湿数增加，实际接触角与理论值的偏差增大
        ew_number = feature_map.get("electrowetting_number", 0.0)
        feature_map["young_lippmann_dev"] = ew_number**2  # 简化模型
        
        return feature_map
    
    def validate_dynamic_parameters(self, dynamic_features):
        """
        验证动态参数的物理合理性
        
        参数:
        - dynamic_features: 动态特征字典
        
        返回:
        - 验证结果字典
        """
        results = {"valid": True, "warnings": []}
        
        # 验证电润湿数
        if "electrowetting_number" in dynamic_features:
            ew_number = dynamic_features["electrowetting_number"]
            if ew_number > 1.0:
                results["valid"] = False
                results["warnings"].append({
                    "parameter": "electrowetting_number",
                    "value": ew_number,
                    "message": "电润湿数超过理论最大值1.0"
                })
        
        # 验证时间参数
        if "T_norm" in dynamic_features:
            t_norm = dynamic_features["T_norm"]
            if t_norm < 0 or t_norm > 1:
                results["valid"] = False
                results["warnings"].append({
                    "parameter": "T_norm",
                    "value": t_norm,
                    "message": "归一化时间必须在0-1范围内"
                })
        
        # 验证松弛时间
        if "relaxation_time" in dynamic_features and "介电层" in self.materials:
            rel_time_norm = dynamic_features["relaxation_time"]
            actual_rel_time = self.calculate_relaxation_time("介电层")
            expected_rel_time_norm = self.normalize_time(actual_rel_time)
            
            # 允许10%的误差
            if abs(rel_time_norm - expected_rel_time_norm) > 0.1:
                results["warnings"].append({
                    "parameter": "relaxation_time",
                    "value": rel_time_norm,
                    "expected": expected_rel_time_norm,
                    "message": "松弛时间与材料参数计算值偏差较大"
                })
        
        return results

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
            "介电层": {
                "厚度": 0.4e-6,
                "相对介电常数": 10.0,
                "电导率": 1e-12,
                "层索引": 2
            },
            "疏水层": {
                "厚度": 0.4e-6,
                "表面张力": 0.015,
                "层索引": 3
            },
            "极性液体层": {
                "厚度": 17e-6,
                "层索引": 5
            }
        }
        
        # 示例尺寸信息
        dimensions = {
            "电学参数": {
                "最大工作电压": 30.0,
                "频率范围": [10, 1000]
            }
        }
    
    # 创建动态参数优化器实例
    optimizer = EWPDynamicParameterOptimizer()
    
    # 设置材料参数
    optimizer.set_material_parameters(material_structure, dimensions)
    
    print("=== EWP动态参数优化器示例 ===")
    
    # 计算关键动态参数
    relaxation_time = optimizer.calculate_relaxation_time()
    print(f"介电层松弛时间: {relaxation_time:.2e} 秒")
    
    # 计算不同电压下的电润湿数
    for voltage in [0, 15, 30]:
        ew_number = optimizer.calculate_electrowetting_number(voltage)
        print(f"电压 {voltage}V 时的电润湿数: {ew_number:.4f}")
    
    # 优化动态参数范围
    optimized_ranges = optimizer.optimize_dynamic_ranges()
    print(f"\n优化后的动态参数范围数量: {len(optimized_ranges)}")
    
    # 创建动态特征映射
    current_time = 0.005  # 5ms
    voltage = 25.0  # 25V
    frequency = 100.0  # 100Hz
    
    feature_map = optimizer.create_dynamic_feature_map(current_time, voltage, frequency)
    print(f"\n动态特征映射示例 (时间={current_time}s, 电压={voltage}V, 频率={frequency}Hz):")
    for key, value in feature_map.items():
        print(f"  {key}: {value:.4f}")
    
    # 验证动态参数
    validation = optimizer.validate_dynamic_parameters(feature_map)
    print(f"\n动态参数验证: {'有效' if validation['valid'] else '无效'}")
    if validation['warnings']:
        print("验证警告:")
        for warning in validation['warnings']:
            print(f"  - {warning['message']}")
    
    print("\n动态参数优化器功能演示完成！")