"""
EWP一致性验证器

此模块负责在EWPINNInputLayer中添加与3D结构一致性的验证函数，
确保输入特征的物理合理性，避免超出3D模型定义的参数范围。

主要功能：
- 输入特征的物理合理性验证
- 3D结构与输入层参数映射的一致性检查
- 材料参数范围的有效性验证
- 动态参数的物理约束验证
- 错误报告和修复建议生成
"""

import numpy as np
import os
import sys

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class EWPConsistencyValidator:
    """
    电润湿像素一致性验证器类
    负责验证输入层与3D结构的一致性
    """
    
    # 物理参数的标准范围约束
    PHYSICAL_CONSTRAINTS = {
        # 几何参数约束
        "dimensions": {
            "pixel_size_x": (0.0, 1.0e-3),  # 像素宽度 (m)
            "pixel_size_y": (0.0, 1.0e-3),  # 像素高度 (m)
            "total_thickness": (0.0, 5.0e-5),  # 总厚度 (m)
            "layer_thickness_min": 0.0,  # 最小层厚度 (m)
            "layer_thickness_max": 1.0e-4,  # 最大层厚度 (m)
        },
        
        # 材料参数约束
        "materials": {
            "relative_permittivity": (1.0, 100.0),  # 相对介电常数
            "conductivity": (1.0e-16, 1.0e5),  # 电导率 (S/m)
            "surface_tension": (0.001, 0.1),  # 表面张力 (N/m)
            "contact_angle": (0.0, 180.0),  # 接触角 (度)
            "young_modulus": (1.0e6, 1.0e12),  # 杨氏模量 (Pa)
            "viscosity": (1.0e-6, 1.0),  # 粘度 (Pa·s)
        },
        
        # 电学参数约束
        "electrical": {
            "voltage": (0.0, 100.0),  # 电压 (V)
            "frequency": (0.0, 1.0e9),  # 频率 (Hz)
            "electric_field": (0.0, 1.0e9),  # 电场强度 (V/m)
        },
        
        # 动态参数约束
        "dynamic": {
            "relaxation_time": (1.0e-12, 1.0e3),  # 松弛时间 (s)
            "response_time": (1.0e-9, 1.0e3),  # 响应时间 (s)
            "electrowetting_number": (0.0, 1.0),  # 电润湿数
            "capillary_number": (0.0, 1.0e-2),  # 毛细管数
            "reynolds_number": (0.0, 1.0e6),  # 雷诺数
        },
        
        # 归一化特征约束
        "normalized_features": {
            "range_min": -2.0,  # 归一化最小值（允许一些余量）
            "range_max": 2.0,  # 归一化最大值（允许一些余量）
            "valid_range_min": 0.0,  # 标准有效范围最小值
            "valid_range_max": 1.0,  # 标准有效范围最大值
        }
    }
    
    # 物理参数到归一化特征的映射关系
    PARAMETER_TO_FEATURE_MAPPING = {
        # 几何参数映射
        "dimensions": {
            "pixel_size_x": "几何参数:像素宽度",
            "pixel_size_y": "几何参数:像素高度",
            "total_thickness": "几何参数:总厚度",
            "layer_thicknesses": "几何参数:层厚度",
        },
        
        # 材料参数映射
        "materials": {
            "relative_permittivity": "材料界面:相对介电常数",
            "surface_tension": "材料界面:表面张力",
            "conductivity": "材料界面:电导率",
            "contact_angle": "电润湿特异性:接触角",
        },
        
        # 电学参数映射
        "electrical": {
            "voltage": "基础时空电压:电压",
            "electric_field": "电场:电场强度",
            "frequency": "电场:频率",
        },
        
        # 动态参数映射
        "dynamic": {
            "relaxation_time": "时间动态:松弛时间",
            "response_time": "时间动态:响应时间",
            "electrowetting_number": "电润湿特异性:电润湿数",
        }
    }
    
    def __init__(self):
        # 存储3D结构参考数据
        self.reference_structure = {
            "material_structure": {},
            "dimensions": {}
        }
        # 存储验证结果
        self.validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "suggestions": []
        }
        # 存储输入层特征信息
        self.input_layer_info = {}
    
    def set_reference_structure(self, material_structure, dimensions):
        """
        设置3D结构参考数据
        
        参数:
        - material_structure: 材料结构字典
        - dimensions: 维度信息字典
        """
        self.reference_structure["material_structure"] = material_structure
        self.reference_structure["dimensions"] = dimensions
        
        # 重置验证结果
        self._reset_validation_results()
    
    def set_input_layer_info(self, feature_groups, normalization_ranges):
        """
        设置输入层特征信息
        
        参数:
        - feature_groups: 特征组字典
        - normalization_ranges: 归一化范围字典
        """
        self.input_layer_info["feature_groups"] = feature_groups
        self.input_layer_info["normalization_ranges"] = normalization_ranges
    
    def _reset_validation_results(self):
        """
        重置验证结果
        """
        self.validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "suggestions": []
        }
    
    def validate_material_parameters(self):
        """
        验证材料参数的物理合理性
        
        返回:
        - 验证结果字典
        """
        material_structure = self.reference_structure["material_structure"]
        constraints = self.PHYSICAL_CONSTRAINTS["materials"]
        
        for layer_name, layer_params in material_structure.items():
            # 验证相对介电常数
            if "相对介电常数" in layer_params:
                value = layer_params["相对介电常数"]
                min_val, max_val = constraints["relative_permittivity"]
                if value < min_val or value > max_val:
                    self._add_error(
                        f"材料参数",
                        f"{layer_name}的相对介电常数({value})超出有效范围[{min_val}, {max_val}]",
                        f"调整{layer_name}的相对介电常数到有效范围内"
                    )
            
            # 验证电导率
            if "电导率" in layer_params:
                value = layer_params["电导率"]
                min_val, max_val = constraints["conductivity"]
                if value < min_val or value > max_val:
                    self._add_error(
                        f"材料参数",
                        f"{layer_name}的电导率({value})超出有效范围[{min_val}, {max_val}]",
                        f"调整{layer_name}的电导率到有效范围内"
                    )
            
            # 验证表面张力
            if "表面张力" in layer_params:
                value = layer_params["表面张力"]
                min_val, max_val = constraints["surface_tension"]
                if value < min_val or value > max_val:
                    self._add_error(
                        f"材料参数",
                        f"{layer_name}的表面张力({value})超出有效范围[{min_val}, {max_val}]",
                        f"调整{layer_name}的表面张力到有效范围内"
                    )
            
            # 验证接触角
            if "接触角" in layer_params:
                value = layer_params["接触角"]
                min_val, max_val = constraints["contact_angle"]
                if value < min_val or value > max_val:
                    self._add_error(
                        f"材料参数",
                        f"{layer_name}的接触角({value})超出有效范围[{min_val}, {max_val}]度",
                        f"调整{layer_name}的接触角到有效范围内"
                    )
        
        return self.validation_results.copy()
    
    def validate_geometric_parameters(self):
        """
        验证几何参数的物理合理性
        
        返回:
        - 验证结果字典
        """
        dimensions = self.reference_structure["dimensions"]
        material_structure = self.reference_structure["material_structure"]
        geom_constraints = self.PHYSICAL_CONSTRAINTS["dimensions"]
        
        # 验证像素尺寸
        pixel_size_x = dimensions.get("像素尺寸_x", 0)
        pixel_size_y = dimensions.get("像素尺寸_y", 0)
        
        min_size, max_size = geom_constraints["pixel_size_x"]
        if pixel_size_x < min_size or pixel_size_x > max_size:
            self._add_error(
                "几何参数",
                f"像素宽度({pixel_size_x})超出有效范围[{min_size}, {max_size}]米",
                f"调整像素宽度到有效范围内"
            )
        
        if pixel_size_y < min_size or pixel_size_y > max_size:
            self._add_error(
                "几何参数",
                f"像素高度({pixel_size_y})超出有效范围[{min_size}, {max_size}]米",
                f"调整像素高度到有效范围内"
            )
        
        # 验证层厚度
        total_thickness = 0.0
        min_thickness = geom_constraints["layer_thickness_min"]
        max_thickness = geom_constraints["layer_thickness_max"]
        
        for layer_name, layer_params in material_structure.items():
            if "厚度" in layer_params:
                thickness = layer_params["厚度"]
                total_thickness += thickness
                
                if thickness < min_thickness:
                    self._add_error(
                        "几何参数",
                        f"{layer_name}的厚度({thickness})小于最小允许值({min_thickness})米",
                        f"增加{layer_name}的厚度到最小允许值以上"
                    )
                elif thickness > max_thickness:
                    self._add_error(
                        "几何参数",
                        f"{layer_name}的厚度({thickness})大于最大允许值({max_thickness})米",
                        f"减少{layer_name}的厚度到最大允许值以下"
                    )
        
        # 验证总厚度
        min_total, max_total = geom_constraints["total_thickness"]
        if total_thickness < min_total:
            self._add_warning(
                "几何参数",
                f"总厚度({total_thickness})过小，可能影响模拟精度",
                f"考虑增加关键层的厚度以提高模拟精度"
            )
        elif total_thickness > max_total:
            self._add_error(
                "几何参数",
                f"总厚度({total_thickness})超出最大允许值({max_total})米",
                f"减少各层厚度以确保总厚度在有效范围内"
            )
        
        # 检查层索引的连续性
        layer_indices = [params.get("层索引", -1) for params in material_structure.values()]
        layer_indices = [idx for idx in layer_indices if idx >= 0]
        
        if layer_indices and sorted(layer_indices) != list(range(min(layer_indices), max(layer_indices) + 1)):
            self._add_warning(
                "几何参数",
                "层索引不连续，可能导致模型错误",
                "确保层索引从0开始连续递增"
            )
        
        return self.validation_results.copy()
    
    def validate_electrical_parameters(self):
        """
        验证电学参数的物理合理性
        
        返回:
        - 验证结果字典
        """
        dimensions = self.reference_structure["dimensions"]
        electrical_constraints = self.PHYSICAL_CONSTRAINTS["electrical"]
        
        if "电学参数" in dimensions:
            electrical = dimensions["电学参数"]
            
            # 验证电压
            if "最大工作电压" in electrical:
                max_voltage = electrical["最大工作电压"]
                min_val, max_val = electrical_constraints["voltage"]
                if max_voltage < min_val or max_voltage > max_val:
                    self._add_error(
                        "电学参数",
                        f"最大工作电压({max_voltage})超出有效范围[{min_val}, {max_val}]V",
                        f"调整最大工作电压到有效范围内"
                    )
            
            # 验证频率范围
            if "频率范围" in electrical:
                freq_range = electrical["频率范围"]
                if len(freq_range) == 2:
                    min_freq, max_freq = freq_range
                    min_val, max_val = electrical_constraints["frequency"]
                    
                    if min_freq < min_val or min_freq > max_val:
                        self._add_error(
                            "电学参数",
                            f"最小频率({min_freq})超出有效范围[{min_val}, {max_val}]Hz",
                            f"调整最小频率到有效范围内"
                        )
                    
                    if max_freq < min_val or max_freq > max_val:
                        self._add_error(
                            "电学参数",
                            f"最大频率({max_freq})超出有效范围[{min_val}, {max_val}]Hz",
                            f"调整最大频率到有效范围内"
                        )
                    
                    if max_freq <= min_freq:
                        self._add_error(
                            "电学参数",
                            f"频率范围无效，最大频率({max_freq})应大于最小频率({min_freq})",
                            f"调整频率范围，确保最大值大于最小值"
                        )
        
        return self.validation_results.copy()
    
    def validate_feature_ranges(self, input_features):
        """
        验证输入特征的范围合理性
        
        参数:
        - input_features: 输入特征字典或数组
        
        返回:
        - 验证结果字典
        """
        norm_constraints = self.PHYSICAL_CONSTRAINTS["normalized_features"]
        min_range = norm_constraints["range_min"]
        max_range = norm_constraints["range_max"]
        valid_min = norm_constraints["valid_range_min"]
        valid_max = norm_constraints["valid_range_max"]
        
        # 处理字典类型输入
        if isinstance(input_features, dict):
            for feature_name, value in input_features.items():
                if value < min_range or value > max_range:
                    self._add_error(
                        "输入特征范围",
                        f"特征'{feature_name}'的值({value})超出允许范围[{min_range}, {max_range}]",
                        f"确保所有特征值在允许范围内"
                    )
                elif value < valid_min or value > valid_max:
                    self._add_warning(
                        "输入特征范围",
                        f"特征'{feature_name}'的值({value})超出标准有效范围[{valid_min}, {valid_max}]",
                        f"考虑调整特征值到标准有效范围内以获得更好的物理一致性"
                    )
        
        # 处理数组类型输入
        elif isinstance(input_features, (list, np.ndarray)):
            input_array = np.array(input_features)
            outliers = np.where((input_array < min_range) | (input_array > max_range))[0]
            warnings = np.where((input_array >= min_range) & (input_array < valid_min) | 
                              (input_array > valid_max) & (input_array <= max_range))[0]
            
            for idx in outliers:
                self._add_error(
                    "输入特征范围",
                    f"特征索引{idx}的值({input_array[idx]})超出允许范围[{min_range}, {max_range}]",
                    f"修正索引{idx}处的特征值"
                )
            
            for idx in warnings:
                self._add_warning(
                    "输入特征范围",
                    f"特征索引{idx}的值({input_array[idx]})超出标准有效范围[{valid_min}, {valid_max}]",
                    f"考虑调整索引{idx}处的特征值到标准有效范围内"
                )
        
        return self.validation_results.copy()
    
    def validate_parameter_mapping(self):
        """
        验证3D结构与输入层参数映射的一致性
        
        返回:
        - 验证结果字典
        """
        # 检查材料结构中的关键参数是否在输入层中有对应特征
        material_structure = self.reference_structure["material_structure"]
        feature_groups = self.input_layer_info.get("feature_groups", {})
        
        # 展平特征组，获取所有特征名称
        all_features = []
        for group_name, group_features in feature_groups.items():
            all_features.extend(group_features)
        
        # 检查关键材料参数的映射
        key_parameters = ["相对介电常数", "电导率", "表面张力", "接触角"]
        missing_parameters = []
        
        for layer_name, layer_params in material_structure.items():
            for param_name in key_parameters:
                if param_name in layer_params:
                    # 查找对应的特征
                    mapped = False
                    for category, mappings in self.PARAMETER_TO_FEATURE_MAPPING.items():
                        if param_name in mappings.values() or param_name in mappings:
                            mapped = True
                            break
                    
                    if not mapped and param_name not in all_features:
                        missing_parameters.append((layer_name, param_name))
        
        if missing_parameters:
            for layer, param in missing_parameters:
                self._add_warning(
                    "参数映射",
                    f"{layer}中的参数'{param}'在输入层中没有明确的映射特征",
                    f"建议在输入层中添加对应于{param}的特征，或更新参数映射配置"
                )
        
        # 检查归一化范围配置
        normalization_ranges = self.input_layer_info.get("normalization_ranges", {})
        if not normalization_ranges:
            self._add_warning(
                "归一化配置",
                "未提供输入层的归一化范围配置",
                "添加归一化范围配置以确保物理参数正确映射"
            )
        
        return self.validation_results.copy()
    
    def validate_input_layer_with_structure(self, input_vector, stage=3):
        """
        验证输入向量与3D结构的一致性
        
        参数:
        - input_vector: 输入向量（数组或字典）
        - stage: 输入层阶段 (1, 2, 或 3)
        
        返回:
        - 验证结果字典
        """
        # 重置验证结果
        self._reset_validation_results()
        
        # 验证输入特征范围
        self.validate_feature_ranges(input_vector)
        
        # 验证参数映射
        self.validate_parameter_mapping()
        
        # 根据阶段进行特定验证
        if stage == 1:
            # 阶段1：基础特征验证
            if isinstance(input_vector, (list, np.ndarray)) and len(input_vector) >= 10:
                # 验证电压相关特征
                voltage_features = input_vector[:2]  # 假设前两个是电压相关
                for idx, val in enumerate(voltage_features):
                    if val < 0 or val > 1:
                        self._add_warning(
                            "电压特征",
                            f"电压特征索引{idx}的值({val})应在[0, 1]范围内",
                            f"确保电压特征正确归一化"
                        )
        
        elif stage == 2:
            # 阶段2：增加了材料和几何特征
            pass
        
        elif stage == 3:
            # 阶段3：完整特征集
            # 验证动态参数的一致性
            if isinstance(input_vector, dict):
                self._validate_dynamic_consistency(input_vector)
        
        # 生成总体建议
        self._generate_suggestions()
        
        return self.validation_results.copy()
    
    def _validate_dynamic_consistency(self, dynamic_features):
        """
        验证动态参数的物理一致性
        
        参数:
        - dynamic_features: 动态特征字典
        """
        # 验证电润湿数与接触角的关系
        if "electrowetting_number" in dynamic_features and "contact_angle" in dynamic_features:
            ew_number = dynamic_features["electrowetting_number"]
            contact_angle = dynamic_features["contact_angle"]
            
            # Young-Lippmann方程的简化验证
            # cos(theta) = cos(theta0) - 2*We
            # 这里只做粗略检查
            if ew_number > 0.5:
                self._add_warning(
                    "动态参数一致性",
                    f"电润湿数({ew_number})较高，接触角({contact_angle})应相应减小",
                    f"确保接触角与电润湿数符合Young-Lippmann方程关系"
                )
        
        # 验证松弛时间与频率的关系
        if "relaxation_time" in dynamic_features and "电场:频率" in dynamic_features:
            rel_time = dynamic_features["relaxation_time"]
            frequency = dynamic_features["电场:频率"]
            
            # 归一化值需要转换为实际值才能进行正确验证
            # 这里只做简化检查
            if rel_time > 0.5 and frequency > 0.5:
                self._add_warning(
                    "动态参数一致性",
                    f"高频({frequency})下松弛时间({rel_time})较长，可能存在物理不一致",
                    f"确保频率和松弛时间的物理关系合理"
                )
    
    def _add_error(self, category, message, suggestion):
        """
        添加错误消息
        """
        self.validation_results["valid"] = False
        self.validation_results["errors"].append({
            "category": category,
            "message": message,
            "suggestion": suggestion
        })
    
    def _add_warning(self, category, message, suggestion):
        """
        添加警告消息
        """
        self.validation_results["warnings"].append({
            "category": category,
            "message": message,
            "suggestion": suggestion
        })
    
    def _generate_suggestions(self):
        """
        基于验证结果生成综合建议
        """
        # 如果有错误，建议修复所有错误
        if self.validation_results["errors"]:
            self.validation_results["suggestions"].append(
                "请修复所有错误后再进行模拟，特别是超出物理范围的参数值"
            )
        
        # 如果有多个警告，建议检查一致性
        if len(self.validation_results["warnings"]) > 3:
            self.validation_results["suggestions"].append(
                "存在多个警告，建议全面检查3D结构与输入层的参数一致性"
            )
        
        # 材料参数建议
        if any("材料参数" in warn["category"] for warn in self.validation_results["warnings"]):
            self.validation_results["suggestions"].append(
                "确保材料参数与实际材料特性相符，可参考材料数据库进行校准"
            )
        
        # 几何参数建议
        if any("几何参数" in warn["category"] for warn in self.validation_results["warnings"]):
            self.validation_results["suggestions"].append(
                "检查几何参数是否符合实际像素设计，特别注意层厚度的比例关系"
            )
    
    def get_fix_suggestions(self, input_vector):
        """
        获取输入向量的修复建议
        
        参数:
        - input_vector: 需要修复的输入向量
        
        返回:
        - 修复后的输入向量和修复说明
        """
        fixed_vector = None
        fixes = []
        
        # 处理字典类型
        if isinstance(input_vector, dict):
            fixed_vector = input_vector.copy()
            norm_constraints = self.PHYSICAL_CONSTRAINTS["normalized_features"]
            
            for feature_name, value in input_vector.items():
                if value < norm_constraints["range_min"]:
                    fixed_vector[feature_name] = norm_constraints["range_min"]
                    fixes.append({
                        "feature": feature_name,
                        "original": value,
                        "fixed": norm_constraints["range_min"],
                        "reason": "超出最小允许范围"
                    })
                elif value > norm_constraints["range_max"]:
                    fixed_vector[feature_name] = norm_constraints["range_max"]
                    fixes.append({
                        "feature": feature_name,
                        "original": value,
                        "fixed": norm_constraints["range_max"],
                        "reason": "超出最大允许范围"
                    })
        
        # 处理数组类型
        elif isinstance(input_vector, (list, np.ndarray)):
            fixed_vector = np.array(input_vector).copy()
            norm_constraints = self.PHYSICAL_CONSTRAINTS["normalized_features"]
            
            # 修复超出范围的值
            min_mask = fixed_vector < norm_constraints["range_min"]
            max_mask = fixed_vector > norm_constraints["range_max"]
            
            if np.any(min_mask):
                for idx in np.where(min_mask)[0]:
                    fixes.append({
                        "index": int(idx),
                        "original": fixed_vector[idx],
                        "fixed": norm_constraints["range_min"],
                        "reason": "超出最小允许范围"
                    })
                fixed_vector[min_mask] = norm_constraints["range_min"]
            
            if np.any(max_mask):
                for idx in np.where(max_mask)[0]:
                    fixes.append({
                        "index": int(idx),
                        "original": fixed_vector[idx],
                        "fixed": norm_constraints["range_max"],
                        "reason": "超出最大允许范围"
                    })
                fixed_vector[max_mask] = norm_constraints["range_max"]
        
        return {
            "fixed_vector": fixed_vector,
            "fixes_applied": fixes,
            "total_fixes": len(fixes)
        }
    
    def generate_validation_report(self):
        """
        生成验证报告
        
        返回:
        - 格式化的验证报告字符串
        """
        report = ["=== EWP一致性验证报告 ==="]
        
        # 总体状态
        report.append(f"验证状态: {'通过' if self.validation_results['valid'] else '失败'}")
        report.append("")
        
        # 错误信息
        if self.validation_results["errors"]:
            report.append("[错误] 发现以下错误:")
            for i, error in enumerate(self.validation_results["errors"], 1):
                report.append(f"  {i}. [{error['category']}] {error['message']}")
                report.append(f"     建议: {error['suggestion']}")
            report.append("")
        
        # 警告信息
        if self.validation_results["warnings"]:
            report.append("[警告] 发现以下警告:")
            for i, warning in enumerate(self.validation_results["warnings"], 1):
                report.append(f"  {i}. [{warning['category']}] {warning['message']}")
                report.append(f"     建议: {warning['suggestion']}")
            report.append("")
        
        # 建议信息
        if self.validation_results["suggestions"]:
            report.append("[建议]")
            for suggestion in self.validation_results["suggestions"]:
                report.append(f"  - {suggestion}")
            report.append("")
        
        # 总结
        if self.validation_results["valid"]:
            report.append("验证总结: 3D结构与输入层参数一致，可以进行后续模拟。")
        else:
            report.append("验证总结: 发现错误，需要修复后再进行模拟。")
        
        return "\n".join(report)

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
                "厚度": 0.18e-3,
                "相对介电常数": 10.0,
                "电导率": 5e6,
                "层索引": 0
            },
            "围堰": {
                "厚度": 15e-6,
                "相对介电常数": 4.0,
                "层索引": 1
            },
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
            "油墨层": {
                "厚度": 5e-6,
                "层索引": 4
            },
            "极性液体层": {
                "厚度": 17e-6,
                "层索引": 5
            },
            "顶层ITO层": {
                "厚度": 0.18e-3,
                "相对介电常数": 10.0,
                "电导率": 5e6,
                "层索引": 6
            }
        }
        
        # 示例尺寸信息
        dimensions = {
            "像素尺寸_x": 184e-6,  # 184μm
            "像素尺寸_y": 184e-6,  # 184μm
            "电学参数": {
                "最大工作电压": 30.0,
                "频率范围": [10, 1000]
            }
        }
    
    # 尝试从ewp_pinn_input_layer.py导入特征信息
    try:
        from ewp_pinn_input_layer import EWPINNInputLayer
        input_layer = EWPINNInputLayer()
        feature_groups = input_layer.feature_groups
        normalization_ranges = input_layer.normalization_ranges
        print("成功导入输入层特征信息")
    except ImportError:
        print("无法导入ewp_pinn_input_layer模块，使用示例特征信息")
        
        # 示例特征组
        feature_groups = {
            "基础时空电压": ["V_norm", "T_norm", "x_norm", "y_norm", "z_norm", "T_phase"],
            "几何参数": ["pixel_width", "pixel_height", "layer_thickness"],
            "材料界面": ["relative_permittivity", "surface_tension", "conductivity"],
            "电场": ["electric_field", "frequency", "voltage_gradient"],
            "流体动力学": ["viscosity_ratio", "density_ratio", "capillary_number"],
            "时间动态": ["relaxation_time", "response_time", "time_step"],
            "电润湿特异性": ["contact_angle", "electrowetting_number", "hysteresis"]
        }
        
        # 示例归一化范围
        normalization_ranges = {
            "V_norm": (0.0, 1.0),
            "T_norm": (0.0, 1.0),
            "x_norm": (0.0, 1.0),
            "y_norm": (0.0, 1.0),
            "z_norm": (0.0, 1.0),
            "relative_permittivity": (1.0, 100.0),
            "surface_tension": (0.001, 0.1),
            "contact_angle": (0.0, 180.0)
        }
    
    # 创建验证器实例
    validator = EWPConsistencyValidator()
    
    # 设置参考结构
    validator.set_reference_structure(material_structure, dimensions)
    
    # 设置输入层信息
    validator.set_input_layer_info(feature_groups, normalization_ranges)
    
    print("开始验证3D结构与输入层的一致性...")
    
    # 验证材料参数
    validator.validate_material_parameters()
    
    # 验证几何参数
    validator.validate_geometric_parameters()
    
    # 验证电学参数
    validator.validate_electrical_parameters()
    
    # 创建示例输入向量进行验证
    # 生成一个62维的示例输入向量（阶段3）
    example_input = np.random.rand(62)
    
    # 添加一些故意的错误值以测试验证
    example_input[0] = 1.5  # 超出范围的电压
    example_input[5] = -0.5  # 负的时间相位
    
    # 验证输入向量
    validator.validate_input_layer_with_structure(example_input, stage=3)
    
    # 获取修复建议
    fix_result = validator.get_fix_suggestions(example_input)
    print(f"\n发现并修复了 {fix_result['total_fixes']} 个超出范围的值")
    
    # 生成验证报告
    report = validator.generate_validation_report()
    print("\n" + report)
    
    print("\n一致性验证完成！")