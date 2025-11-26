# 电磁场案例研究

## 概述

本案例研究展示EFD3D在电磁场模拟领域的应用，涵盖静态场、时谐场、波传播、天线设计等典型电磁问题。

## 案例1：静电场分析

### 问题描述
模拟复杂几何形状下的静电场分布，分析电场强度和电势分布。

### 物理模型
```python
import torch
import numpy as np
from efd_pinns import OptimizedEWPINN
from efd_pinns.constraints import ElectrostaticConstraint

class ElectrostaticCase:
    """静电场分析案例"""
    
    def __init__(self, geometry_config):
        self.geometry = geometry_config
        self.permittivity = 1.0  # 相对介电常数
    
    def setup_electrostatic_constraints(self):
        """设置静电场约束"""
        return ElectrostaticConstraint(
            permittivity=self.permittivity,
            charge_density=self._calculate_charge_distribution()
        )
    
    def simulate_potential_distribution(self):
        """模拟电势分布"""
        model_config = {
            'hidden_layers': [128, 128, 128],
            'activation': 'tanh',
            'learning_rate': 1e-4,
            'output_variables': ['potential']
        }
        
        model = OptimizedEWPINN(model_config)
        
        # 添加静电场约束
        electrostatic_constraint = self.setup_electrostatic_constraints()
        model.add_constraints(electrostatic_constraint)
        
        # 边界条件
        boundary_conditions = self._setup_boundary_conditions()
        
        # 训练模型
        training_data = self.generate_training_data()
        history = model.train(training_data, epochs=10000)
        
        return model, history
    
    def analyze_electric_field(self, model):
        """分析电场特性"""
        # 计算电场强度
        electric_field = model.predict_electric_field()
        
        # 计算电场能量密度
        energy_density = self._calculate_energy_density(electric_field)
        
        # 分析场强分布
        field_strength_analysis = self._analyze_field_strength(electric_field)
        
        return {
            'electric_field': electric_field,
            'energy_density': energy_density,
            'field_strength': field_strength_analysis
        }
```

### 复杂电极结构分析
```python
class ComplexElectrodeAnalysis(ElectrostaticCase):
    """复杂电极结构分析"""
    
    def __init__(self, electrode_config):
        super().__init__(electrode_config)
        self.electrode_shapes = electrode_config['shapes']
        self.voltages = electrode_config['voltages']
    
    def _setup_boundary_conditions(self):
        """设置复杂边界条件"""
        boundary_conditions = {}
        
        for i, shape in enumerate(self.electrode_shapes):
            # 为每个电极设置电压
            boundary_conditions[f'electrode_{i}'] = {
                'type': 'dirichlet',
                'value': self.voltages[i],
                'geometry': shape
            }
        
        # 远场边界条件
        boundary_conditions['far_field'] = {
            'type': 'neumann',
            'value': 0.0  # 电场法向分量为零
        }
        
        return boundary_conditions
    
    def optimize_electrode_design(self, objective='field_uniformity'):
        """优化电极设计"""
        from efd_pinns.optimization import ElectrodeOptimizer
        
        optimizer = ElectrodeOptimizer(
            base_design=self.electrode_shapes,
            optimization_objective=objective,
            constraints=['manufacturing_constraints', 'voltage_limits']
        )
        
        optimized_design = optimizer.optimize()
        
        return optimized_design
```

## 案例2：时谐电磁场

### 问题描述
模拟时谐电磁场（交流场），分析电磁波传播和能量分布。

### 实现代码
```python
class TimeHarmonicEMCase:
    """时谐电磁场案例"""
    
    def __init__(self, frequency=1e9, material_properties=None):
        self.frequency = frequency  # 频率 (Hz)
        self.angular_frequency = 2 * np.pi * frequency
        self.material_properties = material_properties or {}
    
    def setup_maxwell_constraints(self):
        """设置时谐麦克斯韦方程约束"""
        from efd_pinns.constraints import TimeHarmonicMaxwell
        
        return TimeHarmonicMaxwell(
            frequency=self.frequency,
            permittivity=self.material_properties.get('permittivity', 1.0),
            permeability=self.material_properties.get('permeability', 1.0),
            conductivity=self.material_properties.get('conductivity', 0.0)
        )
    
    def simulate_wave_propagation(self, source_config):
        """模拟波传播"""
        model_config = {
            'hidden_layers': [256, 256, 256, 256],
            'activation': 'sin',  # 对波动问题使用正弦激活
            'learning_rate': 1e-4,
            'complex_arithmetic': True  # 复数运算
        }
        
        model = OptimizedEWPINN(model_config)
        
        # 添加时谐麦克斯韦约束
        maxwell_constraint = self.setup_maxwell_constraints()
        model.add_constraints(maxwell_constraint)
        
        # 设置源条件
        source_conditions = self._setup_source_conditions(source_config)
        
        # 训练模型
        training_data = self.generate_wave_training_data()
        history = model.train(training_data, epochs=15000)
        
        return model, history
    
    def analyze_wave_characteristics(self, model):
        """分析波特性"""
        # 计算传播常数
        propagation_constant = self._calculate_propagation_constant(model)
        
        # 分析衰减特性
        attenuation_analysis = self._analyze_attenuation(model)
        
        # 计算能流密度
        power_flow = self._calculate_poynting_vector(model)
        
        # 分析驻波比
        vswr = self._calculate_vswr(model)
        
        return {
            'propagation_constant': propagation_constant,
            'attenuation': attenuation_analysis,
            'power_flow': power_flow,
            'vswr': vswr
        }
```

## 案例3：天线设计与优化

### 问题描述
设计并优化天线结构，分析辐射特性和阻抗匹配。

### 实现代码
```python
class AntennaDesignCase:
    """天线设计案例"""
    
    def __init__(self, frequency, antenna_type='dipole'):
        self.frequency = frequency
        self.wavelength = 3e8 / frequency  # 自由空间波长
        self.antenna_type = antenna_type
    
    def design_antenna_structure(self):
        """设计天线结构"""
        if self.antenna_type == 'dipole':
            return self._design_dipole_antenna()
        elif self.antenna_type == 'patch':
            return self._design_patch_antenna()
        elif self.antenna_type == 'yagi':
            return self._design_yagi_antenna()
        else:
            raise ValueError(f"不支持的天线类型: {self.antenna_type}")
    
    def _design_dipole_antenna(self):
        """设计偶极天线"""
        # 偶极天线长度约为半波长
        length = 0.48 * self.wavelength  # 考虑末端效应
        radius = 0.001 * self.wavelength  # 导线半径
        
        antenna_config = {
            'type': 'dipole',
            'length': length,
            'radius': radius,
            'feed_point': 'center'
        }
        
        return antenna_config
    
    def analyze_radiation_pattern(self, model):
        """分析辐射方向图"""
        # 在球坐标系中采样
        theta = torch.linspace(0, np.pi, 180)  # 俯仰角
        phi = torch.linspace(0, 2*np.pi, 360)  # 方位角
        
        radiation_pattern = {}
        
        for freq_component in ['E_theta', 'E_phi']:
            pattern = self._calculate_field_pattern(model, theta, phi, freq_component)
            radiation_pattern[freq_component] = pattern
        
        # 计算方向性系数
        directivity = self._calculate_directivity(radiation_pattern)
        
        # 计算增益
        gain = self._calculate_gain(radiation_pattern)
        
        return {
            'radiation_pattern': radiation_pattern,
            'directivity': directivity,
            'gain': gain
        }
    
    def optimize_antenna_performance(self, initial_design):
        """优化天线性能"""
        from efd_pinns.optimization import AntennaOptimizer
        
        optimizer = AntennaOptimizer(
            initial_design=initial_design,
            objectives=['maximize_gain', 'minimize_vswr', 'control_beamwidth'],
            constraints=['size_limits', 'bandwidth_requirements']
        )
        
        optimized_design = optimizer.optimize()
        
        return optimized_design
```

## 案例4：电磁兼容分析

### 问题描述
分析电子设备间的电磁干扰，确保电磁兼容性。

### 实现代码
```python
class EMCAnalysisCase:
    """电磁兼容分析案例"""
    
    def __init__(self, device_configurations):
        self.devices = device_configurations
        self.frequency_range = self._determine_frequency_range()
    
    def analyze_crosstalk(self):
        """分析串扰"""
        crosstalk_results = {}
        
        for source_device, victim_device in self._get_device_pairs():
            # 计算耦合系数
            coupling_coefficient = self._calculate_coupling(
                source_device, victim_device
            )
            
            # 分析干扰水平
            interference_level = self._assess_interference(
                source_device, victim_device, coupling_coefficient
            )
            
            crosstalk_results[f'{source_device}_to_{victim_device}'] = {
                'coupling': coupling_coefficient,
                'interference': interference_level,
                'compliance': self._check_emc_compliance(interference_level)
            }
        
        return crosstalk_results
    
    def simulate_shielding_effectiveness(self, shield_config):
        """模拟屏蔽效能"""
        model_config = {
            'hidden_layers': [128, 128, 128],
            'activation': 'tanh',
            'learning_rate': 1e-4
        }
        
        model = OptimizedEWPINN(model_config)
        
        # 添加屏蔽约束
        shielding_constraint = self._setup_shielding_constraints(shield_config)
        model.add_constraints(shielding_constraint)
        
        # 训练模型
        training_data = self.generate_emc_training_data()
        history = model.train(training_data, epochs=10000)
        
        # 计算屏蔽效能
        shielding_effectiveness = self._calculate_se(model)
        
        return shielding_effectiveness
    
    def optimize_emc_design(self):
        """优化EMC设计"""
        from efd_pinns.optimization import EMCOptimizer
        
        optimizer = EMCOptimizer(
            device_configurations=self.devices,
            emc_standards=['FCC', 'CE', 'CISPR'],
            optimization_objectives=['minimize_emissions', 'maximize_immunity']
        )
        
        optimized_layout = optimizer.optimize_layout()
        optimized_shielding = optimizer.optimize_shielding()
        
        return {
            'layout': optimized_layout,
            'shielding': optimized_shielding,
            'compliance_report': optimizer.generate_compliance_report()
        }
```

## 案例5：光子晶体与超材料

### 问题描述
设计光子晶体和超材料结构，控制电磁波传播。

### 实现代码
```python
class PhotonicCrystalCase:
    """光子晶体案例"""
    
    def __init__(self, lattice_type, unit_cell_config):
        self.lattice_type = lattice_type  # square, triangular, etc.
        self.unit_cell = unit_cell_config
        self.band_structure = None
    
    def calculate_band_structure(self):
        """计算能带结构"""
        model_config = {
            'hidden_layers': [256, 256, 256, 256],
            'activation': 'sin',
            'learning_rate': 1e-4,
            'periodic_boundary': True
        }
        
        model = OptimizedEWPINN(model_config)
        
        # 添加周期性边界条件
        periodic_constraint = self._setup_periodic_constraints()
        model.add_constraints(periodic_constraint)
        
        # 计算布里渊区路径
        brillouin_zone_path = self._generate_brillouin_zone_path()
        
        band_structure = {}
        
        for wave_vector in brillouin_zone_path:
            # 求解本征值问题
            eigenvalues = self._solve_eigenproblem(model, wave_vector)
            band_structure[tuple(wave_vector)] = eigenvalues
        
        self.band_structure = band_structure
        return band_structure
    
    def analyze_photonic_band_gap(self):
        """分析光子带隙"""
        if self.band_structure is None:
            self.calculate_band_structure()
        
        # 寻找带隙
        band_gaps = self._find_band_gaps(self.band_structure)
        
        # 分析带隙特性
        gap_analysis = {
            'gap_frequencies': band_gaps,
            'gap_widths': self._calculate_gap_widths(band_gaps),
            'midgap_frequencies': self._calculate_midgap_frequencies(band_gaps)
        }
        
        return gap_analysis
    
    def design_metamaterial_structure(self, target_properties):
        """设计超材料结构"""
        from efd_pinns.metamaterials import MetamaterialDesigner
        
        designer = MetamaterialDesigner(
            target_permittivity=target_properties.get('permittivity'),
            target_permeability=target_properties.get('permeability'),
            frequency_range=target_properties.get('frequency_range')
        )
        
        optimized_structure = designer.optimize_structure()
        
        return optimized_structure
```

## 验证与基准测试

### 与解析解对比
```python
def validate_against_analytical_solutions(self, model):
    """与解析解对比验证"""
    
    # 简单几何的解析解
    analytical_solution = self._get_analytical_solution()
    
    # PINN数值解
    pinn_solution = model.predict()
    
    # 计算误差
    error_metrics = self._calculate_error_metrics(
        analytical_solution, pinn_solution
    )
    
    validation_result = {
        'analytical_solution': analytical_solution,
        'pinn_solution': pinn_solution,
        'error_metrics': error_metrics,
        'validation_status': 'PASS' if error_metrics['relative_error'] < 0.01 else 'FAIL'
    }
    
    return validation_result
```

## 性能优化建议

1. **频率选择**: 根据问题尺度选择合适的频率范围
2. **网格策略**: 对场强变化剧烈区域使用加密网格
3. **复数处理**: 对时谐问题使用复数表示
4. **对称性利用**: 利用几何对称性减少计算域

## 结论

本案例研究展示了EFD3D在电磁场模拟领域的广泛应用，从基础静电场到时谐场、天线设计、电磁兼容和光子晶体等高级应用。通过物理信息神经网络，我们能够在保证电磁场方程正确性的同时，高效解决传统数值方法难以处理的复杂电磁问题。