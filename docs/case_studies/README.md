# 案例研究

## 概述

本目录包含EFD3D在不同工程和科学领域的应用案例研究，展示物理信息神经网络在解决复杂多物理场问题中的强大能力。

## 案例研究列表

### 1. 流体力学案例
- **文件**: `fluid_dynamics.md`
- **内容**: 圆柱绕流模拟、湍流通道流动、多相流模拟
- **应用领域**: 航空航天、汽车工程、环境流体力学

### 2. 电磁场案例  
- **文件**: `electromagnetics.md`
- **内容**: 静电场分析、时谐电磁场、天线设计与优化
- **应用领域**: 电子设备、通信系统、电磁兼容

### 3. 热传导与结构力学案例
- **文件**: `thermal_structural.md`
- **内容**: 稳态/瞬态热传导、热应力分析、结构优化设计、疲劳寿命预测、多物理场耦合
- **应用领域**: 电子散热、能源系统、机械设计、航空航天、复杂系统工程

### 4. 材料科学与生物力学案例
- **文件**: `materials_biomechanics.md`
- **内容**: 材料特性预测、复合材料性能、生物组织力学、医疗器械设计、组织工程支架、生物材料界面分析、个性化医疗设备
- **应用领域**: 新材料开发、生物医学工程、个性化医疗、医疗器械优化

### 5. 快速开始示例
- **文件**: `quick_examples.md`
- **内容**: 基础流体模拟、电磁场模拟、热传导分析
- **应用领域**: 初学者入门、快速原型验证

### 6. 高级应用示例
- **文件**: `advanced_applications.md`
- **内容**: 多物理场耦合、复杂几何处理、大规模并行计算
- **应用领域**: 科研项目、工业级应用开发

### 7. 研究案例
- **文件**: `research_cases.md`
- **内容**: 新型材料特性预测与发现、生物力学前沿研究、量子计算与量子系统模拟、环境科学与气候变化研究
- **应用领域**: 学术研究、前沿技术探索

### 8. 高级AI技术应用
- **文件**: `advanced_ai_applications.md`
- **内容**: 强化学习集成、生成对抗网络、元学习、联邦学习
- **应用领域**: 人工智能、机器学习、智能控制、隐私保护

### 9. 工业应用案例
- **文件**: `industrial_applications.md`
- **内容**: 航空航天工程、汽车工程、能源系统、智能制造
- **应用领域**: 工业制造、工程设计、能源管理、生产优化

## 快速开始示例

### 基础流体模拟示例
```python
import torch
import numpy as np
from efd_pinns import OptimizedEWPINN
from efd_pinns.constraints import NavierStokesConstraint

# 创建简单的2D流体模拟案例
class SimpleFlowCase:
    def __init__(self):
        self.domain = {
            'x_min': 0.0, 'x_max': 1.0,
            'y_min': 0.0, 'y_max': 1.0
        }
        
    def generate_training_data(self, n_points=1000):
        """生成训练数据"""
        # 边界条件数据
        boundary_points = self._generate_boundary_points(n_points//4)
        
        # 内部点数据
        interior_points = self._generate_interior_points(n_points//2)
        
        # 初始条件数据
        initial_points = self._generate_initial_points(n_points//4)
        
        return {
            'boundary': boundary_points,
            'interior': interior_points,
            'initial': initial_points
        }
    
    def setup_model(self):
        """设置PINN模型"""
        config = {
            'hidden_layers': [64, 64, 64, 64],
            'activation': 'tanh',
            'learning_rate': 1e-3,
            'constraints': [
                NavierStokesConstraint(reynolds_number=100)
            ]
        }
        
        return OptimizedEWPINN(config)
```

### 电磁场模拟示例
```python
class ElectromagneticCase:
    """电磁场模拟案例"""
    
    def __init__(self, frequency=1e9):  # 1GHz
        self.frequency = frequency
        self.wavelength = 3e8 / frequency  # 光速/频率
        
    def setup_maxwell_constraints(self):
        """设置麦克斯韦方程约束"""
        from efd_pinns.constraints import MaxwellConstraints
        
        constraints = MaxwellConstraints(
            frequency=self.frequency,
            permittivity=1.0,  # 相对介电常数
            permeability=1.0   # 相对磁导率
        )
        
        return constraints
    
    def simulate_wave_propagation(self, domain_size=2.0):
        """模拟波传播"""
        # 创建点源
        source_position = [domain_size/2, domain_size/2]
        
        # 生成训练数据
        training_data = self._generate_wave_data(
            source_position, 
            domain_size
        )
        
        # 配置模型
        model_config = {
            'hidden_layers': [128, 128, 128, 128],
            'activation': 'sin',  # 对波动问题使用正弦激活
            'learning_rate': 1e-4
        }
        
        model = OptimizedEWPINN(model_config)
        
        # 添加约束
        constraints = self.setup_maxwell_constraints()
        model.add_constraints(constraints)
        
        return model, training_data
```

## 高级应用示例

### 多物理场耦合案例
```python
class MultiPhysicsCase:
    """多物理场耦合案例"""
    
    def __init__(self):
        self.physics_fields = ['fluid', 'thermal', 'structural']
        
    def setup_coupled_simulation(self):
        """设置耦合模拟"""
        from efd_pinns.constraints import (
            NavierStokesConstraint,
            HeatConductionConstraint,
            ElasticityConstraint
        )
        
        # 流体约束
        fluid_constraint = NavierStokesConstraint(reynolds_number=100)
        
        # 热传导约束
        thermal_constraint = HeatConductionConstraint(
            thermal_diffusivity=1e-5
        )
        
        # 结构约束
        structural_constraint = ElasticityConstraint(
            youngs_modulus=2e11,  # 钢的杨氏模量
            poissons_ratio=0.3
        )
        
        # 耦合约束（热膨胀）
        coupling_constraints = self._setup_thermal_expansion_coupling()
        
        return {
            'fluid': fluid_constraint,
            'thermal': thermal_constraint,
            'structural': structural_constraint,
            'coupling': coupling_constraints
        }
    
    def train_coupled_model(self, epochs=10000):
        """训练耦合模型"""
        model_config = {
            'hidden_layers': [256, 256, 256, 256],
            'activation': 'tanh',
            'learning_rate': 1e-4,
            'multi_physics': True
        }
        
        model = OptimizedEWPINN(model_config)
        
        # 添加所有约束
        constraints = self.setup_coupled_simulation()
        for constraint in constraints.values():
            model.add_constraints(constraint)
        
        # 生成训练数据
        training_data = self._generate_coupled_data()
        
        # 训练模型
        history = model.train(
            training_data, 
            epochs=epochs,
            validation_data=self._generate_validation_data()
        )
        
        return model, history
```

### 工业应用案例

#### 汽车空气动力学优化
```python
class AutomotiveAerodynamics:
    """汽车空气动力学优化案例"""
    
    def __init__(self, vehicle_geometry):
        self.geometry = vehicle_geometry
        self.reynolds_number = 1e6  # 典型汽车行驶雷诺数
        
    def optimize_drag_coefficient(self):
        """优化阻力系数"""
        from efd_pinns.optimization import ShapeOptimizer
        
        # 创建形状优化器
        optimizer = ShapeOptimizer(
            base_geometry=self.geometry,
            objective='minimize_drag',
            constraints=['volume_constraint', 'manufacturing_constraints']
        )
        
        # 设置流体约束
        fluid_constraint = NavierStokesConstraint(
            reynolds_number=self.reynolds_number
        )
        
        # 运行优化
        optimized_geometry, performance_metrics = optimizer.optimize(
            max_iterations=100,
            fluid_constraint=fluid_constraint
        )
        
        return optimized_geometry, performance_metrics
```

#### 电子设备散热设计
```python
class ElectronicsCooling:
    """电子设备散热设计案例"""
    
    def __init__(self, chip_layout, power_density):
        self.chip_layout = chip_layout
        self.power_density = power_density
        
    def analyze_thermal_performance(self):
        """分析热性能"""
        from efd_pinns.constraints import (
            HeatConductionConstraint,
            ConvectionConstraint
        )
        
        # 热传导约束
        conduction_constraint = HeatConductionConstraint(
            thermal_conductivity=400  # 铜的热导率 W/m·K
        )
        
        # 对流约束
        convection_constraint = ConvectionConstraint(
            heat_transfer_coefficient=25  # 强制对流系数 W/m²·K
        )
        
        # 配置模型
        model_config = {
            'hidden_layers': [128, 128, 128],
            'activation': 'tanh',
            'learning_rate': 1e-4
        }
        
        model = OptimizedEWPINN(model_config)
        model.add_constraints(conduction_constraint)
        model.add_constraints(convection_constraint)
        
        # 热源设置
        heat_sources = self._setup_heat_sources()
        
        return model, heat_sources
    
    def optimize_cooling_system(self):
        """优化冷却系统"""
        # 分析当前热性能
        baseline_temperature = self._analyze_baseline_temperature()
        
        # 优化散热器设计
        optimized_heatsink = self._optimize_heatsink_design()
        
        # 优化风扇布局
        optimized_fan_layout = self._optimize_fan_layout()
        
        return {
            'baseline_temperature': baseline_temperature,
            'optimized_heatsink': optimized_heatsink,
            'optimized_fan_layout': optimized_fan_layout,
            'performance_improvement': self._calculate_improvement()
        }
```

## 研究案例

### 学术研究应用

#### 新型材料特性预测
```python
class MaterialPropertyPrediction:
    """材料特性预测案例"""
    
    def __init__(self, material_composition):
        self.composition = material_composition
        
    def predict_mechanical_properties(self):
        """预测力学性能"""
        from efd_pinns.materials import MaterialPropertyPredictor
        
        predictor = MaterialPropertyPredictor()
        
        # 预测弹性模量
        youngs_modulus = predictor.predict_youngs_modulus(
            self.composition
        )
        
        # 预测屈服强度
        yield_strength = predictor.predict_yield_strength(
            self.composition
        )
        
        # 预测断裂韧性
        fracture_toughness = predictor.predict_fracture_toughness(
            self.composition
        )
        
        return {
            'youngs_modulus': youngs_modulus,
            'yield_strength': yield_strength,
            'fracture_toughness': fracture_toughness
        }
```

#### 生物力学模拟
```python
class BiomechanicsSimulation:
    """生物力学模拟案例"""
    
    def __init__(self, tissue_properties):
        self.tissue_properties = tissue_properties
        
    def simulate_blood_flow(self):
        """模拟血流"""
        from efd_pinns.biomechanics import BloodFlowSimulator
        
        simulator = BloodFlowSimulator()
        
        # 设置血管几何
        vessel_geometry = self._create_vessel_geometry()
        
        # 设置血流参数
        flow_parameters = self._setup_flow_parameters()
        
        # 运行模拟
        flow_field, wall_shear_stress = simulator.simulate(
            vessel_geometry,
            flow_parameters
        )
        
        return flow_field, wall_shear_stress
    
    def analyze_bone_stress(self):
        """分析骨骼应力"""
        from efd_pinns.constraints import OrthotropicElasticityConstraint
        
        # 正交各向异性弹性约束
        bone_constraint = OrthotropicElasticityConstraint(
            youngs_modulus=[15e9, 15e9, 20e9],  # 骨骼的杨氏模量 (GPa)
            shear_modulus=[5e9, 5e9, 5e9],
            poissons_ratio=[0.3, 0.3, 0.3]
        )
        
        return bone_constraint
```

## 使用指南

### 运行案例研究

1. **选择案例**: 根据你的研究领域选择合适的案例
2. **准备数据**: 按照案例要求准备输入数据
3. **配置模型**: 调整模型参数以适应具体问题
4. **运行模拟**: 执行训练和模拟过程
5. **分析结果**: 使用提供的可视化工具分析结果

### 自定义案例

要创建自定义案例：

1. 继承基础案例类
2. 实现特定的物理约束
3. 提供自定义的数据生成方法
4. 配置适合的模型参数

### 性能优化建议

- 对于简单问题，使用较小的网络架构
- 对于复杂多物理场问题，使用深层网络
- 根据问题特性选择合适的激活函数
- 使用混合精度训练加速计算

## 贡献指南

欢迎贡献新的案例研究！请遵循以下步骤：

1. 在相应的子目录中创建案例文件
2. 提供完整的代码示例和文档
3. 包含测试用例验证正确性
4. 提交Pull Request进行审核

通过研究这些案例，你可以更好地理解如何将EFD3D应用于实际工程和科学研究问题。