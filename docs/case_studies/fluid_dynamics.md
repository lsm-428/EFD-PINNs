# 流体动力学案例研究

## 概述

本案例研究展示EFD3D在流体动力学领域的应用，包括层流、湍流、多相流等复杂流动问题的模拟与优化。

## 案例1：圆柱绕流模拟

### 问题描述
模拟流体绕圆柱流动，分析尾流结构和阻力特性。

### 物理模型
```python
import torch
import numpy as np
from efd_pinns import OptimizedEWPINN
from efd_pinns.constraints import NavierStokesConstraint

class CylinderFlowCase:
    """圆柱绕流案例"""
    
    def __init__(self, reynolds_number=100, cylinder_radius=0.05):
        self.reynolds_number = reynolds_number
        self.cylinder_radius = cylinder_radius
        self.domain = {
            'x_min': -2.0, 'x_max': 8.0,
            'y_min': -2.0, 'y_max': 2.0
        }
    
    def generate_training_data(self, n_points=5000):
        """生成训练数据"""
        # 边界点
        boundary_points = self._generate_boundary_points(n_points//5)
        
        # 圆柱表面点
        cylinder_points = self._generate_cylinder_surface_points(n_points//5)
        
        # 内部点
        interior_points = self._generate_interior_points(n_points//2)
        
        # 入口边界条件
        inlet_points = self._generate_inlet_points(n_points//10)
        
        return {
            'boundary': boundary_points,
            'cylinder': cylinder_points,
            'interior': interior_points,
            'inlet': inlet_points
        }
    
    def setup_navier_stokes_constraint(self):
        """设置Navier-Stokes方程约束"""
        return NavierStokesConstraint(
            reynolds_number=self.reynolds_number,
            density=1.0,
            viscosity=1.0/self.reynolds_number
        )
    
    def train_model(self, epochs=20000):
        """训练模型"""
        model_config = {
            'hidden_layers': [128, 128, 128, 128],
            'activation': 'tanh',
            'learning_rate': 1e-3,
            'adaptive_weights': True
        }
        
        model = OptimizedEWPINN(model_config)
        
        # 添加物理约束
        ns_constraint = self.setup_navier_stokes_constraint()
        model.add_constraints(ns_constraint)
        
        # 生成训练数据
        training_data = self.generate_training_data()
        
        # 训练
        history = model.train(training_data, epochs=epochs)
        
        return model, history
    
    def analyze_results(self, model):
        """分析结果"""
        # 计算阻力系数
        drag_coefficient = self._calculate_drag_coefficient(model)
        
        # 分析尾流结构
        wake_analysis = self._analyze_wake_structure(model)
        
        # 可视化流场
        flow_field = self._visualize_flow_field(model)
        
        return {
            'drag_coefficient': drag_coefficient,
            'wake_analysis': wake_analysis,
            'flow_field': flow_field
        }
```

### 结果分析

#### 阻力系数计算
```python
def _calculate_drag_coefficient(self, model):
    """计算圆柱阻力系数"""
    # 在圆柱表面采样点
    theta = torch.linspace(0, 2*np.pi, 100)
    x_surface = self.cylinder_radius * torch.cos(theta)
    y_surface = self.cylinder_radius * torch.sin(theta)
    
    # 计算表面压力
    points = torch.stack([x_surface, y_surface], dim=1)
    pressure = model.predict_pressure(points)
    
    # 计算压力阻力
    pressure_drag = self._integrate_pressure_force(pressure, theta)
    
    # 计算摩擦阻力
    friction_drag = self._calculate_friction_drag(model, theta)
    
    total_drag = pressure_drag + friction_drag
    drag_coefficient = 2 * total_drag / (self.reynolds_number * self.cylinder_radius)
    
    return drag_coefficient
```

## 案例2：湍流通道流动

### 问题描述
模拟湍流在通道中的发展过程，分析湍流统计特性。

### 实现代码
```python
class TurbulentChannelFlow:
    """湍流通道流动案例"""
    
    def __init__(self, reynolds_tau=180, channel_height=2.0):
        self.reynolds_tau = reynolds_tau  # 基于摩擦速度的雷诺数
        self.channel_height = channel_height
        self.domain = {
            'x_min': 0.0, 'x_max': 4*np.pi,
            'y_min': -1.0, 'y_max': 1.0,
            'z_min': 0.0, 'z_max': 2*np.pi
        }
    
    def setup_turbulence_model(self):
        """设置湍流模型"""
        from efd_pinns.turbulence import RANSConstraints
        
        # RANS方程约束
        rans_constraints = RANSConstraints(
            reynolds_number=self.reynolds_tau,
            turbulence_model='k-epsilon'
        )
        
        return rans_constraints
    
    def generate_turbulent_inflow(self, n_samples=1000):
        """生成湍流入口条件"""
        # 使用合成湍流方法生成入口条件
        inflow_data = self._synthetic_eddy_method(n_samples)
        
        return inflow_data
    
    def analyze_turbulence_statistics(self, model):
        """分析湍流统计特性"""
        # 平均速度剖面
        mean_velocity = self._calculate_mean_velocity(model)
        
        # 雷诺应力
        reynolds_stresses = self._calculate_reynolds_stresses(model)
        
        # 湍流动能
        turbulent_kinetic_energy = self._calculate_tke(model)
        
        # 湍流强度
        turbulence_intensity = self._calculate_turbulence_intensity(model)
        
        return {
            'mean_velocity': mean_velocity,
            'reynolds_stresses': reynolds_stresses,
            'tke': turbulent_kinetic_energy,
            'turbulence_intensity': turbulence_intensity
        }
```

## 案例3：多相流模拟

### 问题描述
模拟气液两相流动，分析界面演化和相分布。

### 实现代码
```python
class MultiphaseFlowCase:
    """多相流案例"""
    
    def __init__(self, density_ratio=1000, viscosity_ratio=100):
        self.density_ratio = density_ratio  # 液气密度比
        self.viscosity_ratio = viscosity_ratio  # 液气粘度比
    
    def setup_multiphase_constraints(self):
        """设置多相流约束"""
        from efd_pinns.multiphase import (
            TwoPhaseNavierStokes,
            InterfaceTrackingConstraint
        )
        
        # 两相Navier-Stokes方程
        multiphase_ns = TwoPhaseNavierStokes(
            density_ratio=self.density_ratio,
            viscosity_ratio=self.viscosity_ratio,
            surface_tension=0.072  # 水-空气表面张力 N/m
        )
        
        # 界面追踪约束
        interface_tracking = InterfaceTrackingConstraint(
            interface_sharpness=0.01
        )
        
        return [multiphase_ns, interface_tracking]
    
    def simulate_droplet_impact(self, droplet_radius=0.1, impact_velocity=1.0):
        """模拟液滴撞击"""
        # 初始条件：球形液滴
        initial_condition = self._create_droplet_initial_condition(
            droplet_radius, impact_velocity
        )
        
        # 配置模型
        model_config = {
            'hidden_layers': [256, 256, 256, 256],
            'activation': 'tanh',
            'learning_rate': 1e-4,
            'time_dependent': True
        }
        
        model = OptimizedEWPINN(model_config)
        
        # 添加多相流约束
        constraints = self.setup_multiphase_constraints()
        for constraint in constraints:
            model.add_constraints(constraint)
        
        # 时间相关训练
        time_history = model.train_time_dependent(
            initial_condition,
            time_steps=100,
            total_time=1.0
        )
        
        return model, time_history
    
    def analyze_interface_dynamics(self, model):
        """分析界面动力学"""
        # 界面形状演化
        interface_evolution = self._track_interface_evolution(model)
        
        # 接触线动力学
        contact_line_dynamics = self._analyze_contact_line(model)
        
        # 飞溅特性
        splashing_characteristics = self._analyze_splashing(model)
        
        return {
            'interface_evolution': interface_evolution,
            'contact_line': contact_line_dynamics,
            'splashing': splashing_characteristics
        }
```

## 高级应用：流动控制优化

### 主动流动控制
```python
class ActiveFlowControl:
    """主动流动控制案例"""
    
    def __init__(self, base_flow_case):
        self.base_case = base_flow_case
    
    def design_control_strategy(self, objective='drag_reduction'):
        """设计控制策略"""
        from efd_pinns.control import FlowController
        
        controller = FlowController(
            base_flow=self.base_case,
            control_objective=objective,
            control_method='adjoint_based'
        )
        
        # 优化控制参数
        optimized_control = controller.optimize_control_parameters()
        
        return optimized_control
    
    def evaluate_control_performance(self, controlled_flow):
        """评估控制性能"""
        # 比较控制前后的流动特性
        baseline_metrics = self.base_case.analyze_results()
        controlled_metrics = controlled_flow.analyze_results()
        
        performance_improvement = {}
        for metric in ['drag_coefficient', 'lift_coefficient', 'separation_point']:
            if metric in baseline_metrics and metric in controlled_metrics:
                improvement = (baseline_metrics[metric] - controlled_metrics[metric]) / baseline_metrics[metric]
                performance_improvement[metric] = improvement
        
        return performance_improvement
```

## 验证与基准测试

### 与经典结果对比
```python
def validate_against_classical_results(self, model):
    """与经典结果对比验证"""
    
    # 层流圆柱绕流经典结果
    classical_drag = self._get_classical_drag_coefficient(self.reynolds_number)
    
    # PINN计算结果
    pinn_drag = self._calculate_drag_coefficient(model)
    
    # 计算误差
    error = abs(pinn_drag - classical_drag) / classical_drag
    
    validation_result = {
        'classical_value': classical_drag,
        'pinn_value': pinn_drag,
        'relative_error': error,
        'validation_status': 'PASS' if error < 0.05 else 'FAIL'
    }
    
    return validation_result
```

## 性能优化建议

1. **网格策略**: 对于边界层区域使用加密网格
2. **权重调整**: 对物理约束使用自适应权重
3. **并行计算**: 对大尺度问题使用分布式训练
4. **混合精度**: 使用FP16加速训练过程

## 结论

本案例研究展示了EFD3D在流体动力学领域的强大能力，能够准确模拟从简单层流到复杂湍流和多相流的各种流动问题。通过物理信息神经网络的结合，我们能够在保证物理正确性的同时，高效解决传统数值方法难以处理的复杂问题。