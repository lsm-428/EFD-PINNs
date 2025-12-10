"""
EWP Flow Field Solver Property-Based Tests

使用 Hypothesis 进行属性测试，验证流场求解器的正确性属性。

测试框架: pytest + hypothesis
配置: 每个属性测试运行 100 次迭代

Author: EFD-PINNs Team
Date: 2025-12-03
"""

import pytest
import numpy as np
from hypothesis import given, settings, assume, strategies as st
from hypothesis.extra.numpy import arrays

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.solvers.flow_solver import (
    MeshGenerator, Mesh, InterfaceTracker, ContactLineHandler,
    create_initial_conditions, compute_aperture_ratio_from_phi,
    DOMAIN_PARAMETERS, FLUID_PROPERTIES,
    MeshGenerationError
)


# ============================================================
# 测试策略定义
# ============================================================

# 网格尺寸策略（合理范围内的正整数）
mesh_size_strategy = st.integers(min_value=4, max_value=64)

# 电压策略
voltage_strategy = st.floats(min_value=0.0, max_value=40.0, allow_nan=False, allow_infinity=False)

# 时间策略
time_strategy = st.floats(min_value=0.0, max_value=0.1, allow_nan=False, allow_infinity=False)

# 体积分数策略（0-1 范围）
volume_fraction_strategy = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


# ============================================================
# Property 3: 输出场完整性
# **Feature: ewp-flow-field-solver, Property 3: 输出场完整性**
# **Validates: Requirements 1.4**
# ============================================================

class TestMeshGeneratorProperties:
    """MeshGenerator 属性测试"""
    
    @given(
        nx=mesh_size_strategy,
        ny=mesh_size_strategy,
        nz=mesh_size_strategy
    )
    @settings(max_examples=100, deadline=None)
    def test_mesh_dimensions_match_request(self, nx: int, ny: int, nz: int):
        """
        Property 3: 输出场完整性 - 网格尺寸正确性
        
        *For any* requested mesh dimensions (nx, ny, nz), the generated mesh
        SHALL have exactly those dimensions.
        
        **Feature: ewp-flow-field-solver, Property 3: 输出场完整性**
        **Validates: Requirements 1.4**
        """
        generator = MeshGenerator()
        mesh = generator.generate_structured_mesh(nx=nx, ny=ny, nz=nz)
        
        # 验证网格数量
        assert mesh.nx == nx, f"Expected nx={nx}, got {mesh.nx}"
        assert mesh.ny == ny, f"Expected ny={ny}, got {mesh.ny}"
        assert mesh.nz == nz, f"Expected nz={nz}, got {mesh.nz}"
        
        # 验证节点数组尺寸
        assert len(mesh.x) == nx + 1, f"Expected {nx+1} x nodes, got {len(mesh.x)}"
        assert len(mesh.y) == ny + 1, f"Expected {ny+1} y nodes, got {len(mesh.y)}"
        assert len(mesh.z) == nz + 1, f"Expected {nz+1} z nodes, got {len(mesh.z)}"
        
        # 验证单元中心数组尺寸
        assert len(mesh.xc) == nx, f"Expected {nx} xc values, got {len(mesh.xc)}"
        assert len(mesh.yc) == ny, f"Expected {ny} yc values, got {len(mesh.yc)}"
        assert len(mesh.zc) == nz, f"Expected {nz} zc values, got {len(mesh.zc)}"
    
    @given(
        nx=mesh_size_strategy,
        ny=mesh_size_strategy,
        nz=mesh_size_strategy
    )
    @settings(max_examples=100, deadline=None)
    def test_mesh_total_cells_consistent(self, nx: int, ny: int, nz: int):
        """
        Property 3: 输出场完整性 - 总单元数一致性
        
        *For any* mesh, the total_cells property SHALL equal nx * ny * nz.
        
        **Feature: ewp-flow-field-solver, Property 3: 输出场完整性**
        **Validates: Requirements 1.4**
        """
        generator = MeshGenerator()
        mesh = generator.generate_structured_mesh(nx=nx, ny=ny, nz=nz)
        
        expected_total = nx * ny * nz
        assert mesh.total_cells == expected_total, \
            f"Expected total_cells={expected_total}, got {mesh.total_cells}"
    
    @given(
        nx=mesh_size_strategy,
        ny=mesh_size_strategy,
        nz=mesh_size_strategy
    )
    @settings(max_examples=100, deadline=None)
    def test_mesh_spacing_positive(self, nx: int, ny: int, nz: int):
        """
        Property 3: 输出场完整性 - 网格间距正值
        
        *For any* mesh, the grid spacing (dx, dy, dz) SHALL be positive.
        
        **Feature: ewp-flow-field-solver, Property 3: 输出场完整性**
        **Validates: Requirements 1.4**
        """
        generator = MeshGenerator()
        mesh = generator.generate_structured_mesh(nx=nx, ny=ny, nz=nz)
        
        assert mesh.dx > 0, f"dx must be positive, got {mesh.dx}"
        assert mesh.dy > 0, f"dy must be positive, got {mesh.dy}"
        assert mesh.dz > 0, f"dz must be positive, got {mesh.dz}"
    
    @given(
        nx=mesh_size_strategy,
        ny=mesh_size_strategy,
        nz=mesh_size_strategy
    )
    @settings(max_examples=100, deadline=None)
    def test_mesh_cell_centers_within_domain(self, nx: int, ny: int, nz: int):
        """
        Property 3: 输出场完整性 - 单元中心在域内
        
        *For any* mesh, all cell centers SHALL be within the domain boundaries.
        
        **Feature: ewp-flow-field-solver, Property 3: 输出场完整性**
        **Validates: Requirements 1.4**
        """
        generator = MeshGenerator()
        mesh = generator.generate_structured_mesh(nx=nx, ny=ny, nz=nz)
        
        # 单元中心应在节点范围内
        assert np.all(mesh.xc >= mesh.x[0]), "xc values below domain"
        assert np.all(mesh.xc <= mesh.x[-1]), "xc values above domain"
        assert np.all(mesh.yc >= mesh.y[0]), "yc values below domain"
        assert np.all(mesh.yc <= mesh.y[-1]), "yc values above domain"
        assert np.all(mesh.zc >= mesh.z[0]), "zc values below domain"
        assert np.all(mesh.zc <= mesh.z[-1]), "zc values above domain"
    
    @given(
        nx=mesh_size_strategy,
        ny=mesh_size_strategy,
        nz=mesh_size_strategy
    )
    @settings(max_examples=100, deadline=None)
    def test_mesh_boundary_cells_exist(self, nx: int, ny: int, nz: int):
        """
        Property 3: 输出场完整性 - 边界单元存在
        
        *For any* mesh, boundary cells SHALL be identified for all six faces.
        
        **Feature: ewp-flow-field-solver, Property 3: 输出场完整性**
        **Validates: Requirements 1.4**
        """
        generator = MeshGenerator()
        mesh = generator.generate_structured_mesh(nx=nx, ny=ny, nz=nz)
        
        # 检查所有边界都被识别
        required_boundaries = ['bottom', 'top', 'front', 'back', 'left', 'right', 'walls']
        for boundary in required_boundaries:
            assert boundary in mesh.boundary_cells, f"Missing boundary: {boundary}"
            assert len(mesh.boundary_cells[boundary]) > 0, f"Empty boundary: {boundary}"
    
    @given(
        nx=mesh_size_strategy,
        ny=mesh_size_strategy,
        nz=mesh_size_strategy
    )
    @settings(max_examples=100, deadline=None)
    def test_mesh_boundary_indices_valid(self, nx: int, ny: int, nz: int):
        """
        Property 3: 输出场完整性 - 边界索引有效
        
        *For any* mesh, all boundary cell indices SHALL be valid (within total_cells).
        
        **Feature: ewp-flow-field-solver, Property 3: 输出场完整性**
        **Validates: Requirements 1.4**
        """
        generator = MeshGenerator()
        mesh = generator.generate_structured_mesh(nx=nx, ny=ny, nz=nz)
        
        max_idx = mesh.total_cells - 1
        for boundary_name, indices in mesh.boundary_cells.items():
            assert np.all(indices >= 0), f"Negative index in {boundary_name}"
            assert np.all(indices <= max_idx), \
                f"Index out of range in {boundary_name}: max={indices.max()}, limit={max_idx}"


# ============================================================
# Property 5: 体积分数边界
# **Feature: ewp-flow-field-solver, Property 5: 体积分数边界**
# **Validates: Requirements 2.1**
# ============================================================

class TestInterfaceTrackerProperties:
    """InterfaceTracker 属性测试"""
    
    @given(
        nx=st.integers(min_value=8, max_value=32),
        ny=st.integers(min_value=8, max_value=32),
        nz=st.integers(min_value=4, max_value=16)
    )
    @settings(max_examples=100, deadline=None)
    def test_volume_fraction_bounds_after_advection(self, nx: int, ny: int, nz: int):
        """
        Property 5: 体积分数边界
        
        *For any* simulation, the volume fraction field φ SHALL be bounded 
        in [0, 1] at all grid points after advection.
        
        **Feature: ewp-flow-field-solver, Property 5: 体积分数边界**
        **Validates: Requirements 2.1**
        """
        # 创建网格
        generator = MeshGenerator()
        mesh = generator.generate_structured_mesh(nx=nx, ny=ny, nz=nz)
        
        # 创建初始条件
        initial = create_initial_conditions(mesh)
        phi = initial['phi']
        
        # 创建小的随机速度场
        u = np.random.uniform(-0.001, 0.001, (nx, ny, nz))
        v = np.random.uniform(-0.001, 0.001, (nx, ny, nz))
        w = np.random.uniform(-0.001, 0.001, (nx, ny, nz))
        
        # 初始化追踪器并执行平流
        tracker = InterfaceTracker(mesh)
        dt = 1e-6  # 小时间步
        phi_new = tracker.advect(phi, u, v, w, dt)
        
        # 验证边界
        assert np.all(phi_new >= 0.0), f"phi below 0: min={phi_new.min()}"
        assert np.all(phi_new <= 1.0), f"phi above 1: max={phi_new.max()}"
    
    @given(
        nx=st.integers(min_value=8, max_value=32),
        ny=st.integers(min_value=8, max_value=32),
        nz=st.integers(min_value=4, max_value=16)
    )
    @settings(max_examples=100, deadline=None)
    def test_clip_volume_fraction_bounds(self, nx: int, ny: int, nz: int):
        """
        Property 5: 体积分数边界 - clip 函数
        
        *For any* volume fraction field, clip_volume_fraction SHALL ensure
        all values are in [0, 1].
        
        **Feature: ewp-flow-field-solver, Property 5: 体积分数边界**
        **Validates: Requirements 2.1**
        """
        generator = MeshGenerator()
        mesh = generator.generate_structured_mesh(nx=nx, ny=ny, nz=nz)
        tracker = InterfaceTracker(mesh)
        
        # 创建可能越界的体积分数场
        phi = np.random.uniform(-0.5, 1.5, (nx, ny, nz))
        
        # 应用 clip
        phi_clipped = tracker.clip_volume_fraction(phi)
        
        # 验证边界
        assert np.all(phi_clipped >= 0.0), f"phi below 0 after clip"
        assert np.all(phi_clipped <= 1.0), f"phi above 1 after clip"


# ============================================================
# Property 6: 界面锐度
# **Feature: ewp-flow-field-solver, Property 6: 界面锐度**
# **Validates: Requirements 2.2**
# ============================================================

class TestInterfaceSharpnessProperties:
    """界面锐度属性测试"""
    
    @given(
        nx=st.integers(min_value=16, max_value=32),
        ny=st.integers(min_value=16, max_value=32),
        nz=st.integers(min_value=8, max_value=16)
    )
    @settings(max_examples=50, deadline=None)
    def test_initial_interface_sharpness(self, nx: int, ny: int, nz: int):
        """
        Property 6: 界面锐度
        
        *For any* initial condition, the interface thickness SHALL be 
        less than 3 grid cells.
        
        **Feature: ewp-flow-field-solver, Property 6: 界面锐度**
        **Validates: Requirements 2.2**
        """
        generator = MeshGenerator()
        mesh = generator.generate_structured_mesh(nx=nx, ny=ny, nz=nz)
        
        # 创建初始条件
        initial = create_initial_conditions(mesh)
        phi = initial['phi']
        
        # 计算界面厚度
        tracker = InterfaceTracker(mesh)
        thickness = tracker.compute_interface_thickness(phi)
        
        # 界面厚度应小于 3 个网格单元
        assert thickness < 3.0, f"Interface too thick: {thickness} cells"


# ============================================================
# Property 7: 接触角一致性
# **Feature: ewp-flow-field-solver, Property 7: 接触角一致性**
# **Validates: Requirements 3.1**
# ============================================================

class TestContactLineHandlerProperties:
    """ContactLineHandler 属性测试"""
    
    @given(
        voltage=voltage_strategy,
        time=time_strategy
    )
    @settings(max_examples=100, deadline=None)
    def test_contact_angle_consistency_with_predictor(self, voltage: float, time: float):
        """
        Property 7: 接触角一致性
        
        *For any* voltage V and time t, the contact angle from ContactLineHandler
        SHALL match the value from HybridPredictor within 1°.
        
        **Feature: ewp-flow-field-solver, Property 7: 接触角一致性**
        **Validates: Requirements 3.1**
        """
        handler = ContactLineHandler()
        
        # 从 handler 获取接触角
        theta_handler = handler.get_dynamic_contact_angle(voltage, time)
        
        # 直接从 predictor 获取
        theta_predictor = handler.predictor.predict(voltage, time)
        
        # 验证一致性（1° 容差）
        diff = abs(theta_handler - theta_predictor)
        assert diff <= 1.0, f"Contact angle mismatch: handler={theta_handler}, predictor={theta_predictor}"
    
    @given(voltage=voltage_strategy)
    @settings(max_examples=100, deadline=None)
    def test_equilibrium_contact_angle_range(self, voltage: float):
        """
        Property 7: 接触角一致性 - 平衡角范围
        
        *For any* voltage, the equilibrium contact angle SHALL be in 
        a physically reasonable range (0° to 180°).
        
        **Feature: ewp-flow-field-solver, Property 7: 接触角一致性**
        **Validates: Requirements 3.1**
        """
        handler = ContactLineHandler()
        theta = handler.get_equilibrium_contact_angle(voltage)
        
        assert 0 <= theta <= 180, f"Contact angle out of range: {theta}°"


# ============================================================
# Property 8: 滑移模型有限性
# **Feature: ewp-flow-field-solver, Property 8: 滑移模型有限性**
# **Validates: Requirements 3.3**
# ============================================================

class TestSlipModelProperties:
    """滑移模型属性测试"""
    
    @given(
        nx=st.integers(min_value=8, max_value=16),
        ny=st.integers(min_value=8, max_value=16),
        nz=st.integers(min_value=4, max_value=8)
    )
    @settings(max_examples=50, deadline=None)
    def test_slip_model_finite_velocity(self, nx: int, ny: int, nz: int):
        """
        Property 8: 滑移模型有限性
        
        *For any* simulation, the velocity at the contact line SHALL be 
        finite after applying the slip model.
        
        **Feature: ewp-flow-field-solver, Property 8: 滑移模型有限性**
        **Validates: Requirements 3.3**
        """
        generator = MeshGenerator()
        mesh = generator.generate_structured_mesh(nx=nx, ny=ny, nz=nz)
        
        # 创建初始条件
        initial = create_initial_conditions(mesh)
        phi = initial['phi']
        
        # 创建随机速度场
        u = np.random.uniform(-0.1, 0.1, (nx, ny, nz))
        v = np.random.uniform(-0.1, 0.1, (nx, ny, nz))
        w = np.random.uniform(-0.1, 0.1, (nx, ny, nz))
        
        # 检测接触线单元
        handler = ContactLineHandler()
        contact_cells = handler.detect_contact_line_cells(phi, mesh)
        
        # 应用滑移模型
        u_new, v_new, w_new = handler.apply_slip_model(u, v, w, mesh, contact_cells)
        
        # 验证速度有限
        assert np.all(np.isfinite(u_new)), "u contains non-finite values"
        assert np.all(np.isfinite(v_new)), "v contains non-finite values"
        assert np.all(np.isfinite(w_new)), "w contains non-finite values"


# ============================================================
# 运行测试
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])


# ============================================================
# Property 1: 质量守恒
# **Feature: ewp-flow-field-solver, Property 1: 质量守恒**
# **Validates: Requirements 1.2**
# ============================================================

class TestMassConservationProperties:
    """质量守恒属性测试"""
    
    @given(
        nx=st.integers(min_value=8, max_value=16),
        ny=st.integers(min_value=8, max_value=16),
        nz=st.integers(min_value=4, max_value=8)
    )
    @settings(max_examples=20, deadline=None)
    def test_mass_conservation_initial(self, nx: int, ny: int, nz: int):
        """
        Property 1: 质量守恒 - 初始状态
        
        *For any* simulation, the initial mass error SHALL be zero.
        
        **Feature: ewp-flow-field-solver, Property 1: 质量守恒**
        **Validates: Requirements 1.2**
        """
        from src.solvers.flow_solver import FlowSolver
        
        generator = MeshGenerator()
        mesh = generator.generate_structured_mesh(nx=nx, ny=ny, nz=nz)
        
        solver = FlowSolver(mesh)
        initial = create_initial_conditions(mesh)
        solver.set_initial_conditions(initial['phi'])
        
        mass_error = solver.compute_mass_conservation_error()
        
        assert mass_error['total_error'] == 0.0, \
            f"Initial mass error should be 0, got {mass_error['total_error']}"
    
    @given(
        nx=st.integers(min_value=8, max_value=12),
        ny=st.integers(min_value=8, max_value=12),
        nz=st.integers(min_value=4, max_value=6)
    )
    @settings(max_examples=10, deadline=None)
    def test_mass_conservation_after_step(self, nx: int, ny: int, nz: int):
        """
        Property 1: 质量守恒 - 单步后
        
        *For any* simulation, the mass error after one time step SHALL be 
        within 0.1% tolerance.
        
        **Feature: ewp-flow-field-solver, Property 1: 质量守恒**
        **Validates: Requirements 1.2**
        """
        from src.solvers.flow_solver import FlowSolver
        
        generator = MeshGenerator()
        mesh = generator.generate_structured_mesh(nx=nx, ny=ny, nz=nz)
        
        solver = FlowSolver(mesh)
        initial = create_initial_conditions(mesh)
        solver.set_initial_conditions(initial['phi'])
        solver.set_boundary_conditions({'walls': 'no_slip'})
        
        # 执行一步
        dt = 1e-7  # 小时间步
        solver.solve_step(dt)
        
        mass_error = solver.compute_mass_conservation_error()
        
        # 质量误差应在 0.1% 以内
        assert mass_error['total_error'] < 0.001, \
            f"Mass error {mass_error['total_error']*100:.4f}% exceeds 0.1% tolerance"


# ============================================================
# Property 4: 无滑移边界条件
# **Feature: ewp-flow-field-solver, Property 4: 无滑移边界条件**
# **Validates: Requirements 1.5**
# ============================================================

class TestNoSlipBoundaryProperties:
    """无滑移边界条件属性测试"""
    
    @given(
        nx=st.integers(min_value=8, max_value=16),
        ny=st.integers(min_value=8, max_value=16),
        nz=st.integers(min_value=4, max_value=8)
    )
    @settings(max_examples=50, deadline=None)
    def test_no_slip_boundary_initial(self, nx: int, ny: int, nz: int):
        """
        Property 4: 无滑移边界条件 - 初始状态
        
        *For any* simulation with no-slip boundary conditions, the velocity 
        at solid walls SHALL be zero initially.
        
        **Feature: ewp-flow-field-solver, Property 4: 无滑移边界条件**
        **Validates: Requirements 1.5**
        """
        from src.solvers.flow_solver import FlowSolver
        
        generator = MeshGenerator()
        mesh = generator.generate_structured_mesh(nx=nx, ny=ny, nz=nz)
        
        solver = FlowSolver(mesh)
        initial = create_initial_conditions(mesh)
        solver.set_initial_conditions(initial['phi'])
        solver.set_boundary_conditions({'walls': 'no_slip'})
        
        wall_vel = solver.get_wall_velocity()
        
        # 所有壁面速度应为零
        for wall_name, vel in wall_vel.items():
            max_vel = np.max(vel)
            assert max_vel < 1e-10, \
                f"Wall '{wall_name}' has non-zero velocity: {max_vel}"
    
    @given(
        nx=st.integers(min_value=8, max_value=12),
        ny=st.integers(min_value=8, max_value=12),
        nz=st.integers(min_value=4, max_value=6)
    )
    @settings(max_examples=20, deadline=None)
    def test_no_slip_boundary_after_step(self, nx: int, ny: int, nz: int):
        """
        Property 4: 无滑移边界条件 - 单步后
        
        *For any* simulation with no-slip boundary conditions, the velocity 
        at solid walls SHALL remain zero after time stepping.
        
        **Feature: ewp-flow-field-solver, Property 4: 无滑移边界条件**
        **Validates: Requirements 1.5**
        """
        from src.solvers.flow_solver import FlowSolver
        
        generator = MeshGenerator()
        mesh = generator.generate_structured_mesh(nx=nx, ny=ny, nz=nz)
        
        solver = FlowSolver(mesh)
        initial = create_initial_conditions(mesh)
        solver.set_initial_conditions(initial['phi'])
        solver.set_boundary_conditions({'walls': 'no_slip'})
        
        # 执行一步
        dt = 1e-7
        solver.solve_step(dt)
        
        wall_vel = solver.get_wall_velocity()
        
        # 所有壁面速度应为零（数值容差）
        for wall_name, vel in wall_vel.items():
            max_vel = np.max(vel)
            assert max_vel < 1e-8, \
                f"Wall '{wall_name}' has non-zero velocity after step: {max_vel}"


# ============================================================
# Property 10: PINN 精度
# **Feature: ewp-flow-field-solver, Property 10: PINN 精度**
# **Validates: Requirements 5.3**
# ============================================================

class TestPINNSolverProperties:
    """PINNSolver 属性测试"""
    
    def test_pinn_network_output_shape(self):
        """
        Property 10: PINN 精度 - 输出形状
        
        *For any* PINN model, the output SHALL have 5 components (u, v, w, p, phi).
        
        **Feature: ewp-flow-field-solver, Property 10: PINN 精度**
        **Validates: Requirements 5.3**
        """
        from src.solvers.flow_solver import PINNSolver
        
        solver = PINNSolver()
        solver.build_network()
        
        # 测试预测
        x = np.random.rand(10, 3) * 1e-4
        result = solver.predict(x, t=0.001)
        
        assert 'u' in result, "Missing u in output"
        assert 'v' in result, "Missing v in output"
        assert 'w' in result, "Missing w in output"
        assert 'p' in result, "Missing p in output"
        assert 'phi' in result, "Missing phi in output"
        
        assert result['u'].shape == (10,), f"Wrong u shape: {result['u'].shape}"
        assert result['phi'].shape == (10,), f"Wrong phi shape: {result['phi'].shape}"
    
    def test_pinn_output_finite(self):
        """
        Property 10: PINN 精度 - 输出有限
        
        *For any* PINN prediction, all output values SHALL be finite.
        
        **Feature: ewp-flow-field-solver, Property 10: PINN 精度**
        **Validates: Requirements 5.3**
        """
        from src.solvers.flow_solver import PINNSolver
        
        solver = PINNSolver()
        solver.build_network()
        
        # 测试多个随机输入
        for _ in range(5):
            x = np.random.rand(20, 3) * 1e-4
            t = np.random.rand() * 0.02
            result = solver.predict(x, t=t)
            
            for key, val in result.items():
                assert np.all(np.isfinite(val)), f"{key} contains non-finite values"
    
    def test_pinn_save_load_consistency(self):
        """
        Property 10: PINN 精度 - 保存加载一致性
        
        *For any* saved and loaded PINN model, predictions SHALL be identical.
        
        **Feature: ewp-flow-field-solver, Property 10: PINN 精度**
        **Validates: Requirements 5.3**
        """
        import tempfile
        from src.solvers.flow_solver import PINNSolver
        
        solver1 = PINNSolver()
        solver1.build_network()
        
        # 预测
        x = np.random.rand(10, 3) * 1e-4
        result1 = solver1.predict(x, t=0.001)
        
        # 保存
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            save_path = f.name
        
        solver1.save_model(save_path)
        
        # 加载到新求解器
        solver2 = PINNSolver()
        solver2.load_model(save_path)
        
        result2 = solver2.predict(x, t=0.001)
        
        # 验证一致性
        for key in result1:
            np.testing.assert_array_almost_equal(
                result1[key], result2[key], decimal=6,
                err_msg=f"{key} differs after save/load"
            )
        
        # 清理
        import os
        os.remove(save_path)


# ============================================================
# Property 9: 开口率积分
# **Feature: ewp-flow-field-solver, Property 9: 开口率积分**
# **Validates: Requirements 4.2**
# ============================================================

class TestApertureRatioProperties:
    """开口率积分属性测试"""
    
    @given(voltage=st.floats(min_value=0.0, max_value=40.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=50, deadline=None)
    def test_aperture_ratio_range(self, voltage: float):
        """
        Property 9: 开口率积分 - 范围
        
        *For any* voltage, the aperture ratio SHALL be in [0, 1].
        
        **Feature: ewp-flow-field-solver, Property 9: 开口率积分**
        **Validates: Requirements 4.2**
        """
        from src.solvers.flow_solver import FlowFieldSimulator
        
        simulator = FlowFieldSimulator()
        result = simulator.simulate(voltage=voltage, duration=0.001, method='hybrid')
        
        for eta in result.aperture_ratio:
            assert 0.0 <= eta <= 1.0, f"Aperture ratio {eta} out of range [0, 1]"
    
    def test_aperture_ratio_consistency_with_model(self):
        """
        Property 9: 开口率积分 - 与模型一致性
        
        *For any* simulation, the aperture ratio SHALL match EnhancedApertureModel 
        within 1% for hybrid method.
        
        **Feature: ewp-flow-field-solver, Property 9: 开口率积分**
        **Validates: Requirements 4.2**
        """
        from src.solvers.flow_solver import FlowFieldSimulator
        
        simulator = FlowFieldSimulator()
        
        for voltage in [0, 15, 30]:
            comparison = simulator.compare_with_aperture_model(voltage=voltage, duration=0.01)
            
            # 对于 hybrid 方法，应该完全一致
            assert comparison['relative_error'] < 0.01, \
                f"Aperture ratio mismatch at {voltage}V: {comparison['relative_error']*100:.2f}%"


# ============================================================
# Property 11: 验证误差标记
# **Feature: ewp-flow-field-solver, Property 11: 验证误差标记**
# **Validates: Requirements 6.4**
# ============================================================

class TestValidationProperties:
    """验证功能属性测试"""
    
    def test_validation_error_flag_high_error(self):
        """
        Property 11: 验证误差标记 - 高误差
        
        *For any* validation with relative error > 10%, the error_flag SHALL be True.
        
        **Feature: ewp-flow-field-solver, Property 11: 验证误差标记**
        **Validates: Requirements 6.4**
        """
        from src.solvers.flow_solver import FlowFieldSimulator
        
        simulator = FlowFieldSimulator()
        
        # 创建与模拟结果差异很大的实验数据
        exp_data = {
            't': np.linspace(0, 0.01, 10),
            'aperture_ratio': np.ones(10) * 0.9,  # 故意设置很高的值
            'voltage': 30.0
        }
        
        validation = simulator.validate_against_experiment(exp_data)
        
        # 由于差异很大，应该标记错误
        assert validation['error_flag'] == True, \
            f"Error flag should be True for high error, got {validation['error_flag']}"
    
    def test_validation_returns_required_fields(self):
        """
        Property 11: 验证误差标记 - 返回字段
        
        *For any* validation, the result SHALL contain mae, rmse, max_error, 
        relative_error, and error_flag.
        
        **Feature: ewp-flow-field-solver, Property 11: 验证误差标记**
        **Validates: Requirements 6.4**
        """
        from src.solvers.flow_solver import FlowFieldSimulator
        
        simulator = FlowFieldSimulator()
        
        exp_data = {
            't': np.linspace(0, 0.01, 10),
            'aperture_ratio': np.linspace(0, 0.3, 10),
            'voltage': 30.0
        }
        
        validation = simulator.validate_against_experiment(exp_data)
        
        required_fields = ['mae', 'rmse', 'max_error', 'relative_error', 'error_flag']
        for field in required_fields:
            assert field in validation, f"Missing field: {field}"


# ============================================================
# Property 12: 导出格式兼容性
# **Feature: ewp-flow-field-solver, Property 12: 导出格式兼容性**
# **Validates: Requirements 7.4**
# ============================================================

class TestExportProperties:
    """导出功能属性测试"""
    
    def test_export_npz_loadable(self):
        """
        Property 12: 导出格式兼容性 - NPZ 可加载
        
        *For any* exported NPZ file, the file SHALL be loadable by NumPy.
        
        **Feature: ewp-flow-field-solver, Property 12: 导出格式兼容性**
        **Validates: Requirements 7.4**
        """
        import tempfile
        import os
        from src.solvers.flow_solver import FlowFieldSimulator
        
        simulator = FlowFieldSimulator()
        result = simulator.simulate(voltage=30, duration=0.001, method='hybrid')
        
        with tempfile.TemporaryDirectory() as tmpdir:
            simulator.export_results(result, tmpdir, format='npz')
            
            # 验证文件存在
            npz_file = os.path.join(tmpdir, "simulation_result.npz")
            assert os.path.exists(npz_file), "NPZ file not created"
            
            # 验证可加载
            data = np.load(npz_file, allow_pickle=True)
            assert 't' in data, "Missing 't' in NPZ"
            assert 'aperture_ratio' in data, "Missing 'aperture_ratio' in NPZ"
    
    def test_export_json_loadable(self):
        """
        Property 12: 导出格式兼容性 - JSON 可加载
        
        *For any* exported JSON file, the file SHALL be loadable by Python json module.
        
        **Feature: ewp-flow-field-solver, Property 12: 导出格式兼容性**
        **Validates: Requirements 7.4**
        """
        import tempfile
        import os
        import json
        from src.solvers.flow_solver import FlowFieldSimulator
        
        simulator = FlowFieldSimulator()
        result = simulator.simulate(voltage=30, duration=0.001, method='hybrid')
        
        with tempfile.TemporaryDirectory() as tmpdir:
            simulator.export_results(result, tmpdir, format='json')
            
            # 验证文件存在
            json_file = os.path.join(tmpdir, "simulation_summary.json")
            assert os.path.exists(json_file), "JSON file not created"
            
            # 验证可加载
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            assert 't' in data, "Missing 't' in JSON"
            assert 'aperture_ratio' in data, "Missing 'aperture_ratio' in JSON"
            assert 'method' in data, "Missing 'method' in JSON"
