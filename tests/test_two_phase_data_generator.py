"""
两相流 PINN DataGenerator 测试

测试 src/models/pinn_two_phase.py 中的 DataGenerator 类

Author: EFD-PINNs Team
Date: 2025-12-12
"""

import pytest
import numpy as np
import torch
from hypothesis import given, strategies as st, settings, assume, HealthCheck

from src.models.pinn_two_phase import DataGenerator, PHYSICS, DEFAULT_CONFIG


# ============================================================
# 测试策略
# ============================================================

voltage_strategy = st.floats(min_value=0, max_value=30, allow_nan=False, allow_infinity=False)
time_strategy = st.floats(min_value=0, max_value=0.05, allow_nan=False, allow_infinity=False)
position_strategy = st.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False)


# ============================================================
# Fixture
# ============================================================

@pytest.fixture
def data_generator():
    """创建 DataGenerator 实例"""
    device = torch.device('cpu')
    return DataGenerator(DEFAULT_CONFIG, device)


# ============================================================
# 测试：接触角计算
# ============================================================

class TestContactAngleCalculation:
    """测试接触角计算的正确性"""
    
    @given(voltage=voltage_strategy, time=time_strategy)
    @settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_contact_angle_range(self, data_generator, voltage, time):
        """接触角应在合理范围内 [60°, 120°]"""
        theta = data_generator.get_contact_angle(voltage, time)
        assert 60 <= theta <= 120, f"接触角 {theta}° 超出范围"
    
    def test_contact_angle_at_zero_voltage(self, data_generator):
        """零电压时接触角应接近初始值 120°"""
        theta = data_generator.get_contact_angle(0, 0.02)
        assert 115 <= theta <= 120, f"零电压接触角 {theta}° 不正确"
    
    def test_contact_angle_decreases_with_voltage(self, data_generator):
        """电压增加时接触角应减小"""
        theta_0V = data_generator.get_contact_angle(0, 0.02)
        theta_10V = data_generator.get_contact_angle(10, 0.02)
        theta_30V = data_generator.get_contact_angle(30, 0.02)
        
        assert theta_0V > theta_10V > theta_30V, "接触角未随电压单调递减"
    
    def test_contact_angle_dynamics(self, data_generator):
        """接触角应随时间动态变化"""
        V = 30
        theta_0ms = data_generator.get_contact_angle(V, 0.000)
        theta_5ms = data_generator.get_contact_angle(V, 0.005)
        theta_20ms = data_generator.get_contact_angle(V, 0.020)
        
        # 初始时刻接近 θ₀
        assert theta_0ms > 115
        # 5ms 时已经下降
        assert theta_5ms < theta_0ms
        # 20ms 时继续下降（欠阻尼振荡）
        assert theta_20ms < theta_5ms


# ============================================================
# 测试：开口率计算
# ============================================================

class TestApertureRatioCalculation:
    """测试开口率计算的正确性"""
    
    @given(voltage=voltage_strategy, time=time_strategy)
    @settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_aperture_ratio_range(self, data_generator, voltage, time):
        """开口率应在 [0, 0.85] 范围内"""
        eta = data_generator.get_opening_rate(voltage, time)
        assert 0 <= eta <= 0.85, f"开口率 {eta} 超出范围"
    
    def test_aperture_ratio_at_zero_voltage(self, data_generator):
        """零电压时开口率应接近零"""
        eta = data_generator.get_opening_rate(0, 0.02)
        assert eta < 0.05, f"零电压开口率 {eta} 过大"
    
    def test_aperture_ratio_increases_with_voltage(self, data_generator):
        """电压增加时开口率应增加"""
        eta_0V = data_generator.get_opening_rate(0, 0.02)
        eta_10V = data_generator.get_opening_rate(10, 0.02)
        eta_30V = data_generator.get_opening_rate(30, 0.02)
        
        assert eta_0V < eta_10V < eta_30V, "开口率未随电压单调递增"


# ============================================================
# 测试：φ 场计算
# ============================================================

class TestPhiFieldCalculation:
    """测试 φ 场计算的正确性"""
    
    @given(
        x_frac=position_strategy,
        y_frac=position_strategy,
        z_frac=position_strategy,
        voltage=voltage_strategy,
        time=time_strategy
    )
    @settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_phi_range(self, data_generator, x_frac, y_frac, z_frac, voltage, time):
        """φ 值应在 [0, 1] 范围内"""
        x = x_frac * PHYSICS["Lx"]
        y = y_frac * PHYSICS["Ly"]
        z = z_frac * PHYSICS["Lz"]
        
        phi = data_generator.target_phi_3d(x, y, z, time, voltage)
        assert 0 <= phi <= 1, f"φ={phi} 超出范围"
    
    def test_phi_initial_condition(self, data_generator):
        """初始条件：底部油墨，上部透明"""
        Lx, Ly = PHYSICS["Lx"], PHYSICS["Ly"]
        h_ink = PHYSICS["h_ink"]
        
        # 底部中心应该是油墨
        phi_bottom = data_generator.target_phi_3d(Lx/2, Ly/2, h_ink/2, 0, 0)
        assert phi_bottom > 0.9, f"底部中心 φ={phi_bottom} 应接近1"
        
        # 顶部应该是透明液体
        phi_top = data_generator.target_phi_3d(Lx/2, Ly/2, PHYSICS["Lz"]*0.9, 0, 0)
        assert phi_top < 0.1, f"顶部 φ={phi_top} 应接近0"
    
    def test_phi_voltage_response(self, data_generator):
        """施加电压后中心应变透明"""
        Lx, Ly = PHYSICS["Lx"], PHYSICS["Ly"]
        h_ink = PHYSICS["h_ink"]
        
        # 0V：中心有油墨
        phi_0V = data_generator.target_phi_3d(Lx/2, Ly/2, h_ink/2, 0.02, 0)
        
        # 30V：中心透明
        phi_30V = data_generator.target_phi_3d(Lx/2, Ly/2, h_ink/2, 0.02, 30)
        
        assert phi_30V < phi_0V, "施加电压后中心应更透明"
        assert phi_30V < 0.3, f"30V时中心 φ={phi_30V} 应接近0"


# ============================================================
# 测试：降压过程
# ============================================================

class TestVoltageDownProcess:
    """测试降压过程的物理正确性"""
    
    def test_voltage_down_aperture_decay(self, data_generator):
        """降压后开口率应指数衰减"""
        V_prev = 30
        t_step = 0.015
        
        # 降压前的开口率
        eta_before = data_generator.get_opening_rate(V_prev, t_step)
        
        # 降压后不同时刻的 φ 场（通过积分估算开口率）
        times = [0.016, 0.020, 0.030, 0.050]
        etas = []
        
        for t in times:
            # 在底面采样计算开口率
            n_samples = 20
            Lx, Ly = PHYSICS["Lx"], PHYSICS["Ly"]
            h_ink = PHYSICS["h_ink"]
            
            phi_samples = []
            for i in range(n_samples):
                for j in range(n_samples):
                    x = i * Lx / (n_samples - 1)
                    y = j * Ly / (n_samples - 1)
                    phi = data_generator.target_phi_3d(x, y, h_ink/2, t, 0, V_prev=V_prev, t_step=t_step)
                    phi_samples.append(phi)
            
            # 开口率 = 透明区域比例
            eta = np.mean(np.array(phi_samples) < 0.5)
            etas.append(eta)
        
        # 检查单调递减
        for i in range(len(etas) - 1):
            assert etas[i] > etas[i+1], f"开口率未单调递减: {etas}"
        
        # 最终应接近零
        assert etas[-1] < 0.1, f"最终开口率 {etas[-1]} 未恢复到接近零"
    
    def test_voltage_down_phi_distribution(self, data_generator):
        """降压后油墨应从边缘向中心铺展"""
        Lx, Ly = PHYSICS["Lx"], PHYSICS["Ly"]
        h_ink = PHYSICS["h_ink"]
        V_prev = 30
        t_step = 0.015
        t = 0.025  # 降压后10ms
        
        # 中心
        phi_center = data_generator.target_phi_3d(Lx/2, Ly/2, h_ink/2, t, 0, V_prev=V_prev, t_step=t_step)
        
        # 边缘
        phi_edge = data_generator.target_phi_3d(Lx*0.1, Ly*0.1, h_ink/2, t, 0, V_prev=V_prev, t_step=t_step)
        
        # 边缘应该比中心更多油墨（油墨从边缘向中心铺展）
        assert phi_edge > phi_center, "降压后边缘应比中心有更多油墨"


# ============================================================
# 测试：数据生成
# ============================================================

class TestDataGeneration:
    """测试数据生成功能"""
    
    def test_generate_all_data(self, data_generator):
        """测试完整数据生成"""
        data = data_generator.generate_all_data()
        
        # 检查数据结构
        required_keys = ['interface_points', 'interface_targets', 'contact_points', 
                        'contact_theta', 'ic_points', 'ic_values', 'bc_points', 
                        'bc_values', 'domain_points']
        for key in required_keys:
            assert key in data, f"缺少键: {key}"
        
        # 检查数据维度
        for key in ['interface_points', 'contact_points', 'ic_points', 'bc_points', 'domain_points']:
            assert data[key].shape[1] == 6, f"{key} 应为6维 (x,y,z,V_from,V_to,t_since)"
        
        # 检查数据范围
        for key in ['interface_points', 'contact_points', 'ic_points', 'bc_points', 'domain_points']:
            points = data[key].cpu().numpy() if torch.is_tensor(data[key]) else data[key]
            x_data = points[:, 0]
            y_data = points[:, 1]
            z_data = points[:, 2]
            V_from = points[:, 3]
            V_to = points[:, 4]
            t_data = points[:, 5]
            
            assert np.all((x_data >= 0) & (x_data <= PHYSICS["Lx"])), f"{key}: x超出范围"
            assert np.all((y_data >= 0) & (y_data <= PHYSICS["Ly"])), f"{key}: y超出范围"
            assert np.all((z_data >= 0) & (z_data <= PHYSICS["Lz"])), f"{key}: z超出范围"
            assert np.all((t_data >= 0) & (t_data <= PHYSICS["t_max"])), f"{key}: t超出范围"
            assert np.all((V_from >= 0) & (V_from <= 30)), f"{key}: V_from超出范围"
            assert np.all((V_to >= 0) & (V_to <= 30)), f"{key}: V_to超出范围"


# ============================================================
# 测试：辅助方法
# ============================================================

class TestHelperMethods:
    """测试辅助方法"""
    
    def test_compute_contact_angle_gradient(self, data_generator):
        """测试接触角梯度计算"""
        # 90度：cos=0, sin=1
        cos_90, sin_90 = data_generator.compute_contact_angle_gradient(90)
        assert abs(cos_90) < 0.01
        assert abs(sin_90 - 1) < 0.01
        
        # 0度：cos=1, sin=0
        cos_0, sin_0 = data_generator.compute_contact_angle_gradient(0)
        assert abs(cos_0 - 1) < 0.01
        assert abs(sin_0) < 0.01
        
        # 180度：cos=-1, sin=0
        cos_180, sin_180 = data_generator.compute_contact_angle_gradient(180)
        assert abs(cos_180 + 1) < 0.01
        assert abs(sin_180) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
