#!/usr/bin/env python3
"""
EnhancedApertureModel 属性测试
==============================

使用 hypothesis 库进行属性测试，验证模型的正确性属性。

作者: EFD-PINNs Team
日期: 2025-12-02
"""

import numpy as np
import pytest
from hypothesis import given, strategies as st, settings, assume

import sys
sys.path.insert(0, '.')

from src.models.aperture_model import EnhancedApertureModel


# 配置 hypothesis 运行至少 100 次迭代
settings.register_profile("ci", max_examples=100)
settings.load_profile("ci")


class TestEffectiveVoltageProperty:
    """
    Property 1: 有效电压公式正确性
    
    **Feature: ewp-multiphysics-aperture, Property 1: 有效电压公式正确性**
    **Validates: Requirements 2.2, 2.5**
    """
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """创建模型实例"""
        self.model = EnhancedApertureModel()
    
    @given(
        V_target=st.floats(min_value=0.1, max_value=50.0),
        t=st.floats(min_value=0.0, max_value=0.1)
    )
    def test_effective_voltage_formula(self, V_target, t):
        """
        **Feature: ewp-multiphysics-aperture, Property 1: 有效电压公式正确性**
        
        *For any* 目标电压 V_target > 0 和时间 t ≥ 0，有效电压 V_eff 应满足：
        V_eff = V_target × (1 - exp(-t/τ_RC))
        """
        V_eff = self.model.effective_voltage(V_target, t)
        expected = V_target * (1 - np.exp(-t / self.model.tau_rc))
        
        # 允许浮点误差
        assert abs(V_eff - expected) < 1e-10, \
            f"V_eff={V_eff}, expected={expected}, diff={abs(V_eff - expected)}"
    
    @given(V_target=st.floats(min_value=0.1, max_value=50.0))
    def test_effective_voltage_at_zero(self, V_target):
        """
        **Feature: ewp-multiphysics-aperture, Property 1: 有效电压公式正确性**
        
        当 t = 0 时，V_eff = 0
        """
        V_eff = self.model.effective_voltage(V_target, 0.0)
        assert V_eff == 0.0, f"V_eff at t=0 should be 0, got {V_eff}"
    
    @given(V_target=st.floats(min_value=0.1, max_value=50.0))
    def test_effective_voltage_at_large_t(self, V_target):
        """
        **Feature: ewp-multiphysics-aperture, Property 1: 有效电压公式正确性**
        
        当 t >> τ_RC 时，|V_eff - V_target| / V_target < 0.01
        """
        # t = 100 * tau_rc 应该足够大
        t_large = 100 * self.model.tau_rc
        V_eff = self.model.effective_voltage(V_target, t_large)
        
        relative_error = abs(V_eff - V_target) / V_target
        assert relative_error < 0.01, \
            f"Relative error {relative_error:.6f} >= 0.01 at t={t_large}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


class TestVolumeConservationProperty:
    """
    Property 2: 体积守恒
    
    **Feature: ewp-multiphysics-aperture, Property 2: 体积守恒**
    **Validates: Requirements 3.1**
    """
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """创建模型实例"""
        self.model = EnhancedApertureModel()
    
    @given(theta=st.floats(min_value=60.0, max_value=120.0))
    def test_volume_conservation(self, theta):
        """
        **Feature: ewp-multiphysics-aperture, Property 2: 体积守恒**
        
        *For any* 有效接触角 θ ∈ [60°, 120°]，计算的油墨分布应满足体积守恒：
        |∫∫ h(r) dA - V_ink| / V_ink < 0.001 (0.1%)
        """
        r, h = self.model.calculate_ink_distribution_enhanced(theta)
        error = self.model.verify_volume_conservation(r, h)
        
        assert error < 0.1, \
            f"Volume conservation error {error:.6f}% >= 0.1% at theta={theta}°"


class TestTransparentRegionBoundaryProperty:
    """
    Property 3: 透明区域边界条件
    
    **Feature: ewp-multiphysics-aperture, Property 3: 透明区域边界条件**
    **Validates: Requirements 3.3**
    """
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """创建模型实例"""
        self.model = EnhancedApertureModel()
    
    @given(theta=st.floats(min_value=60.0, max_value=119.0))
    def test_transparent_region_boundary(self, theta):
        """
        **Feature: ewp-multiphysics-aperture, Property 3: 透明区域边界条件**
        
        *For any* 产生开口的接触角 θ < θ₀，油墨高度在透明区域内应为零：
        ∀r < r_open: h(r) = 0
        """
        r, h = self.model.calculate_ink_distribution_enhanced(theta)
        aperture_ratio = self.model.contact_angle_to_aperture_ratio(theta)
        
        # 只有当有开口时才测试
        if aperture_ratio > 0.01:
            r_open = self.model.aperture_ratio_to_open_radius(aperture_ratio)
            
            # 透明区域内油墨高度应为 0
            mask_open = r < r_open
            h_in_open = h[mask_open]
            
            assert np.all(h_in_open == 0), \
                f"Found non-zero ink height in transparent region at theta={theta}°"


class TestApertureRatioRangeProperty:
    """
    Property 4: 开口率范围约束
    
    **Feature: ewp-multiphysics-aperture, Property 4: 开口率范围约束**
    **Validates: Requirements 6.2**
    """
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """创建模型实例"""
        self.model = EnhancedApertureModel()
    
    @settings(deadline=None)  # 禁用超时检查，因为模型初始化可能较慢
    @given(
        V_end=st.floats(min_value=0.0, max_value=30.0),
        duration=st.floats(min_value=0.01, max_value=0.05)
    )
    def test_aperture_ratio_range(self, V_end, duration):
        """
        **Feature: ewp-multiphysics-aperture, Property 4: 开口率范围约束**
        
        *For any* 有效输入，开口率应在物理合理范围内：
        0 ≤ η ≤ 1
        """
        t, eta = self.model.aperture_step_response(
            V_start=0.0, V_end=V_end, duration=duration, t_step=0.002
        )
        
        assert np.all(eta >= 0), f"Found negative aperture ratio: min={np.min(eta)}"
        assert np.all(eta <= 1), f"Found aperture ratio > 1: max={np.max(eta)}"


class TestApertureMonotonicityProperty:
    """
    Property 5: 开口率单调性
    
    **Feature: ewp-multiphysics-aperture, Property 5: 开口率单调性**
    **Validates: Requirements 6.3**
    """
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """创建模型实例"""
        self.model = EnhancedApertureModel()
    
    @given(voltages=st.lists(
        st.floats(min_value=0.0, max_value=30.0),
        min_size=2, max_size=10, unique=True
    ))
    def test_aperture_monotonicity(self, voltages):
        """
        **Feature: ewp-multiphysics-aperture, Property 5: 开口率单调性**
        
        *For any* 电压序列 V₁ < V₂ < ... < Vₙ（稳态条件下），
        对应的开口率应单调递增：
        η(V₁) ≤ η(V₂) ≤ ... ≤ η(Vₙ)
        """
        voltages = sorted(voltages)
        apertures = []
        
        for V in voltages:
            theta = self.model.get_contact_angle(V)
            eta = self.model.contact_angle_to_aperture_ratio(theta)
            apertures.append(eta)
        
        # 检查单调递增
        for i in range(len(apertures) - 1):
            assert apertures[i] <= apertures[i + 1] + 1e-10, \
                f"Monotonicity violated: η({voltages[i]:.2f}V)={apertures[i]:.4f} > η({voltages[i+1]:.2f}V)={apertures[i+1]:.4f}"


class TestNoOvershootProperty:
    """
    Property 6: 无超调时超调为零
    
    **Feature: ewp-multiphysics-aperture, Property 6: 无超调时超调为零**
    **Validates: Requirements 4.5**
    """
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """创建模型实例"""
        self.model = EnhancedApertureModel()
    
    def test_no_overshoot_when_monotonic(self):
        """
        **Feature: ewp-multiphysics-aperture, Property 6: 无超调时超调为零**
        
        *For any* 单调递增的开口率响应曲线，超调百分比应为零：
        若 η(t) 单调递增，则 overshoot_percent = 0
        """
        # 创建一个单调递增的响应
        t = np.linspace(0, 0.02, 100)
        eta = np.linspace(0, 0.5, 100)  # 完全单调递增
        
        metrics = self.model.get_aperture_metrics(t, eta, t_step=0.002)
        
        assert metrics['overshoot_percent'] == 0.0, \
            f"Expected 0% overshoot for monotonic response, got {metrics['overshoot_percent']:.2f}%"
    
    def test_overshoot_detected_when_present(self):
        """
        验证当存在超调时能正确检测
        """
        # 创建一个有超调的响应
        t = np.linspace(0, 0.02, 100)
        eta = np.zeros(100)
        eta[:20] = 0.0  # 初始阶段
        eta[20:50] = np.linspace(0, 0.6, 30)  # 上升阶段，超过最终值
        eta[50:] = 0.5  # 稳态
        
        metrics = self.model.get_aperture_metrics(t, eta, t_step=0.002)
        
        # 超调应该被检测到
        assert metrics['overshoot_percent'] > 0, \
            f"Expected positive overshoot, got {metrics['overshoot_percent']:.2f}%"


class TestConfigSerializationProperty:
    """
    Property 7: 配置序列化往返一致性
    
    **Feature: ewp-multiphysics-aperture, Property 7: 配置序列化往返一致性**
    **Validates: Requirements 8.1, 8.2, 8.3**
    """
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """创建模型实例"""
        self.model = EnhancedApertureModel()
    
    def test_config_round_trip(self):
        """
        **Feature: ewp-multiphysics-aperture, Property 7: 配置序列化往返一致性**
        
        *For any* 有效的模型配置，保存后加载应产生等价配置：
        save_config(path) 后 from_config(path) 应产生相同参数
        """
        import tempfile
        import os
        
        # 获取原始配置
        original_config = self.model.get_config()
        
        # 保存到临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            self.model.save_config(temp_path)
            
            # 从文件加载
            loaded_model = EnhancedApertureModel.from_config(temp_path)
            loaded_config = loaded_model.get_config()
            
            # 验证所有参数匹配
            for key in original_config:
                assert original_config[key] == loaded_config[key], \
                    f"Config mismatch for {key}: {original_config[key]} != {loaded_config[key]}"
        finally:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    @given(tau_rc=st.floats(min_value=1e-5, max_value=1e-2))
    def test_config_round_trip_with_different_tau_rc(self, tau_rc):
        """
        **Feature: ewp-multiphysics-aperture, Property 7: 配置序列化往返一致性**
        
        测试不同 tau_rc 值的配置往返一致性
        """
        import tempfile
        import os
        
        # 创建具有特定 tau_rc 的模型
        model = EnhancedApertureModel(tau_rc=tau_rc)
        original_config = model.get_config()
        
        # 保存到临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            model.save_config(temp_path)
            
            # 从文件加载
            loaded_model = EnhancedApertureModel.from_config(temp_path)
            loaded_config = loaded_model.get_config()
            
            # 验证 tau_rc 匹配
            assert abs(original_config['tau_rc'] - loaded_config['tau_rc']) < 1e-15, \
                f"tau_rc mismatch: {original_config['tau_rc']} != {loaded_config['tau_rc']}"
        finally:
            # 清理临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)
