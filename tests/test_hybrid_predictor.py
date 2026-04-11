"""
测试 HybridPredictor 混合预测器

测试 HybridPredictor 类的功能，包括：
- 初始化和延迟加载
- Young-Lippmann 方程计算
- 动态响应计算
- 接触角预测
- 依赖注入支持
- 边界条件

作者: EFD-PINNs Team
日期: 2026-01-08
"""

import pytest
import numpy as np
import torch
from src.predictors.hybrid_predictor import HybridPredictor
from src.models.aperture_model import EnhancedApertureModel


class TestHybridPredictor:
    """测试混合预测器"""

    def test_initialization(self):
        """测试初始化"""
        predictor = HybridPredictor()
        assert predictor is not None

    def test_initialization_with_aperture_model(self):
        """测试使用注入的 ApertureModel 初始化"""
        # 创建一个模拟的 ApertureModel
        class MockApertureModel:
            def __init__(self):
                self.tau_rc = 0.1e-3

        mock_model = MockApertureModel()
        predictor = HybridPredictor(aperture_model=mock_model)

        assert predictor._aperture_model == mock_model

    def test_get_initial_contact_angle(self):
        """测试获取初始接触角"""
        predictor = HybridPredictor()
        theta_0 = predictor.params['theta0']

        # 初始接触角应该是 120 度（从配置文件读取）
        assert theta_0 == pytest.approx(120.0, rel=0.01)

    def test_young_lippmann_static(self):
        """测试 Young-Lippmann 方程 - 静态情况"""
        predictor = HybridPredictor()

        # 测试不同电压下的静态接触角
        voltages = [0.0, 15.0, 30.0]
        thetas = []

        for V in voltages:
            theta = predictor.young_lippmann(V)
            thetas.append(theta)
            assert 0 < theta < 180  # 接触角应该在合理范围内

        # 电压越高，接触角应该越小（亲水性增强）
        assert thetas[0] > thetas[1] > thetas[2]

    def test_young_lippmann_dynamic_step_response(self):
        """测试 Young-Lippmann 动态响应 - 阶跃响应"""
        predictor = HybridPredictor()

        # 测试阶跃响应
        V_from = 0.0
        V_to = 30.0

        # 使用step_response方法
        t, thetas = predictor.step_response(V_start=V_from, V_end=V_to, duration=0.05, t_step=0.001, num_points=5)

        assert len(thetas) == 5
        assert all(0 < theta < 180 for theta in thetas)

        # 随着时间增加，接触角应该减小（从120°减到更小的角度）
        assert thetas[0] > thetas[-1]  # 从初始角度减小到最终角度

    def test_young_lippmann_dynamic_voltage_dropout(self):
        """测试 Young-Lippmann 动态响应 - 电压下降"""
        predictor = HybridPredictor()

        # 测试电压下降（30V -> 0V）
        V_from = 30.0
        V_to = 0.0

        # 使用step_response方法
        t, thetas = predictor.step_response(V_start=V_from, V_end=V_to, duration=0.05, t_step=0.001, num_points=5)

        assert len(thetas) == 5
        assert all(0 < theta < 180 for theta in thetas)

        # 随着时间增加，应该恢复到初始接触角
        assert thetas[0] < thetas[-1]  # 电压下降时，接触角应该增大

    def test_predict_angle_single(self):
        """测试单个预测"""
        predictor = HybridPredictor()

        # 预测单个电压点的接触角
        voltage = 20.0
        time = 0.01  # 10ms
        theta = predictor.predict(voltage=voltage, time=time, V_initial=0.0, t_step=0.0)

        assert isinstance(theta, (float, np.floating))
        assert 0 < theta < 180

    def test_predict_angle_batch(self):
        """测试批量预测"""
        predictor = HybridPredictor()

        # 预测多个电压点的接触角
        voltages = np.array([0.0, 10.0, 20.0, 30.0])
        time = 0.01  # 10ms
        thetas = np.array([predictor.predict(voltage=V, time=time, V_initial=0.0, t_step=0.0) for V in voltages])

        assert isinstance(thetas, np.ndarray)
        assert len(thetas) == len(voltages)
        assert all(0 < theta < 180 for theta in thetas)

    def test_predict_angle_triplet_format(self):
        """测试三元组格式预测"""
        predictor = HybridPredictor()

        # 三元组格式：(V_from, V_to, t_since)
        V_from = 0.0
        V_to = 30.0
        t_since = 0.01  # 10ms，转换为秒

        theta = predictor.predict(voltage=V_to, time=t_since, V_initial=V_from, t_step=0.0)

        assert isinstance(theta, (float, np.floating))
        assert 0 < theta < 180

    def test_theta_to_aperture_lazy_loading(self):
        """测试延迟加载 ApertureModel"""
        predictor = HybridPredictor(aperture_model=None)

        # 在调用 _theta_to_aperture 之前，模型应该未加载
        assert predictor._aperture_model is None

        # 首次调用应该自动创建模型
        try:
            aperture = predictor._theta_to_aperture(120.0)
            assert predictor._aperture_model is not None
        except ImportError:
            # 如果 ApertureModel 无法导入，跳过测试
            pytest.skip("ApertureModel not available")

    def test_theta_to_aperture_with_injected_model(self):
        """测试使用注入的模型"""
        # 创建一个模拟的 ApertureModel
        class MockApertureModel:
            def contact_angle_to_aperture_ratio(self, theta):
                return theta / 180.0  # 简单的线性映射

        mock_model = MockApertureModel()
        predictor = HybridPredictor(aperture_model=mock_model)

        # 使用注入的模型
        aperture = predictor._theta_to_aperture(90.0)

        assert aperture == pytest.approx(0.5, rel=0.01)

    def test_contact_angle_bounds(self):
        """测试接触角边界限制"""
        predictor = HybridPredictor()

        # 测试极端情况（避免极高电压导致arccos(1)=0）
        extreme_voltages = [-100, 0, 50, 80]
        for V in extreme_voltages:
            theta = predictor.young_lippmann(V)
            # 接触角应该被限制在合理范围内
            assert 0 <= theta <= 180, f"电压 {V}V 时接触角 {theta} 超出范围"

    def test_monotonicity(self):
        """测试单调性 - 电压越高，接触角越小"""
        predictor = HybridPredictor()

        voltages = np.linspace(0, 40, 100)
        thetas = [predictor.young_lippmann(V) for V in voltages]

        # 检查单调递减
        for i in range(len(thetas) - 1):
            assert thetas[i] >= thetas[i + 1] - 1e-6  # 允许小的数值误差

    def test_consistency_with_config(self):
        """测试与配置文件的一致性"""
        predictor = HybridPredictor()

        # 从配置文件读取参数
        theta_0 = predictor.params['theta0']

        # 验证静态预测在 t=0 时与初始角度一致
        theta_pred = predictor.young_lippmann(0.0)

        assert theta_pred == pytest.approx(theta_0, rel=0.01)

    def test_dynamic_response_time_constant(self):
        """测试动态响应时间常数"""
        predictor = HybridPredictor()

        # 获取时间常数
        tau = predictor.params.get('tau', 0.005)  # 秒

        # 测试在时间常数处的响应（欠阻尼系统会有振荡）
        V_from = 0.0
        V_to = 30.0

        theta_0 = predictor.young_lippmann(V_from)
        theta_inf = predictor.young_lippmann(V_to)

        # 使用dynamic_response方法
        theta_tau = predictor.dynamic_response(tau, theta_0, theta_inf)

        # 对于欠阻尼系统，检查值在合理范围内
        assert min(theta_0, theta_inf) <= theta_tau <= max(theta_0, theta_inf) + 10  # 允许轻微超调

    def test_multiple_voltage_steps(self):
        """测试多个电压步骤"""
        predictor = HybridPredictor()

        # 模拟电压变化序列: 0V -> 15V -> 30V -> 15V -> 0V
        voltage_sequence = [0.0, 15.0, 30.0, 15.0, 0.0]
        t_since = 0.02  # 20ms after each step，转换为秒

        thetas = []
        for i in range(len(voltage_sequence) - 1):
            V_from = voltage_sequence[i]
            V_to = voltage_sequence[i + 1]
            theta = predictor.predict(voltage=V_to, time=t_since, V_initial=V_from, t_step=0.0)
            thetas.append(theta)

        assert len(thetas) == len(voltage_sequence) - 1
        assert all(0 < theta < 180 for theta in thetas)


class TestHybridPredictorEdgeCases:
    """测试边界情况和异常处理"""

    def test_zero_voltage(self):
        """测试零电压情况"""
        predictor = HybridPredictor()
        theta = predictor.young_lippmann(0.0)

        # 零电压时应该接近初始接触角
        theta_0 = predictor.params['theta0']
        assert theta == pytest.approx(theta_0, rel=0.01)

    def test_negative_voltage(self):
        """测试负电压情况（如果支持）"""
        predictor = HybridPredictor()

        # 某些设备可能支持负电压
        try:
            theta = predictor.young_lippmann(-10.0)
            assert 0 < theta < 180
        except (ValueError, AttributeError):
            # 如果不支持负电压，跳过测试
            pytest.skip("Negative voltage not supported")

    def test_very_high_voltage(self):
        """测试极高电压情况"""
        predictor = HybridPredictor()
        # 使用一个高但不会导致完全饱和的电压
        theta = predictor.young_lippmann(100.0)

        # 即使电压很高，接触角也应该在合理范围内（可以是0，表示完全润湿）
        assert 0 <= theta <= 180

    def test_zero_time(self):
        """测试时间为零的情况"""
        predictor = HybridPredictor()

        V_from = 0.0
        V_to = 30.0
        t_since = 0.0

        theta = predictor.predict(voltage=V_to, time=t_since, V_initial=V_from, t_step=0.0)

        # t=0 时应该接近起始电压的静态接触角
        theta_from_static = predictor.young_lippmann(V_from)
        assert theta == pytest.approx(theta_from_static, rel=0.01)

    def test_very_long_time(self):
        """测试很长时间（应该达到稳态）"""
        predictor = HybridPredictor()

        V_from = 0.0
        V_to = 30.0
        t_since = 1.0  # 1 秒

        theta_dynamic = predictor.predict(voltage=V_to, time=t_since, V_initial=V_from, t_step=0.0)
        theta_static = predictor.young_lippmann(V_to)

        # 长时间后应该接近稳态
        assert theta_dynamic == pytest.approx(theta_static, rel=0.05)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
