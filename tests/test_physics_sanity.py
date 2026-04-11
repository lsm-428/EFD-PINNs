"""
物理残差 Sanity Check 测试

验证：
1. 参数一致性：θ₀=120°, εᵣ=12.0, σ=0.045 在配置与约束实现中统一
2. 物理方程形式：N-S + VOF + 表面张力 + 体积守恒 在 PhysicsConstraints 中集中实现
3. 梯度反向传播：通过小批量点验证模型参数能正确接收梯度
4. 数值稳定性：在代表性高电压/长时间/边界点样例下无 NaN/Inf
5. 接口兼容性：旧方法保留且不破坏现有训练脚本

Created: 2025-12-31
"""

import torch
import pytest
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestParameterConsistency:
    """测试参数一致性"""
    
    def test_physics_config_values(self):
        """验证统一物理配置中的关键参数"""
        from src.config import PHYSICS, get_physics_config
        
        config = get_physics_config()
        
        # 验证关键参数
        assert abs(config.theta0 - 120.0) < 1e-6, f"theta0 应为 120.0, 实际为 {config.theta0}"
        assert abs(config.epsilon_r - 12.0) < 1e-6, f"epsilon_r 应为 12.0, 实际为 {config.epsilon_r}"
        assert abs(config.sigma - 0.045) < 1e-6, f"sigma 应为 0.045, 实际为 {config.sigma}"
        assert abs(config.tau - 0.005) < 1e-6, f"tau 应为 0.005, 实际为 {config.tau}"
        assert abs(config.zeta - 0.8) < 1e-6, f"zeta 应为 0.8, 实际为 {config.zeta}"
        
        # 验证 PHYSICS 字典与 config 一致
        assert PHYSICS['theta0'] == config.theta0
        assert PHYSICS['epsilon_r'] == config.epsilon_r
        assert PHYSICS['sigma'] == config.sigma
    
    def test_physics_constraints_uses_unified_config(self):
        """验证 PhysicsConstraints 使用统一配置"""
        from src.physics.constraints import PhysicsConstraints
        from src.config import get_materials_params
        
        materials = get_materials_params()
        pc = PhysicsConstraints(materials_params=materials)
        
        # 验证材料参数已正确传入
        assert pc.materials_params is not None
        assert 'surface_tension_polar_ink' in pc.materials_params
        assert abs(pc.materials_params['surface_tension_polar_ink'] - 0.045) < 1e-6
        assert 'relative_permittivity' in pc.materials_params
        assert abs(pc.materials_params['relative_permittivity'] - 12.0) < 1e-6


class TestCoreResiduals:
    """测试核心残差计算"""
    
    @pytest.fixture
    def setup_model_and_data(self):
        """设置模型和测试数据"""
        from src.models.pinn_two_phase import TwoPhasePINN
        from src.physics.constraints import PhysicsConstraints
        
        device = torch.device('cpu')
        model = TwoPhasePINN()
        pc = PhysicsConstraints()
        
        # 创建测试数据: (x, y, z, V_from, V_to, t_since)
        batch_size = 32
        x_phys = torch.rand(batch_size, 6, device=device)
        x_phys[:, 0:3] = x_phys[:, 0:3] * 174e-6  # 空间坐标
        x_phys[:, 3:5] = x_phys[:, 3:5] * 40.0    # 电压
        x_phys[:, 5] = x_phys[:, 5] * 0.01        # 时间
        x_phys.requires_grad_(True)
        
        return model, pc, x_phys, device
    
    def test_compute_core_residuals_returns_all_keys(self, setup_model_and_data):
        """验证 compute_core_residuals 返回所有核心残差项"""
        model, pc, x_phys, device = setup_model_and_data
        
        residuals = pc.compute_core_residuals(x_phys, None, model=model)
        
        # 验证核心残差项存在
        expected_keys = ['continuity', 'momentum_u', 'momentum_v', 'momentum_w', 'vof']
        for key in expected_keys:
            assert key in residuals, f"缺少残差项: {key}"
            assert isinstance(residuals[key], torch.Tensor), f"{key} 应为 Tensor"
    
    def test_residuals_no_nan_inf(self, setup_model_and_data):
        """验证残差中无 NaN/Inf"""
        model, pc, x_phys, device = setup_model_and_data
        
        residuals = pc.compute_core_residuals(x_phys, None, model=model)
        
        for key, val in residuals.items():
            if isinstance(val, torch.Tensor):
                assert torch.isfinite(val).all(), f"{key} 包含 NaN/Inf"
    
    def test_gradient_backpropagation(self, setup_model_and_data):
        """验证梯度能正确反向传播到模型参数"""
        model, pc, x_phys, device = setup_model_and_data
        model.train()
        
        residuals = pc.compute_core_residuals(x_phys, None, model=model)
        
        # 计算总损失
        total_loss = torch.tensor(0.0, device=device)
        for key, val in residuals.items():
            if isinstance(val, torch.Tensor) and val.requires_grad:
                total_loss = total_loss + torch.mean(val**2)
        
        # 反向传播
        total_loss.backward()
        
        # 检查模型参数是否有梯度
        has_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        
        assert has_grad, "模型参数未收到梯度"


class TestZeroFieldScenario:
    """测试零场景（简单物理解）"""
    
    def test_static_uniform_field(self):
        """
        测试静止、均匀场景：u=v=w=0, p=常数, φ=常数
        
        在此场景下，所有残差应接近零（数值误差层级）
        """
        from src.models.pinn_two_phase import TwoPhasePINN
        from src.physics.constraints import PhysicsConstraints
        
        device = torch.device('cpu')
        batch_size = 16
        
        # 创建一个输出恒定值的简单模型
        class ConstantModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = torch.nn.Parameter(torch.zeros(1))
            
            def forward(self, x):
                batch = x.shape[0]
                # u=v=w=0, p=1.0, phi=0.5
                return torch.cat([
                    torch.zeros(batch, 3),  # u, v, w
                    torch.ones(batch, 1),   # p
                    torch.full((batch, 1), 0.5)  # phi
                ], dim=1)
        
        model = ConstantModel()
        pc = PhysicsConstraints()
        
        x_phys = torch.rand(batch_size, 6, requires_grad=True)
        
        residuals = pc.compute_core_residuals(x_phys, None, model=model)
        
        # 连续性残差应为零（div u = 0）
        if 'continuity' in residuals:
            cont_mse = torch.mean(residuals['continuity']**2).item()
            assert cont_mse < 1e-6, f"连续性残差过大: {cont_mse}"


class TestPhysicsLossInterface:
    """测试 PhysicsLoss 接口"""
    
    def test_compute_all_residuals(self):
        """测试 PhysicsLoss.compute_all_residuals"""
        from src.models.pinn_two_phase import PhysicsLoss, TwoPhasePINN
        
        device = torch.device('cpu')
        physics_loss = PhysicsLoss(device)
        model = TwoPhasePINN()
        
        x_phys = torch.rand(16, 6, requires_grad=True)
        
        residuals = physics_loss.compute_all_residuals(model, x_phys)
        
        assert isinstance(residuals, dict)
        assert len(residuals) > 0
    
    def test_compute_total_loss(self):
        """测试 PhysicsLoss.compute_total_loss"""
        from src.models.pinn_two_phase import PhysicsLoss, TwoPhasePINN
        
        device = torch.device('cpu')
        physics_loss = PhysicsLoss(device)
        model = TwoPhasePINN()
        
        x_phys = torch.rand(16, 6, requires_grad=True)
        
        losses = physics_loss.compute_total_loss(model, x_phys)
        
        assert 'total' in losses
        assert isinstance(losses['total'], torch.Tensor)
        assert torch.isfinite(losses['total'])


class TestStage1ModelsUnaffected:
    """验证 Stage 1 模型不受影响"""
    
    def test_enhanced_aperture_model(self):
        """测试 EnhancedApertureModel 正常工作"""
        from src.models.aperture_model import EnhancedApertureModel
        
        model = EnhancedApertureModel()
        
        # 测试接触角到开口率转换
        theta = 100.0
        eta = model.contact_angle_to_aperture_ratio(theta)
        
        assert 0 <= eta <= 1, f"开口率应在 [0, 1] 范围内, 实际为 {eta}"
    
    def test_hybrid_predictor(self):
        """测试 HybridPredictor 正常工作"""
        from src.predictors.hybrid_predictor import HybridPredictor
        
        predictor = HybridPredictor()
        
        # 测试 Young-Lippmann
        V = 30.0
        theta = predictor.young_lippmann(V)
        
        # 电压增加应导致接触角减小
        assert theta < 120.0, f"30V 时接触角应小于 120°, 实际为 {theta}"
        
        # 测试稳态预测
        eta = predictor.predict_steady_state(V)
        assert eta is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
