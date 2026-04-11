"""
测试动态物理损失权重调度器

测试 DynamicPhysicsWeightScheduler 类的功能，包括：
- 初始化和参数设置
- 基于阶段的权重调整
- 自适应权重调整
- 基于性能的权重调整
- 组合策略权重调整
- 权重边界和限制

作者: EFD-PINNs Team
日期: 2026-01-08
"""

import pytest
import numpy as np
import torch
from src.training.scheduler import DynamicPhysicsWeightScheduler, PhysicsWeightIntegration


class TestDynamicPhysicsWeightScheduler:
    """测试动态物理损失权重调度器"""

    def test_initialization(self):
        """测试初始化"""
        scheduler = DynamicPhysicsWeightScheduler(
            initial_weight=0.1,
            min_weight=0.01,
            max_weight=5.0
        )
        assert scheduler.current_weight == 0.1
        assert scheduler.min_weight == 0.01
        assert scheduler.max_weight == 5.0
        assert scheduler.step_count == 0

    def test_initial_weight_from_dict(self):
        """测试从字典初始化 initial_weight"""
        scheduler = DynamicPhysicsWeightScheduler(
            initial_weight={'value': 0.2, 'other': 0.3}
        )
        assert scheduler.current_weight == 0.2

    def test_default_initialization(self):
        """测试默认参数初始化"""
        scheduler = DynamicPhysicsWeightScheduler()
        assert scheduler.current_weight == 0.1
        assert scheduler.min_weight == 0.01
        assert scheduler.max_weight == 5.0
        assert scheduler.adjustment_strategy == "adaptive"

    def test_stage_based_adjustment(self):
        """测试基于阶段的权重调整"""
        scheduler = DynamicPhysicsWeightScheduler()

        # 测试各个阶段的权重
        assert scheduler._stage_based_adjustment(1) == 0.05
        assert scheduler._stage_based_adjustment(2) == 0.1
        assert scheduler._stage_based_adjustment(3) == 0.5
        assert scheduler._stage_based_adjustment(4) == 1.0

        # 测试超出范围的情况
        assert scheduler._stage_based_adjustment(5) == 1.0
        assert scheduler._stage_based_adjustment(10) == 1.0

    def test_adaptive_adjustment_increase(self):
        """测试自适应权重调整 - 数据损失大于物理损失"""
        scheduler = DynamicPhysicsWeightScheduler()
        scheduler.current_weight = 1.0

        # 数据损失远大于物理损失，应增加权重
        new_weight = scheduler._adaptive_adjustment(
            data_loss=2.0,
            physics_loss=1.0
        )
        assert new_weight > scheduler.current_weight

    def test_adaptive_adjustment_decrease(self):
        """测试自适应权重调整 - 物理损失大于数据损失"""
        scheduler = DynamicPhysicsWeightScheduler()
        scheduler.current_weight = 1.0

        # 物理损失远大于数据损失，应减少权重
        new_weight = scheduler._adaptive_adjustment(
            data_loss=1.0,
            physics_loss=2.0
        )
        assert new_weight < scheduler.current_weight

    def test_adaptive_adjustment_no_change(self):
        """测试自适应权重调整 - 比例接近目标"""
        scheduler = DynamicPhysicsWeightScheduler()
        scheduler.current_weight = 1.0

        # 比例接近 1.0，应保持不变
        new_weight = scheduler._adaptive_adjustment(
            data_loss=1.0,
            physics_loss=1.0
        )
        assert new_weight == scheduler.current_weight

    def test_performance_based_adjustment_improving(self):
        """测试基于性能的调整 - 验证损失改善"""
        scheduler = DynamicPhysicsWeightScheduler()
        scheduler.best_val_loss = 1.0

        # 验证损失改善，保持当前权重
        new_weight = scheduler._performance_based_adjustment(val_loss=0.9)
        assert new_weight == scheduler.current_weight

    def test_performance_based_adjustment_no_improvement(self):
        """测试基于性能的调整 - 验证损失无改善"""
        scheduler = DynamicPhysicsWeightScheduler()
        scheduler.best_val_loss = 1.0
        scheduler.steps_without_improvement = 600
        scheduler.current_weight = 0.5  # 使用较低的权重
        scheduler.patience = 500

        # 超过耐心值，权重较低，应增加权重尝试改善
        new_weight = scheduler._performance_based_adjustment(val_loss=1.1)
        assert new_weight > scheduler.current_weight

    def test_combined_strategy(self):
        """测试组合策略"""
        scheduler = DynamicPhysicsWeightScheduler(
            adjustment_strategy="combined"
        )

        # 组合策略应该结合 stage, adaptive, performance
        new_weight = scheduler._compute_new_weight(
            data_loss=1.5,
            physics_loss=1.0,
            val_loss=1.0,
            epoch=100,
            stage=2
        )

        # 权重应该在合理范围内
        assert scheduler.min_weight <= new_weight <= scheduler.max_weight

    def test_weight_smoothing(self):
        """测试权重平滑处理"""
        scheduler = DynamicPhysicsWeightScheduler(
            smoothing_factor=0.9,
            adjustment_interval=1
        )

        old_weight = scheduler.current_weight
        scheduler.update(
            data_loss=2.0,
            physics_loss=1.0,
            val_loss=None
        )

        # 权重变化应该被平滑
        weight_change = abs(scheduler.current_weight - old_weight)
        assert weight_change < old_weight  # 变化不应该太剧烈

    def test_weight_bounds(self):
        """测试权重边界限制"""
        scheduler = DynamicPhysicsWeightScheduler(
            min_weight=0.01,
            max_weight=5.0,
            adjustment_interval=1
        )

        # 测试极端情况
        scheduler.update(
            data_loss=1e-10,
            physics_loss=1e10,
            val_loss=None
        )

        # 权重应该在边界内
        assert 0.01 <= scheduler.current_weight <= 5.0

    def test_update_increments_step_count(self):
        """测试更新增加步数"""
        scheduler = DynamicPhysicsWeightScheduler()
        initial_count = scheduler.step_count

        scheduler.update(
            data_loss=1.0,
            physics_loss=1.0,
            val_loss=None
        )

        assert scheduler.step_count == initial_count + 1

    def test_loss_history_tracking(self):
        """测试损失历史记录"""
        scheduler = DynamicPhysicsWeightScheduler()

        scheduler.update(
            data_loss=1.0,
            physics_loss=2.0,
            val_loss=3.0
        )

        history = scheduler.get_loss_history()
        assert len(history['data_loss']) == 1
        assert len(history['physics_loss']) == 1
        assert len(history['val_loss']) == 1
        assert history['data_loss'][0] == 1.0
        assert history['physics_loss'][0] == 2.0
        assert history['val_loss'][0] == 3.0

    def test_adjustment_interval(self):
        """测试调整间隔"""
        scheduler = DynamicPhysicsWeightScheduler(
            initial_weight=0.1,
            adjustment_interval=100
        )

        # 前 99 步权重不应改变
        for i in range(99):
            scheduler.update(
                data_loss=2.0,
                physics_loss=1.0,
                val_loss=None
            )
        assert scheduler.current_weight == 0.1

        # 第 100 步权重应该改变
        scheduler.update(
            data_loss=2.0,
            physics_loss=1.0,
            val_loss=None
        )
        assert scheduler.current_weight != 0.1

    def test_reset(self):
        """测试重置调度器"""
        scheduler = DynamicPhysicsWeightScheduler()

        # 运行一些更新
        for i in range(10):
            scheduler.update(
                data_loss=1.0,
                physics_loss=1.0,
                val_loss=1.0
            )

        # 重置
        scheduler.reset()

        # 检查所有状态已重置
        assert scheduler.current_weight == scheduler.initial_weight
        assert scheduler.step_count == 0
        assert scheduler.best_val_loss == float('inf')
        assert scheduler.steps_without_improvement == 0
        assert len(scheduler.loss_history['data_loss']) == 0
        assert len(scheduler.loss_history['physics_loss']) == 0
        assert len(scheduler.loss_history['val_loss']) == 0


class TestPhysicsWeightIntegration:
    """测试物理权重集成类"""

    def test_initialization(self):
        """测试初始化"""
        scheduler = DynamicPhysicsWeightScheduler()
        integration = PhysicsWeightIntegration(scheduler)

        assert integration.weight_scheduler == scheduler
        assert integration.integration_method == "multiplicative"

    def test_multiplicative_integration(self):
        """测试乘法集成方法"""
        scheduler = DynamicPhysicsWeightScheduler(initial_weight=2.0)
        integration = PhysicsWeightIntegration(
            scheduler,
            integration_method="multiplicative"
        )

        physics_loss = torch.tensor(1.0)
        data_loss = torch.tensor(2.0)

        result = integration.apply_dynamic_weight(
            base_physics_loss=physics_loss,
            data_loss=data_loss
        )

        expected = physics_loss * 2.0
        assert torch.allclose(result, expected)

    def test_additive_integration(self):
        """测试加法集成方法"""
        scheduler = DynamicPhysicsWeightScheduler(initial_weight=2.0)
        integration = PhysicsWeightIntegration(
            scheduler,
            integration_method="additive"
        )

        physics_loss = torch.tensor(1.0)
        data_loss = torch.tensor(2.0)

        result = integration.apply_dynamic_weight(
            base_physics_loss=physics_loss,
            data_loss=data_loss
        )

        expected = physics_loss + 2.0
        assert torch.allclose(result, expected)

    def test_get_current_weight(self):
        """测试获取当前权重"""
        scheduler = DynamicPhysicsWeightScheduler(initial_weight=0.5)
        integration = PhysicsWeightIntegration(scheduler)

        assert integration.get_current_weight() == 0.5

    def test_tensor_conversion(self):
        """测试张量转换"""
        scheduler = DynamicPhysicsWeightScheduler(initial_weight=2.0)
        integration = PhysicsWeightIntegration(scheduler)

        # 传入非张量输入
        physics_loss = 1.0  # float
        data_loss = 2.0

        result = integration.apply_dynamic_weight(
            base_physics_loss=physics_loss,
            data_loss=data_loss
        )

        assert isinstance(result, torch.Tensor)

    def test_device_preservation(self):
        """测试设备保持"""
        scheduler = DynamicPhysicsWeightScheduler(initial_weight=2.0)
        integration = PhysicsWeightIntegration(scheduler)

        # CUDA 设备测试（如果可用）
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        physics_loss = torch.tensor(1.0, device=device)
        data_loss = torch.tensor(2.0, device=device)

        result = integration.apply_dynamic_weight(
            base_physics_loss=physics_loss,
            data_loss=data_loss
        )

        assert result.device == physics_loss.device


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
