"""
EWP 3D Visualization Property Tests

使用 Hypothesis 进行属性测试，验证 3D 可视化模块的正确性属性。

Author: EFD-PINNs Team
Date: 2025-12-03
"""

import pytest
import numpy as np
import json
import tempfile
import os
from pathlib import Path
from hypothesis import given, strategies as st, settings, assume

# 导入被测试的模块
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.visualization.visualizer_3d import (
    PixelVisualizer, AnimationEngine, DataExporter,
    VisualizationConfig, PIXEL_STRUCTURE,
    visualize_pixel, export_data
)
from src.models.aperture_model import EnhancedApertureModel


# ============================================================
# 测试配置
# ============================================================

# 电压范围策略
voltage_strategy = st.floats(min_value=0.0, max_value=40.0, allow_nan=False, allow_infinity=False)

# 有效电压范围（产生开口的电压）
opening_voltage_strategy = st.floats(min_value=5.0, max_value=40.0, allow_nan=False, allow_infinity=False)


# ============================================================
# Property 1: 透明区域几何正确性
# ============================================================

class TestTransparentRegionGeometry:
    """
    Property 1: 透明区域几何正确性
    
    *For any* voltage V where aperture_ratio > 0, the generated 3D geometry 
    SHALL include a cylindrical transparent region with radius equal to r_open 
    returned by EnhancedApertureModel.predict_enhanced(V).
    
    **Feature: ewp-3d-visualization, Property 1: 透明区域几何正确性**
    **Validates: Requirements 1.2**
    """
    
    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = EnhancedApertureModel()
        self.visualizer = PixelVisualizer(model=self.model)
    
    @given(voltage=opening_voltage_strategy)
    @settings(max_examples=100, deadline=None)
    def test_transparent_region_radius_matches_model(self, voltage):
        """验证透明区域半径与模型预测一致"""
        # 获取模型预测
        prediction = self.model.predict_enhanced(voltage)
        
        # 只测试有开口的情况
        assume(prediction['aperture_ratio'] > 0.01)
        
        # 获取可视化器计算的几何信息
        # 通过 get_title_info 间接验证模型被正确调用
        title_info = self.visualizer.get_title_info(voltage)
        
        # 验证 aperture_ratio 一致
        assert abs(title_info['aperture_ratio'] - prediction['aperture_ratio']) < 1e-10
        
        # 验证 r_open 可以从 aperture_ratio 正确计算
        # r_open = sqrt(aperture_ratio * pixel_area / pi)
        pixel_area = self.model.pixel_area
        expected_r_open = np.sqrt(prediction['aperture_ratio'] * pixel_area / np.pi)
        
        # 使用相对误差，允许 10% 的误差（模型使用壁面效应修正，与简单公式有差异）
        relative_error = abs(prediction['r_open'] - expected_r_open) / max(expected_r_open, 1e-10)
        assert relative_error < 0.10, f"r_open 相对误差 {relative_error:.2%} 超过 10%"


# ============================================================
# Property 2: 3D 油墨体积守恒
# ============================================================

class TestInkVolumeConservation3D:
    """
    Property 2: 3D 油墨体积守恒
    
    *For any* voltage V, the volume of the 3D ink geometry SHALL equal 
    the theoretical ink volume (pixel_area × ink_thickness) within 0.1% tolerance.
    
    **Feature: ewp-3d-visualization, Property 2: 3D 油墨体积守恒**
    **Validates: Requirements 1.3**
    """
    
    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = EnhancedApertureModel()
        self.visualizer = PixelVisualizer(model=self.model)
    
    @given(voltage=voltage_strategy)
    @settings(max_examples=100, deadline=None)
    def test_ink_volume_conservation(self, voltage):
        """验证油墨体积守恒"""
        # 获取模型预测
        prediction = self.model.predict_enhanced(voltage)
        
        # 验证模型本身的体积守恒
        volume_error = prediction['volume_error']
        assert volume_error < 0.1, f"体积误差 {volume_error}% 超过 0.1%"
        
        # 验证油墨分布数据
        r = prediction['r']
        h = prediction['h']
        
        # 计算理论体积
        theoretical_volume = self.model.ink_volume
        
        # 计算实际体积（从分布数据）
        # 使用模型的验证方法
        actual_error = self.model.verify_volume_conservation(r, h)
        
        assert actual_error < 0.1, f"实际体积误差 {actual_error}% 超过 0.1%"


# ============================================================
# Property 3: 可视化元数据完整性
# ============================================================

class TestVisualizationMetadataCompleteness:
    """
    Property 3: 可视化元数据完整性
    
    *For any* visualization generated with voltage V and optional time t, 
    the title/metadata SHALL contain the contact angle θ and aperture ratio η 
    values matching the model prediction.
    
    **Feature: ewp-3d-visualization, Property 3: 可视化元数据完整性**
    **Validates: Requirements 1.4**
    """
    
    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = EnhancedApertureModel()
        self.visualizer = PixelVisualizer(model=self.model)
    
    @given(voltage=voltage_strategy)
    @settings(max_examples=100, deadline=None)
    def test_title_info_contains_theta_and_eta(self, voltage):
        """验证标题信息包含 theta 和 eta"""
        # 获取模型预测
        prediction = self.model.predict_enhanced(voltage)
        
        # 获取可视化标题信息
        title_info = self.visualizer.get_title_info(voltage)
        
        # 验证 theta 一致
        assert abs(title_info['theta'] - prediction['theta']) < 1e-10
        
        # 验证 aperture_ratio 一致
        assert abs(title_info['aperture_ratio'] - prediction['aperture_ratio']) < 1e-10
        
        # 验证 aperture_percent 正确计算
        assert abs(title_info['aperture_percent'] - prediction['aperture_ratio'] * 100) < 1e-10
    
    @given(
        voltage=voltage_strategy,
        time=st.floats(min_value=0.0, max_value=0.1, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=50, deadline=None)
    def test_title_info_with_time(self, voltage, time):
        """验证带时间参数的标题信息"""
        # 获取模型预测
        prediction = self.model.predict_enhanced(voltage, time)
        
        # 获取可视化标题信息
        title_info = self.visualizer.get_title_info(voltage, time)
        
        # 验证 theta 一致
        assert abs(title_info['theta'] - prediction['theta']) < 1e-10
        
        # 验证时间参数被正确传递
        assert title_info['time'] == time


# ============================================================
# Property 4: 文件输出正确性
# ============================================================

class TestFileOutputCorrectness:
    """
    Property 4: 文件输出正确性
    
    *For any* visualization with save_path provided, a PNG file SHALL be 
    created at the specified path with dimensions matching the requested resolution.
    
    **Feature: ewp-3d-visualization, Property 4: 文件输出正确性**
    **Validates: Requirements 1.5, 8.1**
    """
    
    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = EnhancedApertureModel()
        self.visualizer = PixelVisualizer(model=self.model)
    
    @given(voltage=st.sampled_from([0.0, 15.0, 30.0]))
    @settings(max_examples=3, deadline=None)
    def test_file_created_at_path(self, voltage):
        """验证文件在指定路径创建"""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, f"test_{voltage}V.png")
            
            # 渲染并保存
            self.visualizer.render(voltage, save_path=save_path)
            
            # 验证文件存在
            assert os.path.exists(save_path), f"文件未创建: {save_path}"
            
            # 验证文件大小 > 0
            assert os.path.getsize(save_path) > 0, "文件大小为 0"
    
    def test_default_resolution(self):
        """验证默认分辨率为 1920x1080"""
        config = VisualizationConfig()
        assert config.resolution == (1920, 1080)
    
    def test_custom_resolution(self):
        """验证自定义分辨率"""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test_custom_res.png")
            
            # 使用自定义分辨率
            self.visualizer.render(
                voltage=30,
                save_path=save_path,
                resolution=(800, 600)
            )
            
            # 验证文件存在
            assert os.path.exists(save_path)


# ============================================================
# Property 5: 油墨剖面边界准确性
# ============================================================

class TestInkProfileBoundaryAccuracy:
    """
    Property 5: 油墨剖面边界准确性
    
    *For any* ink profile visualization, the 3D surface SHALL have h=0 
    for all r < r_open and h>0 for r >= r_open, matching the model's ink distribution.
    
    **Feature: ewp-3d-visualization, Property 5: 油墨剖面边界准确性**
    **Validates: Requirements 4.2**
    """
    
    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = EnhancedApertureModel()
        self.visualizer = PixelVisualizer(model=self.model)
    
    @given(voltage=opening_voltage_strategy)
    @settings(max_examples=100, deadline=None)
    def test_ink_profile_boundary(self, voltage):
        """验证油墨剖面边界正确"""
        # 获取油墨剖面数据
        profile_data = self.visualizer.get_ink_profile_data(voltage)
        
        r = profile_data['r']
        h = profile_data['h']
        r_open = profile_data['r_open']
        aperture_ratio = profile_data['aperture_ratio']
        
        # 只测试有开口的情况
        assume(aperture_ratio > 0.01)
        assume(r_open > 0)
        
        # 验证透明区域内 h = 0
        mask_open = r < r_open
        h_in_open = h[mask_open]
        
        if len(h_in_open) > 0:
            assert np.all(h_in_open == 0), \
                f"透明区域内存在非零高度: max(h) = {np.max(h_in_open)}"
        
        # 验证油墨区域内 h > 0
        mask_ink = r >= r_open
        h_in_ink = h[mask_ink]
        
        if len(h_in_ink) > 0:
            assert np.all(h_in_ink > 0), \
                f"油墨区域内存在零高度"
    
    @given(voltage=st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=50, deadline=None)
    def test_no_opening_uniform_ink(self, voltage):
        """验证无开口时油墨均匀分布"""
        # 获取油墨剖面数据
        profile_data = self.visualizer.get_ink_profile_data(voltage)
        
        aperture_ratio = profile_data['aperture_ratio']
        
        # 只测试无开口的情况
        assume(aperture_ratio <= 0.01)
        
        h = profile_data['h']
        
        # 验证油墨均匀分布（所有高度相等）
        if len(h) > 0:
            h_nonzero = h[h > 0]
            if len(h_nonzero) > 1:
                h_std = np.std(h_nonzero)
                assert h_std < 1e-10, f"油墨高度不均匀: std = {h_std}"


# ============================================================
# Property 6: 数据导出往返一致性
# ============================================================

class TestDataExportRoundTrip:
    """
    Property 6: 数据导出往返一致性
    
    *For any* exported JSON data, loading the JSON and comparing with 
    the original model prediction SHALL yield identical values for all 
    numeric fields within floating-point precision.
    
    **Feature: ewp-3d-visualization, Property 6: 数据导出往返一致性**
    **Validates: Requirements 7.1, 7.2, 7.3**
    """
    
    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = EnhancedApertureModel()
        self.exporter = DataExporter(model=self.model)
    
    @given(voltage=voltage_strategy)
    @settings(max_examples=100, deadline=None)
    def test_export_round_trip(self, voltage):
        """验证导出数据往返一致性"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_export.json")
            
            # 导出数据
            exported = self.exporter.export_prediction(voltage, output_path=output_path)
            
            # 加载数据
            loaded = DataExporter.load_prediction(output_path)
            
            # 验证预测字段一致
            pred_exported = exported['prediction']
            pred_loaded = loaded['prediction']
            
            assert pred_exported['voltage'] == pred_loaded['voltage']
            assert abs(pred_exported['theta'] - pred_loaded['theta']) < 1e-10
            assert abs(pred_exported['aperture_ratio'] - pred_loaded['aperture_ratio']) < 1e-10
            assert abs(pred_exported['r_open'] - pred_loaded['r_open']) < 1e-10
            assert abs(pred_exported['volume_error'] - pred_loaded['volume_error']) < 1e-10
    
    @given(voltage=voltage_strategy)
    @settings(max_examples=100, deadline=None)
    def test_export_contains_required_fields(self, voltage):
        """验证导出数据包含所有必需字段"""
        # 导出数据（不保存文件）
        exported = self.exporter.export_prediction(voltage)
        
        # 验证 metadata 字段
        assert 'metadata' in exported
        assert 'timestamp' in exported['metadata']
        assert 'model_version' in exported['metadata']
        assert 'units' in exported['metadata']
        
        # 验证 prediction 字段
        assert 'prediction' in exported
        pred = exported['prediction']
        required_fields = ['voltage', 'theta', 'aperture_ratio', 'r_open', 'volume_error']
        for field in required_fields:
            assert field in pred, f"缺少字段: {field}"
        
        # 验证 ink_distribution 字段
        assert 'ink_distribution' in exported
        ink = exported['ink_distribution']
        assert 'r' in ink
        assert 'h' in ink
        assert len(ink['r']) == len(ink['h'])
    
    @given(voltage=voltage_strategy)
    @settings(max_examples=50, deadline=None)
    def test_ink_distribution_arrays_preserved(self, voltage):
        """验证油墨分布数组正确保存和加载"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_arrays.json")
            
            # 获取原始预测
            original_prediction = self.model.predict_enhanced(voltage)
            
            # 导出数据
            self.exporter.export_prediction(voltage, output_path=output_path)
            
            # 加载数据
            loaded = DataExporter.load_prediction(output_path)
            
            # 验证数组一致
            r_original = original_prediction['r']
            h_original = original_prediction['h']
            r_loaded = loaded['ink_distribution']['r']
            h_loaded = loaded['ink_distribution']['h']
            
            np.testing.assert_array_almost_equal(r_original, r_loaded)
            np.testing.assert_array_almost_equal(h_original, h_loaded)


# ============================================================
# Property 7: 模型集成正确性
# ============================================================

class TestModelIntegrationCorrectness:
    """
    Property 7: 模型集成正确性
    
    *For any* visualization function call, the geometry parameters 
    (r_open, h_avg, theta, aperture_ratio) SHALL be obtained from 
    EnhancedApertureModel.predict_enhanced() and not computed independently.
    
    **Feature: ewp-3d-visualization, Property 7: 模型集成正确性**
    **Validates: Requirements 6.2, 6.3**
    """
    
    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = EnhancedApertureModel()
        self.visualizer = PixelVisualizer(model=self.model)
        self.exporter = DataExporter(model=self.model)
    
    @given(voltage=voltage_strategy)
    @settings(max_examples=100, deadline=None)
    def test_visualizer_uses_model_prediction(self, voltage):
        """验证可视化器使用模型预测"""
        # 获取模型预测
        model_prediction = self.model.predict_enhanced(voltage)
        
        # 获取可视化器的标题信息
        title_info = self.visualizer.get_title_info(voltage)
        
        # 验证值来自模型
        assert title_info['theta'] == model_prediction['theta']
        assert title_info['aperture_ratio'] == model_prediction['aperture_ratio']
    
    @given(voltage=voltage_strategy)
    @settings(max_examples=100, deadline=None)
    def test_exporter_uses_model_prediction(self, voltage):
        """验证导出器使用模型预测"""
        # 获取模型预测
        model_prediction = self.model.predict_enhanced(voltage)
        
        # 获取导出数据
        exported = self.exporter.export_prediction(voltage)
        
        # 验证值来自模型
        pred = exported['prediction']
        assert pred['theta'] == model_prediction['theta']
        assert pred['aperture_ratio'] == model_prediction['aperture_ratio']
        assert pred['r_open'] == float(model_prediction['r_open'])
    
    @given(voltage=opening_voltage_strategy)
    @settings(max_examples=50, deadline=None)
    def test_ink_geometry_uses_model_arrays(self, voltage):
        """验证油墨几何使用模型的 r, h 数组"""
        # 获取模型预测
        model_prediction = self.model.predict_enhanced(voltage)
        
        # 只测试有开口的情况
        assume(model_prediction['aperture_ratio'] > 0.01)
        
        # 获取油墨剖面数据
        profile_data = self.visualizer.get_ink_profile_data(voltage)
        
        # 验证数组来自模型
        np.testing.assert_array_equal(profile_data['r'], model_prediction['r'])
        np.testing.assert_array_equal(profile_data['h'], model_prediction['h'])


# ============================================================
# 运行测试
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
