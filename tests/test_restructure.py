#!/usr/bin/env python3
"""
项目重构验证测试

验证目录结构、模块导入和核心功能
"""

import os
import sys
import pytest
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestDirectoryStructure:
    """测试目录结构是否正确"""
    
    def test_src_directory_exists(self):
        """验证 src/ 目录存在"""
        assert (PROJECT_ROOT / "src").is_dir()
    
    def test_src_subdirectories_exist(self):
        """验证 src/ 子目录存在"""
        subdirs = ["models", "physics", "training", "predictors", "solvers", "visualization", "utils"]
        for subdir in subdirs:
            assert (PROJECT_ROOT / "src" / subdir).is_dir(), f"src/{subdir} 不存在"
    
    def test_models_layers_directory_exists(self):
        """验证 src/models/layers/ 目录存在"""
        assert (PROJECT_ROOT / "src" / "models" / "layers").is_dir()
    
    def test_scripts_directory_exists(self):
        """验证 scripts/ 目录存在"""
        assert (PROJECT_ROOT / "scripts").is_dir()
    
    def test_scripts_subdirectories_exist(self):
        """验证 scripts/ 子目录存在"""
        subdirs = ["analysis", "validation", "visualization"]
        for subdir in subdirs:
            assert (PROJECT_ROOT / "scripts" / subdir).is_dir(), f"scripts/{subdir} 不存在"
    
    def test_config_directory_exists(self):
        """验证 config/ 目录存在"""
        assert (PROJECT_ROOT / "config").is_dir()
    
    def test_init_files_exist(self):
        """验证 __init__.py 文件存在"""
        init_paths = [
            "src/__init__.py",
            "src/models/__init__.py",
            "src/models/layers/__init__.py",
            "src/physics/__init__.py",
            "src/training/__init__.py",
            "src/predictors/__init__.py",
            "src/solvers/__init__.py",
            "src/visualization/__init__.py",
            "src/utils/__init__.py",
        ]
        for path in init_paths:
            assert (PROJECT_ROOT / path).is_file(), f"{path} 不存在"


class TestModuleImports:
    """测试模块导入是否正常"""
    
    def test_import_src_package(self):
        """验证可以导入 src 包"""
        import src
        assert hasattr(src, "__version__")
    
    def test_import_models_module(self):
        """验证可以导入 models 模块"""
        from src import models
        assert hasattr(models, "layers")
    
    def test_import_layers(self):
        """验证可以导入 layers 模块"""
        from src.models.layers import EWPINNInputLayer, EWPINNOutputLayer
        assert EWPINNInputLayer is not None
        assert EWPINNOutputLayer is not None
    
    def test_import_aperture_model(self):
        """验证可以导入 ApertureModel"""
        from src.models.aperture_model import ApertureModel
        assert ApertureModel is not None
    
    def test_import_physics_constraints(self):
        """验证可以导入 PhysicsConstraints"""
        from src.physics.constraints import PhysicsConstraints
        assert PhysicsConstraints is not None
    
    def test_import_data_generator(self):
        """验证可以导入 ElectrowettingPhysicsGenerator"""
        from src.physics.data_generator import ElectrowettingPhysicsGenerator
        assert ElectrowettingPhysicsGenerator is not None


class TestConfigFiles:
    """测试配置文件迁移"""
    
    def test_config_files_exist(self):
        """验证配置文件已迁移到 config/ 目录"""
        config_files = [
            "stage6_wall_effect.json",
            "stage2_optimized.json",
        ]
        for config_file in config_files:
            assert (PROJECT_ROOT / "config" / config_file).is_file(), f"config/{config_file} 不存在"


class TestScriptFiles:
    """测试脚本文件迁移"""
    
    def test_analysis_scripts_exist(self):
        """验证分析脚本已迁移"""
        scripts = [
            "analyze_pinn_predictions.py",
        ]
        for script in scripts:
            assert (PROJECT_ROOT / "scripts" / "analysis" / script).is_file(), f"scripts/analysis/{script} 不存在"
    
    def test_validation_scripts_exist(self):
        """验证验证脚本已迁移"""
        scripts = [
            "validate_pinn_model.py",
            "validate_two_phase_pinn.py",
            "verify_parameters.py",
            "validate_stage1.py",
        ]
        for script in scripts:
            assert (PROJECT_ROOT / "scripts" / "validation" / script).is_file(), f"scripts/validation/{script} 不存在"
    
    def test_visualization_scripts_exist(self):
        """验证可视化脚本已迁移"""
        scripts = [
            "generate_paper_figures.py",
            "generate_pyvista_3d.py",
        ]
        for script in scripts:
            assert (PROJECT_ROOT / "scripts" / "visualization" / script).is_file(), f"scripts/visualization/{script} 不存在"


class TestOutputsDirectory:
    """测试输出目录整合"""
    
    def test_outputs_directory_exists(self):
        """验证 outputs/ 或 outputs_pinn_* 目录存在"""
        # 检查 outputs/ 目录或任何 outputs_pinn_* 目录
        outputs_exists = (PROJECT_ROOT / "outputs").is_dir()
        pinn_outputs_exist = any(PROJECT_ROOT.glob("outputs_pinn_*"))
        assert outputs_exists or pinn_outputs_exist, "outputs/ 或 outputs_pinn_* 目录不存在"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
