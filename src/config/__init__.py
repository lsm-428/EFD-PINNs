# src/config/__init__.py
"""EFD3D 配置模块"""

from .physics_config import (
    PhysicsConfig,
    get_physics_config,
    get_materials_params,
    PHYSICS,
)

from .paths import (
    PROJECT_ROOT,
    CONFIG_PATH,
    DEFAULT_CONFIG_PATH,
    OUTPUT_DIR,
    get_config_path,
    get_output_dir,
)

__all__ = [
    # 物理配置
    'PhysicsConfig',
    'get_physics_config',
    'get_materials_params',
    'PHYSICS',
    # 路径管理
    'PROJECT_ROOT',
    'CONFIG_PATH',
    'DEFAULT_CONFIG_PATH',
    'OUTPUT_DIR',
    'get_config_path',
    'get_output_dir',
]
