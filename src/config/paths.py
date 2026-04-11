"""
路径管理模块
============

集中管理项目中的所有路径常量，避免硬编码。
支持通过环境变量覆盖默认路径。

使用方法:
    from src.config.paths import CONFIG_PATH, PROJECT_ROOT
    
    # 加载配置
    with open(CONFIG_PATH) as f:
        config = json.load(f)

环境变量:
    EFD_CONFIG_PATH: 覆盖默认配置文件路径
    EFD_OUTPUT_DIR: 覆盖默认输出目录

作者: EFD-PINNs Team
日期: 2026-01-08
"""

import os
from pathlib import Path
from typing import Optional

# ============================================================================
# 项目根目录
# ============================================================================
# 从当前文件位置推断: src/config/paths.py -> 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()

# ============================================================================
# 配置文件路径
# ============================================================================
# 默认配置文件
DEFAULT_CONFIG_FILENAME = "device_calibrated_physics.json"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / DEFAULT_CONFIG_FILENAME

# 支持环境变量覆盖
_env_config_path = os.getenv("EFD_CONFIG_PATH")
if _env_config_path:
    CONFIG_PATH = Path(_env_config_path)
else:
    CONFIG_PATH = DEFAULT_CONFIG_PATH

# ============================================================================
# 输出目录
# ============================================================================
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"

_env_output_dir = os.getenv("EFD_OUTPUT_DIR")
if _env_output_dir:
    OUTPUT_DIR = Path(_env_output_dir)
else:
    OUTPUT_DIR = DEFAULT_OUTPUT_DIR

# ============================================================================
# 辅助函数
# ============================================================================

def get_config_path(config_path: Optional[str] = None) -> Path:
    """
    获取配置文件路径
    
    优先级:
    1. 显式传入的 config_path
    2. 环境变量 EFD_CONFIG_PATH
    3. 默认路径 config/device_calibrated_physics.json
    
    Args:
        config_path: 可选的配置文件路径
        
    Returns:
        配置文件的 Path 对象
        
    Raises:
        FileNotFoundError: 配置文件不存在时
    """
    if config_path:
        path = Path(config_path)
        # 如果是相对路径，相对于项目根目录
        if not path.is_absolute():
            path = PROJECT_ROOT / path
    else:
        path = CONFIG_PATH
    
    if not path.exists():
        raise FileNotFoundError(
            f"配置文件未找到: {path}\n"
            f"请确保文件存在，或设置环境变量 EFD_CONFIG_PATH"
        )
    
    return path


def get_output_dir(subdir: Optional[str] = None, create: bool = True) -> Path:
    """
    获取输出目录路径
    
    Args:
        subdir: 可选的子目录名
        create: 是否自动创建目录
        
    Returns:
        输出目录的 Path 对象
    """
    path = OUTPUT_DIR
    if subdir:
        path = path / subdir
    
    if create and not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    
    return path


# ============================================================================
# 导出
# ============================================================================
__all__ = [
    "PROJECT_ROOT",
    "CONFIG_PATH", 
    "DEFAULT_CONFIG_PATH",
    "OUTPUT_DIR",
    "DEFAULT_OUTPUT_DIR",
    "get_config_path",
    "get_output_dir",
]
