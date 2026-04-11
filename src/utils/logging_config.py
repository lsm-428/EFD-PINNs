"""
统一日志配置模块
================

为 EFD3D 项目提供统一的日志配置和管理。
支持控制台输出、文件输出、日志级别控制等功能。

使用方法:
    from src.utils.logging_config import setup_logging, get_logger

    # 设置日志系统
    setup_logging(level="INFO", log_file="training.log")

    # 获取 logger
    logger = get_logger(__name__)
    logger.info("训练开始")

环境变量:
    EFD_LOG_LEVEL: 日志级别 (DEBUG, INFO, WARNING, ERROR)
    EFD_LOG_FILE: 日志文件路径

作者: EFD-PINNs Team
日期: 2026-01-08
"""

import logging
import sys
from pathlib import Path
from typing import Optional


# 默认日志格式
DEFAULT_FORMAT = "[%(asctime)s] %(levelname)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# 详细格式（包含文件名和行号）
VERBOSE_FORMAT = "[%(asctime)s] %(levelname)s [%(filename)s:%(lineno)d] | %(message)s"


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    enable_console: bool = True,
    verbose: bool = False,
    format_type: str = "default"
) -> None:
    """
    配置统一的日志系统

    Args:
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: 日志文件路径（可选）
        enable_console: 是否启用控制台输出
        verbose: 是否使用详细格式（包含文件名和行号）
        format_type: 格式类型 ("default" 或 "verbose")

    Examples:
        >>> setup_logging(level="INFO")
        >>> setup_logging(level="DEBUG", log_file="debug.log", verbose=True)
    """
    # 获取根 logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # 设置最低级别，由 handler 控制

    # 清除现有 handlers
    root_logger.handlers.clear()

    # 选择格式
    if format_type == "verbose" or verbose:
        log_format = VERBOSE_FORMAT
    else:
        log_format = DEFAULT_FORMAT

    formatter = logging.Formatter(log_format, datefmt=DEFAULT_DATE_FORMAT)

    # 控制台处理器
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # 文件处理器
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)  # 文件记录所有级别
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    获取命名 logger

    Args:
        name: logger 名称，通常使用 __name__

    Returns:
        Logger 实例

    Examples:
        >>> logger = get_logger(__name__)
        >>> logger.info("信息日志")
        >>> logger.error("错误日志")
    """
    return logging.getLogger(name)


def log_level_from_env(env_var: str = "EFD_LOG_LEVEL", default: str = "INFO") -> str:
    """
    从环境变量获取日志级别

    Args:
        env_var: 环境变量名
        default: 默认级别

    Returns:
        日志级别字符串
    """
    import os
    return os.getenv(env_var, default)


def setup_logging_from_env() -> None:
    """
    从环境变量设置日志系统

    环境变量:
        EFD_LOG_LEVEL: 日志级别
        EFD_LOG_FILE: 日志文件路径
        EFD_LOG_VERBOSE: 是否使用详细格式 (1/0, true/false)
    """
    import os

    level = log_level_from_env()
    log_file = os.getenv("EFD_LOG_FILE")
    verbose = os.getenv("EFD_LOG_VERBOSE", "0").lower() in ("1", "true", "yes")

    setup_logging(level=level, log_file=log_file, verbose=verbose)


class LoggerMixin:
    """
    Logger 混入类，为类提供 logger 属性

    Examples:
        >>> class MyClass(LoggerMixin):
        ...     def my_method(self):
        ...         self.logger.info("使用混入的 logger")
    """

    @property
    def logger(self) -> logging.Logger:
        """获取当前类的 logger"""
        return logging.getLogger(self.__class__.__name__)


# ============================================================================
# 便捷函数
# ============================================================================

def set_log_level(level: str) -> None:
    """
    设置所有 logger 的日志级别

    Args:
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logging.getLogger().setLevel(getattr(logging, level.upper(), logging.INFO))


def disable_third_party_logging() -> None:
    """
    禁用第三方库的详细日志
    """
    for name in ['matplotlib', 'PIL', 'numba']:
        logging.getLogger(name).setLevel(logging.WARNING)


def enable_debug_mode() -> None:
    """
    启用调试模式：设置 DEBUG 级别并使用详细格式
    """
    setup_logging(level="DEBUG", verbose=True)


# 模块级别的默认 logger（用于向后兼容）
_default_logger = logging.getLogger("EFD3D")


# ============================================================================
# 导出
# ============================================================================
__all__ = [
    "setup_logging",
    "get_logger",
    "log_level_from_env",
    "setup_logging_from_env",
    "LoggerMixin",
    "set_log_level",
    "disable_third_party_logging",
    "enable_debug_mode",
]
