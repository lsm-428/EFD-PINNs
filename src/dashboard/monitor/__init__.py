"""Dashboard monitor module for training log analysis.

This module provides non-invasive incremental log parsing capability.

Usage:
    from src.dashboard.monitor import LogParser, parse_training_log

    parser = LogParser()
    records = parser.parse("outputs/train/pinn_*/training.log")

    # Or use the simple function
    records = parse_training_log("outputs/train/pinn_*/training.log")
"""

__all__ = [
    "LogParser",
    "TAG_MAP",
    "PATTERN_MAIN",
    "PATTERN_LR",
    "find_log_path",
    "parse_training_log",
    "save_csv",
    "summarize_tail",
    "analyze_volume_trend",
    "analyze_rmse_per_voltage",
    "plot_loss_components",
    "plot_learning_curve",
    "generate_html_report",
    "LOSS_COLORS",
]


def __getattr__(name: str):
    if name == "LogParser":
        from .log_parser import LogParser

        return LogParser
    if name in (
        "TAG_MAP",
        "PATTERN_MAIN",
        "PATTERN_LR",
        "find_log_path",
        "parse_training_log",
        "save_csv",
    ):
        from .log_parsing import (
            TAG_MAP,
            PATTERN_MAIN,
            PATTERN_LR,
            find_log_path,
            parse_training_log,
            save_csv,
        )

        return locals()[name]
    if name in ("summarize_tail", "analyze_volume_trend", "analyze_rmse_per_voltage"):
        from .performance_metrics import (
            summarize_tail,
            analyze_volume_trend,
            analyze_rmse_per_voltage,
        )

        return locals()[name]
    if name in (
        "plot_loss_components",
        "plot_learning_curve",
        "generate_html_report",
        "LOSS_COLORS",
    ):
        from .visualization_output import (
            plot_loss_components,
            plot_learning_curve,
            generate_html_report,
            LOSS_COLORS,
        )

        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
