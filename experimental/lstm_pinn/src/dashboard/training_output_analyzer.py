"""
训练输出分析组件
===============

扫描和分析训练输出目录，提取训练运行信息。
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import pandas as pd
import numpy as np
import warnings

# Dashboard inference engine
from src.dashboard.inference import PINNInferenceEngine


# 安全限制配置
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB 最大文件读取大小


def validate_path_safe(path: str, base_dir: str = "outputs/train") -> bool:
    """验证路径是否安全，防止路径遍历攻击

    Args:
        path: 要验证的路径
        base_dir: 基础目录，默认为 outputs/train

    Returns:
        如果路径安全则返回 True，否则返回 False

    安全检查:
    - 不包含 ".."（防止路径遍历）
    - 不是绝对路径（防止访问系统文件）
    - 解析后的路径在基础目录内
    """
    try:
        # 检查路径中是否包含 ".."
        if ".." in path:
            warnings.warn(f"路径包含 '..'，拒绝访问: {path}")
            return False

        # 检查是否为绝对路径
        path_obj = Path(path)
        if path_obj.is_absolute():
            warnings.warn(f"路径为绝对路径，拒绝访问: {path}")
            return False

        # 解析为相对于基础目录的绝对路径
        base = Path(base_dir).resolve()
        full_path = (base / path_obj).resolve()

        # 检查是否在基础目录内
        try:
            full_path.relative_to(base)
            return True
        except ValueError:
            warnings.warn(f"路径不在基础目录内，拒绝访问: {path}")
            return False

    except Exception as e:
        warnings.warn(f"验证路径时出错: {path}, 错误: {e}")
        return False


def check_file_size(file_path: str, max_size: int = MAX_FILE_SIZE) -> bool:
    """检查文件大小是否在允许范围内

    Args:
        file_path: 文件路径
        max_size: 最大允许文件大小（字节），默认为 50 MB

    Returns:
        如果文件大小在允许范围内则返回 True，否则返回 False
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return True  # 文件不存在，让调用者处理

        file_size = path.stat().st_size
        if file_size > max_size:
            warnings.warn(
                f"文件大小超过限制 ({file_size / 1024 / 1024:.2f} MB > {max_size / 1024 / 1024:.2f} MB): {file_path}"
            )
            return False
        return True

    except Exception as e:
        warnings.warn(f"检查文件大小时出错: {file_path}, 错误: {e}")
        return False


@dataclass
class TrainingRunInfo:
    """训练运行信息数据类"""

    name: str
    path: str
    creation_time: datetime
    model_files: List[str]
    config_path: Optional[str]
    has_loss_csv: bool
    has_metrics: bool


class TrainingOutputScanner:
    """训练输出扫描器类"""

    def __init__(self, train_outputs_dir: str = "outputs/train"):
        """初始化训练输出扫描器

        Args:
            train_outputs_dir: 训练输出目录路径，默认为 outputs/train
        """
        self.train_outputs_dir = train_outputs_dir
        self.supported_model_extensions = [".pth", ".pt"]

    def scan_training_outputs(self) -> List[TrainingRunInfo]:
        """扫描所有训练输出目录

        Returns:
            按创建时间倒序排列的训练运行信息列表
        """
        runs = []

        # 验证基础目录路径是否安全
        if not validate_path_safe(self.train_outputs_dir, base_dir="."):
            warnings.warn(f"训练输出目录路径不安全: {self.train_outputs_dir}")
            return runs

        train_dir = Path(self.train_outputs_dir)
        if not train_dir.exists():
            return runs

        try:
            for run_dir in train_dir.iterdir():
                if run_dir.is_dir():
                    # 验证子目录路径是否安全
                    relative_path = run_dir.relative_to(train_dir)
                    if not validate_path_safe(
                        str(relative_path), base_dir=self.train_outputs_dir
                    ):
                        warnings.warn(f"子目录路径不安全: {run_dir.name}")
                        continue

                    run_info = self._extract_run_info(run_dir)
                    if run_info:
                        runs.append(run_info)
        except Exception as e:
            warnings.warn(f"扫描训练输出目录 {self.train_outputs_dir} 时出错: {e}")

        return sorted(runs, key=lambda x: x.creation_time, reverse=True)

    def _extract_run_info(self, run_dir: Path) -> Optional[TrainingRunInfo]:
        """提取单个训练运行目录的信息

        Args:
            run_dir: 训练运行目录路径

        Returns:
            训练运行信息对象，如果提取失败则返回 None
        """
        try:
            # 获取目录创建时间
            stat = run_dir.stat()
            creation_time = datetime.fromtimestamp(stat.st_ctime)

            # 扫描模型文件
            model_files = self._find_model_files(run_dir)

            # 检查配置文件
            config_path = self._find_config_file(run_dir)

            # 检查是否有 loss CSV
            has_loss_csv = self._check_loss_csv(run_dir)

            # 检查是否有指标文件
            has_metrics = self._check_metrics_files(run_dir)

            return TrainingRunInfo(
                name=run_dir.name,
                path=str(run_dir),
                creation_time=creation_time,
                model_files=model_files,
                config_path=config_path,
                has_loss_csv=has_loss_csv,
                has_metrics=has_metrics,
            )

        except Exception as e:
            warnings.warn(f"提取训练运行信息 {run_dir.name} 时出错: {e}")
            return None

    def _find_model_files(self, run_dir: Path) -> List[str]:
        """查找目录中的模型文件

        Args:
            run_dir: 训练运行目录路径

        Returns:
            模型文件路径列表
        """
        model_files = []
        try:
            for file_path in run_dir.iterdir():
                if (
                    file_path.is_file()
                    and file_path.suffix in self.supported_model_extensions
                ):
                    model_files.append(str(file_path))
        except Exception:
            pass
        return sorted(model_files)

    def _find_config_file(self, run_dir: Path) -> Optional[str]:
        """查找配置文件

        Args:
            run_dir: 训练运行目录路径

        Returns:
            配置文件路径，如果不存在则返回 None
        """
        config_path = run_dir / "config.json"
        if config_path.exists() and config_path.is_file():
            return str(config_path)
        return None

    def _check_loss_csv(self, run_dir: Path) -> bool:
        """检查是否存在损失 CSV 文件

        Args:
            run_dir: 训练运行目录路径

        Returns:
            如果存在损失 CSV 文件则返回 True
        """
        # 检查常见的损失 CSV 文件名
        loss_csv_patterns = [
            "loss_breakdown.csv",
            "loss.csv",
            "training_loss.csv",
            "training_losses.csv",
        ]

        for pattern in loss_csv_patterns:
            csv_path = run_dir / pattern
            if csv_path.exists() and csv_path.is_file():
                return True
        return False

    def _check_metrics_files(self, run_dir: Path) -> bool:
        """检查是否存在指标文件

        Args:
            run_dir: 训练运行目录路径

        Returns:
            如果存在指标文件则返回 True
        """
        # 检查常见的指标文件
        metrics_patterns = [
            "rmse_per_voltage.csv",
            "volume_trend_stats.csv",
            "metrics.csv",
            "training_metrics.csv",
        ]

        for pattern in metrics_patterns:
            metrics_path = run_dir / pattern
            if metrics_path.exists() and metrics_path.is_file():
                return True
        return False

    def get_training_config(
        self, run_info: TrainingRunInfo
    ) -> Optional[Dict[str, Any]]:
        """获取训练配置

        Args:
            run_info: 训练运行信息对象

        Returns:
            训练配置字典，如果读取失败则返回 None
        """
        if run_info.config_path is None:
            return None

        try:
            config_path = Path(run_info.config_path)
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            import warnings

            warnings.warn(f"读取配置文件 {run_info.config_path} 时出错: {e}")
            return None


class TrainingConfigParser:
    """训练配置解析器类"""

    @staticmethod
    def parse(config_path: str) -> Optional[Dict[str, Any]]:
        """解析训练配置文件

        Args:
            config_path: 配置文件路径

        Returns:
            结构化的配置字典，包含：
            - metadata: 元数据信息
            - model_architecture: 模型架构参数
            - training_parameters: 训练参数
            - physics_weights: 物理权重
            - data_config: 数据配置
            - dynamic_weight_config: 动态权重配置
            如果解析失败则返回 None
        """
        try:
            path = Path(config_path)
            if not path.exists():
                warnings.warn(f"配置文件不存在: {config_path}")
                return None

            # Check file size to prevent DoS
            if not check_file_size(str(path)):
                return None

            with open(path, "r", encoding="utf-8") as f:
                raw_config = json.load(f)

            # 提取并结构化配置
            structured_config = {
                "metadata": TrainingConfigParser._extract_metadata(raw_config),
                "model_architecture": TrainingConfigParser._extract_model_architecture(
                    raw_config
                ),
                "training_parameters": TrainingConfigParser._extract_training_parameters(
                    raw_config
                ),
                "physics_weights": TrainingConfigParser._extract_physics_weights(
                    raw_config
                ),
                "data_config": TrainingConfigParser._extract_data_config(raw_config),
                "dynamic_weight_config": TrainingConfigParser._extract_dynamic_weight_config(
                    raw_config
                ),
            }

            return structured_config

        except Exception as e:
            warnings.warn(f"解析配置文件 {config_path} 时出错: {e}")
            return None

    @staticmethod
    def _extract_metadata(config: Dict[str, Any]) -> Dict[str, Any]:
        """提取元数据信息

        Args:
            config: 原始配置字典

        Returns:
            元数据字典
        """
        metadata_section = config.get("metadata", {})
        return {
            "stage": metadata_section.get("stage"),
            "version": metadata_section.get("version"),
            "created_at": metadata_section.get("created_at"),
            "description": metadata_section.get("description"),
            "based_on": metadata_section.get("based_on", {}),
        }

    @staticmethod
    def _extract_model_architecture(config: Dict[str, Any]) -> Dict[str, Any]:
        """提取模型架构参数

        Args:
            config: 原始配置字典

        Returns:
            模型架构字典
        """
        model_section = config.get("model", {})
        return {
            "input_format": model_section.get("input_format"),
            "hidden_phi": model_section.get("hidden_phi", []),
            "hidden_vel": model_section.get("hidden_vel", []),
        }

    @staticmethod
    def _extract_training_parameters(config: Dict[str, Any]) -> Dict[str, Any]:
        """提取训练参数

        Args:
            config: 原始配置字典

        Returns:
            训练参数字典
        """
        training_section = config.get("training", {})
        return {
            "epochs": training_section.get("epochs"),
            "batch_size": training_section.get("batch_size"),
            "learning_rate": training_section.get("learning_rate"),
            "min_lr": training_section.get("min_lr"),
            "gradient_clip": training_section.get("gradient_clip"),
            "stage1_epochs": training_section.get("stage1_epochs"),
            "stage2_epochs": training_section.get("stage2_epochs"),
            "stage3_epochs": training_section.get("stage3_epochs"),
            "early_stop_patience": training_section.get("early_stop_patience"),
            "warmup_epochs": training_section.get("warmup_epochs"),
            "use_lbfgs": training_section.get("use_lbfgs"),
            "lbfgs_iter": training_section.get("lbfgs_iter"),
            "stage1_eta_tutor": config.get("stage1_eta_tutor"),
        }

    @staticmethod
    def _extract_physics_weights(config: Dict[str, Any]) -> Dict[str, Any]:
        """提取物理权重

        Args:
            config: 原始配置字典

        Returns:
            物理权重字典
        """
        physics_section = config.get("physics", {})
        return {
            "interface_weight": physics_section.get("interface_weight"),
            "ic_weight": physics_section.get("ic_weight"),
            "bc_weight": physics_section.get("bc_weight"),
            "continuity_weight": physics_section.get("continuity_weight"),
            "vof_weight": physics_section.get("vof_weight"),
            "ns_weight": physics_section.get("ns_weight"),
            "surface_tension_weight": physics_section.get("surface_tension_weight"),
            "sharpening_weight": physics_section.get("sharpening_weight"),
            "explicit_volume_weight": physics_section.get("explicit_volume_weight"),
        }

    @staticmethod
    def _extract_data_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """提取数据配置

        Args:
            config: 原始配置字典

        Returns:
            数据配置字典
        """
        data_section = config.get("data", {})
        return {
            "n_interface": data_section.get("n_interface"),
            "n_initial": data_section.get("n_initial"),
            "n_boundary": data_section.get("n_boundary"),
            "n_domain": data_section.get("n_domain"),
            "voltages": data_section.get("voltages", []),
            "times": data_section.get("times"),
        }

    @staticmethod
    def _extract_dynamic_weight_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """提取动态权重配置

        Args:
            config: 原始配置字典

        Returns:
            动态权重配置字典
        """
        dynamic_weight_section = config.get("dynamic_weight", {})
        return {
            "enable": dynamic_weight_section.get("enable"),
            "initial_weight": dynamic_weight_section.get("initial_weight"),
            "min_weight": dynamic_weight_section.get("min_weight"),
            "max_weight": dynamic_weight_section.get("max_weight"),
            "adjustment_strategy": dynamic_weight_section.get("adjustment_strategy"),
            "smoothing_factor": dynamic_weight_section.get("smoothing_factor"),
            "adjustment_interval": dynamic_weight_section.get("adjustment_interval"),
            "target_loss_ratio": dynamic_weight_section.get("target_loss_ratio"),
            "patience": dynamic_weight_section.get("patience"),
        }


class LossDataParser:
    """损失数据解析器类"""

    @staticmethod
    def parse(csv_path: str) -> Optional[pd.DataFrame]:
        """解析损失数据CSV文件

        Args:
            csv_path: 损失CSV文件路径

        Returns:
            包含所有损失数据的pandas DataFrame，包含列：
            - epoch: 训练轮次
            - stage: 训练阶段
            - loss_total: 总损失
            - lr: 学习率
            - low_voltage: 低电压损失
            - volume_conservation: 体积守恒损失
            - contact_angle: 接触角损失
            - phi_spatial: 空间phi损失
            - continuity: 连续性损失
            - interface: 界面损失
            如果解析失败则返回 None
        """
        try:
            path = Path(csv_path)
            if not path.exists():
                warnings.warn(f"损失CSV文件不存在: {csv_path}")
                return None

            # Check file size to prevent DoS
            if not check_file_size(str(path)):
                return None

            df = pd.read_csv(csv_path)

            # 验证必需的列是否存在
            required_columns = [
                "epoch",
                "stage",
                "loss_total",
                "lr",
                "low_voltage",
                "volume_conservation",
                "contact_angle",
                "phi_spatial",
                "continuity",
                "interface",
            ]

            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                warnings.warn(f"损失CSV文件缺少必需的列: {missing_columns}")
                return None

            return df

        except Exception as e:
            warnings.warn(f"解析损失CSV文件 {csv_path} 时出错: {e}")
            return None


class MetricsParser:
    """评估指标解析器类"""

    @staticmethod
    def parse_rmse(csv_path: str) -> Optional[pd.DataFrame]:
        """解析RMSE数据CSV文件

        Args:
            csv_path: RMSE CSV文件路径

        Returns:
            包含RMSE数据的pandas DataFrame，包含列：
            - Voltage: 电压值
            - RMSE: 均方根误差
            - Rating: 评估等级
            如果解析失败则返回 None
        """
        try:
            path = Path(csv_path)
            if not path.exists():
                warnings.warn(f"RMSE CSV文件不存在: {csv_path}")
                return None

            # Check file size to prevent DoS
            if not check_file_size(str(path)):
                return None

            df = pd.read_csv(csv_path)

            # 验证必需的列是否存在
            required_columns = ["Voltage", "RMSE", "Rating"]

            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                warnings.warn(f"RMSE CSV文件缺少必需的列: {missing_columns}")
                return None

            return df

        except Exception as e:
            warnings.warn(f"解析RMSE CSV文件 {csv_path} 时出错: {e}")
            return None

    @staticmethod
    def parse_volume_stats(csv_path: str) -> Optional[pd.DataFrame]:
        """解析体积统计数据CSV文件

        Args:
            csv_path: 体积统计CSV文件路径

        Returns:
            包含体积统计数据的pandas DataFrame，包含列：
            - stage: 训练阶段
            - start_epoch: 起始轮次
            - end_epoch: 结束轮次
            - mean: 平均值
            - std: 标准差
            - min: 最小值
            - max: 最大值
            - final: 最终值
            如果解析失败则返回 None
        """
        try:
            path = Path(csv_path)
            if not path.exists():
                warnings.warn(f"体积统计CSV文件不存在: {csv_path}")
                return None

            # Check file size to prevent DoS
            if not check_file_size(str(path)):
                return None

            df = pd.read_csv(csv_path)

            # 验证必需的列是否存在
            required_columns = [
                "stage",
                "start_epoch",
                "end_epoch",
                "mean",
                "std",
                "min",
                "max",
                "final",
            ]

            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                warnings.warn(f"体积统计CSV文件缺少必需的列: {missing_columns}")
                return None

            return df

        except Exception as e:
            warnings.warn(f"解析体积统计CSV文件 {csv_path} 时出错: {e}")
            return None


@dataclass
class ModelInfo:
    """模型信息数据类"""

    model_path: str
    model_type: str
    epoch: Optional[int]
    file_size: int
    architecture: Dict[str, Any]


class ModelLoader:
    """模型加载器类

    负责从训练输出目录加载 TwoPhasePINN 模型。

    支持的模型类型:
    - "best": 最佳模型 (best_model.pth)
    - "final": 最终模型 (final_model.pth)
    - "latest": 最新模型 (latest_model.pth)
    - "epoch_N": 指定轮次的模型 (best_model_epoch_N.pth)

    回退机制:
    1. 如果指定的模型不存在，尝试按顺序回退:
       - "best" → "latest" → "final" → 任意找到的 .pth 文件
       - "final" → "latest" → "best" → 任意找到的 .pth 文件
       - "latest" → "best" → "final" → 任意找到的 .pth 文件
       - "epoch_N" → "best" → "latest" → "final" → 任意找到的 .pth 文件
    2. 如果没有任何模型文件，返回 (None, None)
    """

    # 模型文件名模式
    MODEL_PATTERNS = {
        "best": "best_model.pth",
        "final": "final_model.pth",
        "latest": "latest_model.pth",
    }

    # 回退顺序
    FALLBACK_ORDER = {
        "best": ["latest", "final"],
        "final": ["latest", "best"],
        "latest": ["best", "final"],
    }

    @staticmethod
    def load_model(
        run_path: str,
        model_type: str = "best",
        device: str = "cpu",
        config_path: Optional[str] = None,
    ) -> tuple:
        """加载模型

        Args:
            run_path: 训练运行目录路径
            model_type: 模型类型 ("best", "final", "latest", "epoch_N")
            device: 加载设备 ("cpu" 或 "cuda")
            config_path: 配置文件路径（可选，默认从 run_path 中查找）

        Returns:
            (model, model_info) 元组:
            - model: 加载的 TwoPhasePINN 模型，如果加载失败则为 None
            - model_info: ModelInfo 对象，包含模型元数据，如果加载失败则为 None

        Raises:
            ImportError: 如果 torch 或 TwoPhasePINN 未安装
        """
        try:
            import torch
            from src.models.pinn_two_phase import TwoPhasePINN
        except ImportError as e:
            warnings.warn(f"导入必要的模块失败: {e}")
            return None, None

        run_dir = Path(run_path)
        if not run_dir.exists():
            warnings.warn(f"训练运行目录不存在: {run_path}")
            return None, None

        # 1. 解析模型类型并确定模型路径
        model_path, actual_model_type, epoch = ModelLoader._resolve_model_path(
            run_dir, model_type
        )

        if model_path is None:
            warnings.warn(f"无法找到任何模型文件: {run_path}")
            return None, None

        # 2. 加载配置并构建模型
        config = ModelLoader._load_config(run_dir, config_path)
        if config is None:
            warnings.warn(f"无法加载配置文件，使用默认配置")
            config = {}

        model_config = ModelLoader._build_model_config(config)

        # 3. 初始化模型
        try:
            model = TwoPhasePINN(config=model_config)
        except Exception as e:
            warnings.warn(f"初始化模型失败: {e}")
            return None, None

        # 4. 加载权重
        try:
            # 尝试加载完整 checkpoint（包含 model_state_dict 等元数据）
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)

            # 提取 state_dict
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                # 直接是 state_dict
                state_dict = checkpoint

            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()  # 设置为评估模式
        except Exception as e:
            warnings.warn(f"加载模型权重失败: {e}")
            return None, None

        # 5. 构建模型信息
        model_info = ModelInfo(
            model_path=str(model_path),
            model_type=actual_model_type,
            epoch=epoch,
            file_size=model_path.stat().st_size if model_path.exists() else 0,
            architecture={
                "hidden_phi": model_config.get("model", {}).get("hidden_phi", []),
                "hidden_vel": model_config.get("model", {}).get("hidden_vel", []),
            },
        )

        return model, model_info

    @staticmethod
    def _resolve_model_path(run_dir: Path, model_type: str) -> tuple:
        """解析模型路径

        Args:
            run_dir: 训练运行目录
            model_type: 模型类型

        Returns:
            (model_path, actual_model_type, epoch) 元组
        """
        epoch = None

        # 处理 epoch_N 格式
        if model_type.startswith("epoch_"):
            try:
                epoch = int(model_type.split("_")[1])
                pattern = f"best_model_epoch_{epoch}.pth"
                model_path = run_dir / pattern
                if model_path.exists():
                    return model_path, model_type, epoch
            except (IndexError, ValueError):
                pass

            # epoch_N 不存在，回退到 best
            warnings.warn(f"指定轮次的模型不存在: {model_type}，尝试回退")
            model_type = "best"

        # 检查标准模型类型
        if model_type in ModelLoader.MODEL_PATTERNS:
            pattern = ModelLoader.MODEL_PATTERNS[model_type]
            model_path = run_dir / pattern
            if model_path.exists():
                return model_path, model_type, epoch

            # 回退机制
            fallback_order = ModelLoader.FALLBACK_ORDER.get(model_type, [])
            for fallback_type in fallback_order:
                fallback_pattern = ModelLoader.MODEL_PATTERNS[fallback_type]
                fallback_path = run_dir / fallback_pattern
                if fallback_path.exists():
                    warnings.warn(f"模型 {model_type} 不存在，回退到 {fallback_type}")
                    return fallback_path, fallback_type, epoch

        # 最后尝试：查找任意 .pth 文件
        all_pth_files = sorted(run_dir.glob("*.pth"))
        if all_pth_files:
            warnings.warn(f"使用找到的第一个模型文件: {all_pth_files[0].name}")
            return all_pth_files[0], "fallback", epoch

        return None, None, None

    @staticmethod
    def _load_config(
        run_dir: Path, config_path: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """加载配置文件

        Args:
            run_dir: 训练运行目录
            config_path: 配置文件路径（可选）

        Returns:
            配置字典，如果加载失败则返回 None
        """
        if config_path:
            config_file = Path(config_path)
        else:
            config_file = run_dir / "config.json"

        if not config_file.exists():
            return None

        try:
            with open(config_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            warnings.warn(f"加载配置文件失败: {e}")
            return None

    @staticmethod
    def _build_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """构建模型配置

        从完整配置中提取模型架构参数。

        Args:
            config: 完整配置字典

        Returns:
            模型配置字典，包含 model.hidden_phi 和 model.hidden_vel
        """
        model_config = {}

        # 提取模型架构参数
        model_section = config.get("model", {})
        hidden_phi = model_section.get("hidden_phi", [128, 128, 64, 32])
        hidden_vel = model_section.get("hidden_vel", [64, 64, 32])

        model_config["model"] = {
            "hidden_phi": hidden_phi,
            "hidden_vel": hidden_vel,
        }

        return model_config

    @staticmethod
    def list_available_models(run_path: str) -> List[Dict[str, Any]]:
        """列出训练运行目录中所有可用的模型

        Args:
            run_path: 训练运行目录路径

        Returns:
            模型信息列表，每个元素包含:
            - name: 模型文件名
            - path: 模型文件路径
            - type: 模型类型 (best, final, latest, epoch_N, unknown)
            - size: 文件大小 (bytes)
            - epoch: 轮次（如果是 epoch_N 类型）
        """
        run_dir = Path(run_path)
        if not run_dir.exists():
            return []

        models = []

        try:
            for pth_file in sorted(run_dir.glob("*.pth")):
                model_info = {
                    "name": pth_file.name,
                    "path": str(pth_file),
                    "type": "unknown",
                    "size": pth_file.stat().st_size,
                    "epoch": None,
                }

                # 判断模型类型
                if pth_file.name == "best_model.pth":
                    model_info["type"] = "best"
                elif pth_file.name == "final_model.pth":
                    model_info["type"] = "final"
                elif pth_file.name == "latest_model.pth":
                    model_info["type"] = "latest"
                elif pth_file.name.startswith("best_model_epoch_"):
                    model_info["type"] = "epoch_N"
                    try:
                        model_info["epoch"] = int(
                            pth_file.stem.replace("best_model_epoch_", "")
                        )
                    except ValueError:
                        pass

                models.append(model_info)
        except Exception as e:
            warnings.warn(f"列出模型文件时出错: {e}")

        return models


class TrainingOutputAnalyzer:
    """训练输出分析器 UI 组件

    提供 Streamlit 界面用于分析训练输出目录中的训练运行。
    包含 5 个标签页：训练曲线、评估指标、配置信息、可视化图集、实时推理。
    """

    def __init__(self, train_outputs_dir: str = "outputs/train"):
        """初始化训练输出分析器

        Args:
            train_outputs_dir: 训练输出目录路径，默认为 outputs/train
        """
        self.train_outputs_dir = train_outputs_dir
        self.scanner = TrainingOutputScanner(train_outputs_dir)
        self._runs_cache: Optional[List[TrainingRunInfo]] = None

    def _get_runs(self, force_refresh: bool = False) -> List[TrainingRunInfo]:
        """获取训练运行列表（带缓存）

        Args:
            force_refresh: 是否强制刷新缓存

        Returns:
            训练运行信息列表
        """
        if self._runs_cache is None or force_refresh:
            self._runs_cache = self.scanner.scan_training_outputs()
        return self._runs_cache

    def render(self) -> None:
        """渲染训练输出分析器 UI

        主入口方法，渲染完整的分析界面，包括：
        - 训练运行选择器
        - 5 个分析标签页
        """
        try:
            import streamlit as st
        except ImportError:
            raise ImportError("Streamlit 未安装。请运行: pip install streamlit")

        st.title("📊 训练输出分析器")
        st.markdown("分析和可视化训练输出目录中的训练运行。")

        runs = self._get_runs()

        if not runs:
            st.warning(f"在 `{self.train_outputs_dir}` 目录中未找到训练运行。")
            st.info("请确保已完成至少一次训练，或检查目录路径是否正确。")
            return

        st.sidebar.subheader("📁 训练运行选择")

        # 使用 session state 跟踪选中的运行
        if "selected_run_index" not in st.session_state:
            st.session_state.selected_run_index = 0

        # 显示训练运行卡片列表
        st.sidebar.markdown("**点击卡片选择训练运行：**")

        for idx, run in enumerate(runs):
            # 创建卡片样式
            is_selected = idx == st.session_state.selected_run_index
            card_border = "2px solid #1f77b4" if is_selected else "1px solid #e0e0e0"
            card_bg = "#f0f7ff" if is_selected else "#ffffff"

            # 格式化日期
            date_str = run.creation_time.strftime("%Y-%m-%d")
            time_str = run.creation_time.strftime("%H:%M")

            # 文件状态指示器
            config_status = "🟢" if run.config_path else "🔴"
            loss_status = "🟢" if run.has_loss_csv else "🔴"
            metrics_status = "🟢" if run.has_metrics else "🔴"

            # 构建卡片 HTML
            card_html = f"""
            <div style="
                border: {card_border};
                border-radius: 8px;
                padding: 12px;
                margin: 8px 0;
                background-color: {card_bg};
                cursor: pointer;
                transition: all 0.2s;
            ">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: bold; font-size: 14px;">{run.name}</span>
                    <span style="color: #666; font-size: 12px;">{date_str}</span>
                </div>
                <div style="margin-top: 8px; font-size: 12px; color: #666;">
                    <span>🕐 {time_str}</span>
                    <span style="margin-left: 12px;">📦 {len(run.model_files)} 模型</span>
                </div>
                <div style="margin-top: 6px; font-size: 11px;">
                    <span title="配置文件">{config_status} 配置</span>
                    <span style="margin-left: 8px;" title="损失数据">{loss_status} 损失</span>
                    <span style="margin-left: 8px;" title="指标数据">{metrics_status} 指标</span>
                </div>
            </div>
            """

            # 显示卡片
            st.sidebar.markdown(card_html, unsafe_allow_html=True)

            # 创建不可见的按钮用于点击处理
            if st.sidebar.button(
                f"选择: {run.name}",
                key=f"select_run_{idx}",
                use_container_width=True,
                type="primary" if is_selected else "secondary",
            ):
                st.session_state.selected_run_index = idx
                st.rerun()

        selected_run = runs[st.session_state.selected_run_index]

        # 显示选中运行的详细信息（折叠面板）
        with st.sidebar.expander("📋 选中运行详情", expanded=True):
            st.markdown(f"**名称:** `{selected_run.name}`")
            st.markdown(f"**路径:** `{selected_run.path}`")
            st.markdown(
                f"**创建时间:** {selected_run.creation_time.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            st.markdown(f"**模型文件数:** {len(selected_run.model_files)}")
            st.markdown(f"**配置文件:** {'✅' if selected_run.config_path else '❌'}")
            st.markdown(f"**损失数据:** {'✅' if selected_run.has_loss_csv else '❌'}")
            st.markdown(f"**指标数据:** {'✅' if selected_run.has_metrics else '❌'}")

        if st.sidebar.button("🔄 刷新列表", use_container_width=True):
            self._runs_cache = None
            st.session_state.selected_run_index = 0
            st.rerun()

        tab_names = [
            "📈 训练曲线",
            "📊 评估指标",
            "⚙️ 配置信息",
            "🖼️ 可视化图集",
            "🔮 实时推理",
        ]
        tabs = st.tabs(tab_names)

        with tabs[0]:
            self._render_training_curves_tab(selected_run)

        with tabs[1]:
            self._render_metrics_tab(selected_run)

        with tabs[2]:
            self._render_config_tab(selected_run)

        with tabs[3]:
            self._render_visualization_tab(selected_run)

        with tabs[4]:
            self._render_inference_tab(selected_run)

    def _render_training_curves_tab(self, run: TrainingRunInfo) -> None:
        """渲染训练曲线标签页

        Args:
            run: 选中的训练运行信息
        """
        import streamlit as st

        st.markdown("### 📈 训练曲线")

        if not run.has_loss_csv:
            st.warning("此训练运行没有损失数据文件。")
            return

        # 查找并解析损失 CSV 文件
        loss_csv_path = self._find_loss_csv(run.path)
        if not loss_csv_path:
            st.error("无法找到损失 CSV 文件。")
            return

        df = LossDataParser.parse(loss_csv_path)
        if df is None:
            st.error("无法解析损失数据。")
            return

        st.success(f"✅ 成功加载 {len(df)} 个训练周期的损失数据")

        # 图表控制面板
        with st.expander("📊 图表控制", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                show_log_scale = st.checkbox("对数刻度 (Total Loss)", value=True)
            with col2:
                show_stage_markers = st.checkbox("显示阶段标记", value=True)
            with col3:
                smooth_window = st.slider("平滑窗口", 1, 50, 10)

        # Chart 1: Total Loss over Epochs
        st.markdown("#### 1️⃣ 总损失曲线")
        fig1 = self._create_total_loss_chart(
            df, show_log_scale, show_stage_markers, smooth_window
        )
        st.plotly_chart(fig1, use_container_width=True)

        # Chart 2: Loss Components Breakdown
        st.markdown("#### 2️⃣ 损失组件分解")
        fig2 = self._create_loss_components_chart(df, smooth_window)
        st.plotly_chart(fig2, use_container_width=True)

        # Chart 3: Learning Rate over Epochs
        st.markdown("#### 3️⃣ 学习率变化")
        fig3 = self._create_lr_chart(df, show_stage_markers)
        st.plotly_chart(fig3, use_container_width=True)

        # 数据摘要
        with st.expander("📋 数据摘要", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("最终总损失", f"{float(df['loss_total'].iloc[-1]):.4e}")
                st.metric("最小总损失", f"{float(df['loss_total'].min()):.4e}")
            with col2:
                st.metric("最终学习率", f"{float(df['lr'].iloc[-1]):.4e}")
                st.metric("平均学习率", f"{float(df['lr'].mean()):.4e}")
            with col3:
                st.metric("训练阶段数", int(df["stage"].nunique()))
                st.metric("总轮次", int(len(df)))

    def _find_loss_csv(self, run_path: str) -> Optional[str]:
        """查找训练运行目录中的损失 CSV 文件

        Args:
            run_path: 训练运行目录路径

        Returns:
            损失 CSV 文件路径，如果未找到则返回 None
        """
        loss_csv_patterns = [
            "loss_breakdown.csv",
            "loss.csv",
            "training_loss.csv",
            "training_losses.csv",
        ]

        run_dir = Path(run_path)
        for pattern in loss_csv_patterns:
            csv_path = run_dir / pattern
            if csv_path.exists() and csv_path.is_file():
                return str(csv_path)
        return None

    def _create_total_loss_chart(
        self,
        df: pd.DataFrame,
        show_log_scale: bool = True,
        show_stage_markers: bool = True,
        smooth_window: int = 10,
    ) -> "go.Figure":
        """创建总损失曲线图表

        Args:
            df: 损失数据 DataFrame
            show_log_scale: 是否使用对数刻度
            show_stage_markers: 是否显示阶段标记
            smooth_window: 平滑窗口大小

        Returns:
            Plotly Figure 对象
        """
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # 计算平滑后的损失（使用指数移动平均）
        alpha = 2 / (smooth_window + 1)
        df_smooth = df.copy()
        df_smooth["loss_total_smooth"] = (
            df["loss_total"].ewm(alpha=alpha, adjust=False).mean()
        )

        fig = make_subplots(specs=[[{"secondary_y": False}]])

        # 添加原始数据（半透明）
        fig.add_trace(
            go.Scatter(
                x=df["epoch"],
                y=df["loss_total"],
                mode="lines",
                name="原始损失",
                line=dict(color="rgba(150, 150, 150, 0.3)", width=1),
                hovertemplate="轮次: %{x}<br>损失: %{y:.4e}<extra></extra>",
            )
        )

        # 添加平滑后的数据
        fig.add_trace(
            go.Scatter(
                x=df_smooth["epoch"],
                y=df_smooth["loss_total_smooth"],
                mode="lines",
                name="平滑损失",
                line=dict(color="#1f77b4", width=2.5),
                hovertemplate="轮次: %{x}<br>平滑损失: %{y:.4e}<extra></extra>",
            )
        )

        # 添加阶段标记
        if show_stage_markers:
            stage_colors = {1: "#ff7f0e", 2: "#2ca02c", 3: "#d62728"}
            stage_names = {1: "阶段1 (几何)", 2: "阶段2 (运动学)", 3: "阶段3 (物理)"}

            for stage in sorted(df["stage"].unique()):
                stage_df = df[df["stage"] == stage]
                if len(stage_df) > 0:
                    first_epoch = stage_df["epoch"].iloc[0]
                    last_epoch = stage_df["epoch"].iloc[-1]

                    # 阶段开始标记
                    fig.add_vline(
                        x=first_epoch,
                        line=dict(
                            color=stage_colors.get(stage, "gray"), dash="dash", width=1
                        ),
                        annotation_text=stage_names.get(stage, f"阶段{stage}"),
                        annotation_position="top",
                        annotation=dict(
                            font_size=10, font_color=stage_colors.get(stage, "gray")
                        ),
                    )

        # 设置布局
        fig.update_layout(
            title="总损失随训练轮次变化",
            xaxis_title="训练轮次 (Epoch)",
            yaxis_title="总损失 (Total Loss)",
            hovermode="x unified",
            showlegend=True,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            template="plotly_white",
        )

        if show_log_scale:
            fig.update_yaxes(type="log")

        return fig

    def _create_loss_components_chart(
        self, df: pd.DataFrame, smooth_window: int = 10
    ) -> "go.Figure":
        """创建损失组件分解图表

        Args:
            df: 损失数据 DataFrame
            smooth_window: 平滑窗口大小

        Returns:
            Plotly Figure 对象
        """
        import plotly.graph_objects as go

        # 定义损失组件及其颜色
        loss_components = {
            "low_voltage": {"name": "低电压损失", "color": "#1f77b4"},
            "volume_conservation": {"name": "体积守恒", "color": "#ff7f0e"},
            "contact_angle": {"name": "接触角", "color": "#2ca02c"},
            "phi_spatial": {"name": "空间 Phi", "color": "#d62728"},
            "continuity": {"name": "连续性", "color": "#9467bd"},
            "interface": {"name": "界面", "color": "#8c564b"},
        }

        alpha = 2 / (smooth_window + 1)

        fig = go.Figure()

        for col, config in loss_components.items():
            if col in df.columns:
                # 平滑数据
                smooth_values = df[col].ewm(alpha=alpha, adjust=False).mean()

                fig.add_trace(
                    go.Scatter(
                        x=df["epoch"],
                        y=smooth_values,
                        mode="lines",
                        name=config["name"],
                        line=dict(color=config["color"], width=2),
                        hovertemplate=f"{config['name']}: %{{y:.4e}}<extra></extra>",
                    )
                )

        fig.update_layout(
            title="损失组件分解（平滑后）",
            xaxis_title="训练轮次 (Epoch)",
            yaxis_title="损失值",
            hovermode="x unified",
            showlegend=True,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            template="plotly_white",
        )

        return fig

    def _create_lr_chart(
        self, df: pd.DataFrame, show_stage_markers: bool = True
    ) -> "go.Figure":
        """创建学习率变化图表

        Args:
            df: 损失数据 DataFrame
            show_stage_markers: 是否显示阶段标记

        Returns:
            Plotly Figure 对象
        """
        import plotly.graph_objects as go

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=df["epoch"],
                y=df["lr"],
                mode="lines",
                name="学习率",
                line=dict(color="#17becf", width=2.5),
                fill="tozeroy",
                fillcolor="rgba(23, 190, 207, 0.1)",
                hovertemplate="轮次: %{x}<br>学习率: %{y:.4e}<extra></extra>",
            )
        )

        # 添加阶段标记
        if show_stage_markers:
            stage_colors = {1: "#ff7f0e", 2: "#2ca02c", 3: "#d62728"}

            for stage in sorted(df["stage"].unique()):
                stage_df = df[df["stage"] == stage]
                if len(stage_df) > 0:
                    first_epoch = stage_df["epoch"].iloc[0]
                    fig.add_vline(
                        x=first_epoch,
                        line=dict(
                            color=stage_colors.get(stage, "gray"), dash="dash", width=1
                        ),
                    )

        fig.update_layout(
            title="学习率随训练轮次变化",
            xaxis_title="训练轮次 (Epoch)",
            yaxis_title="学习率 (Learning Rate)",
            hovermode="x unified",
            showlegend=True,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            template="plotly_white",
            yaxis=dict(type="log"),
        )

        return fig

    def _render_metrics_tab(self, run: TrainingRunInfo) -> None:
        """渲染评估指标标签页

        Args:
            run: 选中的训练运行信息
        """
        import streamlit as st
        import plotly.graph_objects as go

        st.markdown("### 📊 评估指标")

        if not run.has_metrics:
            st.warning("此训练运行没有指标数据文件。")
            return

        # 查找指标文件
        rmse_path = self._find_metrics_file(run.path, "rmse_per_voltage.csv")
        volume_path = self._find_metrics_file(run.path, "volume_trend_stats.csv")

        # 渲染摘要信息卡片
        self._render_metrics_summary_cards(run, rmse_path, volume_path)

        # 如果有 RMSE 数据，渲染图表
        if rmse_path:
            df_rmse = MetricsParser.parse_rmse(rmse_path)
            if df_rmse is not None:
                self._render_rmse_chart(df_rmse)
            else:
                st.error("无法解析 RMSE 数据。")
        else:
            st.info("未找到 RMSE 数据文件 (rmse_per_voltage.csv)")

        # 如果有体积统计数据，渲染表格
        if volume_path:
            df_volume = MetricsParser.parse_volume_stats(volume_path)
            if df_volume is not None:
                self._render_volume_stats_table(df_volume)
            else:
                st.error("无法解析体积统计数据。")
        else:
            st.info("未找到体积统计数据文件 (volume_trend_stats.csv)")

    def _find_metrics_file(self, run_path: str, filename: str) -> Optional[str]:
        """在训练运行目录中查找指标文件

        Args:
            run_path: 训练运行目录路径
            filename: 要查找的文件名

        Returns:
            文件路径，如果未找到则返回 None
        """
        file_path = Path(run_path) / filename
        if file_path.exists() and file_path.is_file():
            return str(file_path)
        return None

    def _render_metrics_summary_cards(
        self, run: TrainingRunInfo, rmse_path: Optional[str], volume_path: Optional[str]
    ) -> None:
        """渲染指标摘要信息卡片

        Args:
            run: 训练运行信息
            rmse_path: RMSE 文件路径
            volume_path: 体积统计文件路径
        """
        import streamlit as st

        # 创建三列布局
        col1, col2, col3 = st.columns(3)

        with col1:
            # 运行名称卡片
            st.markdown(
                """
                <div style="
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    border-radius: 12px;
                    padding: 20px;
                    color: white;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                ">
                    <div style="font-size: 12px; opacity: 0.8; margin-bottom: 8px;">📁 训练运行</div>
                    <div style="font-size: 18px; font-weight: bold;">{}</div>
                </div>
                """.format(run.name),
                unsafe_allow_html=True,
            )

        with col2:
            # 创建时间卡片
            time_str = run.creation_time.strftime("%Y-%m-%d %H:%M")
            st.markdown(
                """
                <div style="
                    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                    border-radius: 12px;
                    padding: 20px;
                    color: white;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                ">
                    <div style="font-size: 12px; opacity: 0.8; margin-bottom: 8px;">🕐 创建时间</div>
                    <div style="font-size: 18px; font-weight: bold;">{}</div>
                </div>
                """.format(time_str),
                unsafe_allow_html=True,
            )

        with col3:
            # 模型数量卡片
            st.markdown(
                """
                <div style="
                    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                    border-radius: 12px;
                    padding: 20px;
                    color: white;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                ">
                    <div style="font-size: 12px; opacity: 0.8; margin-bottom: 8px;">📦 模型文件</div>
                    <div style="font-size: 24px; font-weight: bold;">{}</div>
                    <div style="font-size: 12px; opacity: 0.8;">个检查点</div>
                </div>
                """.format(len(run.model_files)),
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # 数据可用性状态
        col1, col2 = st.columns(2)
        with col1:
            if rmse_path:
                st.success("✅ RMSE 数据可用")
            else:
                st.error("❌ RMSE 数据不可用")
        with col2:
            if volume_path:
                st.success("✅ 体积统计数据可用")
            else:
                st.error("❌ 体积统计数据不可用")

    def _render_rmse_chart(self, df: pd.DataFrame) -> None:
        """渲染 RMSE 柱状图

        Args:
            df: RMSE 数据 DataFrame
        """
        import streamlit as st
        import plotly.graph_objects as go

        st.markdown("#### 📈 RMSE 按电压分布")

        # 定义评级颜色映射
        rating_colors = {
            "Excellent": "#2ecc71",  # 绿色
            "Acceptable": "#f39c12",  # 黄色/橙色
            "Poor": "#e74c3c",  # 红色
        }

        # 为每个评级创建颜色列表
        colors = [rating_colors.get(rating, "#95a5a6") for rating in df["Rating"]]

        # 创建柱状图
        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=df["Voltage"].astype(str) + "V",
                y=df["RMSE"],
                marker_color=colors,
                text=df["Rating"],
                textposition="auto",
                hovertemplate="<b>%{x}</b><br>RMSE: %{y:.4f}<br>评级: %{text}<extra></extra>",
            )
        )

        # 添加参考线
        fig.add_hline(
            y=0.05,
            line_dash="dash",
            line_color="#2ecc71",
            annotation_text="Excellent 阈值 (0.05)",
            annotation_position="right",
        )
        fig.add_hline(
            y=0.10,
            line_dash="dash",
            line_color="#f39c12",
            annotation_text="Acceptable 阈值 (0.10)",
            annotation_position="right",
        )

        fig.update_layout(
            title="各电压下的 RMSE 误差",
            xaxis_title="电压 (V)",
            yaxis_title="RMSE",
            template="plotly_white",
            showlegend=False,
            height=400,
        )

        st.plotly_chart(fig, use_container_width=True)

        # 显示统计摘要
        col1, col2, col3 = st.columns(3)
        with col1:
            avg_rmse = df["RMSE"].mean()
            st.metric("平均 RMSE", f"{avg_rmse:.4f}")
        with col2:
            min_rmse = df["RMSE"].min()
            st.metric("最小 RMSE", f"{min_rmse:.4f}")
        with col3:
            max_rmse = df["RMSE"].max()
            st.metric("最大 RMSE", f"{max_rmse:.4f}")

        # 评级分布
        st.markdown("**评级分布:**")
        rating_counts = df["Rating"].value_counts()
        cols = st.columns(len(rating_counts))
        for idx, (rating, count) in enumerate(rating_counts.items()):
            color = rating_colors.get(rating, "#95a5a6")
            with cols[idx]:
                st.markdown(
                    f"""
                    <div style="
                        background-color: {color}20;
                        border: 2px solid {color};
                        border-radius: 8px;
                        padding: 10px;
                        text-align: center;
                    ">
                        <div style="font-size: 24px; font-weight: bold; color: {color};">{count}</div>
                        <div style="font-size: 12px; color: #666;">{rating}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    def _render_volume_stats_table(self, df: pd.DataFrame) -> None:
        """渲染体积统计数据表格

        Args:
            df: 体积统计数据 DataFrame
        """
        import streamlit as st

        st.markdown("---")
        st.markdown("#### 📊 体积守恒统计")

        # 阶段名称映射
        stage_names = {
            "stage1": "阶段 1 (几何)",
            "stage2": "阶段 2 (运动学)",
            "stage3": "阶段 3 (物理)",
            1: "阶段 1 (几何)",
            2: "阶段 2 (运动学)",
            3: "阶段 3 (物理)",
        }

        # 创建显示用的 DataFrame 副本
        display_df = df.copy()

        # 转换阶段名称为中文
        if "stage" in display_df.columns:
            display_df["阶段"] = display_df["stage"].map(
                lambda x: stage_names.get(x, f"阶段 {x}")
            )

        # 重命名列
        column_mapping = {
            "start_epoch": "起始轮次",
            "end_epoch": "结束轮次",
            "mean": "平均值",
            "std": "标准差",
            "min": "最小值",
            "max": "最大值",
            "final": "最终值",
        }
        display_df = display_df.rename(columns=column_mapping)

        # 选择要显示的列
        display_columns = [
            "阶段",
            "起始轮次",
            "结束轮次",
            "平均值",
            "标准差",
            "最小值",
            "最大值",
            "最终值",
        ]
        display_columns = [col for col in display_columns if col in display_df.columns]

        # 格式化数值列
        numeric_columns = ["平均值", "标准差", "最小值", "最大值", "最终值"]
        for col in numeric_columns:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:.4e}" if pd.notna(x) else "N/A"
                )

        # 显示表格
        st.dataframe(
            display_df[display_columns],
            use_container_width=True,
            hide_index=True,
        )

        # 显示趋势分析
        st.markdown("**体积误差趋势分析:**")
        if "final" in df.columns:
            final_values = df["final"].values
            stages = df["stage"].values if "stage" in df.columns else range(len(df))

            # 判断趋势
            if len(final_values) >= 2:
                if final_values[-1] < final_values[0]:
                    trend = "📉 体积误差呈下降趋势"
                    trend_color = "#2ecc71"
                elif final_values[-1] > final_values[0]:
                    trend = "📈 体积误差呈上升趋势"
                    trend_color = "#e74c3c"
                else:
                    trend = "➡️ 体积误差保持稳定"
                    trend_color = "#f39c12"

                st.markdown(
                    f"""
                    <div style="
                        background-color: {trend_color}20;
                        border-left: 4px solid {trend_color};
                        padding: 12px;
                        border-radius: 4px;
                    ">
                        <span style="color: {trend_color}; font-weight: bold;">{trend}</span><br>
                        <span style="color: #666; font-size: 12px;">
                            初始阶段误差: {final_values[0]:.4e} → 最终阶段误差: {final_values[-1]:.4e}
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    def _render_config_tab(self, run: TrainingRunInfo) -> None:
        """渲染配置信息标签页

        Args:
            run: 选中的训练运行信息
        """
        import streamlit as st

        st.markdown("### ⚙️ 配置信息")

        if not run.config_path:
            st.warning("此训练运行没有配置文件。")
            return

        # 解析配置文件
        config = TrainingConfigParser.parse(run.config_path)
        if config is None:
            st.error("无法解析配置文件。")
            return

        st.success("✅ 配置文件解析成功！")
        st.markdown(f"**配置文件路径:** `{run.config_path}`")

        # 定义各部分的显示名称和图标
        section_info = {
            "metadata": {"title": "📋 元数据信息", "expanded": True},
            "model_architecture": {"title": "🏗️ 模型架构", "expanded": True},
            "training_parameters": {"title": "⚡ 训练参数", "expanded": True},
            "physics_weights": {"title": "⚛️ 物理权重", "expanded": False},
            "data_config": {"title": "📊 数据配置", "expanded": False},
            "dynamic_weight_config": {"title": "🔄 动态权重配置", "expanded": False},
        }

        # 渲染各部分
        for section_key, info in section_info.items():
            section_data = config.get(section_key)
            if section_data is None:
                continue

            with st.expander(info["title"], expanded=info["expanded"]):
                st.json(section_data)

    def _render_visualization_tab(self, run: TrainingRunInfo) -> None:
        """渲染可视化图集标签页

        显示训练输出目录中的所有 PNG/PDF 图像文件，按类型分组：
        - 训练曲线: training_curve, learning_curve, loss_components, loss_fraction
        - 3D 可视化: interface_3d, phi_grid_evolution, z_profile, mass_conservation, volume_trend
        - 仪表板: pro_dashboard, response_times, rmse_per_voltage, dynamic_curves

        Args:
            run: 选中的训练运行信息
        """
        import streamlit as st

        st.markdown("### 🖼️ 可视化图集")
        st.markdown("浏览训练过程中生成的所有可视化图像。")

        # 查找所有图像文件
        image_files = self._find_image_files(run.path)

        if not image_files:
            st.warning("此训练运行目录中没有找到 PNG 或 PDF 图像文件。")
            return

        st.success(f"✅ 找到 {len(image_files)} 个图像文件")

        # 按类型分组
        grouped_images = self._group_images_by_type(image_files)

        # 定义分组显示配置
        group_configs = {
            "training_curves": {
                "title": "📈 训练曲线",
                "description": "训练过程中的损失曲线和学习率变化",
                "expanded": True,
            },
            "3d_visualization": {
                "title": "🎨 3D 可视化",
                "description": "3D 界面、网格演化和物理场分布",
                "expanded": True,
            },
            "dashboard": {
                "title": "📊 仪表板",
                "description": "综合性能仪表板和响应时间分析",
                "expanded": False,
            },
            "other": {
                "title": "📁 其他图像",
                "description": "其他生成的可视化图像",
                "expanded": False,
            },
        }

        # 渲染每个分组
        for group_key, config in group_configs.items():
            images = grouped_images.get(group_key, [])
            if images:
                with st.expander(
                    f"{config['title']} ({len(images)} 张)", expanded=config["expanded"]
                ):
                    st.markdown(f"*{config['description']}*")
                    self._render_image_grid(images)

    def _find_image_files(self, run_path: str) -> List[str]:
        """查找训练运行目录中的所有图像文件

        Args:
            run_path: 训练运行目录路径

        Returns:
            图像文件路径列表（PNG 和 PDF）
        """
        image_files = []
        run_dir = Path(run_path)

        if not run_dir.exists():
            return image_files

        try:
            for file_path in run_dir.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in [
                    ".png",
                    ".pdf",
                ]:
                    image_files.append(str(file_path))
        except Exception as e:
            warnings.warn(f"查找图像文件时出错: {e}")

        return sorted(image_files)

    def _group_images_by_type(self, image_files: List[str]) -> Dict[str, List[str]]:
        """按类型分组图像文件

        Args:
            image_files: 图像文件路径列表

        Returns:
            按类型分组的字典
        """
        groups = {
            "training_curves": [],
            "3d_visualization": [],
            "dashboard": [],
            "other": [],
        }

        # 定义文件名模式映射
        patterns = {
            "training_curves": [
                "training_curve",
                "learning_curve",
                "loss_components",
                "loss_fraction",
                "training_curves",
            ],
            "3d_visualization": [
                "interface_3d",
                "phi_grid_evolution",
                "z_profile",
                "mass_conservation",
                "volume_trend",
            ],
            "dashboard": [
                "pro_dashboard",
                "response_times",
                "rmse_per_voltage",
                "dynamic_curves",
            ],
        }

        for image_path in image_files:
            filename = Path(image_path).stem.lower()
            assigned = False

            for group_key, group_patterns in patterns.items():
                for pattern in group_patterns:
                    if pattern in filename:
                        groups[group_key].append(image_path)
                        assigned = True
                        break
                if assigned:
                    break

            if not assigned:
                groups["other"].append(image_path)

        return groups

    def _render_image_grid(self, image_files: List[str], columns: int = 2) -> None:
        """渲染图像网格

        Args:
            image_files: 图像文件路径列表
            columns: 每行显示的列数，默认为 2
        """
        import streamlit as st

        if not image_files:
            return

        # 批量显示图像
        for i in range(0, len(image_files), columns):
            cols = st.columns(columns)
            for j, col in enumerate(cols):
                idx = i + j
                if idx < len(image_files):
                    image_path = image_files[idx]
                    filename = Path(image_path).name

                    with col:
                        # 使用 st.image 显示图像（支持 PNG 和 PDF）
                        try:
                            st.image(
                                image_path,
                                caption=filename,
                                use_container_width=True,
                            )
                        except Exception as e:
                            st.error(f"无法加载图像: {filename}")
                            st.markdown(f"路径: `{image_path}`")
                            if st.checkbox(
                                f"显示错误详情: {filename}", key=f"err_{idx}"
                            ):
                                st.code(str(e))

    def _render_inference_tab(self, run: TrainingRunInfo) -> None:
        """渲染实时推理标签页

        提供6D Triad输入界面，支持对选中的训练模型进行实时推理。
        包含三个子标签页：单点推理、轨迹预测、切片可视化
        6D输入: [x, y, z, V_from, V_to, t_since]
        5D输出: [u, v, w, p, phi]

        Args:
            run: 选中的训练运行信息
        """
        import streamlit as st

        st.markdown("### 🔮 实时推理")
        st.markdown("使用6D Triad输入对训练好的PINN模型进行实时推理。")

        # 导入物理参数
        try:
            from src.config import PHYSICS

            Lx = PHYSICS["Lx"]
            Ly = PHYSICS["Ly"]
            Lz = PHYSICS["Lz"]
            V_max = PHYSICS["V_max"]
            t_max = PHYSICS["t_max"]
        except ImportError:
            st.error("无法导入物理参数配置。请确保 src.config 模块可用。")
            return

        # 检查是否有模型文件
        if not run.model_files:
            st.warning("⚠️ 此训练运行没有可用的模型文件。")
            return

        # 模型选择 (在子标签页之外，所有子标签页共用)
        st.markdown("#### 📦 模型选择")
        available_models = ModelLoader.list_available_models(run.path)

        if not available_models:
            st.error("无法列出可用模型。")
            return

        model_options = {
            f"{m['type']} ({m['name']})": m["type"] for m in available_models
        }
        selected_model_label = st.selectbox(
            "选择模型",
            options=list(model_options.keys()),
            index=0,
            key="inference_model_select",
        )
        selected_model_type = model_options[selected_model_label]

        # 显示模型信息
        selected_model_info = next(
            (m for m in available_models if m["type"] == selected_model_type), None
        )
        if selected_model_info:
            size_mb = selected_model_info["size"] / (1024 * 1024)
            st.caption(
                f"模型大小: {size_mb:.2f} MB | 路径: `{selected_model_info['path']}`"
            )

        st.markdown("---")

        # 创建子标签页
        tab_single, tab_trajectory, tab_slice = st.tabs(
            ["📍 单点推理", "📈 轨迹预测", "🔲 切片可视化"]
        )

        # ============ 子标签页 1: 单点推理 ============
        with tab_single:
            self._render_single_point_inference(
                selected_model_info, Lx, Ly, Lz, V_max, t_max
            )

        # ============ 子标签页 2: 轨迹预测 ============
        with tab_trajectory:
            self._render_trajectory_prediction(
                selected_model_info, Lx, Ly, Lz, V_max, t_max
            )

        # ============ 子标签页 3: 切片可视化 ============
        with tab_slice:
            self._render_slice_visualization(
                selected_model_info, Lx, Ly, Lz, V_max, t_max
            )

    def _render_single_point_inference(
        self, selected_model_info, Lx, Ly, Lz, V_max, t_max
    ):
        """渲染单点推理子标签页"""
        import streamlit as st

        st.markdown("#### 🎛️ 6D Triad 输入参数")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**空间坐标 (m)**")
            x_input = st.number_input(
                "x",
                min_value=0.0,
                max_value=float(Lx),
                value=float(Lx / 2),
                format="%.2e",
                help=f"x坐标范围: 0 ~ {Lx:.2e} m (像素宽度)",
                key="sp_x",
            )
            y_input = st.number_input(
                "y",
                min_value=0.0,
                max_value=float(Ly),
                value=float(Ly / 2),
                format="%.2e",
                help=f"y坐标范围: 0 ~ {Ly:.2e} m (像素长度)",
                key="sp_y",
            )
            z_input = st.number_input(
                "z",
                min_value=0.0,
                max_value=float(Lz),
                value=float(Lz / 2),
                format="%.2e",
                help=f"z坐标范围: 0 ~ {Lz:.2e} m (围堰高度)",
                key="sp_z",
            )

        with col2:
            st.markdown("**电压 (V)**")
            V_from_input = st.slider(
                "V_from (起始电压)",
                min_value=0.0,
                max_value=float(V_max),
                value=0.0,
                step=1.0,
                help="电压跳变前的电压值",
                key="sp_vfrom",
            )
            V_to_input = st.slider(
                "V_to (目标电压)",
                min_value=0.0,
                max_value=float(V_max),
                value=20.0,
                step=1.0,
                help="电压跳变后的电压值",
                key="sp_vto",
            )

        with col3:
            st.markdown("**时间 (s)**")
            t_since_input = st.slider(
                "t_since (跳变后时间)",
                min_value=0.0,
                max_value=float(t_max),
                value=0.01,
                step=0.001,
                format="%.3f",
                help=f"自电压跳变以来的时间 (最大 {t_max} s)",
                key="sp_t",
            )

        # 输入汇总
        st.markdown("---")
        st.markdown("#### 📋 输入汇总")

        input_data = {
            "x (m)": x_input,
            "y (m)": y_input,
            "z (m)": z_input,
            "V_from (V)": V_from_input,
            "V_to (V)": V_to_input,
            "t_since (s)": t_since_input,
        }

        col1, col2 = st.columns([1, 2])
        with col1:
            st.json(input_data)
        with col2:
            # 归一化值显示
            st.markdown("**归一化值 (模型内部使用):**")
            norm_data = {
                "x_norm": x_input / Lx,
                "y_norm": y_input / Ly,
                "z_norm": z_input / Lz,
                "V_from_norm": V_from_input / V_max,
                "V_to_norm": V_to_input / V_max,
                "t_norm": t_since_input / t_max,
            }
            st.json({k: f"{v:.4f}" for k, v in norm_data.items()})

        # 预测按钮
        st.markdown("---")
        predict_button = st.button(
            "🔮 预测", type="primary", use_container_width=True, key="sp_predict_btn"
        )

        if predict_button:
            if selected_model_info is None:
                st.error("❌ 无法获取选中的模型信息。")
                return
            self._run_inference(
                checkpoint_path=selected_model_info["path"],
                inputs=(
                    x_input,
                    y_input,
                    z_input,
                    V_from_input,
                    V_to_input,
                    t_since_input,
                ),
            )

    def _render_trajectory_prediction(
        self, selected_model_info, Lx, Ly, Lz, V_max, t_max
    ):
        """渲染轨迹预测子标签页 (Task 5)"""
        import streamlit as st
        import numpy as np

        st.markdown("#### 🎯 轨迹预测")
        st.markdown("预测特定空间点在时间序列上的物理量演化。")

        # 空间点输入
        st.markdown("**📍 空间点坐标**")
        col1, col2, col3 = st.columns(3)
        with col1:
            traj_x = st.number_input(
                "x (m)",
                min_value=0.0,
                max_value=float(Lx),
                value=float(Lx / 2),
                format="%.2e",
                key="traj_x",
            )
        with col2:
            traj_y = st.number_input(
                "y (m)",
                min_value=0.0,
                max_value=float(Ly),
                value=float(Ly / 2),
                format="%.2e",
                key="traj_y",
            )
        with col3:
            traj_z = st.number_input(
                "z (m)",
                min_value=0.0,
                max_value=float(Lz),
                value=float(Lz / 2),
                format="%.2e",
                key="traj_z",
            )

        # 电压和时间设置
        st.markdown("**⚡ 电压与时间设置**")
        col1, col2 = st.columns(2)
        with col1:
            traj_voltage = st.slider(
                "目标电压 (V)",
                min_value=0.0,
                max_value=float(V_max),
                value=20.0,
                step=1.0,
                key="traj_voltage",
            )
        with col2:
            traj_n_points = st.slider(
                "时间点数",
                min_value=10,
                max_value=200,
                value=50,
                step=10,
                key="traj_n_points",
            )

        # 时间范围
        col1, col2 = st.columns(2)
        with col1:
            traj_t_start = st.number_input(
                "起始时间 (s)",
                min_value=0.0,
                max_value=float(t_max),
                value=0.0,
                format="%.4f",
                key="traj_t_start",
            )
        with col2:
            traj_t_end = st.number_input(
                "结束时间 (s)",
                min_value=0.0,
                max_value=float(t_max),
                value=float(t_max * 0.5),
                format="%.4f",
                key="traj_t_end",
            )

        # 生成时间数组
        t_array = np.linspace(traj_t_start, traj_t_end, traj_n_points)

        st.markdown("---")
        traj_predict_btn = st.button(
            "📈 生成轨迹",
            type="primary",
            use_container_width=True,
            key="traj_predict_btn",
        )

        if traj_predict_btn:
            if selected_model_info is None:
                st.error("❌ 无法获取选中的模型信息。")
                return

            self._run_trajectory_prediction(
                checkpoint_path=selected_model_info["path"],
                point=(traj_x, traj_y, traj_z),
                voltage=traj_voltage,
                t_array=t_array,
            )

    def _run_trajectory_prediction(
        self, checkpoint_path: str, point: tuple, voltage: float, t_array: np.ndarray
    ):
        """执行轨迹预测并显示结果"""
        import streamlit as st
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import numpy as np

        st.markdown("#### 📊 轨迹预测结果")

        with st.spinner("正在加载模型..."):
            try:
                engine = PINNInferenceEngine(checkpoint_path, device="cpu")
            except Exception as e:
                st.error(f"❌ 模型加载失败: {e}")
                return

        st.success(f"✅ 模型加载成功")

        try:
            with st.spinner("正在计算轨迹..."):
                result = engine.predict_point_trajectory(voltage, t_array, point)

            # 创建图表
            fig = make_subplots(
                rows=3,
                cols=2,
                subplot_titles=(
                    "速度分量 (u, v, w)",
                    "VOF分数 (φ)",
                    "压力 (p)",
                    "速度大小 |v|",
                ),
                specs=[
                    [{"secondary_y": False}, {"secondary_y": False}],
                    [{"secondary_y": False}, {"secondary_y": False}],
                    [{"secondary_y": False}, {"secondary_y": False}],
                ],
            )

            # 速度分量
            fig.add_trace(
                go.Scatter(
                    x=result["t"],
                    y=result["u"],
                    mode="lines",
                    name="u",
                    line=dict(color="#1f77b4"),
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=result["t"],
                    y=result["v"],
                    mode="lines",
                    name="v",
                    line=dict(color="#ff7f0e"),
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=result["t"],
                    y=result["w"],
                    mode="lines",
                    name="w",
                    line=dict(color="#2ca02c"),
                ),
                row=1,
                col=1,
            )

            # VOF分数
            fig.add_trace(
                go.Scatter(
                    x=result["t"],
                    y=result["phi"],
                    mode="lines",
                    name="φ",
                    line=dict(color="#9467bd"),
                    fill="tozeroy",
                ),
                row=1,
                col=2,
            )

            # 压力
            fig.add_trace(
                go.Scatter(
                    x=result["t"],
                    y=result["p"],
                    mode="lines",
                    name="p",
                    line=dict(color="#d62728"),
                ),
                row=2,
                col=1,
            )

            # 速度大小
            vel_mag = np.sqrt(result["u"] ** 2 + result["v"] ** 2 + result["w"] ** 2)
            fig.add_trace(
                go.Scatter(
                    x=result["t"],
                    y=vel_mag,
                    mode="lines",
                    name="|v|",
                    line=dict(color="#17becf"),
                ),
                row=2,
                col=2,
            )

            fig.update_layout(
                height=700,
                showlegend=True,
                template="plotly_white",
                title_text=f"轨迹预测: 点 ({point[0]:.2e}, {point[1]:.2e}, {point[2]:.2e}), 电压 {voltage}V",
            )

            fig.update_xaxes(title_text="时间 (s)", row=1, col=1)
            fig.update_xaxes(title_text="时间 (s)", row=1, col=2)
            fig.update_xaxes(title_text="时间 (s)", row=2, col=1)
            fig.update_xaxes(title_text="时间 (s)", row=2, col=2)

            fig.update_yaxes(title_text="速度 (m/s)", row=1, col=1)
            fig.update_yaxes(title_text="VOF分数", row=1, col=2)
            fig.update_yaxes(title_text="压力 (Pa)", row=2, col=1)
            fig.update_yaxes(title_text="速度大小 (m/s)", row=2, col=2)

            st.plotly_chart(fig, use_container_width=True, key="trajectory_chart")

            # 数据表格
            st.markdown("**📋 轨迹数据**")
            import pandas as pd

            df = pd.DataFrame(
                {
                    "时间 (s)": result["t"],
                    "u (m/s)": result["u"],
                    "v (m/s)": result["v"],
                    "w (m/s)": result["w"],
                    "p (Pa)": result["p"],
                    "φ": result["phi"],
                }
            )
            st.dataframe(df, use_container_width=True)

        except Exception as e:
            st.error(f"❌ 轨迹预测失败: {e}")
            st.exception(e)

    def _render_slice_visualization(
        self, selected_model_info, Lx, Ly, Lz, V_max, t_max
    ):
        """渲染切片可视化子标签页 (Task 7)"""
        import streamlit as st
        import numpy as np

        st.markdown("#### 🔲 3D切片可视化")
        st.markdown("选择切片平面和位置，可视化二维截面上的物理量分布。")

        # 切片设置
        st.markdown("**📐 切片设置**")
        col1, col2, col3 = st.columns(3)

        with col1:
            slice_axis = st.selectbox(
                "切片平面",
                options=[
                    ("xy (z=常数)", "z"),
                    ("xz (y=常数)", "y"),
                    ("yz (x=常数)", "x"),
                ],
                format_func=lambda x: x[0],
                key="slice_axis",
            )[1]

        with col2:
            slice_pos = st.slider(
                "切片位置 (相对)",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                key="slice_pos",
            )

        with col3:
            slice_res = st.slider(
                "分辨率",
                min_value=20,
                max_value=100,
                value=50,
                step=10,
                key="slice_res",
            )

        # 电压和时间设置
        st.markdown("**⚡ 工况设置**")
        col1, col2, col3 = st.columns(3)
        with col1:
            slice_v_from = st.slider(
                "起始电压 (V)",
                min_value=0.0,
                max_value=float(V_max),
                value=0.0,
                step=1.0,
                key="slice_vfrom",
            )
        with col2:
            slice_v_to = st.slider(
                "目标电压 (V)",
                min_value=0.0,
                max_value=float(V_max),
                value=20.0,
                step=1.0,
                key="slice_vto",
            )
        with col3:
            slice_time = st.slider(
                "时间 (s)",
                min_value=0.0,
                max_value=float(t_max),
                value=0.01,
                step=0.001,
                format="%.3f",
                key="slice_time",
            )

        st.markdown("---")
        slice_predict_btn = st.button(
            "🔲 生成切片",
            type="primary",
            use_container_width=True,
            key="slice_predict_btn",
        )

        if slice_predict_btn:
            if selected_model_info is None:
                st.error("❌ 无法获取选中的模型信息。")
                return

            self._run_slice_prediction(
                checkpoint_path=selected_model_info["path"],
                axis=slice_axis,
                pos=slice_pos,
                res=slice_res,
                voltage_from=slice_v_from,
                voltage_to=slice_v_to,
                time=slice_time,
                Lx=Lx,
                Ly=Ly,
                Lz=Lz,
            )

    def _run_slice_prediction(
        self,
        checkpoint_path: str,
        axis: str,
        pos: float,
        res: int,
        voltage_from: float,
        voltage_to: float,
        time: float,
        Lx: float,
        Ly: float,
        Lz: float,
    ):
        """执行切片预测并显示结果"""
        import streamlit as st
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import numpy as np

        st.markdown("#### 📊 切片可视化结果")

        with st.spinner("正在加载模型..."):
            try:
                engine = PINNInferenceEngine(checkpoint_path, device="cpu")
            except Exception as e:
                st.error(f"❌ 模型加载失败: {e}")
                return

        st.success(f"✅ 模型加载成功")

        try:
            with st.spinner("正在计算切片..."):
                result = engine.predict_field_slice(
                    voltage=voltage_to,
                    time=time,
                    axis=axis,
                    pos=pos,
                    res=res,
                    voltage_from=voltage_from,
                )

            # 根据切片平面确定坐标轴标签
            if axis == "z":
                x_label, y_label = "x (m)", "y (m)"
                x_coords = result["X"][0, :] * 1e6  # 转换为微米
                y_coords = result["Y"][:, 0] * 1e6
            elif axis == "y":
                x_label, y_label = "x (m)", "z (m)"
                x_coords = result["X"][0, :] * 1e6
                y_coords = result["Z"][:, 0] * 1e6
            else:  # axis == "x"
                x_label, y_label = "y (m)", "z (m)"
                x_coords = result["Y"][0, :] * 1e6
                y_coords = result["Z"][:, 0] * 1e6

            # 创建子图
            fig = make_subplots(
                rows=2,
                cols=3,
                subplot_titles=(
                    "VOF分数 (φ)",
                    "速度大小 |v|",
                    "压力 (p)",
                    "u 分量",
                    "v 分量",
                    "w 分量",
                ),
                specs=[
                    [{"type": "heatmap"}, {"type": "heatmap"}, {"type": "heatmap"}],
                    [{"type": "heatmap"}, {"type": "heatmap"}, {"type": "heatmap"}],
                ],
            )

            # VOF分数
            fig.add_trace(
                go.Heatmap(
                    z=result["phi"],
                    x=x_coords,
                    y=y_coords,
                    colorscale="RdBu",
                    zmid=0.5,
                    colorbar=dict(title="φ", x=0.15),
                    name="φ",
                ),
                row=1,
                col=1,
            )

            # 速度大小
            fig.add_trace(
                go.Heatmap(
                    z=result["vel_mag"],
                    x=x_coords,
                    y=y_coords,
                    colorscale="Viridis",
                    colorbar=dict(title="|v| (m/s)", x=0.48),
                    name="|v|",
                ),
                row=1,
                col=2,
            )

            # 压力
            fig.add_trace(
                go.Heatmap(
                    z=result["p"],
                    x=x_coords,
                    y=y_coords,
                    colorscale="Plasma",
                    colorbar=dict(title="p (Pa)", x=0.82),
                    name="p",
                ),
                row=1,
                col=3,
            )

            # 速度分量
            fig.add_trace(
                go.Heatmap(
                    z=result["u"],
                    x=x_coords,
                    y=y_coords,
                    colorscale="RdBu",
                    zmid=0,
                    colorbar=dict(title="u (m/s)", x=0.15),
                    name="u",
                ),
                row=2,
                col=1,
            )

            fig.add_trace(
                go.Heatmap(
                    z=result["v"],
                    x=x_coords,
                    y=y_coords,
                    colorscale="RdBu",
                    zmid=0,
                    colorbar=dict(title="v (m/s)", x=0.48),
                    name="v",
                ),
                row=2,
                col=2,
            )

            fig.add_trace(
                go.Heatmap(
                    z=result["w"],
                    x=x_coords,
                    y=y_coords,
                    colorscale="RdBu",
                    zmid=0,
                    colorbar=dict(title="w (m/s)", x=0.82),
                    name="w",
                ),
                row=2,
                col=3,
            )

            fig.update_layout(
                height=700,
                title_text=f"切片可视化: {axis.upper()}平面 @ 位置{pos * 100:.0f}% | 电压 {voltage_from}V→{voltage_to}V | 时间 {time:.3f}s",
                showlegend=False,
            )

            # 更新所有子图的坐标轴标签
            for i in range(1, 3):
                for j in range(1, 4):
                    fig.update_xaxes(title_text=f"{x_label} (μm)", row=i, col=j)
                    fig.update_yaxes(title_text=f"{y_label} (μm)", row=i, col=j)

            st.plotly_chart(fig, use_container_width=True, key="slice_chart")

            # 显示切片信息
            st.markdown("**📋 切片信息**")
            slice_info = {
                "切片平面": f"{axis.upper()}平面",
                "相对位置": f"{pos * 100:.1f}%",
                "绝对位置": f"{pos * (Lz if axis == 'z' else Ly if axis == 'y' else Lx):.2e} m",
                "分辨率": f"{res}×{res}",
                "起始电压": f"{voltage_from} V",
                "目标电压": f"{voltage_to} V",
                "时间": f"{time} s",
            }
            st.json(slice_info)

        except Exception as e:
            st.error(f"❌ 切片预测失败: {e}")
            st.exception(e)

    def _run_inference(
        self,
        checkpoint_path: str,
        inputs: tuple,
    ) -> None:
        """执行推理并显示结果

        Args:
            checkpoint_path: 模型检查点路径 (.pth 文件)
            inputs: (x, y, z, V_from, V_to, t_since) 元组
        """
        import streamlit as st
        import plotly.graph_objects as go

        st.markdown("#### 📊 推理结果")

        # 使用 PINNInferenceEngine 加载模型
        with st.spinner("正在加载模型..."):
            try:
                engine = PINNInferenceEngine(checkpoint_path, device="cpu")
            except Exception as e:
                st.error(f"❌ 模型加载失败: {e}")
                return

        st.success(f"✅ 模型加载成功: {checkpoint_path}")

        # 执行推理
        try:
            x, y, z, V_from, V_to, t_since = inputs

            with st.spinner("正在推理..."):
                result = engine.predict_point(x, y, z, V_from, V_to, t_since)

            # 解析输出: result is a dict with u, v, w, p, phi
            u = result["u"]
            v = result["v"]
            w = result["w"]
            p = result["p"]
            phi = result["phi"]

            # 显示结果
            result_data = {
                "u (x方向速度, m/s)": f"{u:.6e}",
                "v (y方向速度, m/s)": f"{v:.6e}",
                "w (z方向速度, m/s)": f"{w:.6e}",
                "p (压力, Pa)": f"{p:.6e}",
                "φ (VOF分数, 0-1)": f"{phi:.6f}",
            }

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**原始输出值:**")
                st.json(result_data)

            with col2:
                # 物理意义解释
                st.markdown("**物理意义:**")
                if phi < 0.1:
                    phase = "极性液体 (φ ≈ 0)"
                    phase_color = "#e3f2fd"
                elif phi > 0.9:
                    phase = "油墨 (φ ≈ 1)"
                    phase_color = "#fff3e0"
                else:
                    phase = "界面区域 (0.1 < φ < 0.9)"
                    phase_color = "#f3e5f5"

                velocity_mag = (u**2 + v**2 + w**2) ** 0.5

                st.markdown(
                    f"""
                    <div style="
                        background-color: {phase_color};
                        border-radius: 8px;
                        padding: 12px;
                        border: 1px solid #ddd;
                    ">
                        <b>相态:</b> {phase}<br>
                        <b>速度大小:</b> {velocity_mag:.6e} m/s<br>
                        <b>压力:</b> {p:.6e} Pa<br>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # 结果可视化
            st.markdown("---")
            st.markdown("#### 📈 结果可视化")

            # 使用列显示不同可视化
            col1, col2, col3 = st.columns(3)

            with col1:
                # 速度分量条形图
                import plotly.graph_objects as go

                fig = go.Figure()
                fig.add_trace(
                    go.Bar(
                        x=["u", "v", "w"],
                        y=[u, v, w],
                        marker_color=["#1f77b4", "#ff7f0e", "#2ca02c"],
                    )
                )
                fig.update_layout(
                    title="速度分量",
                    yaxis_title="速度 (m/s)",
                    template="plotly_white",
                    height=250,
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True, key="velocity_bars")

            with col2:
                # VOF分数仪表图
                fig = go.Figure()
                fig.add_trace(
                    go.Indicator(
                        mode="gauge+number",
                        value=phi,
                        domain={"x": [0, 1], "y": [0, 1]},
                        title={"text": "φ (VOF)"},
                        gauge={
                            "axis": {"range": [0, 1]},
                            "bar": {"color": "#9467bd"},
                            "steps": [
                                {"range": [0, 0.1], "color": "#e3f2fd"},
                                {"range": [0.1, 0.9], "color": "#f3e5f5"},
                                {"range": [0.9, 1], "color": "#fff3e0"},
                            ],
                        },
                    )
                )
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True, key="phi_gauge")

            with col3:
                # 压力显示
                fig = go.Figure()
                fig.add_trace(
                    go.Indicator(
                        mode="number",
                        value=p,
                        title={"text": "压力 (Pa)"},
                        number={"suffix": " Pa", "font": {"size": 24}},
                    )
                )
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True, key="pressure_indicator")

            # ============ Task 6: 物理验证显示 ============
            st.markdown("---")
            st.markdown("#### 🔬 物理验证")

            try:
                physics_check = engine.check_point_physics(
                    x, y, z, V_from, V_to, t_since
                )

                phys_col1, phys_col2, phys_col3 = st.columns(3)

                with phys_col1:
                    cont_resid = physics_check.get("continuity_residual", 0.0)
                    st.metric(
                        label="连续性残差 (∇·u)",
                        value=f"{cont_resid:.2e}",
                        delta="✅ 良好" if abs(cont_resid) < 1e-3 else "⚠️ 偏高",
                        delta_color="normal" if abs(cont_resid) < 1e-3 else "inverse",
                    )

                with phys_col2:
                    mom_resid = physics_check.get("momentum_residual", 0.0)
                    st.metric(
                        label="动量残差",
                        value=f"{mom_resid:.2e}",
                        delta="✅ 良好" if abs(mom_resid) < 1e-3 else "⚠️ 偏高",
                        delta_color="normal" if abs(mom_resid) < 1e-3 else "inverse",
                    )

                with phys_col3:
                    mass_err = physics_check.get("mass_conservation_error", 0.0)
                    st.metric(
                        label="质量守恒误差",
                        value=f"{mass_err:.2e}",
                        delta="✅ 良好" if abs(mass_err) < 1e-3 else "⚠️ 偏高",
                        delta_color="normal" if abs(mass_err) < 1e-3 else "inverse",
                    )

            except Exception as phys_e:
                st.warning(f"物理验证计算失败: {phys_e}")

            # ============ Task 8: 质量守恒检查面板 ============
            st.markdown("---")
            st.markdown("#### ⚖️ 质量守恒检查")
            st.markdown("检查整个3D体积的墨水质量守恒情况。")

            # 质量守恒检查参数
            mc_col1, mc_col2, mc_col3 = st.columns(3)

            with mc_col1:
                mc_resolution = st.slider(
                    "分辨率",
                    min_value=20,
                    max_value=50,
                    value=30,
                    step=5,
                    key="mc_resolution",
                )

            with mc_col2:
                st.markdown("<br>", unsafe_allow_html=True)
                check_mass_btn = st.button(
                    "⚖️ 检查质量守恒",
                    type="secondary",
                    use_container_width=True,
                    key="check_mass_btn",
                )

            if check_mass_btn:
                try:
                    with st.spinner("正在计算体积积分..."):
                        volume = engine.check_mass_conservation(
                            t=t_since, voltage_from=V_from, voltage_to=V_to
                        )

                    # 计算预期体积（基于初始条件）
                    from src.config import PHYSICS

                    expected_volume = (
                        PHYSICS.get("h_ink", 3e-6) * PHYSICS["Lx"] * PHYSICS["Ly"]
                    )
                    volume_error = abs(volume - expected_volume) / expected_volume * 100

                    # 显示结果
                    result_col1, result_col2, result_col3 = st.columns(3)

                    with result_col1:
                        st.metric(label="计算体积", value=f"{volume:.2e} m³")

                    with result_col2:
                        st.metric(label="预期体积", value=f"{expected_volume:.2e} m³")

                    with result_col3:
                        st.metric(
                            label="体积误差",
                            value=f"{volume_error:.2f}%",
                            delta="✅ 良好" if volume_error < 1.0 else "⚠️ 偏高",
                            delta_color="normal" if volume_error < 1.0 else "inverse",
                        )

                    # 状态指示器
                    if volume_error < 1.0:
                        st.success(f"✅ 质量守恒良好! 误差 {volume_error:.2f}% < 1%")
                    elif volume_error < 5.0:
                        st.warning(f"⚠️ 质量守恒可接受。误差 {volume_error:.2f}%")
                    else:
                        st.error(f"❌ 质量守恒偏差较大。误差 {volume_error:.2f}%")

                except Exception as mc_e:
                    st.error(f"质量守恒检查失败: {mc_e}")
                    st.exception(mc_e)

        except Exception as e:
            st.error(f"❌ 推理过程中发生错误: {e}")
            st.exception(e)
