"""
训练输出分析器集成测试
=======================

测试完整的工作流程：扫描 → 解析 → 加载 → 推理
"""

import pytest
import os
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

from src.dashboard.training_output_analyzer import (
    TrainingOutputScanner,
    TrainingRunInfo,
    TrainingConfigParser,
    LossDataParser,
    MetricsParser,
    ModelLoader,
    ModelInfo,
)


class TestTrainingOutputAnalyzerIntegration:
    """训练输出分析器集成测试类"""

    def test_full_scan_parse_flow(self):
        """测试完整的扫描和解析流程"""
        # 1. 扫描训练输出
        scanner = TrainingOutputScanner(train_outputs_dir="outputs/train")
        runs = scanner.scan_training_outputs()

        # 验证扫描结果
        assert len(runs) > 0, "应该能扫描到至少一个训练输出目录"
        assert all(isinstance(run, TrainingRunInfo) for run in runs), (
            "所有返回项应该是 TrainingRunInfo 类型"
        )

        # 2. 选择第一个有完整数据的训练运行
        target_run = next(
            (
                run
                for run in runs
                if run.config_path
                and os.path.exists(run.config_path)
                and run.has_loss_csv
                and run.has_metrics
            ),
            None,
        )

        if target_run is None:
            pytest.skip("没有找到有完整数据的训练运行")

        # 3. 解析配置
        config = TrainingConfigParser.parse(target_run.config_path)
        assert config is not None, "配置解析应该成功"
        assert "metadata" in config, "配置应该包含 metadata"
        assert "model_architecture" in config, "配置应该包含 model_architecture"
        assert "training_parameters" in config, "配置应该包含 training_parameters"

        # 4. 解析损失数据
        loss_csv_path = os.path.join(target_run.path, "loss_breakdown.csv")
        if os.path.exists(loss_csv_path):
            loss_df = LossDataParser.parse(loss_csv_path)
            assert loss_df is not None, "损失数据解析应该成功"
            assert len(loss_df) > 0, "损失数据不应该为空"
            assert "epoch" in loss_df.columns, "损失数据应该包含 epoch 列"
            assert "loss_total" in loss_df.columns, "损失数据应该包含 loss_total 列"

        # 5. 解析评估指标
        rmse_csv_path = os.path.join(target_run.path, "rmse_per_voltage.csv")
        if os.path.exists(rmse_csv_path):
            rmse_df = MetricsParser.parse_rmse(rmse_csv_path)
            assert rmse_df is not None, "RMSE 数据解析应该成功"
            assert len(rmse_df) > 0, "RMSE 数据不应该为空"
            assert "Voltage" in rmse_df.columns, "RMSE 数据应该包含 Voltage 列"
            assert "RMSE" in rmse_df.columns, "RMSE 数据应该包含 RMSE 列"

        volume_csv_path = os.path.join(target_run.path, "volume_trend_stats.csv")
        if os.path.exists(volume_csv_path):
            volume_df = MetricsParser.parse_volume_stats(volume_csv_path)
            assert volume_df is not None, "体积统计数据解析应该成功"
            assert len(volume_df) > 0, "体积统计数据不应该为空"
            assert "stage" in volume_df.columns, "体积统计应该包含 stage 列"
            assert "mean" in volume_df.columns, "体积统计应该包含 mean 列"

    def test_model_loading_integration(self):
        """测试模型加载集成流程"""
        # 1. 扫描训练输出
        scanner = TrainingOutputScanner(train_outputs_dir="outputs/train")
        runs = scanner.scan_training_outputs()

        if not runs:
            pytest.skip("没有找到训练运行目录")

        # 2. 查找有模型文件的训练运行
        target_run = next(
            (run for run in runs if len(run.model_files) > 0),
            None,
        )
        if target_run is None:
            pytest.skip("没有找到有模型文件的训练运行")

        # 3. 列出可用模型
        available_models = ModelLoader.list_available_models(target_run.path)
        assert len(available_models) > 0, "应该至少有一个模型文件"
        assert all(isinstance(model, dict) for model in available_models), (
            "每个模型应该是字典"
        )

        # 4. 加载最佳模型
        best_model_path = os.path.join(target_run.path, "best_model.pth")
        if os.path.exists(best_model_path):
            model, model_info = ModelLoader.load_model(
                target_run.path, model_type="best"
            )

            assert model is not None, "应该成功加载模型"
            assert model_info is not None, "应该返回模型信息"
            assert isinstance(model_info, ModelInfo), "模型信息应该是 ModelInfo 类型"
            assert model_info.model_type == "best", "模型类型应该是 best"
            assert model_info.file_size > 0, "文件大小应该大于 0"

    def test_end_to_end_workflow(self):
        """测试端到端工作流程"""
        # 1. 扫描所有训练输出
        scanner = TrainingOutputScanner(train_outputs_dir="outputs/train")
        runs = scanner.scan_training_outputs()
        assert len(runs) > 0, "应该能扫描到训练输出"

        # 2. 选择特定的训练运行
        target_run = next(
            (run for run in runs if run.name == "pinn_20260205_174333"),
            None,
        )
        if target_run is None:
            pytest.skip("未找到目标训练运行")

        # 3. 验证训练运行信息
        assert target_run.name == "pinn_20260205_174333", "运行名称应该匹配"
        assert os.path.exists(target_run.path), "运行路径应该存在"
        assert isinstance(target_run.creation_time, datetime), (
            "创建时间应该是 datetime 对象"
        )

        # 4. 获取配置
        config = scanner.get_training_config(target_run)
        if config:
            assert isinstance(config, dict), "配置应该是字典类型"
            assert len(config) > 0, "配置字典不应该为空"

        # 5. 验证模型文件
        if target_run.model_files:
            for model_file in target_run.model_files:
                assert os.path.exists(model_file), f"模型文件应该存在: {model_file}"
                assert model_file.endswith((".pth", ".pt")), (
                    f"模型文件应该是 .pth 或 .pt 格式: {model_file}"
                )

    def test_multiple_runs_consistency(self):
        """测试多个训练运行的一致性"""
        # 1. 扫描所有训练输出
        scanner = TrainingOutputScanner(train_outputs_dir="outputs/train")
        runs = scanner.scan_training_outputs()

        if len(runs) < 2:
            pytest.skip("需要至少两个训练运行来测试一致性")

        # 2. 验证所有运行都有正确的字段
        for run in runs:
            assert isinstance(run.name, str), f"{run.name}: name 应该是字符串"
            assert isinstance(run.path, str), f"{run.name}: path 应该是字符串"
            assert isinstance(run.creation_time, datetime), (
                f"{run.name}: creation_time 应该是 datetime 对象"
            )
            assert isinstance(run.model_files, list), (
                f"{run.name}: model_files 应该是列表"
            )
            assert isinstance(run.has_loss_csv, bool), (
                f"{run.name}: has_loss_csv 应该是布尔值"
            )
            assert isinstance(run.has_metrics, bool), (
                f"{run.name}: has_metrics 应该是布尔值"
            )

        # 3. 验证排序
        for i in range(len(runs) - 1):
            assert runs[i].creation_time >= runs[i + 1].creation_time, (
                "训练运行应该按创建时间倒序排列"
            )

    def test_parser_integration_with_real_data(self):
        """测试解析器与真实数据的集成"""
        # 使用特定的训练运行
        run_path = "outputs/train/pinn_20260205_174333"
        if not os.path.exists(run_path):
            pytest.skip("训练运行目录不存在")

        # 1. 测试配置解析
        config_path = os.path.join(run_path, "config.json")
        if os.path.exists(config_path):
            config = TrainingConfigParser.parse(config_path)
            assert config is not None, "配置解析应该成功"

            # 验证关键字段
            assert "metadata" in config, "应该包含 metadata"
            assert "model_architecture" in config, "应该包含 model_architecture"
            assert "training_parameters" in config, "应该包含 training_parameters"
            assert "physics_weights" in config, "应该包含 physics_weights"
            assert "data_config" in config, "应该包含 data_config"

            # 验证数据完整性
            assert config["metadata"]["stage"] == "2_two_phase_pinn", "stage 应该匹配"
            assert config["model_architecture"]["input_format"] == "triplet", (
                "input_format 应该是 triplet"
            )
            assert config["training_parameters"]["epochs"] > 0, "epochs 应该大于 0"

        # 2. 测试损失数据解析
        loss_csv_path = os.path.join(run_path, "loss_breakdown.csv")
        if os.path.exists(loss_csv_path):
            loss_df = LossDataParser.parse(loss_csv_path)
            assert loss_df is not None, "损失数据解析应该成功"

            # 验证列
            required_columns = [
                "epoch",
                "stage",
                "loss_total",
                "lr",
            ]
            for col in required_columns:
                assert col in loss_df.columns, f"损失数据应该包含列: {col}"

            # 验证数据范围
            assert loss_df["epoch"].min() >= 0, "epoch 应该 >= 0"
            assert loss_df["stage"].min() >= 1, "stage 应该 >= 1"
            assert loss_df["loss_total"].min() >= 0, "loss_total 应该 >= 0"

        # 3. 测试 RMSE 解析
        rmse_csv_path = os.path.join(run_path, "rmse_per_voltage.csv")
        if os.path.exists(rmse_csv_path):
            rmse_df = MetricsParser.parse_rmse(rmse_csv_path)
            assert rmse_df is not None, "RMSE 数据解析应该成功"

            # 验证列
            assert "Voltage" in rmse_df.columns, "应该包含 Voltage 列"
            assert "RMSE" in rmse_df.columns, "应该包含 RMSE 列"
            assert "Rating" in rmse_df.columns, "应该包含 Rating 列"

            # 验证数据范围
            assert rmse_df["Voltage"].min() >= 0, "Voltage 应该 >= 0"
            assert rmse_df["RMSE"].min() >= 0, "RMSE 应该 >= 0"

        # 4. 测试体积统计解析
        volume_csv_path = os.path.join(run_path, "volume_trend_stats.csv")
        if os.path.exists(volume_csv_path):
            volume_df = MetricsParser.parse_volume_stats(volume_csv_path)
            assert volume_df is not None, "体积统计数据解析应该成功"

            # 验证列
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
            for col in required_columns:
                assert col in volume_df.columns, f"体积统计应该包含列: {col}"

            # 验证阶段数
            assert len(volume_df) == 3, "应该有 3 个阶段的统计"

    def test_model_loading_with_config(self):
        """测试使用配置加载模型"""
        run_path = "outputs/train/pinn_20260205_174333"
        if not os.path.exists(run_path):
            pytest.skip("训练运行目录不存在")

        # 1. 加载配置
        config_path = os.path.join(run_path, "config.json")
        if not os.path.exists(config_path):
            pytest.skip("配置文件不存在")

        config = TrainingConfigParser.parse(config_path)
        assert config is not None, "配置解析应该成功"

        # 2. 验证配置中的模型架构
        assert "model_architecture" in config, "配置应该包含 model_architecture"
        model_arch = config["model_architecture"]
        assert "hidden_phi" in model_arch, "应该包含 hidden_phi"
        assert "hidden_vel" in model_arch, "应该包含 hidden_vel"

        # 3. 列出可用模型
        available_models = ModelLoader.list_available_models(run_path)
        assert len(available_models) > 0, "应该至少有一个模型文件"

        # 4. 尝试加载最佳模型
        best_model_path = os.path.join(run_path, "best_model.pth")
        if os.path.exists(best_model_path):
            model, model_info = ModelLoader.load_model(run_path, model_type="best")

            assert model is not None, "应该成功加载模型"
            assert model_info is not None, "应该返回模型信息"

            # 验证模型信息
            assert model_info.model_type == "best", "模型类型应该是 best"
            assert model_info.file_size > 0, "文件大小应该大于 0"
            assert "architecture" in model_info.__dict__, "应该包含架构信息"

            # 验证架构信息与配置匹配
            if model_info.architecture:
                assert model_info.architecture.get("hidden_phi") == model_arch.get(
                    "hidden_phi"
                ), "hidden_phi 应该匹配"
                assert model_info.architecture.get("hidden_vel") == model_arch.get(
                    "hidden_vel"
                ), "hidden_vel 应该匹配"

    def test_error_handling_integration(self):
        """测试错误处理的集成"""
        # 1. 测试不存在的目录
        scanner = TrainingOutputScanner(train_outputs_dir="nonexistent_directory")
        runs = scanner.scan_training_outputs()
        assert runs == [], "对于不存在的目录应该返回空列表"

        # 2. 测试解析不存在的文件
        config = TrainingConfigParser.parse("nonexistent/config.json")
        assert config is None, "不存在的配置文件应该返回 None"

        loss_df = LossDataParser.parse("nonexistent/loss.csv")
        assert loss_df is None, "不存在的损失CSV应该返回 None"

        rmse_df = MetricsParser.parse_rmse("nonexistent/rmse.csv")
        assert rmse_df is None, "不存在的RMSE CSV应该返回 None"

        volume_df = MetricsParser.parse_volume_stats("nonexistent/volume.csv")
        assert volume_df is None, "不存在的体积统计CSV应该返回 None"

        # 3. 测试加载不存在的模型
        model, model_info = ModelLoader.load_model("nonexistent_directory")
        assert model is None, "不存在的目录应该返回 None"
        assert model_info is None, "模型信息应该是 None"

    def test_scanner_with_empty_directory(self):
        """测试扫描器处理空目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            scanner = TrainingOutputScanner(train_outputs_dir=tmpdir)
            runs = scanner.scan_training_outputs()
            assert len(runs) == 0, "空目录应该返回空列表"

    def test_data_integrity_across_components(self):
        """测试跨组件的数据完整性"""
        # 使用特定的训练运行
        run_path = "outputs/train/pinn_20260205_174333"
        if not os.path.exists(run_path):
            pytest.skip("训练运行目录不存在")

        # 1. 扫描获取训练运行信息
        scanner = TrainingOutputScanner(train_outputs_dir="outputs/train")
        runs = scanner.scan_training_outputs()
        target_run = next(
            (run for run in runs if run.path == run_path),
            None,
        )

        if target_run is None:
            pytest.skip("未找到目标训练运行")

        # 2. 验证扫描器检测的标志与实际文件匹配
        loss_csv_path = os.path.join(run_path, "loss_breakdown.csv")
        has_loss_csv_actual = os.path.exists(loss_csv_path)
        assert target_run.has_loss_csv == has_loss_csv_actual, (
            "has_loss_csv 标志应该与实际文件匹配"
        )

        rmse_csv_path = os.path.join(run_path, "rmse_per_voltage.csv")
        volume_csv_path = os.path.join(run_path, "volume_trend_stats.csv")
        has_metrics_actual = os.path.exists(rmse_csv_path) or os.path.exists(
            volume_csv_path
        )
        assert target_run.has_metrics == has_metrics_actual, (
            "has_metrics 标志应该与实际文件匹配"
        )

        # 3. 验证配置路径
        config_path = os.path.join(run_path, "config.json")
        has_config_actual = os.path.exists(config_path)
        assert (target_run.config_path is not None) == has_config_actual, (
            "config_path 应该与实际文件匹配"
        )

        # 4. 验证模型文件列表
        actual_model_files = []
        if os.path.exists(run_path):
            for file in os.listdir(run_path):
                if file.endswith((".pth", ".pt")):
                    actual_model_files.append(os.path.join(run_path, file))

        assert len(target_run.model_files) == len(actual_model_files), (
            "模型文件数量应该匹配"
        )
        for model_file in target_run.model_files:
            assert model_file in actual_model_files, f"模型文件应该存在: {model_file}"
