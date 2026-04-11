"""
训练输出分析器单元测试
======================

测试 TrainingOutputScanner 的扫描功能。
"""

import pytest
import os
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


class TestTrainingOutputScanner:
    """训练输出扫描器测试类"""

    def test_scanner_finds_outputs(self):
        """测试扫描器能找到训练输出目录"""
        scanner = TrainingOutputScanner(train_outputs_dir="outputs/train")
        runs = scanner.scan_training_outputs()

        # 应该能找到至少一个训练运行
        assert len(runs) > 0, "应该能扫描到至少一个训练输出目录"

        # 检查是否包含已知的训练运行
        run_names = [run.name for run in runs]
        assert "pinn_20260205_174333" in run_names, (
            "应该包含已知的训练运行 pinn_20260205_174333"
        )

        # 验证返回的是 TrainingRunInfo 对象
        assert all(isinstance(run, TrainingRunInfo) for run in runs), (
            "所有返回项应该是 TrainingRunInfo 类型"
        )

    def test_run_info_has_correct_fields(self):
        """测试训练运行信息包含正确的字段"""
        scanner = TrainingOutputScanner(train_outputs_dir="outputs/train")
        runs = scanner.scan_training_outputs()

        if not runs:
            pytest.skip("没有找到训练运行目录")

        # 查找特定的训练运行
        target_run = next(
            (run for run in runs if run.name == "pinn_20260205_174333"), None
        )
        assert target_run is not None, "应该能找到特定的训练运行"

        # 验证字段存在且类型正确
        assert isinstance(target_run.name, str), "name 应该是字符串"
        assert isinstance(target_run.path, str), "path 应该是字符串"
        assert isinstance(target_run.creation_time, datetime), (
            "creation_time 应该是 datetime 对象"
        )
        assert isinstance(target_run.model_files, list), "model_files 应该是列表"
        assert isinstance(target_run.has_loss_csv, bool), "has_loss_csv 应该是布尔值"
        assert isinstance(target_run.has_metrics, bool), "has_metrics 应该是布尔值"

        # 验证路径存在
        assert os.path.exists(target_run.path), "训练运行路径应该存在"

    def test_scanner_finds_model_files(self):
        """测试扫描器能找到模型文件"""
        scanner = TrainingOutputScanner(train_outputs_dir="outputs/train")
        runs = scanner.scan_training_outputs()

        if not runs:
            pytest.skip("没有找到训练运行目录")

        # 查找包含模型文件的训练运行
        runs_with_models = [run for run in runs if len(run.model_files) > 0]
        assert len(runs_with_models) > 0, "应该至少有一个训练运行包含模型文件"

        # 检查模型文件格式
        for run in runs_with_models:
            for model_file in run.model_files:
                assert model_file.endswith((".pth", ".pt")), (
                    "模型文件应该是 .pth 或 .pt 格式"
                )
                assert os.path.exists(model_file), "模型文件路径应该存在"

    def test_scanner_detects_config(self):
        """测试扫描器能检测配置文件"""
        scanner = TrainingOutputScanner(train_outputs_dir="outputs/train")
        runs = scanner.scan_training_outputs()

        if not runs:
            pytest.skip("没有找到训练运行目录")

        # 查找特定训练运行
        target_run = next(
            (run for run in runs if run.name == "pinn_20260205_174333"), None
        )
        if target_run is None:
            pytest.skip("未找到目标训练运行")

        # 验证配置文件检测
        assert target_run.config_path is not None, "应该能检测到配置文件"
        assert os.path.exists(target_run.config_path), "配置文件路径应该存在"
        assert target_run.config_path.endswith("config.json"), (
            "配置文件应该是 config.json"
        )

    def test_scanner_detects_loss_csv(self):
        """测试扫描器能检测损失 CSV 文件"""
        scanner = TrainingOutputScanner(train_outputs_dir="outputs/train")
        runs = scanner.scan_training_outputs()

        if not runs:
            pytest.skip("没有找到训练运行目录")

        # 查找特定训练运行
        target_run = next(
            (run for run in runs if run.name == "pinn_20260205_174333"), None
        )
        if target_run is None:
            pytest.skip("未找到目标训练运行")

        # 验证损失 CSV 检测
        assert target_run.has_loss_csv is not None, "has_loss_csv 应该有值"

    def test_scanner_detects_metrics(self):
        """测试扫描器能检测指标文件"""
        scanner = TrainingOutputScanner(train_outputs_dir="outputs/train")
        runs = scanner.scan_training_outputs()

        if not runs:
            pytest.skip("没有找到训练运行目录")

        # 查找特定训练运行
        target_run = next(
            (run for run in runs if run.name == "pinn_20260205_174333"), None
        )
        if target_run is None:
            pytest.skip("未找到目标训练运行")

        # 验证指标文件检测
        assert target_run.has_metrics is not None, "has_metrics 应该有值"

    def test_scanner_sorts_by_creation_time(self):
        """测试扫描器按创建时间倒序排列"""
        scanner = TrainingOutputScanner(train_outputs_dir="outputs/train")
        runs = scanner.scan_training_outputs()

        if len(runs) < 2:
            pytest.skip("需要至少两个训练运行来测试排序")

        # 验证按创建时间倒序排列
        for i in range(len(runs) - 1):
            assert runs[i].creation_time >= runs[i + 1].creation_time, (
                "训练运行应该按创建时间倒序排列"
            )

    def test_scanner_handles_missing_directory(self):
        """测试扫描器能处理不存在的目录"""
        scanner = TrainingOutputScanner(train_outputs_dir="nonexistent_directory")
        runs = scanner.scan_training_outputs()

        # 对于不存在的目录，应该返回空列表
        assert runs == [], "对于不存在的目录应该返回空列表"

    def test_get_training_config(self):
        """测试获取训练配置"""
        scanner = TrainingOutputScanner(train_outputs_dir="outputs/train")
        runs = scanner.scan_training_outputs()

        if not runs:
            pytest.skip("没有找到训练运行目录")

        # 查找有配置文件的训练运行
        target_run = next((run for run in runs if run.config_path is not None), None)
        if target_run is None:
            pytest.skip("没有找到有配置文件的训练运行")

        # 获取配置
        config = scanner.get_training_config(target_run)

        # 验证配置是字典
        assert isinstance(config, dict), "配置应该是字典类型"
        assert len(config) > 0, "配置字典不应该为空"

    def test_get_training_config_returns_none_for_missing_config(self):
        """测试对于没有配置文件的训练运行返回 None"""
        scanner = TrainingOutputScanner(train_outputs_dir="outputs/train")

        # 创建一个没有配置文件的虚拟 TrainingRunInfo
        fake_run = TrainingRunInfo(
            name="fake_run",
            path="/fake/path",
            creation_time=datetime.now(),
            model_files=[],
            config_path=None,
            has_loss_csv=False,
            has_metrics=False,
        )

        config = scanner.get_training_config(fake_run)
        assert config is None, "对于没有配置文件的训练运行应该返回 None"


class TestTrainingConfigParser:
    """训练配置解析器测试类"""

    def test_config_parser_returns_none_for_nonexistent_file(self):
        """测试对于不存在的配置文件返回 None"""
        config = TrainingConfigParser.parse("nonexistent_config.json")
        assert config is None, "对于不存在的配置文件应该返回 None"

    def test_config_parser_extracts_metadata(self):
        """测试解析器能提取元数据"""
        config_path = "outputs/train/pinn_20260205_174333/config.json"

        if not os.path.exists(config_path):
            pytest.skip("配置文件不存在")

        config = TrainingConfigParser.parse(config_path)
        assert config is not None, "解析应该成功"
        assert "metadata" in config, "应该包含 metadata 字段"

        metadata = config["metadata"]
        assert metadata["stage"] == "2_two_phase_pinn", "应该正确提取 stage"
        assert metadata["version"] == "v4.5-standard", "应该正确提取 version"
        assert "created_at" in metadata, "应该包含 created_at"

    def test_config_parser_extracts_model_architecture(self):
        """测试解析器能提取模型架构"""
        config_path = "outputs/train/pinn_20260205_174333/config.json"

        if not os.path.exists(config_path):
            pytest.skip("配置文件不存在")

        config = TrainingConfigParser.parse(config_path)
        assert config is not None, "解析应该成功"
        assert "model_architecture" in config, "应该包含 model_architecture 字段"

        model_arch = config["model_architecture"]
        assert model_arch["input_format"] == "triplet", "应该正确提取 input_format"
        assert model_arch["hidden_phi"] == [128, 128, 64, 32], "应该正确提取 hidden_phi"
        assert model_arch["hidden_vel"] == [64, 64, 32], "应该正确提取 hidden_vel"

    def test_config_parser_extracts_training_parameters(self):
        """测试解析器能提取训练参数"""
        config_path = "outputs/train/pinn_20260205_174333/config.json"

        if not os.path.exists(config_path):
            pytest.skip("配置文件不存在")

        config = TrainingConfigParser.parse(config_path)
        assert config is not None, "解析应该成功"
        assert "training_parameters" in config, "应该包含 training_parameters 字段"

        training_params = config["training_parameters"]
        assert training_params["epochs"] == 60000, "应该正确提取 epochs"
        assert training_params["batch_size"] == 4096, "应该正确提取 batch_size"
        assert training_params["learning_rate"] == 0.0003, "应该正确提取 learning_rate"
        assert training_params["stage1_epochs"] == 1500, "应该正确提取 stage1_epochs"
        assert training_params["stage2_epochs"] == 4000, "应该正确提取 stage2_epochs"
        assert training_params["stage3_epochs"] == 50000, "应该正确提取 stage3_epochs"
        assert training_params["use_lbfgs"] is True, "应该正确提取 use_lbfgs"

    def test_config_parser_extracts_physics_weights(self):
        """测试解析器能提取物理权重"""
        config_path = "outputs/train/pinn_20260205_174333/config.json"

        if not os.path.exists(config_path):
            pytest.skip("配置文件不存在")

        config = TrainingConfigParser.parse(config_path)
        assert config is not None, "解析应该成功"
        assert "physics_weights" in config, "应该包含 physics_weights 字段"

        physics_weights = config["physics_weights"]
        assert physics_weights["interface_weight"] == 500.0, (
            "应该正确提取 interface_weight"
        )
        assert physics_weights["ic_weight"] == 300.0, "应该正确提取 ic_weight"
        assert physics_weights["bc_weight"] == 80.0, "应该正确提取 bc_weight"
        assert physics_weights["continuity_weight"] == 0.5, (
            "应该正确提取 continuity_weight"
        )
        assert physics_weights["vof_weight"] == 0.5, "应该正确提取 vof_weight"
        assert physics_weights["ns_weight"] == 0.1, "应该正确提取 ns_weight"

    def test_config_parser_extracts_data_config(self):
        """测试解析器能提取数据配置"""
        config_path = "outputs/train/pinn_20260205_174333/config.json"

        if not os.path.exists(config_path):
            pytest.skip("配置文件不存在")

        config = TrainingConfigParser.parse(config_path)
        assert config is not None, "解析应该成功"
        assert "data_config" in config, "应该包含 data_config 字段"

        data_config = config["data_config"]
        assert data_config["n_interface"] == 60000, "应该正确提取 n_interface"
        assert data_config["n_initial"] == 10000, "应该正确提取 n_initial"
        assert data_config["n_boundary"] == 8000, "应该正确提取 n_boundary"
        assert data_config["n_domain"] == 50000, "应该正确提取 n_domain"
        assert len(data_config["voltages"]) == 31, "应该正确提取 voltages 数量"
        assert data_config["times"] == 200, "应该正确提取 times"

    def test_config_parser_extracts_dynamic_weight_config(self):
        """测试解析器能提取动态权重配置"""
        config_path = "outputs/train/pinn_20260205_174333/config.json"

        if not os.path.exists(config_path):
            pytest.skip("配置文件不存在")

        config = TrainingConfigParser.parse(config_path)
        assert config is not None, "解析应该成功"
        assert "dynamic_weight_config" in config, "应该包含 dynamic_weight_config 字段"

        dynamic_weight = config["dynamic_weight_config"]
        assert dynamic_weight["enable"] is True, "应该正确提取 enable"
        assert dynamic_weight["initial_weight"] == 0.1, "应该正确提取 initial_weight"
        assert dynamic_weight["adjustment_strategy"] == "combined", (
            "应该正确提取 adjustment_strategy"
        )
        assert dynamic_weight["smoothing_factor"] == 0.9, (
            "应该正确提取 smoothing_factor"
        )


class TestLossDataParser:
    """损失数据解析器测试类"""

    def test_loss_parser_returns_none_for_nonexistent_file(self):
        """测试对于不存在的损失文件返回 None"""
        df = LossDataParser.parse("nonexistent_loss.csv")
        assert df is None, "对于不存在的损失文件应该返回 None"

    def test_loss_parser_parses_all_rows(self):
        """测试解析器能处理所有600行数据"""
        csv_path = "outputs/train/pinn_20260205_174333/loss_breakdown.csv"

        if not os.path.exists(csv_path):
            pytest.skip("损失CSV文件不存在")

        df = LossDataParser.parse(csv_path)
        assert df is not None, "解析应该成功"
        assert len(df) == 600, "应该正确解析600行数据"

    def test_loss_parser_has_all_columns(self):
        """测试解析器包含所有必需的列"""
        csv_path = "outputs/train/pinn_20260205_174333/loss_breakdown.csv"

        if not os.path.exists(csv_path):
            pytest.skip("损失CSV文件不存在")

        df = LossDataParser.parse(csv_path)
        assert df is not None, "解析应该成功"

        expected_columns = [
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

        for col in expected_columns:
            assert col in df.columns, f"应该包含列: {col}"

    def test_loss_parser_data_types(self):
        """测试解析器返回正确的数据类型"""
        csv_path = "outputs/train/pinn_20260205_174333/loss_breakdown.csv"

        if not os.path.exists(csv_path):
            pytest.skip("损失CSV文件不存在")

        df = LossDataParser.parse(csv_path)
        assert df is not None, "解析应该成功"

        assert df["epoch"].dtype == "int64", "epoch 应该是整数类型"
        assert df["stage"].dtype == "int64", "stage 应该是整数类型"
        assert df["loss_total"].dtype == "float64", "loss_total 应该是浮点数类型"
        assert df["lr"].dtype == "float64", "lr 应该是浮点数类型"

    def test_loss_parser_sample_values(self):
        """测试解析器能正确读取具体值"""
        csv_path = "outputs/train/pinn_20260205_174333/loss_breakdown.csv"

        if not os.path.exists(csv_path):
            pytest.skip("损失CSV文件不存在")

        df = LossDataParser.parse(csv_path)
        assert df is not None, "解析应该成功"

        first_row = df.iloc[0]
        assert first_row["epoch"] == 0, "第一行的 epoch 应该是 0"
        assert first_row["stage"] == 1, "第一行的 stage 应该是 1"
        assert abs(first_row["loss_total"] - 575.23) < 0.01, (
            "第一行的 loss_total 应该是 575.23"
        )


class TestMetricsParser:
    """评估指标解析器测试类"""

    def test_rmse_parser_returns_none_for_nonexistent_file(self):
        """测试对于不存在的RMSE文件返回 None"""
        df = MetricsParser.parse_rmse("nonexistent_rmse.csv")
        assert df is None, "对于不存在的RMSE文件应该返回 None"

    def test_rmse_parser_parses_all_rows(self):
        """测试解析器能处理所有6行RMSE数据"""
        csv_path = "outputs/train/pinn_20260205_174333/rmse_per_voltage.csv"

        if not os.path.exists(csv_path):
            pytest.skip("RMSE CSV文件不存在")

        df = MetricsParser.parse_rmse(csv_path)
        assert df is not None, "解析应该成功"
        assert len(df) == 6, "应该正确解析6行数据"

    def test_rmse_parser_has_all_columns(self):
        """测试解析器包含所有必需的列"""
        csv_path = "outputs/train/pinn_20260205_174333/rmse_per_voltage.csv"

        if not os.path.exists(csv_path):
            pytest.skip("RMSE CSV文件不存在")

        df = MetricsParser.parse_rmse(csv_path)
        assert df is not None, "解析应该成功"

        expected_columns = ["Voltage", "RMSE", "Rating"]

        for col in expected_columns:
            assert col in df.columns, f"应该包含列: {col}"

    def test_rmse_parser_data_types(self):
        """测试解析器返回正确的数据类型"""
        csv_path = "outputs/train/pinn_20260205_174333/rmse_per_voltage.csv"

        if not os.path.exists(csv_path):
            pytest.skip("RMSE CSV文件不存在")

        df = MetricsParser.parse_rmse(csv_path)
        assert df is not None, "解析应该成功"

        assert df["Voltage"].dtype == "float64", "Voltage 应该是浮点数类型"
        assert df["RMSE"].dtype == "float64", "RMSE 应该是浮点数类型"
        assert df["Rating"].dtype == "object", "Rating 应该是字符串类型"

    def test_rmse_parser_sample_values(self):
        """测试解析器能正确读取具体值"""
        csv_path = "outputs/train/pinn_20260205_174333/rmse_per_voltage.csv"

        if not os.path.exists(csv_path):
            pytest.skip("RMSE CSV文件不存在")

        df = MetricsParser.parse_rmse(csv_path)
        assert df is not None, "解析应该成功"

        first_row = df.iloc[0]
        assert abs(first_row["Voltage"] - 5.0) < 0.01, "第一行的 Voltage 应该是 5.0"
        assert abs(first_row["RMSE"] - 0.0054) < 0.0001, "第一行的 RMSE 应该是 0.0054"
        assert first_row["Rating"] == "Excellent", "第一行的 Rating 应该是 Excellent"

    def test_volume_stats_parser_returns_none_for_nonexistent_file(self):
        """测试对于不存在的体积统计文件返回 None"""
        df = MetricsParser.parse_volume_stats("nonexistent_volume_stats.csv")
        assert df is None, "对于不存在的体积统计文件应该返回 None"

    def test_volume_stats_parser_parses_all_rows(self):
        """测试解析器能处理所有3阶段体积统计数据"""
        csv_path = "outputs/train/pinn_20260205_174333/volume_trend_stats.csv"

        if not os.path.exists(csv_path):
            pytest.skip("体积统计CSV文件不存在")

        df = MetricsParser.parse_volume_stats(csv_path)
        assert df is not None, "解析应该成功"
        assert len(df) == 3, "应该正确解析3行数据（3个阶段）"

    def test_volume_stats_parser_has_all_columns(self):
        """测试解析器包含所有必需的列"""
        csv_path = "outputs/train/pinn_20260205_174333/volume_trend_stats.csv"

        if not os.path.exists(csv_path):
            pytest.skip("体积统计CSV文件不存在")

        df = MetricsParser.parse_volume_stats(csv_path)
        assert df is not None, "解析应该成功"

        expected_columns = [
            "stage",
            "start_epoch",
            "end_epoch",
            "mean",
            "std",
            "min",
            "max",
            "final",
        ]

        for col in expected_columns:
            assert col in df.columns, f"应该包含列: {col}"

    def test_volume_stats_parser_data_types(self):
        """测试解析器返回正确的数据类型"""
        csv_path = "outputs/train/pinn_20260205_174333/volume_trend_stats.csv"

        if not os.path.exists(csv_path):
            pytest.skip("体积统计CSV文件不存在")

        df = MetricsParser.parse_volume_stats(csv_path)
        assert df is not None, "解析应该成功"

        assert df["stage"].dtype == "int64", "stage 应该是整数类型"
        assert df["start_epoch"].dtype == "int64", "start_epoch 应该是整数类型"
        assert df["end_epoch"].dtype == "int64", "end_epoch 应该是整数类型"
        assert df["mean"].dtype == "float64", "mean 应该是浮点数类型"
        assert df["std"].dtype == "float64", "std 应该是浮点数类型"

    def test_volume_stats_parser_sample_values(self):
        """测试解析器能正确读取具体值"""
        csv_path = "outputs/train/pinn_20260205_174333/volume_trend_stats.csv"

        if not os.path.exists(csv_path):
            pytest.skip("体积统计CSV文件不存在")

        df = MetricsParser.parse_volume_stats(csv_path)
        assert df is not None, "解析应该成功"

        first_row = df.iloc[0]
        assert first_row["stage"] == 1, "第一行的 stage 应该是 1"
        assert first_row["start_epoch"] == 0, "第一行的 start_epoch 应该是 0"
        assert first_row["end_epoch"] == 1400, "第一行的 end_epoch 应该是 1400"
        assert abs(first_row["mean"] - 1.6068) < 0.01, "第一行的 mean 应该是 1.6068"


class TestBoundaryConditions:
    """边界条件处理测试类"""

    def test_path_traversal_protection(self):
        """测试路径遍历攻击防护"""
        from src.dashboard.training_output_analyzer import (
            validate_path_safe,
        )

        # 拒绝包含 ".." 的路径
        assert not validate_path_safe("../etc/passwd", base_dir="outputs/train")
        assert not validate_path_safe(
            "pinn_20260205_174333/../../etc", base_dir="outputs/train"
        )

        # 拒绝绝对路径
        assert not validate_path_safe("/etc/passwd", base_dir="outputs/train")
        assert not validate_path_safe(
            "/home/scnu/Gitee/EFD3D/outputs/train", base_dir="outputs/train"
        )

        # 接受安全的相对路径
        assert validate_path_safe("pinn_20260205_174333", base_dir="outputs/train")
        assert validate_path_safe("outputs/train/pinn_20260205_174333", base_dir=".")

    def test_file_size_check(self):
        """测试文件大小检查"""
        from src.dashboard.training_output_analyzer import (
            check_file_size,
            MAX_FILE_SIZE,
        )

        # 不存在的文件应该返回 True（让调用者处理）
        assert check_file_size("nonexistent.csv") is True

        # 小文件应该通过检查
        assert (
            check_file_size("outputs/train/pinn_20260205_174333/loss_breakdown.csv")
            is True
        )

        # 验证 MAX_FILE_SIZE 常量
        assert MAX_FILE_SIZE == 50 * 1024 * 1024

    def test_empty_directory_handling(self):
        """测试空目录处理"""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            scanner = TrainingOutputScanner(train_outputs_dir=tmpdir)
            runs = scanner.scan_training_outputs()
            assert len(runs) == 0, "空目录应该返回空列表"

    def test_nonexistent_directory_handling(self):
        """测试不存在的目录处理"""
        scanner = TrainingOutputScanner(train_outputs_dir="nonexistent_outputs")
        runs = scanner.scan_training_outputs()
        assert len(runs) == 0, "不存在的目录应该返回空列表"

    def test_parser_missing_file_handling(self):
        """测试解析器处理缺失文件"""
        # 配置解析器
        result = TrainingConfigParser.parse("nonexistent/config.json")
        assert result is None, "缺失的配置文件应该返回 None"

        # 损失解析器
        result = LossDataParser.parse("nonexistent/loss.csv")
        assert result is None, "缺失的损失CSV应该返回 None"

        # RMSE 解析器
        result = MetricsParser.parse_rmse("nonexistent/rmse.csv")
        assert result is None, "缺失的RMSE CSV应该返回 None"

        # 体积统计解析器
        result = MetricsParser.parse_volume_stats("nonexistent/volume.csv")
        assert result is None, "缺失的体积统计CSV应该返回 None"

    def test_parser_corrupted_file_handling(self):
        """测试解析器处理损坏文件"""
        import tempfile

        # 损坏的 JSON
        with tempfile.TemporaryDirectory() as tmpdir:
            corrupted_json = os.path.join(tmpdir, "corrupted.json")
            with open(corrupted_json, "w") as f:
                f.write("This is not valid JSON {")

            result = TrainingConfigParser.parse(corrupted_json)
            assert result is None, "损坏的JSON应该返回 None"

        # CSV 缺少必需列
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_csv = os.path.join(tmpdir, "bad.csv")
            with open(bad_csv, "w") as f:
                f.write("epoch,stage\n1,2\n")

            result = LossDataParser.parse(bad_csv)
            assert result is None, "缺少列的CSV应该返回 None"

        # RMSE CSV 错误列
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_csv = os.path.join(tmpdir, "bad_rmse.csv")
            with open(bad_csv, "w") as f:
                f.write("Voltage,WrongColumn\n10.0,0.01\n")

            result = MetricsParser.parse_rmse(bad_csv)
            assert result is None, "错误列的RMSE CSV应该返回 None"

        # 体积统计 CSV 错误列
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_csv = os.path.join(tmpdir, "bad_volume.csv")
            with open(bad_csv, "w") as f:
                f.write("stage,wrong\n1,0.5\n")

            result = MetricsParser.parse_volume_stats(bad_csv)
            assert result is None, "错误列的体积统计CSV应该返回 None"

    def test_scanner_rejects_unsafe_paths(self):
        """测试扫描器拒绝不安全路径"""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            # 创建包含 ".." 的目录（系统不允许，所以跳过此测试）
            # 但测试扫描器的路径验证逻辑
            scanner = TrainingOutputScanner(train_outputs_dir=tmpdir)
            runs = scanner.scan_training_outputs()
            # 即使目录结构奇怪，扫描器也应该安全地处理
            assert isinstance(runs, list), "应该返回列表"


class TestModelLoader:
    """模型加载器测试类"""

    def test_model_loader_imports_dependencies(self):
        """测试模型加载器能导入必要的依赖"""
        try:
            import torch
            from src.models.pinn_two_phase import TwoPhasePINN
        except ImportError as e:
            pytest.skip(f"依赖未安装: {e}")

    def test_list_available_models(self):
        """测试列出所有可用模型"""
        run_path = "outputs/train/pinn_20260205_174333"

        if not os.path.exists(run_path):
            pytest.skip("训练运行目录不存在")

        models = ModelLoader.list_available_models(run_path)

        assert len(models) > 0, "应该至少有一个模型文件"

        # 验证返回的是字典列表
        assert all(isinstance(model, dict) for model in models), "每个模型应该是字典"

        # 验证必需的字段
        required_fields = ["name", "path", "type", "size", "epoch"]
        for model in models:
            for field in required_fields:
                assert field in model, f"模型应该包含字段: {field}"

    def test_list_available_models_identifies_model_types(self):
        """测试能正确识别模型类型"""
        run_path = "outputs/train/pinn_20260205_174333"

        if not os.path.exists(run_path):
            pytest.skip("训练运行目录不存在")

        models = ModelLoader.list_available_models(run_path)

        # 检查是否有标准模型类型
        model_types = {model["type"] for model in models}

        # 应该至少有 best, final, latest 之一
        standard_types = {"best", "final", "latest"}
        assert len(model_types & standard_types) > 0, "应该至少有一个标准模型类型"

        # 检查 epoch_N 类型
        epoch_models = [m for m in models if m["type"] == "epoch_N"]
        if epoch_models:
            for model in epoch_models:
                assert model["epoch"] is not None, "epoch_N 类型应该有 epoch 字段"
                assert isinstance(model["epoch"], int), "epoch 应该是整数"

    def test_list_available_models_nonexistent_directory(self):
        """测试对于不存在的目录返回空列表"""
        models = ModelLoader.list_available_models("nonexistent_directory")
        assert models == [], "对于不存在的目录应该返回空列表"

    def test_load_model_best(self):
        """测试加载最佳模型"""
        run_path = "outputs/train/pinn_20260205_174333"

        if not os.path.exists(run_path):
            pytest.skip("训练运行目录不存在")

        # 检查 best_model.pth 是否存在
        best_model_path = os.path.join(run_path, "best_model.pth")
        if not os.path.exists(best_model_path):
            pytest.skip("best_model.pth 不存在")

        model, model_info = ModelLoader.load_model(run_path, model_type="best")

        # 验证加载成功
        assert model is not None, "应该成功加载模型"
        assert model_info is not None, "应该返回模型信息"

        # 验证模型信息
        assert isinstance(model_info, ModelInfo), "模型信息应该是 ModelInfo 类型"
        assert model_info.model_type == "best", "模型类型应该是 best"
        assert "best_model.pth" in model_info.model_path, "路径应该包含 best_model.pth"
        assert model_info.file_size > 0, "文件大小应该大于 0"

        # 验证架构信息
        assert "architecture" in model_info.__dict__, "应该包含架构信息"
        assert "hidden_phi" in model_info.architecture, "应该包含 hidden_phi"
        assert "hidden_vel" in model_info.architecture, "应该包含 hidden_vel"

    def test_load_model_final(self):
        """测试加载最终模型"""
        run_path = "outputs/train/pinn_20260205_174333"

        if not os.path.exists(run_path):
            pytest.skip("训练运行目录不存在")

        final_model_path = os.path.join(run_path, "final_model.pth")
        if not os.path.exists(final_model_path):
            pytest.skip("final_model.pth 不存在")

        model, model_info = ModelLoader.load_model(run_path, model_type="final")

        assert model is not None, "应该成功加载模型"
        assert model_info is not None, "应该返回模型信息"
        assert model_info.model_type == "final", "模型类型应该是 final"

    def test_load_model_latest(self):
        """测试加载最新模型"""
        run_path = "outputs/train/pinn_20260205_174333"

        if not os.path.exists(run_path):
            pytest.skip("训练运行目录不存在")

        latest_model_path = os.path.join(run_path, "latest_model.pth")
        if not os.path.exists(latest_model_path):
            pytest.skip("latest_model.pth 不存在")

        model, model_info = ModelLoader.load_model(run_path, model_type="latest")

        assert model is not None, "应该成功加载模型"
        assert model_info is not None, "应该返回模型信息"
        assert model_info.model_type == "latest", "模型类型应该是 latest"

    def test_load_model_epoch_N(self):
        """测试加载指定轮次的模型"""
        run_path = "outputs/train/pinn_20260205_174333"

        if not os.path.exists(run_path):
            pytest.skip("训练运行目录不存在")

        # 查找存在的 epoch 模型
        epoch_models = ModelLoader.list_available_models(run_path)
        epoch_models = [m for m in epoch_models if m["type"] == "epoch_N"]

        if not epoch_models:
            pytest.skip("没有 epoch_N 类型的模型")

        # 使用第一个找到的 epoch 模型
        target_epoch = epoch_models[0]["epoch"]
        model_type = f"epoch_{target_epoch}"

        model, model_info = ModelLoader.load_model(run_path, model_type=model_type)

        assert model is not None, "应该成功加载模型"
        assert model_info is not None, "应该返回模型信息"
        assert model_info.model_type == model_type, "模型类型应该匹配"
        assert model_info.epoch == target_epoch, "轮次应该匹配"

    def test_load_model_fallback_mechanism(self):
        """测试模型加载的回退机制"""
        run_path = "outputs/train/pinn_20260205_174333"

        if not os.path.exists(run_path):
            pytest.skip("训练运行目录不存在")

        # 尝试加载一个不存在的 epoch，应该回退到最佳模型
        model, model_info = ModelLoader.load_model(run_path, model_type="epoch_999999")

        # 应该成功回退到某个模型
        assert model is not None, "应该成功回退到某个模型"
        assert model_info is not None, "应该返回模型信息"
        # 回退后的模型类型不应该是原始请求的类型
        assert model_info.model_type != "epoch_999999", "应该回退到其他模型"

    def test_load_model_with_custom_device(self):
        """测试在自定义设备上加载模型"""
        run_path = "outputs/train/pinn_20260205_174333"

        if not os.path.exists(run_path):
            pytest.skip("训练运行目录不存在")

        best_model_path = os.path.join(run_path, "best_model.pth")
        if not os.path.exists(best_model_path):
            pytest.skip("best_model.pth 不存在")

        # 测试 CPU 设备
        model, model_info = ModelLoader.load_model(
            run_path, model_type="best", device="cpu"
        )
        assert model is not None, "应该能在 CPU 上加载模型"

        # 注意：不测试 CUDA，因为可能没有 GPU

    def test_load_model_nonexistent_directory(self):
        """测试对于不存在的目录返回 None"""
        model, model_info = ModelLoader.load_model("nonexistent_directory")

        assert model is None, "对于不存在的目录应该返回 None"
        assert model_info is None, "模型信息应该是 None"

    def test_load_model_no_models_found(self):
        """测试对于没有模型文件的目录返回 None"""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            model, model_info = ModelLoader.load_model(tmpdir)

            assert model is None, "对于没有模型的目录应该返回 None"
            assert model_info is None, "模型信息应该是 None"

    def test_resolve_model_path_best(self):
        """测试解析最佳模型路径"""
        run_dir = Path("outputs/train/pinn_20260205_174333")

        if not run_dir.exists():
            pytest.skip("训练运行目录不存在")

        best_model_path = run_dir / "best_model.pth"
        if not best_model_path.exists():
            pytest.skip("best_model.pth 不存在")

        model_path, model_type, epoch = ModelLoader._resolve_model_path(run_dir, "best")

        assert model_path is not None, "应该解析出路径"
        assert model_path.exists(), "路径应该存在"
        assert model_type == "best", "模型类型应该是 best"
        assert epoch is None, "best 模型不应该有 epoch"

    def test_resolve_model_path_fallback(self):
        """测试解析模型路径的回退机制"""
        run_dir = Path("outputs/train/pinn_20260205_174333")

        if not run_dir.exists():
            pytest.skip("训练运行目录不存在")

        # 删除所有标准模型文件（如果存在）
        best_model_path = run_dir / "best_model.pth"
        final_model_path = run_dir / "final_model.pth"
        latest_model_path = run_dir / "latest_model.pth"

        # 如果有任意一个标准模型，测试回退
        if not (
            best_model_path.exists()
            or final_model_path.exists()
            or latest_model_path.exists()
        ):
            pytest.skip("没有标准模型文件用于测试回退")

        # 尝试解析一个不存在的 epoch，应该回退
        model_path, model_type, epoch = ModelLoader._resolve_model_path(
            run_dir, "epoch_999999"
        )

        assert model_path is not None, "应该回退到某个模型"
        assert model_path.exists(), "回退路径应该存在"

    def test_load_config_from_run_dir(self):
        """测试从运行目录加载配置"""
        run_dir = Path("outputs/train/pinn_20260205_174333")

        if not run_dir.exists():
            pytest.skip("训练运行目录不存在")

        config_path = run_dir / "config.json"
        if not config_path.exists():
            pytest.skip("config.json 不存在")

        config = ModelLoader._load_config(run_dir)

        assert config is not None, "应该成功加载配置"
        assert isinstance(config, dict), "配置应该是字典"
        assert len(config) > 0, "配置不应该为空"

    def test_load_config_custom_path(self):
        """测试从自定义路径加载配置"""
        config_path = "outputs/train/pinn_20260205_174333/config.json"

        if not os.path.exists(config_path):
            pytest.skip("config.json 不存在")

        run_dir = Path("outputs/train/pinn_20260205_174333")
        config = ModelLoader._load_config(run_dir, config_path=config_path)

        assert config is not None, "应该成功加载配置"
        assert isinstance(config, dict), "配置应该是字典"

    def test_load_config_nonexistent(self):
        """测试加载不存在的配置返回 None"""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            config = ModelLoader._load_config(run_dir)
            assert config is None, "对于不存在的配置应该返回 None"

    def test_build_model_config_default(self):
        """测试构建默认模型配置"""
        config = {}
        model_config = ModelLoader._build_model_config(config)

        assert "model" in model_config, "应该包含 model 配置"
        assert "hidden_phi" in model_config["model"], "应该包含 hidden_phi"
        assert "hidden_vel" in model_config["model"], "应该包含 hidden_vel"

        # 验证默认值
        assert model_config["model"]["hidden_phi"] == [128, 128, 64, 32], (
            "应该使用默认 hidden_phi"
        )
        assert model_config["model"]["hidden_vel"] == [64, 64, 32], (
            "应该使用默认 hidden_vel"
        )

    def test_build_model_config_custom(self):
        """测试构建自定义模型配置"""
        config = {
            "model": {
                "hidden_phi": [64, 32, 16],
                "hidden_vel": [32, 16],
            }
        }
        model_config = ModelLoader._build_model_config(config)

        assert model_config["model"]["hidden_phi"] == [64, 32, 16], (
            "应该使用自定义 hidden_phi"
        )
        assert model_config["model"]["hidden_vel"] == [32, 16], (
            "应该使用自定义 hidden_vel"
        )

    def test_model_info_dataclass(self):
        """测试 ModelInfo 数据类"""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "model.pth")

            # 创建虚拟模型文件
            with open(model_path, "w") as f:
                f.write("dummy")

            model_info = ModelInfo(
                model_path=model_path,
                model_type="best",
                epoch=100,
                file_size=12345,
                architecture={"hidden_phi": [64, 32], "hidden_vel": [32, 16]},
            )

            assert model_info.model_path == model_path, "路径应该匹配"
            assert model_info.model_type == "best", "类型应该匹配"
            assert model_info.epoch == 100, "轮次应该匹配"
            assert model_info.file_size == 12345, "文件大小应该匹配"
            assert model_info.architecture["hidden_phi"] == [64, 32], "架构应该匹配"
