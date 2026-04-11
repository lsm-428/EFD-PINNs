#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSTM-Hybrid-PINN 训练脚本

基于预训练的 TwoPhasePINN，训练 LSTM 编码器处理电压跳变序列。

核心设计：
- LSTM 输入：(V_from, V_to, t_since) 三元组序列
- 每个三元组表示一次电压跳变及其持续时间
- 支持多步序列：0→20→30→20→0

用法：
    python train_lstm_hybrid.py --checkpoint outputs_pinn_xxx/best_model.pth
    python train_lstm_hybrid.py --checkpoint outputs_pinn_xxx/best_model.pth --epochs 10000
"""

import argparse
import datetime
import json
import logging
import os
import time
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from src.models.pinn_two_phase import (
    TwoPhasePINN,
    DataGenerator,
    PhysicsLoss,
    PHYSICS,
    DEFAULT_CONFIG,
    set_seed,
)
from src.models.lstm_pinn import LSTMHybridPINN
from src.models.lstm_pinn.physics_loss import LSTMPINNPhysicsLoss, SimplifiedPhysicsLoss

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("LSTM-Hybrid-Train")


class VoltageSequenceGenerator:
    """
    电压跳变序列生成器

    生成 (V_from, V_to, t_since) 三元组序列用于 LSTM 训练
    """

    def __init__(self, config: Dict[str, Any], device: torch.device):
        self.config = config
        self.device = device
        self.t_max = PHYSICS["t_max"]
        self.max_seq_len = config.get("sequence", {}).get("max_length", 10)

        # 物理参数
        self.tau = PHYSICS.get("tau", 0.005)  # 升压时间常数
        self.tau_recovery = PHYSICS.get("tau_recovery", 0.0075)  # 降压恢复时间常数

        # 复用现有的数据生成器获取目标 φ
        self.data_generator = DataGenerator(config, device)

    def generate_single_step_sequence(
        self, V_from: float, V_to: float, t_since: float
    ) -> torch.Tensor:
        """
        生成单步跳变序列

        Args:
            V_from: 跳变前电压
            V_to: 跳变后电压
            t_since: 跳变后经过的时间

        Returns:
            sequence: (1, 3) - 单个三元组 (V_from_norm, V_to_norm, t_since_norm)
        """
        return torch.tensor(
            [[V_from / 30.0, V_to / 30.0, t_since / self.t_max]],
            dtype=torch.float32,
            device=self.device,
        )

    def generate_multi_step_sequence(
        self, transitions: List[Tuple[float, float, float]]
    ) -> torch.Tensor:
        """
        生成多步跳变序列

        Args:
            transitions: [(V_from, V_to, t_since), ...] 跳变列表

        Returns:
            sequence: (seq_len, 3) - 三元组序列
        """
        seq = []
        for V_from, V_to, t_since in transitions:
            seq.append([V_from / 30.0, V_to / 30.0, t_since / self.t_max])
        return torch.tensor(seq, dtype=torch.float32, device=self.device)

    def generate_training_data(self) -> Dict[str, torch.Tensor]:
        """
        生成完整训练数据集

        数据类型：
        1. 单步升压：0→V (V=10,20,30)
        2. 单步降压：V→0 (V=10,20,30)
        3. 部分升压：V1→V2 (V1<V2)
        4. 部分降压：V2→V1 (V2>V1)
        5. 多步序列：0→20→30→20→0 等
        6. 稳态：V→V

        Returns:
            {
                'spatial_coords': (N, 3),
                'times': (N, 1),
                'voltage_sequences': (N, max_seq_len, 3),
                'seq_lengths': (N,),
                'phi_targets': (N, 1)
            }
        """
        data_cfg = self.config.get("data", {})
        n_samples = data_cfg.get("n_interface", 100000)

        spatial_coords = []
        times = []
        voltage_sequences = []
        seq_lengths = []
        phi_targets = []

        logger.info("生成 LSTM 训练数据...")
        logger.info("  数据格式: (V_from, V_to, t_since) 三元组序列")

        # 1. 单步升压数据 (40%)
        n_rise = int(n_samples * 0.4)
        logger.info(f"  单步升压数据: {n_rise}")
        for _ in range(n_rise):
            V_to = np.random.choice([10, 20, 30])
            V_from = 0
            t_since = np.random.uniform(0.001, 0.030)

            # 空间采样
            eta = self.data_generator.get_opening_rate(V_to, t_since)
            x, y, z = self.data_generator._sample_point_by_eta(eta)

            # 目标 φ
            phi = self.data_generator.target_phi_3d(
                x, y, z, t_since, V_to, V_prev=V_from
            )

            # 序列：单步
            seq = self.generate_single_step_sequence(V_from, V_to, t_since)

            spatial_coords.append([x, y, z])
            times.append([t_since])
            voltage_sequences.append(seq)
            seq_lengths.append(1)
            phi_targets.append([phi])

        # 2. 单步降压数据 (25%)
        # 关键：降压时需要更多边缘样本，因为油墨在边缘回流
        n_fall = int(n_samples * 0.25)
        logger.info(f"  单步降压数据: {n_fall}")
        for _ in range(n_fall):
            V_from = np.random.choice([10, 20, 30])
            V_to = 0
            t_since = np.random.uniform(0.001, 0.030)

            # 降压时的开口率：指数衰减
            eta_initial = self.data_generator.get_opening_rate(V_from, 0.020)
            eta = eta_initial * np.exp(-t_since / self.tau_recovery)

            # 降压采样策略：70% 边缘，30% 中心
            # 边缘区域是油墨所在位置，φ > 0
            if np.random.rand() < 0.7:
                # 边缘采样：r > r_open
                r_open = np.sqrt(eta * PHYSICS["Lx"] * PHYSICS["Ly"] / np.pi)
                r_max = np.sqrt((PHYSICS["Lx"] / 2) ** 2 + (PHYSICS["Ly"] / 2) ** 2)
                r = np.random.uniform(r_open, r_max)
                theta = np.random.uniform(0, 2 * np.pi)
                x = PHYSICS["Lx"] / 2 + r * np.cos(theta)
                y = PHYSICS["Ly"] / 2 + r * np.sin(theta)
                # 限制在像素范围内
                x = np.clip(x, 0, PHYSICS["Lx"])
                y = np.clip(y, 0, PHYSICS["Ly"])
                z = np.random.uniform(0, PHYSICS["h_ink"] * 2)  # 油墨层附近
            else:
                # 中心采样
                x, y, z = self.data_generator._sample_point_by_eta(eta)

            phi = self.data_generator.target_phi_3d(
                x, y, z, t_since, V_to, V_prev=V_from, t_step=0
            )

            seq = self.generate_single_step_sequence(V_from, V_to, t_since)

            spatial_coords.append([x, y, z])
            times.append([t_since])
            voltage_sequences.append(seq)
            seq_lengths.append(1)
            phi_targets.append([phi])

        # 3. 部分升压/降压数据 (15%)
        n_partial = int(n_samples * 0.15)
        logger.info(f"  部分升压/降压数据: {n_partial}")
        partial_transitions = [
            (0, 10),
            (0, 20),
            (10, 20),
            (10, 30),
            (20, 30),  # 升压
            (30, 20),
            (30, 10),
            (20, 10),
            (20, 0),
            (10, 0),  # 降压
        ]
        for _ in range(n_partial):
            V_from, V_to = partial_transitions[
                np.random.randint(len(partial_transitions))
            ]
            t_since = np.random.uniform(0.001, 0.030)

            if V_to > V_from:
                # 升压
                eta = self.data_generator.get_opening_rate(V_to, t_since)
            else:
                # 降压
                eta_initial = self.data_generator.get_opening_rate(V_from, 0.020)
                eta_target = (
                    self.data_generator.get_opening_rate(V_to, 0.020) if V_to > 0 else 0
                )
                eta = eta_target + (eta_initial - eta_target) * np.exp(
                    -t_since / self.tau_recovery
                )

            x, y, z = self.data_generator._sample_point_by_eta(eta)
            phi = self.data_generator.target_phi_3d(
                x, y, z, t_since, V_to, V_prev=V_from
            )

            seq = self.generate_single_step_sequence(V_from, V_to, t_since)

            spatial_coords.append([x, y, z])
            times.append([t_since])
            voltage_sequences.append(seq)
            seq_lengths.append(1)
            phi_targets.append([phi])

        # 4. 多步序列数据 (15%)
        n_multi = int(n_samples * 0.15)
        logger.info(f"  多步序列数据: {n_multi}")
        multi_step_patterns = [
            [(0, 20, 0.010), (20, 30, 0.005)],  # 0→20→30
            [(0, 20, 0.010), (20, 30, 0.005), (30, 20, 0.008)],  # 0→20→30→20
            [(0, 30, 0.015), (30, 0, 0.010)],  # 0→30→0
            [(0, 20, 0.010), (20, 0, 0.015)],  # 0→20→0
            [(0, 10, 0.008), (10, 20, 0.008), (20, 30, 0.008)],  # 0→10→20→30
        ]
        for _ in range(n_multi):
            pattern = multi_step_patterns[np.random.randint(len(multi_step_patterns))]

            # 随机调整每步的持续时间
            transitions = []
            for V_from, V_to, t_base in pattern:
                t_since = t_base * np.random.uniform(0.5, 1.5)
                transitions.append((V_from, V_to, t_since))

            # 最后一步的状态决定当前 φ
            last_V_from, last_V_to, last_t_since = transitions[-1]

            if last_V_to > last_V_from:
                eta = self.data_generator.get_opening_rate(last_V_to, last_t_since)
            else:
                eta_initial = self.data_generator.get_opening_rate(last_V_from, 0.020)
                eta_target = (
                    self.data_generator.get_opening_rate(last_V_to, 0.020)
                    if last_V_to > 0
                    else 0
                )
                eta = eta_target + (eta_initial - eta_target) * np.exp(
                    -last_t_since / self.tau_recovery
                )

            x, y, z = self.data_generator._sample_point_by_eta(eta)
            phi = self.data_generator.target_phi_3d(
                x, y, z, last_t_since, last_V_to, V_prev=last_V_from
            )

            seq = self.generate_multi_step_sequence(transitions)

            spatial_coords.append([x, y, z])
            times.append([last_t_since])
            voltage_sequences.append(seq)
            seq_lengths.append(len(transitions))
            phi_targets.append([phi])

        # 5. 稳态数据 (5%)
        n_steady = n_samples - n_rise - n_fall - n_partial - n_multi
        logger.info(f"  稳态数据: {n_steady}")
        for _ in range(n_steady):
            V = np.random.choice([0, 10, 20, 30])
            t_since = 0.030  # 足够长，已稳定

            eta = self.data_generator.get_opening_rate(V, t_since)
            x, y, z = self.data_generator._sample_point_by_eta(eta)
            phi = self.data_generator.target_phi_3d(x, y, z, t_since, V, V_prev=V)

            seq = self.generate_single_step_sequence(V, V, t_since)

            spatial_coords.append([x, y, z])
            times.append([t_since])
            voltage_sequences.append(seq)
            seq_lengths.append(1)
            phi_targets.append([phi])

        # 填充序列到相同长度
        padded_sequences = self._pad_sequences(voltage_sequences)

        logger.info(f"  总数据点: {len(spatial_coords)}")

        return {
            "spatial_coords": torch.tensor(
                np.array(spatial_coords), dtype=torch.float32, device=self.device
            ),
            "times": torch.tensor(
                np.array(times), dtype=torch.float32, device=self.device
            ),
            "voltage_sequences": padded_sequences,
            "seq_lengths": torch.tensor(
                seq_lengths, dtype=torch.long, device=self.device
            ),
            "phi_targets": torch.tensor(
                np.array(phi_targets), dtype=torch.float32, device=self.device
            ),
        }

    def _pad_sequences(self, sequences: List[torch.Tensor]) -> torch.Tensor:
        """填充序列到相同长度"""
        max_len = max(seq.shape[0] for seq in sequences)
        batch_size = len(sequences)

        padded = torch.zeros(batch_size, max_len, 3, device=self.device)
        for i, seq in enumerate(sequences):
            padded[i, : seq.shape[0], :] = seq

        return padded


class LSTMHybridTrainer:
    """
    LSTM-Hybrid-PINN 训练器

    使用 (V_from, V_to, t_since) 三元组序列训练 LSTM
    """

    def __init__(
        self,
        pretrained_checkpoint: str,
        config: Dict[str, Any] = None,
        freeze_pinn: bool = True,
        device: str = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config or DEFAULT_CONFIG.copy()

        # 加载预训练模型
        logger.info(f"加载预训练模型: {pretrained_checkpoint}")
        checkpoint = torch.load(
            pretrained_checkpoint, map_location=self.device, weights_only=False
        )

        pinn_config = checkpoint.get("config", {})
        pretrained_pinn = TwoPhasePINN(pinn_config)
        pretrained_pinn.load_state_dict(checkpoint["model_state_dict"])

        pretrained_epoch = checkpoint.get("epoch", "N/A")
        pretrained_loss = checkpoint.get("best_loss", "N/A")
        logger.info(f"  预训练 Epoch: {pretrained_epoch}, Loss: {pretrained_loss}")

        # 创建混合模型 - 注意 input_dim=3 (V_from, V_to, t_since)
        lstm_config = self.config.get(
            "lstm",
            {
                "input_dim": 3,  # 三元组输入
                "hidden_dim": 128,
                "num_layers": 2,
                "dropout": 0.0,
            },
        )
        lstm_config["input_dim"] = 3  # 强制设置

        self.model = LSTMHybridPINN(
            pretrained_pinn=pretrained_pinn,
            config={"lstm": lstm_config},
            freeze_pinn=freeze_pinn,
        )
        self.model.to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        logger.info(f"  总参数: {total_params:,}, 可训练: {trainable_params:,}")

        # 训练配置
        training_cfg = self.config.get("training", {})
        self.epochs = training_cfg.get("epochs", 10000)
        self.lr = training_cfg.get("learning_rate", 1e-3)
        self.min_lr = training_cfg.get("min_lr", 1e-6)
        self.batch_size = training_cfg.get("batch_size", 1024)  # 减小以适应梯度计算

        self.stage1_epochs = training_cfg.get("stage1_epochs", 5000)
        self.stage2_epochs = training_cfg.get("stage2_epochs", 10000)

        # 早停参数
        self.patience = training_cfg.get("early_stopping_patience", 2000)
        self.epochs_without_improvement = 0

        # 优化器
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr
        )

        # 序列数据生成器
        self.seq_generator = VoltageSequenceGenerator(self.config, self.device)

        # 物理损失（完整版，包含连续性方程和VOF方程）
        self.physics_loss = LSTMPINNPhysicsLoss(self.config)

        # 输出目录
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        from src.config.paths import get_output_dir

        self.output_dir = str(get_output_dir(f"train/lstm_hybrid_{timestamp}"))

        log_file = os.path.join(self.output_dir, "training.log")
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter(
                "[%(asctime)s] %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            )
        )
        logger.addHandler(file_handler)
        logger.info(f"日志文件: {log_file}")

        # 训练历史 - 分项记录
        self.history = {
            "epoch": [],
            "total_loss": [],
            "data_loss": [],  # 数据拟合损失
            "physics_loss": [],  # 物理损失（总）
            "continuity_loss": [],  # 连续性方程
            "vof_loss": [],  # VOF 方程
            "ns_loss": [],  # NS 方程
            "st_loss": [],  # 表面张力
            "volume_loss": [],  # 体积守恒
            "rise_loss": [],  # 升压损失
            "fall_loss": [],  # 降压损失
            "partial_loss": [],  # 部分升压/降压损失
            "multi_loss": [],  # 多步序列损失
            "steady_loss": [],  # 稳态损失
            "lr": [],
        }
        self.best_loss = float("inf")
        self.freeze_pinn = freeze_pinn

    def _update_optimizer_lr(self, new_lr: float):
        """更新优化器学习率，不重建优化器"""
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr
        logger.info(f"学习率更新为: {new_lr:.2e}")

    def _add_params_to_optimizer(self, params):
        """将新解冻的参数添加到优化器参数组（保持现有参数的动量状态）"""
        new_params = [p for p in params if p.requires_grad]
        if new_params:
            self.optimizer.add_param_group({"params": new_params})
            logger.info(f"添加 {len(new_params)} 个参数到优化器")

    def compute_loss(
        self, data: Dict[str, torch.Tensor], batch_idx: torch.Tensor, epoch: int = 0
    ) -> Dict[str, torch.Tensor]:
        """
        计算批次损失（分项 + 物理损失）

        Returns:
            losses: {
                'total': 总损失,
                'data': 数据拟合损失,
                'rise': 升压损失,
                'fall': 降压损失,
                'partial': 部分升压/降压损失,
                'multi': 多步序列损失,
                'steady': 稳态损失,
                'physics': 物理损失（体积守恒 + φ范围）
                'volume': 体积守恒损失
                'range': φ范围损失
            }
        """
        # 需要梯度来计算物理损失（连续性方程、VOF方程）
        spatial = (
            data["spatial_coords"][batch_idx].clone().detach().requires_grad_(True)
        )
        times = data["times"][batch_idx].clone().detach().requires_grad_(True)
        sequences = data["voltage_sequences"][batch_idx]
        targets = data["phi_targets"][batch_idx]
        seq_lengths = data["seq_lengths"][batch_idx]

        # 前向传播
        output = self.model(spatial, times, sequences)
        phi_pred = output[:, 4:5]
        velocity = output[:, 0:3]  # (u, v, w)
        pressure = output[:, 3:4]  # 压力场 p

        # 获取 LSTM 输出的等效电压
        V_eff, V_prev_eff = self.model.get_effective_voltage(sequences)

        # 数据拟合损失
        data_loss = nn.MSELoss()(phi_pred, targets)

        # 辅助损失：单步场景下 V_eff ≈ V_to, V_prev_eff ≈ V_from
        # 这帮助 LSTM 学习正确的电压语义
        single_step_mask = seq_lengths == 1
        if single_step_mask.any():
            single_idx = single_step_mask.nonzero(as_tuple=True)[0]
            single_seq = sequences[single_idx, 0, :]  # (N, 3)
            V_from_true = single_seq[:, 0:1]  # 已归一化
            V_to_true = single_seq[:, 1:2]  # 已归一化

            V_eff_single = V_eff[single_idx]
            V_prev_eff_single = V_prev_eff[single_idx]

            # 辅助损失：V_eff ≈ V_to, V_prev_eff ≈ V_from
            aux_loss = nn.MSELoss()(V_eff_single, V_to_true) + nn.MSELoss()(
                V_prev_eff_single, V_from_true
            )
        else:
            aux_loss = torch.tensor(0.0, device=self.device)

        # 物理损失权重（渐进式增加）
        physics_cfg = self.config.get("physics", {})
        physics_weight = physics_cfg.get(
            "physics_weight", 0.1
        )  # 默认权重 0.1（避免物理损失主导）

        # 使用完整物理损失（连续性方程 + VOF方程 + NS方程 + 表面张力 + 体积守恒）
        physics_losses = self.physics_loss(phi_pred, velocity, spatial, times, pressure)

        # 阶段 1：纯数据学习，物理损失权重为 0
        # 阶段 2+：逐渐增加物理损失
        if epoch < self.stage1_epochs:
            physics_scale = 0.0
        else:
            # 从 stage1 到 stage2 线性增加到 physics_weight
            progress = min(
                1.0,
                (epoch - self.stage1_epochs)
                / max(1, self.stage2_epochs - self.stage1_epochs),
            )
            physics_scale = progress * physics_weight

        # 组合物理损失（带权重）
        physics_total = physics_losses["total"] * physics_scale

        # 辅助损失权重（帮助 LSTM 学习正确的电压语义）
        aux_weight = 0.5

        # 总损失 = 数据损失 + 物理损失 + 辅助损失
        total_loss = data_loss + physics_total + aux_weight * aux_loss

        # 分项损失 - 根据序列特征分类
        losses = {
            "total": total_loss,
            "data": data_loss,
            "aux": aux_loss,  # 辅助损失
            "physics": physics_total,  # 带 scale 的物理损失
            "physics_raw": physics_losses["total"],  # 原始物理损失（不带 scale）
            # 物理损失分项
            "continuity": physics_losses.get(
                "continuity", torch.tensor(0.0, device=self.device)
            ),
            "vof": physics_losses.get("vof", torch.tensor(0.0, device=self.device)),
            "ns": physics_losses.get("ns", torch.tensor(0.0, device=self.device)),
            "surface_tension": physics_losses.get(
                "surface_tension", torch.tensor(0.0, device=self.device)
            ),
            "volume": physics_losses.get(
                "volume", torch.tensor(0.0, device=self.device)
            ),
            "interface": physics_losses.get(
                "interface", torch.tensor(0.0, device=self.device)
            ),
            "physics_scale": torch.tensor(physics_scale, device=self.device),
        }

        # 分类样本
        # 单步序列 (seq_len == 1)
        single_step = seq_lengths == 1
        # 多步序列 (seq_len > 1)
        multi_step = seq_lengths > 1

        if single_step.any():
            single_idx = single_step.nonzero(as_tuple=True)[0]
            single_seq = sequences[single_idx, 0, :]  # (N, 3) - 第一步
            single_pred = phi_pred[single_idx]
            single_target = targets[single_idx]

            # V_from, V_to
            V_from = single_seq[:, 0]
            V_to = single_seq[:, 1]

            # 升压: V_to > V_from
            rise_mask = V_to > V_from + 0.01
            if rise_mask.any():
                rise_pred = single_pred[rise_mask]
                rise_target = single_target[rise_mask]
                losses["rise"] = nn.MSELoss()(rise_pred, rise_target)

            # 降压: V_to < V_from
            fall_mask = V_to < V_from - 0.01
            if fall_mask.any():
                fall_pred = single_pred[fall_mask]
                fall_target = single_target[fall_mask]
                losses["fall"] = nn.MSELoss()(fall_pred, fall_target)

            # 稳态: V_to ≈ V_from
            steady_mask = (V_to - V_from).abs() <= 0.01
            if steady_mask.any():
                steady_pred = single_pred[steady_mask]
                steady_target = single_target[steady_mask]
                losses["steady"] = nn.MSELoss()(steady_pred, steady_target)

            # 部分升压/降压 (V_from > 0 或 V_to > 0 且不是完全升压/降压)
            partial_mask = ((V_from > 0.01) | (V_to < 0.99)) & (rise_mask | fall_mask)
            if partial_mask.any():
                partial_pred = single_pred[partial_mask]
                partial_target = single_target[partial_mask]
                losses["partial"] = nn.MSELoss()(partial_pred, partial_target)

        # 多步序列损失
        if multi_step.any():
            multi_pred = phi_pred[multi_step]
            multi_target = targets[multi_step]
            losses["multi"] = nn.MSELoss()(multi_pred, multi_target)

        return losses

    def train(self):
        """训练主循环"""
        logger.info("=" * 60)
        logger.info("开始 LSTM-Hybrid-PINN 训练")
        logger.info("  输入格式: (V_from, V_to, t_since) 三元组序列")
        logger.info(f"  冻结 PINN: {self.freeze_pinn}")
        logger.info(f"  总 Epochs: {self.epochs}")
        logger.info("=" * 60)

        # 生成数据
        logger.info("生成训练数据...")
        data = self.seq_generator.generate_training_data()
        n_samples = data["spatial_coords"].shape[0]

        # 保存配置
        with open(f"{self.output_dir}/config.json", "w") as f:
            json.dump(self.config, f, indent=2, default=str)

        start_time = time.time()

        for epoch in range(self.epochs):
            self.model.train()

            # 随机采样批次
            batch_idx = torch.randperm(n_samples, device=self.device)[: self.batch_size]

            self.optimizer.zero_grad()
            losses = self.compute_loss(data, batch_idx, epoch=epoch)
            losses["total"].backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # 记录分项损失
            self.history["epoch"].append(epoch)
            self.history["total_loss"].append(losses["total"].item())
            self.history["data_loss"].append(losses["data"].item())
            self.history["physics_loss"].append(losses["physics"].item())
            # 物理损失分项
            self.history["continuity_loss"].append(
                losses.get("continuity", torch.tensor(0)).item()
            )
            self.history["vof_loss"].append(losses.get("vof", torch.tensor(0)).item())
            self.history["ns_loss"].append(losses.get("ns", torch.tensor(0)).item())
            self.history["st_loss"].append(
                losses.get("surface_tension", torch.tensor(0)).item()
            )
            self.history["volume_loss"].append(
                losses.get("volume", torch.tensor(0)).item()
            )
            # 数据分项
            self.history["rise_loss"].append(losses.get("rise", torch.tensor(0)).item())
            self.history["fall_loss"].append(losses.get("fall", torch.tensor(0)).item())
            self.history["partial_loss"].append(
                losses.get("partial", torch.tensor(0)).item()
            )
            self.history["multi_loss"].append(
                losses.get("multi", torch.tensor(0)).item()
            )
            self.history["steady_loss"].append(
                losses.get("steady", torch.tensor(0)).item()
            )
            self.history["lr"].append(self.optimizer.param_groups[0]["lr"])

            if epoch % 100 == 0:
                if losses["total"].item() < self.best_loss:
                    self.best_loss = losses["total"].item()
                    self.epochs_without_improvement = 0
                    self.model.save_checkpoint(
                        f"{self.output_dir}/best_model.pth",
                        optimizer=self.optimizer,
                        epoch=epoch,
                        config=self.config,
                        loss=losses["total"].item(),
                    )
                else:
                    self.epochs_without_improvement += 1

                # 早停检查
                if self.epochs_without_improvement >= self.patience:
                    logger.info(f"\n早停触发: 连续 {self.patience} epochs 无改善")
                    logger.info(f"最佳损失: {self.best_loss:.6e}")
                    break

                elapsed = time.time() - start_time
                # 分项显示损失
                data_l = losses["data"].item()
                aux_l = losses["aux"].item()
                physics_l = losses["physics"].item()
                # 物理损失分项
                cont_l = losses.get("continuity", torch.tensor(0)).item()
                vof_l = losses.get("vof", torch.tensor(0)).item()
                ns_l = losses.get("ns", torch.tensor(0)).item()
                st_l = losses.get("surface_tension", torch.tensor(0)).item()
                # 数据分项
                rise_l = losses.get("rise", torch.tensor(0)).item()
                fall_l = losses.get("fall", torch.tensor(0)).item()
                multi_l = losses.get("multi", torch.tensor(0)).item()

                # 主日志行
                logger.info(
                    f"Epoch {epoch:5d} | Total: {losses['total'].item():.4e} | "
                    f"Data: {data_l:.2e} | Phys: {physics_l:.2e} | "
                    f"Rise: {rise_l:.2e} | Fall: {fall_l:.2e} | Multi: {multi_l:.2e} | "
                    f"LR: {self.optimizer.param_groups[0]['lr']:.2e} | {elapsed:.0f}s"
                )
                # 物理损失详情（每 500 epoch 显示一次）
                if epoch % 500 == 0:
                    logger.info(
                        f"  Physics: Cont={cont_l:.2e} VOF={vof_l:.2e} NS={ns_l:.2e} ST={st_l:.2e}"
                    )

            # 三阶段渐进解冻
            # Stage 1 → Stage 2: 解冻 phi_net
            if epoch == self.stage1_epochs and self.freeze_pinn:
                logger.info("\n" + "=" * 60)
                logger.info("进入 Stage 2: 解冻 phi_net，保持 vel_net 冻结")
                logger.info("=" * 60)
                self.model.unfreeze_phi_only()
                self._add_params_to_optimizer(self.model.phi_net.parameters())
                self._update_optimizer_lr(1e-4)

            # Stage 2 → Stage 3: 解冻 vel_net
            elif epoch == self.stage2_epochs:
                logger.info("\n" + "=" * 60)
                logger.info("进入 Stage 3: 解冻 vel_net，端到端微调")
                logger.info("=" * 60)
                # 直接解冻 vel_net，避免 unfreeze_pinn() 的冗余操作
                for param in self.model.vel_net.parameters():
                    param.requires_grad = True
                self._add_params_to_optimizer(self.model.vel_net.parameters())
                self._update_optimizer_lr(1e-5)

        # 保存最终模型
        self.model.save_checkpoint(
            f"{self.output_dir}/final_model.pth",
            epoch=self.epochs,
            config=self.config,
            loss=losses["total"].item(),
        )

        self.visualize()

        logger.info("=" * 60)
        logger.info(f"训练完成! 最佳损失: {self.best_loss:.6e}")
        logger.info(f"输出目录: {self.output_dir}")
        logger.info("=" * 60)

    def visualize(self):
        """可视化训练曲线（分项损失）"""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(4, 1, figsize=(12, 18))

        # 第1图：总损失 + 数据损失 + 物理损失
        ax1 = axes[0]
        ax1.semilogy(
            self.history["epoch"],
            self.history["total_loss"],
            "b-",
            linewidth=2,
            label="Total",
        )
        ax1.semilogy(
            self.history["epoch"],
            self.history["data_loss"],
            "g-",
            linewidth=1.5,
            label="Data",
            alpha=0.8,
        )
        ax1.semilogy(
            self.history["epoch"],
            self.history["physics_loss"],
            "r-",
            linewidth=1.5,
            label="Physics",
            alpha=0.8,
        )
        ax1.axvline(
            x=self.stage1_epochs,
            color="orange",
            linestyle="--",
            alpha=0.5,
            label="Stage 1 End",
        )
        ax1.axvline(
            x=self.stage2_epochs,
            color="purple",
            linestyle="--",
            alpha=0.5,
            label="Stage 2 (Unfreeze)",
        )
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("LSTM-Hybrid-PINN Training - Total / Data / Physics Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 第2图：物理损失分项（连续性、VOF、NS、表面张力）
        ax2 = axes[1]
        ax2.semilogy(
            self.history["epoch"],
            self.history["continuity_loss"],
            label="Continuity (∇·u=0)",
            alpha=0.8,
        )
        ax2.semilogy(
            self.history["epoch"],
            self.history["vof_loss"],
            label="VOF (∂φ/∂t+u·∇φ=0)",
            alpha=0.8,
        )
        ax2.semilogy(
            self.history["epoch"],
            self.history["ns_loss"],
            label="NS (Navier-Stokes)",
            alpha=0.8,
        )
        ax2.semilogy(
            self.history["epoch"],
            self.history["st_loss"],
            label="Surface Tension (CSF)",
            alpha=0.8,
        )
        ax2.semilogy(
            self.history["epoch"],
            self.history["volume_loss"],
            label="Volume Conservation",
            alpha=0.8,
        )
        ax2.axvline(x=self.stage1_epochs, color="orange", linestyle="--", alpha=0.3)
        ax2.axvline(x=self.stage2_epochs, color="purple", linestyle="--", alpha=0.3)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.set_title(
            "Physics Component Losses (Continuity / VOF / NS / Surface Tension / Volume)"
        )
        ax2.legend(loc="upper right")
        ax2.grid(True, alpha=0.3)

        # 第3图：数据分项损失
        ax3 = axes[2]
        ax3.semilogy(
            self.history["epoch"], self.history["rise_loss"], label="Rise", alpha=0.8
        )
        ax3.semilogy(
            self.history["epoch"], self.history["fall_loss"], label="Fall", alpha=0.8
        )
        ax3.semilogy(
            self.history["epoch"],
            self.history["partial_loss"],
            label="Partial",
            alpha=0.8,
        )
        ax3.semilogy(
            self.history["epoch"], self.history["multi_loss"], label="Multi", alpha=0.8
        )
        ax3.semilogy(
            self.history["epoch"],
            self.history["steady_loss"],
            label="Steady",
            alpha=0.8,
        )
        ax3.axvline(x=self.stage1_epochs, color="orange", linestyle="--", alpha=0.3)
        ax3.axvline(x=self.stage2_epochs, color="purple", linestyle="--", alpha=0.3)
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Loss")
        ax3.set_title("Data Component Losses (Rise / Fall / Partial / Multi / Steady)")
        ax3.legend(loc="upper right")
        ax3.grid(True, alpha=0.3)

        # 第4图：学习率
        ax4 = axes[3]
        ax4.semilogy(self.history["epoch"], self.history["lr"], "k-", linewidth=1.5)
        ax4.axvline(x=self.stage1_epochs, color="orange", linestyle="--", alpha=0.5)
        ax4.axvline(x=self.stage2_epochs, color="purple", linestyle="--", alpha=0.5)
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("Learning Rate")
        ax4.set_title("Learning Rate Schedule")
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/training_curve.png", dpi=150)
        plt.close()

        # 保存损失历史到 CSV（包含物理损失分项）
        import csv

        with open(f"{self.output_dir}/loss_history.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "epoch",
                    "total",
                    "data",
                    "physics",
                    "continuity",
                    "vof",
                    "ns",
                    "surface_tension",
                    "volume",
                    "rise",
                    "fall",
                    "partial",
                    "multi",
                    "steady",
                    "lr",
                ]
            )
            for i in range(len(self.history["epoch"])):
                writer.writerow(
                    [
                        self.history["epoch"][i],
                        self.history["total_loss"][i],
                        self.history["data_loss"][i],
                        self.history["physics_loss"][i],
                        self.history["continuity_loss"][i],
                        self.history["vof_loss"][i],
                        self.history["ns_loss"][i],
                        self.history["st_loss"][i],
                        self.history["volume_loss"][i],
                        self.history["rise_loss"][i],
                        self.history["fall_loss"][i],
                        self.history["partial_loss"][i],
                        self.history["multi_loss"][i],
                        self.history["steady_loss"][i],
                        self.history["lr"][i],
                    ]
                )

        logger.info(f"损失历史已保存到 {self.output_dir}/loss_history.csv")


def main():
    parser = argparse.ArgumentParser(description="训练 LSTM-Hybrid-PINN")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径（可选）")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="预训练 TwoPhasePINN checkpoint 路径",
    )
    parser.add_argument("--epochs", type=int, default=10000, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--batch", type=int, default=4096, help="批量大小")
    parser.add_argument(
        "--freeze", action="store_true", default=True, help="冻结预训练权重"
    )
    parser.add_argument("--no-freeze", dest="freeze", action="store_false")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--device", type=str, default=None, help="设备")

    args = parser.parse_args()
    set_seed(args.seed)

    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as f:
            config = json.load(f)
    else:
        config = DEFAULT_CONFIG.copy()
    config["training"]["epochs"] = args.epochs
    config["training"]["learning_rate"] = args.lr
    config["training"]["batch_size"] = args.batch

    trainer = LSTMHybridTrainer(
        pretrained_checkpoint=args.checkpoint,
        config=config,
        freeze_pinn=args.freeze,
        device=args.device,
    )
    trainer.train()


if __name__ == "__main__":
    main()
