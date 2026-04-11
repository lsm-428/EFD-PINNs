#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D Level Set PINN 训练脚本
==========================

基于成功的 2D Level Set 实现（10K epochs, 0.00% 体积误差），
训练 3D 版本用于电润湿显示仿真。

使用方法:
    python train_levelset_3d.py --config /config/v5.5_full.json

作者: EFD-PINNs Team
日期: 2026-01-09
"""

import sys
import os
import argparse
import json
import logging
import time
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, List

import matplotlib

matplotlib.use("Agg")  # 设置后端为 Agg (无头模式)，必须在导入 pyplot 之前
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
import random
import importlib.util

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from pinn_levelset_3d import LevelSet3DPINN, PHYSICS

# 尝试导入绘图工具
try:
    spec = importlib.util.spec_from_file_location(
        "plot_training_curves", Path(__file__).parent / "plot_training_curves.py"
    )
    plot_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plot_module)
    # plot_training_curves.py 模块中并没有 plot_training_curves 函数，只有 plot_curves 函数
    # 所以我们只需要保留 plot_module 即可
    HAS_PLOT_TOOL = True
except Exception as e:
    print(f"⚠️  警告: 无法导入绘图工具: {e}")
    HAS_PLOT_TOOL = False

# 使用仓库中的 device_parameters.py 作为物理参数来源
from device_parameters import (
    DEVICE_GEOMETRY,
    INITIAL_STATE,
    MATERIAL_PROPERTIES,
    ELECTRICAL_PROPERTIES,
    SIMULATION_PARAMETERS,
    ELECTROWETTING,
)


# ========================================================================
# 日志配置
# ========================================================================


def setup_logging(output_dir: Path) -> logging.Logger:
    """配置日志系统"""
    logger = logging.getLogger("LevelSet3D_Training")
    logger.setLevel(logging.DEBUG)

    # 文件处理器
    log_file = output_dir / "training.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 格式化器
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# ========================================================================
# 数据生成器
# ========================================================================


class LevelSet3DDataGenerator:
    """
    6D 输入数据生成器: (x, y, z, V_from, V_to, t_since)

    生成训练数据，包括:
    - 内部点（物理约束）
    - 边界点（边界条件）
    - 界面点（Level Set 约束）
    """

    def __init__(
        self,
        config: Dict,
        num_interior: int = 5000,
        num_boundary: int = 1000,
        num_interface: int = 2000,
    ):
        self.config = config
        self.num_interior = num_interior
        self.num_boundary = num_boundary
        self.num_interface = num_interface

        # 几何参数
        self.Lx = PHYSICS["Lx"]
        self.Ly = PHYSICS["Ly"]
        self.Lz = PHYSICS["Lz"]

        # 电压参数 - 从0开始，包含所有电压
        self.V_min = 0.0
        self.V_max = PHYSICS.get("V_max", 30.0)
        self.voltages = np.array([0, 5, 10, 15, 20, 25, 30], dtype=np.float32)

        # 时间参数
        self.t_max = PHYSICS.get("t_max", 0.05)  # 50ms

    def generate_interior_points(self) -> torch.Tensor:
        """生成内部点（使用连续时间采样 + 三种电压场景）"""
        # 空间坐标
        x = np.random.uniform(0, self.Lx, self.num_interior)
        y = np.random.uniform(0, self.Ly, self.num_interior)
        z = np.random.uniform(0, self.Lz, self.num_interior)

        # 三种电压场景
        n_steady = int(self.num_interior * 0.4)
        n_up = int(self.num_interior * 0.3)
        n_down = self.num_interior - n_steady - n_up

        V_from_list = []
        V_to_list = []

        # 稳态
        V_steady = np.random.uniform(0, 30.0, n_steady)
        V_from_list.extend(V_steady)
        V_to_list.extend(V_steady)

        # 升压
        V_up = np.random.uniform(1.0, 30.0, n_up)
        V_from_list.extend(np.zeros(n_up))
        V_to_list.extend(V_up)

        # 降压
        V_down = np.random.uniform(1.0, 30.0, n_down)
        V_from_list.extend(V_down)
        V_to_list.extend(np.zeros(n_down))

        V_from = np.array(V_from_list, dtype=np.float32)
        V_to = np.array(V_to_list, dtype=np.float32)

        # Beta 分布时间采样
        t_since = np.random.beta(0.5, 1.0, self.num_interior) * self.t_max

        # 组装: [N, 6]
        interior = np.stack([x, y, z, V_from, V_to, t_since], axis=1)

        return torch.tensor(interior, dtype=torch.float32, requires_grad=True)

    def generate_boundary_points(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成边界点: [N, 6] 和 边界类型"""
        # 这里简化为随机生成，实际应该按边界分配
        num_per_boundary = self.num_boundary // 6

        boundary_list = []
        boundary_types = []

        # 壁面边界 (z=0 和 z=Lz)
        for z_val in [0, self.Lz]:
            x = np.random.uniform(0, self.Lx, num_per_boundary)
            y = np.random.uniform(0, self.Ly, num_per_boundary)
            z = np.full(num_per_boundary, z_val)

            V_from = np.random.uniform(0, 30.0, num_per_boundary)
            V_to = np.random.uniform(0, 30.0, num_per_boundary)
            t_since = np.random.beta(0.5, 1.0, num_per_boundary) * self.t_max

            boundary = np.stack([x, y, z, V_from, V_to, t_since], axis=1)
            boundary_list.append(boundary)
            boundary_types.extend(["wall"] * num_per_boundary)

        # 侧面边界 (x=0, x=Lx, y=0, y=Ly) - 对称边界
        # 1. x=0 (左边界)
        x = np.zeros(num_per_boundary)
        y = np.random.uniform(0, self.Ly, num_per_boundary)
        z = np.random.uniform(0, self.Lz, num_per_boundary)
        V_from = np.random.choice(self.voltages, num_per_boundary)
        V_to = np.random.choice(self.voltages, num_per_boundary)
        t_since = np.random.uniform(0, self.t_max, num_per_boundary)
        boundary = np.stack([x, y, z, V_from, V_to, t_since], axis=1)
        boundary_list.append(boundary)
        boundary_types.extend(["symmetry"] * num_per_boundary)

        # 2. x=Lx (右边界)
        x = np.full(num_per_boundary, self.Lx)
        y = np.random.uniform(0, self.Ly, num_per_boundary)
        z = np.random.uniform(0, self.Lz, num_per_boundary)
        V_from = np.random.choice(self.voltages, num_per_boundary)
        V_to = np.random.choice(self.voltages, num_per_boundary)
        t_since = np.random.uniform(0, self.t_max, num_per_boundary)
        boundary = np.stack([x, y, z, V_from, V_to, t_since], axis=1)
        boundary_list.append(boundary)
        boundary_types.extend(["symmetry"] * num_per_boundary)

        # 3. y=0 (前边界)
        y = np.zeros(num_per_boundary)
        x = np.random.uniform(0, self.Lx, num_per_boundary)
        z = np.random.uniform(0, self.Lz, num_per_boundary)
        V_from = np.random.choice(self.voltages, num_per_boundary)
        V_to = np.random.choice(self.voltages, num_per_boundary)
        t_since = np.random.uniform(0, self.t_max, num_per_boundary)
        boundary = np.stack([x, y, z, V_from, V_to, t_since], axis=1)
        boundary_list.append(boundary)
        boundary_types.extend(["symmetry"] * num_per_boundary)

        # 4. y=Ly (后边界)
        y = np.full(num_per_boundary, self.Ly)
        x = np.random.uniform(0, self.Lx, num_per_boundary)
        z = np.random.uniform(0, self.Lz, num_per_boundary)
        V_from = np.random.choice(self.voltages, num_per_boundary)
        V_to = np.random.choice(self.voltages, num_per_boundary)
        t_since = np.random.uniform(0, self.t_max, num_per_boundary)
        boundary = np.stack([x, y, z, V_from, V_to, t_since], axis=1)
        boundary_list.append(boundary)
        boundary_types.extend(["symmetry"] * num_per_boundary)

        boundary = np.concatenate(boundary_list, axis=0)

        return (
            torch.tensor(boundary, dtype=torch.float32, requires_grad=True),
            boundary_types,
        )

    def generate_interface_points(self) -> torch.Tensor:
        """
        生成界面点（初始界面位置）

        简化实现: 在 z = h_ink 附近生成点
        """
        h_ink = PHYSICS.get("h_ink", 3e-6)

        x = np.random.uniform(0, self.Lx, self.num_interface)
        y = np.random.uniform(0, self.Ly, self.num_interface)
        z = np.random.normal(h_ink, h_ink * 0.1, self.num_interface)  # 界面附近

        # 电压转换（连续采样 + 三种场景）
        n_steady = int(self.num_interface * 0.4)
        n_up = int(self.num_interface * 0.3)
        n_down = self.num_interface - n_steady - n_up

        V_from_list = []
        V_to_list = []

        # 稳态
        V_steady = np.random.uniform(0, 30.0, n_steady)
        V_from_list.extend(V_steady)
        V_to_list.extend(V_steady)

        # 升压
        V_up = np.random.uniform(1.0, 30.0, n_up)
        V_from_list.extend(np.zeros(n_up))
        V_to_list.extend(V_up)

        # 降压
        V_down = np.random.uniform(1.0, 30.0, n_down)
        V_from_list.extend(V_down)
        V_to_list.extend(np.zeros(n_down))

        V_from = np.array(V_from_list, dtype=np.float32)
        V_to = np.array(V_to_list, dtype=np.float32)

        # 时间（Beta 分布）
        t_since = np.random.beta(0.5, 1.0, self.num_interface) * self.t_max

        interface = np.stack([x, y, z, V_from, V_to, t_since], axis=1)

        return torch.tensor(interface, dtype=torch.float32, requires_grad=True)

    def generate_initial_condition(self) -> torch.Tensor:
        """
        生成初始条件点 (t_since = 0)

        初始状态: 油墨充满下方，极性液体在上方
        """
        num_initial = 2000

        x = np.random.uniform(0, self.Lx, num_initial)
        y = np.random.uniform(0, self.Ly, num_initial)
        z = np.random.uniform(0, self.Lz, num_initial)

        V_from = np.zeros(num_initial)  # 初始无电压
        V_to = np.random.choice(self.voltages, num_initial)
        t_since = np.zeros(num_initial)  # t = 0

        initial = np.stack([x, y, z, V_from, V_to, t_since], axis=1)

        return torch.tensor(initial, dtype=torch.float32, requires_grad=True)

    def get_training_batch(self) -> Dict[str, torch.Tensor]:
        """生成一个训练批次的所有数据"""
        interior = self.generate_interior_points()
        boundary, boundary_types = self.generate_boundary_points()
        interface = self.generate_interface_points()
        initial = self.generate_initial_condition()

        return {
            "interior": interior,
            "boundary": boundary,
            "boundary_types": boundary_types,
            "interface": interface,
            "initial": initial,
        }


# ========================================================================
# 损失函数
# ========================================================================


class LevelSet3DLoss:
    """
    3D Level Set PINN 损失函数

    基于 2D 实现的成功配置
    """

    def __init__(self, config: Dict):
        self.config = config

        # 损失权重（从 2D 成功配置调整，添加新的多物理场约束）
        self.weights = config.get(
            "loss_weights",
            {
                "data": 100.0,  # 数据拟合损失（新增！）
                "levelset_transport": 1.0,  # Level Set 输运方程（核心）
                "continuity_1": 1.0,  # 相1 连续性方程
                "continuity_2": 1.0,  # 相2 连续性方程
                "ns_1": 1.0,  # 相1 Navier-Stokes
                "ns_2": 1.0,  # 相2 Navier-Stokes
                "interface_geometry": 0.5,  # 界面几何约束
                "boundary_wall": 10.0,  # 壁面边界条件
                "boundary_symmetry": 5.0,  # 对称边界条件
                "initial_condition": 10.0,  # 初始条件
                # 新增：多物理场约束
                "contact_angle": 5.0,  # 接触角边界条件 (COMSOL: 润湿壁)
                "electrostatic": 1.0,  # 静电场方程 (COMSOL: AC/DC -> Electrostatics)
                # 新增：符号约束（科研严谨性）
                "sign_constraint": 50.0,  # 强制 ψ 符号约定 (油墨<0, 极性>0)
            },
        )

        self.logger = logging.getLogger("LevelSet3D_Training")

        # 几何参数
        self.Lx = PHYSICS["Lx"]
        self.Ly = PHYSICS["Ly"]
        self.Lz = PHYSICS["Lz"]
        self.h_ink = PHYSICS.get("h_ink", 3e-6)
        self.t_max = PHYSICS.get("t_max", 0.05)

        # 加载训练数据
        self.training_data = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data_path = config.get("training_data_path", "data/levelset_training_data.pt")
        if Path(data_path).exists():
            raw_data = torch.load(data_path)
            # 将数据移到正确的设备
            self.training_data = {
                "interface_points": raw_data["interface_points"].to(self.device),
                "interface_targets": raw_data["interface_targets"].to(self.device),
                "ink_points": raw_data["ink_points"].to(self.device),
                "ink_targets": raw_data["ink_targets"].to(self.device),
                "polar_points": raw_data["polar_points"].to(self.device),
                "polar_targets": raw_data["polar_targets"].to(self.device),
            }
            self.logger.info(f"✅ 加载训练数据: {data_path}")
            self.logger.info(
                f"   界面点: {len(self.training_data['interface_points'])}"
            )
            self.logger.info(f"   油墨点: {len(self.training_data['ink_points'])}")
            self.logger.info(
                f"   极性液体点: {len(self.training_data['polar_points'])}"
            )
            self.logger.info(f"   设备: {self.device}")
        else:
            self.logger.warning(f"⚠️  训练数据文件不存在: {data_path}")
            self.logger.warning("   将使用纯物理约束训练（可能不收敛）")

    def data_loss(self, model: LevelSet3DPINN) -> torch.Tensor:
        """
        数据拟合损失

        使用 Stage 1 解析解生成的真实 ψ 值来约束模型
        """
        if self.training_data is None:
            return torch.tensor(0.0, device=self.device)

        total_data_loss = 0.0
        count = 0

        # 1. 界面数据损失
        if "interface_points" in self.training_data:
            pts = self.training_data["interface_points"]
            tgt = self.training_data["interface_targets"]

            # 随机采样一批
            batch_size = min(5000, len(pts))
            idx = torch.randperm(len(pts))[:batch_size]

            # 前向传播计算 ψ（需要梯度！）
            output = model(pts[idx])
            psi_pred = output[:, 6:7]

            # MSE 损失
            loss_interface = torch.mean((psi_pred.squeeze() - tgt[idx]) ** 2)
            total_data_loss += loss_interface
            count += 1

        # 2. 油墨区域数据损失
        if "ink_points" in self.training_data:
            pts = self.training_data["ink_points"]
            tgt = self.training_data["ink_targets"]

            batch_size = min(5000, len(pts))
            idx = torch.randperm(len(pts))[:batch_size]

            output = model(pts[idx])
            psi_pred = output[:, 6:7]

            loss_ink = torch.mean((psi_pred.squeeze() - tgt[idx]) ** 2)
            total_data_loss += loss_ink
            count += 1

        # 3. 极性液体区域数据损失
        if "polar_points" in self.training_data:
            pts = self.training_data["polar_points"]
            tgt = self.training_data["polar_targets"]

            batch_size = min(5000, len(pts))
            idx = torch.randperm(len(pts))[:batch_size]

            output = model(pts[idx])
            psi_pred = output[:, 6:7]

            loss_polar = torch.mean((psi_pred.squeeze() - tgt[idx]) ** 2)
            total_data_loss += loss_polar
            count += 1

        return (
            total_data_loss / count
            if count > 0
            else torch.tensor(0.0, device=self.device)
        )

    def compute_total_loss(
        self, model: LevelSet3DPINN, data: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算总损失

        Returns:
            total_loss: 总损失
            loss_dict: 各项损失的详细信息
        """
        loss_dict = {}

        # Initialize total_loss on the same device as the model
        device = next(model.parameters()).device
        total_loss = torch.tensor(0.0, device=device)

        # 0. 数据拟合损失（新增！最重要的约束）
        loss_data = self.data_loss(model)
        loss_dict["data"] = loss_data.item()
        total_loss += self.weights["data"] * loss_data

        # 1. Level Set 输运方程（最重要）
        if "interface" in data:
            loss_levelset = self.levelset_transport_loss(model, data["interface"])
            loss_dict["levelset_transport"] = loss_levelset.item()
            total_loss += self.weights["levelset_transport"] * loss_levelset

        # 2. 连续性方程
        if "interior" in data:
            interior = data["interior"]  # 已经有 requires_grad=True
            y_pred = model(interior)

            # 传入完整的 interior,不要只传 xyz
            loss_cont1, loss_cont2 = model.continuity_loss_3d(y_pred, interior)

            loss_dict["continuity_1"] = loss_cont1.item()
            loss_dict["continuity_2"] = loss_cont2.item()

            total_loss += self.weights["continuity_1"] * loss_cont1
            total_loss += self.weights["continuity_2"] * loss_cont2

        # 3. Navier-Stokes 方程（始终计算以确保 V_field 在计算图中，即使权重为0）
        # 这样可以避免 Stage 2 启用 NS 时出现梯度断开问题
        if "interior" in data:
            interior = data["interior"]
            y_pred = model(interior)

            # 只有当任一 NS 权重 > 0 时才计算实际损失值
            if self.weights["ns_1"] > 0 or self.weights["ns_2"] > 0:
                loss_ns1, loss_ns2 = model.navier_stokes_loss_3d(y_pred, interior)

                loss_dict["ns_1"] = loss_ns1.item()
                loss_dict["ns_2"] = loss_ns2.item()

                total_loss += self.weights["ns_1"] * loss_ns1
                total_loss += self.weights["ns_2"] * loss_ns2
            else:
                # 即使权重为0，也要调用模型以确保 V_field 在计算图中
                # 但不将损失加入 total_loss
                _ = y_pred[:, 9:10].sum()  # 使用 V_field (第10个输出)
                loss_dict["ns_1"] = 0.0
                loss_dict["ns_2"] = 0.0
        else:
            loss_dict["ns_1"] = 0.0
            loss_dict["ns_2"] = 0.0

        # 4. 边界条件（包括接触角边界条件）
        if "boundary" in data:
            loss_bc_wall, loss_bc_contact = self.boundary_condition_loss(
                model,
                data["boundary"],
                data["boundary_types"],
                contact_angle_weight=self.weights["contact_angle"],
            )
            loss_dict["boundary_wall"] = loss_bc_wall.item()
            loss_dict["contact_angle"] = loss_bc_contact.item()
            total_loss += self.weights["boundary_wall"] * loss_bc_wall
            total_loss += self.weights["contact_angle"] * loss_bc_contact

        # 5. 静电场方程（如果启用）
        # CRITICAL FIX: Always compute electrostatic loss to maintain V_field gradient connection
        # Even when weight is 0, use tiny weight (1e-4) to prevent gradient disconnection
        if "interior" in data:
            # 使用 V_field (第10个输出) 求解静电场方程
            # ∇·(ε(ψ)∇V) = 0, with boundary conditions: V(z=0) = V_to, V(z=Lz) = 0
            interior = data["interior"].requires_grad_(True)
            y_pred_interior = model(interior)

            V_field = y_pred_interior[:, 9:10]
            psi = y_pred_interior[:, 6:7]
            xyz = interior[:, 0:3]

            # 5a. 静电场方程残差（内部点）
            loss_electrostatic_pde = model.electrostatic_loss(V_field, psi, xyz)

            # 5b. 电势边界条件（底部和顶部电极）
            # 底部 (z=0): V = V_to (施加电压)
            # 顶部 (z=Lz): V = 0 (接地)
            z = xyz[:, 2:3]
            V_to = interior[:, 4:5]

            # 底部边界条件：V(z=0) = V_to
            bottom_mask = (z < 1e-7).flatten()
            # 顶部边界条件：V(z=Lz) = 0
            Lz = model.Lz
            top_mask = (z > Lz - 1e-7).flatten()

            loss_bc_bottom = torch.tensor(0.0, device=data["interior"].device)
            loss_bc_top = torch.tensor(0.0, device=data["interior"].device)

            if bottom_mask.any():
                V_bottom = V_field[bottom_mask]
                V_to_bottom = V_to[bottom_mask]
                loss_bc_bottom = torch.mean((V_bottom - V_to_bottom) ** 2)

            if top_mask.any():
                V_top = V_field[top_mask]
                loss_bc_top = torch.mean(V_top**2)  # 应该为0

            loss_electrostatic = loss_electrostatic_pde + loss_bc_bottom + loss_bc_top
            loss_dict["electrostatic"] = loss_electrostatic.item()

            # GRADIENT CONNECTION FIX: Use actual weight if > 0, else use tiny weight (1e-4)
            # This ensures V_field always maintains gradient connection, preventing
            # loss explosion when electrostatic is suddenly activated at stage transitions
            electrostatic_weight = self.weights["electrostatic"]
            if electrostatic_weight <= 0:
                electrostatic_weight = 1e-4  # Minimal weight to maintain gradient
            total_loss += electrostatic_weight * loss_electrostatic
        else:
            loss_dict["electrostatic"] = 0.0

        # 6. 初始条件
        if "initial" in data:
            loss_ic = self.initial_condition_loss(model, data["initial"])
            loss_dict["initial"] = loss_ic.item()
            total_loss += self.weights["initial_condition"] * loss_ic

        # 7. 符号约束（科研严谨性修复）
        if "interior" in data and self.weights["sign_constraint"] > 0:
            loss_sc = self.sign_constraint_loss(model, data["interior"])
            loss_dict["sign_constraint"] = loss_sc.item()
            total_loss += self.weights["sign_constraint"] * loss_sc
        else:
            loss_dict["sign_constraint"] = 0.0

        # 8. 体积守恒约束（从VOF方法移植）
        if "interior" in data and self.weights["volume_conservation"] > 0:
            loss_vc = self.volume_conservation_loss(model, data["interior"])
            loss_dict["volume_conservation"] = loss_vc.item()
            total_loss += self.weights["volume_conservation"] * loss_vc
        else:
            loss_dict["volume_conservation"] = 0.0

        # 9. ψ空间分布约束（防止模式坍塌）
        if "interior" in data and self.weights["psi_spatial"] > 0:
            loss_ps = self.psi_spatial_loss(model, data["interior"])
            loss_dict["psi_spatial"] = loss_ps.item()
            total_loss += self.weights["psi_spatial"] * loss_ps
        else:
            loss_dict["psi_spatial"] = 0.0

        # 10. 开口率单调性约束（从VOF方法移植）
        if self.weights["aperture_monotonicity"] > 0:
            loss_am = self.aperture_monotonicity_loss(model)
            loss_dict["aperture_monotonicity"] = loss_am.item()
            total_loss += self.weights["aperture_monotonicity"] * loss_am
        else:
            loss_dict["aperture_monotonicity"] = 0.0

        return total_loss, loss_dict

    def levelset_transport_loss(
        self, model: LevelSet3DPINN, data: torch.Tensor
    ) -> torch.Tensor:
        """Level Set 输运方程损失"""
        # data 已经在生成时设置了 requires_grad=True
        y_pred = model(data)

        u1 = y_pred[:, 0:1]
        v1 = y_pred[:, 1:2]
        w1 = y_pred[:, 2:3]
        u2 = y_pred[:, 3:4]
        v2 = y_pred[:, 4:5]
        w2 = y_pred[:, 5:6]
        psi = y_pred[:, 6:7]

        # 直接传入完整的 data,不要分割后重新组合
        return model.levelset_transport_loss_3d(psi, u1, v1, w1, u2, v2, w2, data)

    def boundary_condition_loss(
        self,
        model: LevelSet3DPINN,
        data: torch.Tensor,
        boundary_types: list,
        contact_angle_weight: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        边界条件损失

        Args:
            boundary_types: 边界类型列表 ["wall"] 或 ["symmetry"]

        Returns:
            loss_wall: 壁面速度为零的损失
            loss_contact: 接触角边界条件损失
        """
        y_pred = model(data)
        xyz = data[:, 0:3]

        # 区分 wall 和 symmetry 边界条件
        # wall (z=0): 无滑移 u=v=w=0
        # symmetry (x=0,x=Lx,y=0,y=Ly): 仅法向速度为0

        # 识别 wall 点 (z ≈ 0)
        z = xyz[:, 2:3]
        wall_mask = (z < 1e-7).flatten()

        # 识别 symmetry 点 (x≈0, x≈Lx, y≈0, y≈Ly)
        Lx = 174e-6
        Ly = 174e-6
        x = xyz[:, 0:1]
        y = xyz[:, 1:2]
        symmetry_mask = (
            (x < 1e-7) | (x > Lx - 1e-7) | (y < 1e-7) | (y > Ly - 1e-7)
        ).flatten()

        # 速度分量
        u1 = y_pred[:, 0:1]
        v1 = y_pred[:, 1:2]
        w1 = y_pred[:, 2:3]

        # wall: 无滑移 (u=v=w=0)
        if wall_mask.any():
            loss_wall = torch.mean(
                u1[wall_mask] ** 2 + v1[wall_mask] ** 2 + w1[wall_mask] ** 2
            )
        else:
            loss_wall = torch.tensor(0.0, device=data.device)

        # symmetry: 仅法向速度为0
        # 根据坐标判断法向方向
        if symmetry_mask.any():
            sym_xyz = xyz[symmetry_mask]
            sym_u1 = u1[symmetry_mask]
            sym_v1 = v1[symmetry_mask]
            sym_w1 = w1[symmetry_mask]

            # x 边界: 法向速度 u=0
            x_sym = (sym_xyz[:, 0] < 1e-7) | (sym_xyz[:, 0] > Lx - 1e-7)
            # y 边界: 法向速度 v=0
            y_sym = (sym_xyz[:, 1] < 1e-7) | (sym_xyz[:, 1] > Ly - 1e-7)

            sym_loss = torch.tensor(0.0, device=data.device)
            if x_sym.any():
                sym_loss = sym_loss + torch.mean(sym_u1[x_sym] ** 2)
            if y_sym.any():
                sym_loss = sym_loss + torch.mean(sym_v1[y_sym] ** 2)

            loss_wall = loss_wall + sym_loss

        # 2. 接触角边界条件（仅在底部壁面 z=0）
        # IMPORTANT: Only compute if weight > 0 to avoid expensive finite difference calculation
        loss_contact = torch.tensor(0.0, device=data.device)

        if contact_angle_weight > 0:
            # 从边界点中筛选出底部壁面点
            z = data[:, 2:3]
            bottom_mask = (z < 1e-7).flatten()  # z ≈ 0

            if bottom_mask.any():
                # 获取底部壁面点的数据
                bottom_data = data[bottom_mask]
                bottom_pred = y_pred[bottom_mask]

                # 计算施加电压 V_applied = V_to（假设 V_from 是之前的电压）
                V_from = bottom_data[:, 3:4]
                V_to = bottom_data[:, 4:5]
                V_applied = V_to  # 简化：使用 V_to 作为施加电压

                # 为了计算空间梯度，需要重新前向传播，使用 requires_grad=True 的输入
                bottom_data_grad = bottom_data.clone().detach().requires_grad_(True)
                bottom_pred_grad = model(bottom_data_grad)

                # 调用模型中的 contact_angle_loss 函数
                loss_contact = model.contact_angle_loss(
                    bottom_pred_grad, bottom_data_grad, V_applied
                )

        return loss_wall, loss_contact

    def initial_condition_loss(
        self, model: LevelSet3DPINN, data: torch.Tensor
    ) -> torch.Tensor:
        """
        初始条件损失

        根据 LEVELSET_PHYSICS.md:
        t=0 时，界面应该在 z = h_ink 处（油墨平铺）
        ψ = h_ink（界面高度）

        注意：模型输出是归一化的 ψ ∈ [0,1]，需要用归一化目标
        """
        y_pred = model(data)
        psi = y_pred[:, 6:7]

        psi_initial = torch.full_like(psi, 1.0)  # 归一化后 h_ink=3e-6 对应 1.0

        return torch.mean((psi - psi_initial) ** 2)

    def sign_constraint_loss(
        self, model: LevelSet3DPINN, data: torch.Tensor
    ) -> torch.Tensor:
        y_pred = model(data)
        psi = y_pred[:, 6:7]

        z = data[:, 2:3]
        h_ink = PHYSICS.get("h_ink", 3e-6)

        loss_nonnegative = torch.mean(torch.relu(psi + 0.1) ** 2)

        bottom_zone = z < 0.5 * h_ink
        loss_bottom = torch.tensor(0.0, device=data.device)
        if bottom_zone.any():
            psi_bottom = psi[bottom_zone]
            loss_bottom = torch.mean(torch.relu(0.5 - psi_bottom) ** 2)

        top_zone = z > 2.0 * h_ink
        loss_top = torch.tensor(0.0, device=data.device)
        if top_zone.any():
            psi_top = psi[top_zone]
            loss_top = torch.mean(torch.relu(0.5 - psi_top) ** 2)

        loss_sign = loss_nonnegative + loss_bottom + loss_top

        return loss_sign

    def generate_volume_sample_points(
        self, V_to: float, num_samples: int = 10000
    ) -> torch.Tensor:
        """
        生成指定电压下的体积采样点

        Args:
            V_to: 目标电压
            num_samples: 采样点数量

        Returns:
            sample_points: [num_samples, 6] 的采样点
        """
        # 空间坐标均匀采样
        x = np.random.uniform(0, self.Lx, num_samples)
        y = np.random.uniform(0, self.Ly, num_samples)
        z = np.random.uniform(0, self.Lz, num_samples)

        # 电压设置：稳态场景 (V_from = V_to)
        V_from = np.full(num_samples, V_to, dtype=np.float32)
        V_to_array = np.full(num_samples, V_to, dtype=np.float32)

        # 时间：使用稳态时间 (t_max)
        t_since = np.full(num_samples, self.t_max, dtype=np.float32)

        # 组装
        samples = np.stack([x, y, z, V_from, V_to_array, t_since], axis=1)
        return torch.tensor(samples, dtype=torch.float32, device=self.device)

    def improved_volume_conservation_loss(
        self, model: LevelSet3DPINN, data: torch.Tensor
    ) -> torch.Tensor:
        """
        改进的体积守恒损失：确保所有电压下体积保持一致且等于理论值

        关键改进：
        1. 显式采样多个电压点 (0, 5, 10, 15, 20, 25, 30V)
        2. 约束每个电压下的体积等于理论体积
        3. 使用高分辨率采样 (10000 points per voltage)
        4. 避免依赖训练数据分布的不一致性

        Args:
            model: LevelSet模型
            data: 采样点 [N, 6] (用于设备信息)

        Returns:
            loss: 体积守恒损失
        """
        device = next(model.parameters()).device

        # 定义要测试的电压点
        test_voltages = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
        volumes = []

        # 为每个电压生成高分辨率采样
        for V in test_voltages:
            # 生成该电压下的采样点
            volume_samples = self.generate_volume_sample_points(V, num_samples=10000)
            volume_samples = volume_samples.to(device)

            # 模型推理（需要梯度来学习体积守恒！）
            y_pred = model(volume_samples)
            psi = y_pred[:, 6:7]

            # 新定义: ψ ≥ 0
            # ψ = 0: 界面(开口边界)
            # ψ > 0: 油墨高度
            # 体积 = ψ 均值 (对于 ψ > 0 的点)
            ink_mask = psi > 0
            if ink_mask.sum() > 0:
                ink_fraction = psi[ink_mask].mean() / PHYSICS["Lz"]
            else:
                ink_fraction = torch.tensor(0.0, device=psi.device)
            volumes.append(ink_fraction)

        # 计算理论体积分数
        theoretical_volume = PHYSICS["h_ink"] / PHYSICS["Lz"]

        # 计算每个电压下的体积误差
        volume_losses = []
        for vol in volumes:
            error = (vol - theoretical_volume) ** 2
            volume_losses.append(error)

        # 返回平均损失
        total_loss = torch.mean(torch.stack(volume_losses))
        return total_loss

    def volume_conservation_loss(
        self, model: LevelSet3DPINN, data: torch.Tensor
    ) -> torch.Tensor:
        """
        体积守恒约束（改进版）

        使用改进的体积守恒损失函数
        """
        return self.improved_volume_conservation_loss(model, data)

    def psi_spatial_loss(
        self, model: LevelSet3DPINN, data: torch.Tensor
    ) -> torch.Tensor:
        y_pred = model(data)
        psi = y_pred[:, 6:7]

        psi_variance = torch.var(psi)
        min_variance = 0.1
        loss_variance = torch.relu(min_variance - psi_variance)

        psi_mean = torch.mean(psi)
        loss_mean = (psi_mean - 0.5) ** 2

        return loss_variance + loss_mean

    def aperture_monotonicity_loss(self, model: LevelSet3DPINN) -> torch.Tensor:
        """
        开口率单调性约束（从VOF方法移植）

        物理原理：
        - 电压升高 → 开口率增大（单调）
        - 电压降低 → 开口率减小（单调）

        实现：
        - 在Z=0平面采样
        - 测试不同电压下的ψ分布
        - 约束：V1 < V2 → aperture(V1) < aperture(V2)

        Args:
            model: LevelSet模型

        Returns:
            loss: 单调性损失
        """
        Lx = PHYSICS.get("Lx", 174e-6)
        Ly = PHYSICS.get("Ly", 174e-6)
        t_max = PHYSICS.get("t_max", 0.05)

        # 在Z=0平面采样
        n_sample = 1000
        x = torch.linspace(0, Lx, int(np.sqrt(n_sample)), device=self.device)
        y = torch.linspace(0, Ly, int(np.sqrt(n_sample)), device=self.device)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        Z = torch.zeros_like(X)

        # 展平
        xyz = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=1)

        # 测试4个电压点（包括0V以约束初始状态）
        voltages = [0.0, 10.0, 20.0, 30.0]

        apertures = []
        for V in voltages:
            V_from = torch.zeros(len(xyz), 1, device=self.device)
            V_to = torch.full((len(xyz), 1), V, device=self.device)
            t_since = torch.full((len(xyz), 1), t_max, device=self.device)

            input_data = torch.cat([xyz, V_from, V_to, t_since], dim=1)

            output = model(input_data)
            psi = output[:, 6:7]

            epsilon = 1e-6
            aperture = (psi.squeeze() < epsilon).float().mean()
            apertures.append(aperture)

        # 单调性约束：aperture(0V) < aperture(10V) < aperture(20V) < aperture(30V)
        loss_mono = torch.tensor(0.0, device=self.device)

        for i in range(len(apertures) - 1):
            violation = torch.relu(apertures[i] - apertures[i + 1])
            loss_mono = loss_mono + violation

        # 额外约束：0V aperture 必须很小 (< 5%)
        loss_0V_small = torch.relu(apertures[0] - 0.05)  # Penalize if > 5%

        return loss_mono + 10.0 * loss_0V_small


# ========================================================================
# 可视化工具
# ========================================================================


class TrainingPlotter:
    """
    训练过程可视化工具

    实时绘制训练曲线并保存到输出目录，无需手动运行 plot_training_curves.py
    """

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.logger = logging.getLogger("LevelSet3D_Training")

    def update_plots(self, history_list: List[Dict[str, float]]):
        """根据当前的 loss history 更新图表"""
        if not history_list:
            return

        try:
            # 1. 转换数据格式 (List of Dicts -> Dict of Lists)
            history = self._convert_history(history_list)

            # 2. 绘制图表
            self._plot_training_curve(history)
            self._plot_loss_fractions(history)
            self._plot_symbol_accuracy(history)
            self._plot_dashboard(history)
            self.logger.info(f"训练曲线已更新: {self.output_dir}")

        except Exception as e:
            self.logger.warning(f"绘图失败: {str(e)}")
            # 只有在调试模式下才打印堆栈
            # import traceback
            # self.logger.debug(traceback.format_exc())

    def _convert_history(self, history_list: List[Dict[str, float]]) -> Dict[str, List]:
        """将 list of dicts 转换为 dict of lists"""
        if not history_list:
            return {}

        # 获取所有可能的键
        keys = set()
        for h in history_list:
            keys.update(h.keys())

        # 初始化
        history = {k: [] for k in keys}

        # 填充数据
        for h in history_list:
            for k in keys:
                history[k].append(h.get(k, 0.0))

        return history

    def _plot_training_curve(self, history: Dict[str, List]):
        """绘制总损失和分项损失曲线"""
        fig, ax = plt.subplots(figsize=(12, 7))
        epochs = history.get("epoch", range(len(history["total"])))

        # 绘制主要loss
        if "total" in history:
            ax.semilogy(
                epochs, history["total"], label="Total Loss", linewidth=2, color="black"
            )

        # 绘制分项 Loss (如果存在且非零)
        components = [
            ("data", "Data Loss", 0.7),
            ("levelset_transport", "LevelSet Transport", 0.7),
            ("continuity_1", "Continuity 1", 0.5),
            ("continuity_2", "Continuity 2", 0.5),
            ("contact_angle", "Contact Angle", 0.3),
            ("electrostatic", "Electrostatic", 0.3),
            (
                "sign_constraint",
                "volume_conservation",
                "psi_spatial",
                "aperture_monotonicity",
                "Sign Constraint",
                0.3,
            ),
            ("volume_conservation", "Volume Conservation", 0.3),
            ("psi_spatial", "Psi Spatial", 0.3),
            ("aperture_monotonicity", "Aperture Mono", 0.3),
        ]

        for key, label, alpha in components:
            if key in history and np.any(np.array(history[key]) > 1e-10):
                ax.semilogy(epochs, history[key], label=label, alpha=alpha)

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss (log scale)", fontsize=12)
        ax.set_title("Training Loss Curve", fontsize=14, fontweight="bold")
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "training_curve.png", dpi=100, bbox_inches="tight"
        )
        plt.close()

    def _plot_loss_fractions(self, history: Dict[str, List]):
        """绘制 Loss 占比堆叠图"""
        if "total" not in history:
            return

        fig, ax = plt.subplots(figsize=(12, 7))
        epochs = history.get("epoch", range(len(history["total"])))

        # 计算loss占比
        total = np.array(history["total"])
        total = np.where(total < 1e-10, 1.0, total)  # 避免除以0

        # 定义要堆叠的组件
        stack_components = [
            "data",
            "levelset_transport",
            "continuity_1",
            "continuity_2",
            "contact_angle",
            "electrostatic",
            "sign_constraint",
            "volume_conservation",
            "psi_spatial",
            "aperture_monotonicity",
            "volume_conservation",
            "psi_spatial",
            "aperture_monotonicity",
        ]

        # 过滤掉不存在或全为0的组件
        active_components = []
        for key in stack_components:
            if key in history and np.any(np.array(history[key]) > 0):
                active_components.append(key)

        if not active_components:
            plt.close()
            return

        # 计算累积占比
        accumulated_frac = np.zeros_like(total)

        for key in active_components:
            frac = np.array(history[key]) / total
            ax.fill_between(
                epochs, accumulated_frac, accumulated_frac + frac, label=key, alpha=0.7
            )
            accumulated_frac += frac

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss Fraction", fontsize=12)
        ax.set_title("Loss Components Fraction", fontsize=14, fontweight="bold")
        ax.set_ylim([0, 1.0])
        ax.legend(loc="upper left", fontsize=10, bbox_to_anchor=(1.02, 1))
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "loss_components_fraction.png",
            dpi=100,
            bbox_inches="tight",
        )
        plt.close()

    def _plot_symbol_accuracy(self, history: Dict[str, List]):
        """绘制符号准确率"""
        if "ink_negative_pct" not in history:
            return

        fig, ax = plt.subplots(figsize=(12, 6))
        epochs = history.get("epoch", range(len(history["ink_negative_pct"])))

        if "ink_negative_pct" in history:
            # 转换为百分比 (如果不是的话)
            vals = np.array(history["ink_negative_pct"])
            if np.max(vals) <= 1.0:
                vals *= 100
            ax.plot(
                epochs, vals, label="Ink<0 (ψ<0 correct)", linewidth=2, color="blue"
            )

        if "polar_positive_pct" in history:
            vals = np.array(history["polar_positive_pct"])
            if np.max(vals) <= 1.0:
                vals *= 100
            ax.plot(
                epochs, vals, label="Polar>0 (ψ>0 correct)", linewidth=2, color="red"
            )

        # 理想值线
        ax.axhline(
            y=100.0, color="green", linestyle=":", alpha=0.5, label="Ideal (100%)"
        )

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Accuracy (%)", fontsize=12)
        ax.set_title("Symbol Convention Accuracy", fontsize=14, fontweight="bold")
        ax.set_ylim([0, 105])
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "symbol_accuracy.png", dpi=100, bbox_inches="tight"
        )
        plt.close()

    def _plot_dashboard(self, history: Dict[str, List]):
        """绘制综合仪表盘"""
        if "total" not in history:
            return

        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        epochs = history.get("epoch", range(len(history["total"])))

        # 4.1 Total Loss & Components (左上)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.semilogy(
            epochs, history["total"], label="Total", linewidth=2, color="black"
        )
        if "data" in history:
            ax1.semilogy(epochs, history["data"], label="Data", alpha=0.7)
        if "levelset_transport" in history:
            ax1.semilogy(
                epochs, history["levelset_transport"], label="LevelSet", alpha=0.7
            )
        ax1.set_ylabel("Loss (log scale)", fontsize=11)
        ax1.set_title("Training Loss", fontsize=12, fontweight="bold")
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        # 4.2 Symbol Accuracy (右上)
        ax2 = fig.add_subplot(gs[0, 1])
        if "ink_negative_pct" in history:
            vals = np.array(history["ink_negative_pct"])
            if np.max(vals) <= 1.0:
                vals *= 100
            ax2.plot(epochs, vals, label="Ink<0", linewidth=2, color="blue")
        if "polar_positive_pct" in history:
            vals = np.array(history["polar_positive_pct"])
            if np.max(vals) <= 1.0:
                vals *= 100
            ax2.plot(epochs, vals, label="Polar>0", linewidth=2, color="red")
        ax2.axhline(y=100.0, color="green", linestyle=":", alpha=0.5, label="Ideal")
        ax2.set_ylabel("Accuracy (%)", fontsize=11)
        ax2.set_title("Symbol Convention Accuracy", fontsize=12, fontweight="bold")
        ax2.set_ylim([0, 105])
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        # 4.3 Loss Components Fraction (左下)
        ax3 = fig.add_subplot(gs[1, 0])
        total = np.array(history["total"])
        total = np.where(total < 1e-10, 1.0, total)

        stack_components = [
            "data",
            "levelset_transport",
            "continuity_1",
            "continuity_2",
            "electrostatic",
        ]
        accumulated_frac = np.zeros_like(total)

        for key in stack_components:
            if key in history and np.any(np.array(history[key]) > 0):
                frac = np.array(history[key]) / total
                ax3.fill_between(
                    epochs,
                    accumulated_frac,
                    accumulated_frac + frac,
                    label=key,
                    alpha=0.7,
                )
                accumulated_frac += frac

        ax3.set_xlabel("Epoch", fontsize=11)
        ax3.set_ylabel("Loss Fraction", fontsize=11)
        ax3.set_title("Loss Components Fraction", fontsize=12, fontweight="bold")
        ax3.set_ylim([0, 1.0])
        ax3.legend(loc="upper left", fontsize=8, bbox_to_anchor=(1.02, 1))
        ax3.grid(True, alpha=0.3, axis="y")

        # 4.4 Recent Loss Trend (右下) - 替代原来的 Stage Bar，显示最近 1000 epoch 的趋势
        ax4 = fig.add_subplot(gs[1, 1])
        recent_n = min(len(epochs), 2000)
        if recent_n > 0:
            ax4.semilogy(
                epochs[-recent_n:],
                history["total"][-recent_n:],
                label="Total",
                color="black",
            )
            ax4.set_title(
                f"Recent Trend (Last {recent_n} Epochs)", fontsize=12, fontweight="bold"
            )
            ax4.set_xlabel("Epoch", fontsize=11)
            ax4.grid(True, alpha=0.3)

        plt.suptitle("Training Dashboard", fontsize=16, fontweight="bold", y=0.995)
        plt.savefig(
            self.output_dir / "training_dashboard.png", dpi=100, bbox_inches="tight"
        )
        plt.close()


# ========================================================================
# 训练器
# ========================================================================


class LevelSet3DTrainer:
    """
    3D Level Set PINN 训练器

    实现渐进式训练策略（基于 2D 成功经验）
    """

    def __init__(self, config: Dict, output_dir: Path):
        self.config = config
        self.output_dir = output_dir

        # 创建模型
        self.model = LevelSet3DPINN(config)

        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # 优化器
        lr = config.get("learning_rate", 1e-3)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # 混合精度训练
        self.use_amp = config.get("use_amp", True)
        self.scaler = GradScaler() if self.use_amp else None

        # 梯度累积
        self.gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)

        # 学习率调度器
        scheduler_config = config.get(
            "lr_scheduler", {"type": "StepLR", "step_size": 5000, "gamma": 0.5}
        )

        scheduler_type = scheduler_config.get("type", "StepLR")

        if scheduler_type == "StepLR":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get("step_size", 5000),
                gamma=scheduler_config.get("gamma", 0.5),
            )
        elif scheduler_type == "CosineAnnealingWarmRestarts":
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=scheduler_config.get("T_0", 10000),
                T_mult=scheduler_config.get("T_mult", 1),
                eta_min=scheduler_config.get("eta_min", 1e-6),
            )
        else:
            print(f"Warning: Unknown scheduler type {scheduler_type}, using StepLR")
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer, step_size=5000, gamma=0.5
            )

        # 数据生成器
        num_interior = config.get("num_interior", 5000)
        num_boundary = config.get("num_boundary", 1000)
        num_interface = config.get("num_interface", 2000)

        self.data_generator = LevelSet3DDataGenerator(
            config, num_interior, num_boundary, num_interface
        )

        # 损失函数
        self.loss_fn = LevelSet3DLoss(config)

        # 训练参数
        training_config = config.get("training", {})
        self.epochs = training_config.get("epochs", 50000)
        self.log_interval = config.get("log_interval", 100)
        self.save_interval = config.get("save_interval", 5000)

        # 状态
        self.current_epoch = 0
        self.best_loss = float("inf")
        self.loss_history = []
        self.last_valid_state = None
        self.nan_recovery_count = 0
        self.save_state_interval = 10

        # 日志（使用主 logger 以确保输出到文件）
        self.logger = logging.getLogger("LevelSet3D_Training")

        # 可视化绘图器
        self.plotter = TrainingPlotter(output_dir)

        # 渐进式训练阶段（包含多物理场约束）
        # 支持扁平化配置: config['training_stages'] 或 config['training']['training_stages']
        self.training_stages = config.get(
            "training_stages",
            config.get("training", {}).get(
                "training_stages",
                [
                    {"end_epoch": 5000, "focus": "data"},
                    {"end_epoch": 15000, "focus": "transport"},
                    {"end_epoch": 25000, "focus": "multiphysics"},
                    {"end_epoch": 40000, "focus": "full_physics"},
                    {"end_epoch": 50000, "focus": "refinement"},
                ],
            ),
        )

    def adjust_loss_weights(self, stage: Dict):
        """根据训练阶段调整损失权重"""
        focus = stage["focus"]

        # 优先使用配置文件中定义的loss_weights
        if "loss_weights" in stage:
            for key, value in stage["loss_weights"].items():
                if key in self.loss_fn.weights:
                    self.loss_fn.weights[key] = value
        else:
            # 回退到基于focus的硬编码权重（旧版兼容）
            if focus == "data":
                # Stage 1: 重点学习数据分布
                self.loss_fn.weights["data"] = 500.0  # 数据损失权重最大！
                self.loss_fn.weights["levelset_transport"] = 0.1
                self.loss_fn.weights["continuity_1"] = 0.1
                self.loss_fn.weights["continuity_2"] = 0.1
                self.loss_fn.weights["ns_1"] = 0.0
                self.loss_fn.weights["ns_2"] = 0.0
                self.loss_fn.weights["boundary_wall"] = 1.0
                self.loss_fn.weights["contact_angle"] = 0.0  # 暂时不启用接触角
                self.loss_fn.weights["electrostatic"] = (
                    0.01  # 启用极小权重，让V_field学习空间依赖性
                )
                self.loss_fn.weights["initial_condition"] = 10.0

            elif focus == "transport":
                # Stage 2: 引入输运方程
                self.loss_fn.weights["levelset_transport"] = 1.0
                self.loss_fn.weights["continuity_1"] = 0.5
                self.loss_fn.weights["continuity_2"] = 0.5
                self.loss_fn.weights["ns_1"] = 0.1
                self.loss_fn.weights["ns_2"] = 0.1
                self.loss_fn.weights["boundary_wall"] = 5.0
                self.loss_fn.weights["contact_angle"] = 0.0
                self.loss_fn.weights["electrostatic"] = 0.0

            elif focus == "multiphysics":
                # Stage 3: 引入多物理场约束（接触角边界条件）
                self.loss_fn.weights["levelset_transport"] = 1.0
                self.loss_fn.weights["continuity_1"] = 1.0
                self.loss_fn.weights["continuity_2"] = 1.0
                self.loss_fn.weights["ns_1"] = 0.5
                self.loss_fn.weights["ns_2"] = 0.5
                self.loss_fn.weights["boundary_wall"] = 10.0
                self.loss_fn.weights["contact_angle"] = 5.0  # 启用接触角边界条件
                self.loss_fn.weights["electrostatic"] = 2.0  # 启用静电场方程

            elif focus == "full_physics":
                # Stage 4: 完整物理约束（包括接触角和静电场）
                self.loss_fn.weights["levelset_transport"] = 1.0
                self.loss_fn.weights["continuity_1"] = 1.0
                self.loss_fn.weights["continuity_2"] = 1.0
                self.loss_fn.weights["ns_1"] = 1.0
                self.loss_fn.weights["ns_2"] = 1.0
                self.loss_fn.weights["boundary_wall"] = 10.0
                self.loss_fn.weights["contact_angle"] = 10.0  # 接触角权重提高
                self.loss_fn.weights["electrostatic"] = 5.0  # Maxwell 应力权重提高

            elif focus == "refinement":
                # Stage 5: 精化
                self.loss_fn.weights["levelset_transport"] = 2.0
                self.loss_fn.weights["continuity_1"] = 2.0
                self.loss_fn.weights["continuity_2"] = 2.0
                self.loss_fn.weights["ns_1"] = 2.0
                self.loss_fn.weights["ns_2"] = 2.0
                self.loss_fn.weights["boundary_wall"] = 15.0
                self.loss_fn.weights["contact_angle"] = 15.0  # 最高权重
                self.loss_fn.weights["electrostatic"] = 10.0  # Maxwell 应力最高权重

        # 记录权重调整
        self.logger.info(f"训练阶段: {focus}, 损失权重已调整:")
        self.logger.info(
            f"  data: {self.loss_fn.weights['data']:.1f}, "
            f"levelset_transport: {self.loss_fn.weights['levelset_transport']:.1f}, "
            f"continuity: {self.loss_fn.weights['continuity_1']:.1f}, "
            f"ns: {self.loss_fn.weights['ns_1']:.1f}, "
            f"contact_angle: {self.loss_fn.weights['contact_angle']:.1f}, "
            f"electrostatic: {self.loss_fn.weights['electrostatic']:.1f}"
        )

    def train_epoch(self) -> Dict[str, float]:
        """训练一个 epoch"""
        self.model.train()
        self.optimizer.zero_grad()

        # 生成数据
        data = self.data_generator.get_training_batch()

        # 将数据移到设备
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(self.device)

        # 前向传播（混合精度）
        if self.use_amp:
            with autocast():
                total_loss, loss_dict = self.loss_fn.compute_total_loss(
                    self.model, data
                )
                total_loss = total_loss / self.gradient_accumulation_steps
            # 反向传播
            self.scaler.scale(total_loss).backward()

            # 梯度裁剪
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # 优化器步进（混合精度）
            if (self.current_epoch + 1) % self.gradient_accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
        else:
            total_loss, loss_dict = self.loss_fn.compute_total_loss(self.model, data)
            total_loss = total_loss / self.gradient_accumulation_steps
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            if (self.current_epoch + 1) % self.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

        # 速度场监控（检测平凡解）
        with torch.no_grad():
            if "interior" in data:
                interior_data = data["interior"]
                interior_output = self.model(interior_data)

                u1 = interior_output[:, 0:1]
                v1 = interior_output[:, 1:2]
                w1 = interior_output[:, 2:3]

                vel_mag1 = torch.mean(u1**2 + v1**2 + w1**2).item()
                vel_mag2 = 0.0  # 相2速度（如果有需要可以添加）

                loss_dict["vel_mag1"] = vel_mag1
                loss_dict["vel_mag2"] = vel_mag2
            else:
                loss_dict["vel_mag1"] = 0.0
                loss_dict["vel_mag2"] = 0.0

        # ✅ 符号监控（在训练数据上验证，而不是几何假设）
        with torch.no_grad():
            if self.loss_fn.training_data is not None:
                # 新定义: ψ ≥ 0
                # 1. Ink 数据：应该 ψ > 0 (有油墨，高度 > 0)
                ink_pts = self.loss_fn.training_data["ink_points"]
                ink_output = self.model(ink_pts)
                ink_psi = ink_output[:, 6]
                ink_positive = (ink_psi > 0).sum().item() / len(ink_psi)
                ink_zero = (ink_psi == 0).sum().item() / len(ink_psi)

                polar_pts = self.loss_fn.training_data["polar_points"]
                polar_output = self.model(polar_pts)
                polar_psi = polar_output[:, 6]
                epsilon = 1e-6
                polar_near_zero = (polar_psi < epsilon).sum().item() / len(polar_psi)

                loss_dict["ink_positive_pct"] = ink_positive
                loss_dict["ink_zero_pct"] = ink_zero
                loss_dict["polar_near_zero_pct"] = polar_near_zero
            else:
                loss_dict["ink_positive_pct"] = 0.0
                loss_dict["ink_zero_pct"] = 0.0
                loss_dict["polar_near_zero_pct"] = 0.0

        return {
            "total": total_loss.item() * self.gradient_accumulation_steps,
            **loss_dict,
        }

    def save_checkpoint(self, epoch: int, loss: float):
        """保存检查点"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": loss,
            "config": self.config,
        }

        checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)

        self.logger.info(f"检查点已保存: {checkpoint_path}")

        # 绘制训练曲线
        if HAS_PLOT_TOOL and len(self.loss_history) > 0:
            try:
                self.plot_curves()
            except Exception as e:
                self.logger.warning(f"绘图失败: {e}")

    def save_best_model(self, epoch: int, loss: float):
        """保存最佳模型"""
        if loss < self.best_loss:
            self.best_loss = loss

            best_model_path = self.output_dir / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "loss": loss,
                    "config": self.config,
                },
                best_model_path,
            )

            self.logger.info(f"最佳模型已保存 (loss={loss:.6f}): {best_model_path}")

    def train(self):
        """主训练循环"""
        self.logger.info("=" * 60)
        self.logger.info("开始训练 3D Level Set PINN")
        self.logger.info("=" * 60)
        self.logger.info(f"设备: {self.device}")
        self.logger.info(f"总 epochs: {self.epochs}")
        self.logger.info(f"输出目录: {self.output_dir}")
        self.logger.info(
            f"模型参数量: {sum(p.numel() for p in self.model.parameters())}"
        )
        self.logger.info("=" * 60)

        start_time = time.time()
        current_stage_idx = 0

        for epoch in range(self.current_epoch, self.epochs):
            self.current_epoch = epoch

            # 检查是否需要切换训练阶段
            if current_stage_idx < len(self.training_stages):
                stage = self.training_stages[current_stage_idx]
                if epoch >= stage["end_epoch"]:
                    current_stage_idx += 1
                    if current_stage_idx < len(self.training_stages):
                        self.adjust_loss_weights(
                            self.training_stages[current_stage_idx]
                        )
                elif epoch == 0:
                    self.adjust_loss_weights(stage)

            # 训练一个 epoch
            loss_info = self.train_epoch()
            total_loss = loss_info["total"]

            if torch.isnan(torch.tensor(total_loss)) or torch.isinf(
                torch.tensor(total_loss)
            ):
                self.logger.warning(f"NaN/Inf detected at epoch {epoch}!")
                if self.last_valid_state is not None:
                    self.model.load_state_dict(self.last_valid_state)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] *= 0.5
                    self.nan_recovery_count += 1
                    self.logger.info(
                        f"Restored state and halved LR (Recovery #{self.nan_recovery_count})"
                    )
                    continue
                else:
                    self.logger.error("No valid state to restore from!")

            loss_info["epoch"] = epoch

            if epoch % self.save_state_interval == 0:
                self.last_valid_state = {
                    k: v.clone().detach() for k, v in self.model.state_dict().items()
                }

            # 记录损失
            loss_info["epoch"] = epoch
            self.loss_history.append(loss_info)

            # 实时更新图表 (每 500 epochs 或 save_interval)
            if epoch > 0 and (epoch % 500 == 0 or epoch % self.save_interval == 0):
                self.plotter.update_plots(self.loss_history)

            # 日志输出
            if epoch % self.log_interval == 0:
                self.logger.info(
                    f"Epoch [{epoch}/{self.epochs}] "
                    f"Loss: {loss_info['total']:.6f} "
                    f"(Data: {loss_info.get('data', 0):.2e}, "
                    f"LevelSet: {loss_info.get('levelset_transport', 0):.2e}, "
                    f"Cont1: {loss_info.get('continuity_1', 0):.2e}, "
                    f"Cont2: {loss_info.get('continuity_2', 0):.2e}, "
                    f"|V1|: {loss_info.get('vel_mag1', 0):.2e}, "
                    f"Wall: {loss_info.get('boundary_wall', 0):.2e}, "
                    f"Sign: {loss_info.get('sign_constraint', 0):.2e}, "
                    f"VolCon: {loss_info.get('volume_conservation', 0):.2e}, "
                    f"Spatial: {loss_info.get('psi_spatial', 0):.2e}, "
                    f"Mono: {loss_info.get('aperture_monotonicity', 0):.2e}, "
                    f"Ink>0: {loss_info.get('ink_positive_pct', 0) * 100:.1f}%, "
                    f"Polar≈0: {loss_info.get('polar_near_zero_pct', 0) * 100:.1f}%, "
                    f"Contact: {loss_info.get('contact_angle', 0):.2e}, "
                    f"Electro: {loss_info.get('electrostatic', 0):.2e})"
                )

            # 保存检查点
            if epoch % self.save_interval == 0 and epoch > 0:
                self.save_checkpoint(epoch, loss_info["total"])
                self.save_best_model(epoch, loss_info["total"])

            # 学习率调度
            self.scheduler.step()

        # 训练完成
        total_time = time.time() - start_time

        self.logger.info("=" * 60)
        self.logger.info("训练完成!")
        self.logger.info(f"总时间: {total_time / 3600:.2f} 小时")
        self.logger.info(f"最佳损失: {self.best_loss:.6f}")
        self.logger.info("=" * 60)

        # 保存最终模型
        final_model_path = self.output_dir / "final_model.pt"
        torch.save(
            {
                "epoch": self.epochs,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": self.loss_history[-1]["total"],
                "config": self.config,
                "loss_history": self.loss_history,
            },
            final_model_path,
        )

        self.logger.info(f"最终模型已保存: {final_model_path}")

        # 最终绘图
        self.plotter.update_plots(self.loss_history)

        # ====================================================================
        # 自动执行评估脚本 (Dashboard, 3D, Volume Conservation)
        # ====================================================================
        self.logger.info("=" * 60)
        self.logger.info("正在自动执行模型评估...")
        try:
            evaluate_script = Path(__file__).parent / "evaluate.py"
            if evaluate_script.exists():
                # 使用当前 python 解释器调用 evaluate.py
                # 传递 output_dir 作为参数
                cmd = [sys.executable, str(evaluate_script), str(self.output_dir)]

                self.logger.info(f"执行命令: {' '.join(cmd)}")

                # 使用 subprocess.run 执行，并捕获输出
                result = subprocess.run(
                    cmd,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )

                self.logger.info("评估脚本执行成功!")
                # 将评估脚本的输出记录到日志
                for line in result.stdout.splitlines():
                    self.logger.info(f"[Evaluate] {line}")

            else:
                self.logger.warning(f"找不到评估脚本: {evaluate_script}")

        except subprocess.CalledProcessError as e:
            self.logger.error(f"评估脚本执行失败 (Exit Code {e.returncode})")
            self.logger.error(f"错误输出:\n{e.stderr}")
        except Exception as e:
            self.logger.error(f"执行评估脚本时发生异常: {e}")

        self.logger.info("=" * 60)


# ========================================================================
# 主函数
# ========================================================================


def main():
    parser = argparse.ArgumentParser(description="训练 3D Level Set PINN")
    parser.add_argument(
        "--config", type=str, default="config/baseline.json", help="配置文件路径"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录（默认为 outputs_levelset_<timestamp>）",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="从检查点恢复训练（检查点文件路径）"
    )
    parser.add_argument("--seed", type=int, default=None, help="随机种子（用于可复现）")
    parser.add_argument(
        "--plot-only", action="store_true", help="仅从日志重新生成训练曲线，不进行训练"
    )

    args = parser.parse_args()

    # ========================================================================
    # 仅绘图模式
    # ========================================================================
    if args.plot_only:
        if not args.output_dir:
            print("错误: 使用 --plot-only 时必须指定 --output-dir")
            sys.exit(1)

        output_dir = Path(args.output_dir)
        if not output_dir.exists():
            print(f"错误: 输出目录不存在: {output_dir}")
            sys.exit(1)

        if not HAS_PLOT_TOOL:
            print("错误: 无法导入绘图工具 (plot_training_curves.py)")
            sys.exit(1)

        log_file = output_dir / "training.log"
        if not log_file.exists():
            print(f"错误: 日志文件不存在: {log_file}")
            sys.exit(1)

        print(f"📈 正在处理日志文件: {log_file}")
        try:
            # 调用 plot_training_curves 模块的功能
            epochs, losses = plot_module.parse_log_file(log_file)

            if not epochs:
                print(f"⚠️  警告: 日志文件中未找到 epoch 数据")
                sys.exit(0)

            print(f"   找到 {len(epochs)} 个数据点。正在生成图表...")
            plot_module.plot_curves(epochs, losses, output_dir)
            print(f"✅ 图表已保存至: {output_dir}")
        except Exception as e:
            print(f"❌ 绘图失败: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)

        sys.exit(0)

    # 加载配置
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"错误: 配置文件不存在: {config_path}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # ========================================================================
    # 配置验证（确保所有必需的损失权重都已定义）
    # ========================================================================
    required_loss_weights = [
        "data",
        "levelset_transport",
        "continuity_1",
        "continuity_2",
        "ns_1",
        "ns_2",
        "interface_geometry",
        "boundary_wall",
        "boundary_symmetry",
        "initial_condition",
        "contact_angle",
        "electrostatic",
        "sign_constraint",
        "volume_conservation",
        "psi_spatial",
        "aperture_monotonicity",
    ]

    # 验证全局 loss_weights
    if "loss_weights" in config:
        missing = [w for w in required_loss_weights if w not in config["loss_weights"]]
        if missing:
            print(f"⚠️  警告: 配置文件中缺少以下损失权重: {missing}")
            print(f"   将使用默认值 0.0")
            for w in missing:
                config["loss_weights"][w] = 0.0

    # 验证 training_stages 中的 loss_weights (支持扁平化配置)
    training_stages = config.get(
        "training_stages", config.get("training", {}).get("training_stages", [])
    )
    for stage_idx, stage in enumerate(training_stages):
        if "loss_weights" in stage:
            missing = [
                w for w in required_loss_weights if w not in stage["loss_weights"]
            ]
            if missing:
                print(f"⚠️  警告: Stage {stage_idx + 1} 缺少以下损失权重: {missing}")
                print(f"   将使用默认值 0.0")
                for w in missing:
                    stage["loss_weights"][w] = 0.0

    # 创建输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"outputs_levelset_{timestamp}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # 设置日志
    logger = setup_logging(output_dir)
    logger.info(f"配置文件: {config_path}")
    logger.info(f"输出目录: {output_dir}")

    # 保存配置
    config_backup_path = output_dir / "config.json"
    with open(config_backup_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    logger.info(f"配置已保存: {config_backup_path}")

    # 设置随机种子以保证可复现性（如果提供）
    seed = args.seed or config.get("seed", None)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # 为确定性设置（可能影响性能）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info(f"随机种子已设置: {seed}")

    # 创建训练器
    trainer = LevelSet3DTrainer(config, output_dir)

    # 从检查点恢复
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            logger.info(f"从检查点恢复: {resume_path}")
            checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)

            # 使用检查点中的配置重建模型
            if "config" in checkpoint:
                ckpt_config = checkpoint["config"]
                logger.info("使用检查点中的配置重建模型...")
                # 合并配置（检查点配置优先）
                for key in ckpt_config:
                    if key in ["model", "training"]:
                        config[key] = ckpt_config[key]

                # 重新创建训练器
                trainer = LevelSet3DTrainer(config, output_dir)

            trainer.model.load_state_dict(checkpoint["model_state_dict"])
            trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            trainer.current_epoch = checkpoint["epoch"] + 1
            trainer.best_loss = checkpoint.get("loss", float("inf"))
            logger.info(
                f"恢复到 epoch {trainer.current_epoch}, 损失 {trainer.best_loss:.6f}"
            )
        else:
            logger.warning(f"检查点文件不存在: {resume_path}")

    try:
        trainer.train()
        logger.info("训练脚本执行完成")
    except KeyboardInterrupt:
        logger.info("训练被用户中断")
        if trainer.current_epoch > 0:
            trainer.save_checkpoint(trainer.current_epoch, trainer.best_loss)
            logger.info(f"检查点已保存至 epoch {trainer.current_epoch}")
    except Exception as e:
        logger.error(f"训练出错: {str(e)}")
        if trainer.current_epoch > 0:
            trainer.save_checkpoint(trainer.current_epoch, trainer.best_loss)
            logger.info(f"检查点已保存至 epoch {trainer.current_epoch}")
        raise


if __name__ == "__main__":
    main()
