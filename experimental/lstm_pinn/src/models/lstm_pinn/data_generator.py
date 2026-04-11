"""
LSTM-PINN 序列数据生成器

生成电压时间序列格式的训练数据，支持：
- 单步升压/降压
- 多步电压序列
- 稳态数据
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
import logging

from .physics import ElectrowettingPhysics

logger = logging.getLogger(__name__)


class SequenceDataGenerator:
    """
    序列格式训练数据生成器

    生成数据类型:
        1. 单步升压序列: [0, 0, ..., V_target, V_target, ...]
        2. 单步降压序列: [V_initial, V_initial, ..., 0, 0, ...]
        3. 多步序列: [0, ..., 20, ..., 30, ..., 20, ..., 0]
        4. 稳态序列: [V, V, V, ..., V]
    """

    def __init__(self, config: Dict[str, Any], device: torch.device = None):
        """
        初始化数据生成器

        Args:
            config: 配置字典（从 lstm_dynamic_response.json 加载）
            device: PyTorch 设备
        """
        self.config = config
        self.device = device or torch.device("cpu")

        # 几何参数
        geometry = config.get("geometry", {})
        self.Lx = geometry.get("Lx", 174e-6)
        self.Ly = geometry.get("Ly", 174e-6)
        self.Lz = geometry.get("Lz", 20e-6)
        self.h_ink = geometry.get("ink_thickness", 3e-6)
        self.cx = self.Lx / 2
        self.cy = self.Ly / 2

        # 序列参数
        sequence = config.get("sequence", {})
        self.seq_len = sequence.get("length", 50)
        self.dt = sequence.get("dt", 0.001)
        self.t_max = sequence.get("t_max", 0.05)

        # 物理模型
        self.physics = (
            ElectrowettingPhysics.from_config(
                config.get("_config_path", "config/device_calibrated_physics.json")
            )
            if "_config_path" in config
            else ElectrowettingPhysics(config.get("materials", {}))
        )

        # 归一化参数
        norm = config.get("normalization", {})
        self.V_max = norm.get("voltage", {}).get("max", 30.0)
        self.x_scale = norm.get("spatial", {}).get("x_scale", self.Lx)
        self.y_scale = norm.get("spatial", {}).get("y_scale", self.Ly)
        self.z_scale = norm.get("spatial", {}).get("z_scale", self.Lz)
        self.t_scale = norm.get("time", {}).get("scale", 0.001)

    def generate_voltage_sequence(
        self, transitions: List[Tuple[float, float, float]], dt: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成电压序列

        Args:
            transitions: [(t_start, t_end, V), ...] 电压变化列表
            dt: 时间步长 (s)，默认使用配置值

        Returns:
            voltage_seq: (seq_len,) 电压序列
            time_seq: (seq_len,) 时间序列
        """
        if dt is None:
            dt = self.dt

        # 计算总时间
        t_total = max(t[1] for t in transitions)
        n_steps = int(t_total / dt) + 1

        time_seq = np.linspace(0, t_total, n_steps)
        voltage_seq = np.zeros(n_steps)

        for t_start, t_end, V in transitions:
            mask = (time_seq >= t_start) & (time_seq < t_end)
            voltage_seq[mask] = V

        return voltage_seq, time_seq

    def generate_step_rise_sequence(
        self,
        V_from: float,
        V_to: float,
        t_step: float = 0.002,
        t_total: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成单步升压序列

        Args:
            V_from: 初始电压
            V_to: 目标电压
            t_step: 升压时刻
            t_total: 总时间

        Returns:
            voltage_seq, time_seq
        """
        if t_total is None:
            t_total = self.t_max

        transitions = [(0, t_step, V_from), (t_step, t_total, V_to)]
        return self.generate_voltage_sequence(transitions)

    def generate_step_fall_sequence(
        self,
        V_from: float,
        V_to: float,
        t_step: float = 0.015,
        t_total: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成单步降压序列

        Args:
            V_from: 初始电压
            V_to: 目标电压
            t_step: 降压时刻
            t_total: 总时间

        Returns:
            voltage_seq, time_seq
        """
        if t_total is None:
            t_total = self.t_max

        transitions = [(0, t_step, V_from), (t_step, t_total, V_to)]
        return self.generate_voltage_sequence(transitions)

    def generate_multi_step_sequence(
        self, voltage_steps: List[float], step_duration: float = 0.010
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成多步电压序列

        Args:
            voltage_steps: 电压序列，如 [0, 20, 30, 20, 0]
            step_duration: 每步持续时间

        Returns:
            voltage_seq, time_seq
        """
        transitions = []
        t = 0
        for V in voltage_steps:
            transitions.append((t, t + step_duration, V))
            t += step_duration

        return self.generate_voltage_sequence(transitions)

    def generate_steady_state_sequence(
        self, V: float, t_total: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成稳态序列（恒定电压）

        Args:
            V: 电压
            t_total: 总时间

        Returns:
            voltage_seq, time_seq
        """
        if t_total is None:
            t_total = self.t_max

        transitions = [(0, t_total, V)]
        return self.generate_voltage_sequence(transitions)

    def sample_spatial_point(
        self, eta: float, strategy: str = "interface_weighted"
    ) -> Tuple[float, float, float]:
        """
        采样空间点

        Args:
            eta: 当前开口率
            strategy: 采样策略 ("uniform", "interface_weighted")

        Returns:
            (x, y, z) 坐标
        """
        if strategy == "interface_weighted" and eta > 0.01:
            # 在界面附近加密采样
            if np.random.rand() < 0.4:
                # 在开口边界附近采样
                r_open = np.sqrt(eta * self.Lx * self.Ly / np.pi)
                r = r_open + np.random.randn() * 10e-6
                r = max(0, min(r, np.sqrt(self.cx**2 + self.cy**2)))
                theta = np.random.rand() * 2 * np.pi
                x = self.cx + r * np.cos(theta)
                y = self.cy + r * np.sin(theta)
                x = np.clip(x, 0, self.Lx)
                y = np.clip(y, 0, self.Ly)
            else:
                x = np.random.rand() * self.Lx
                y = np.random.rand() * self.Ly
        else:
            # 均匀采样
            x = np.random.rand() * self.Lx
            y = np.random.rand() * self.Ly

        # z 方向：在油墨层附近加密
        if np.random.rand() < 0.5:
            z = np.random.rand() * self.h_ink * 3
        else:
            z = np.random.rand() * self.Lz

        return x, y, z

    def compute_target_phi(self, x: float, y: float, z: float, eta: float) -> float:
        """
        计算目标 φ 值

        基于中心开口模式：
        - 中心透明 (φ=0)
        - 边缘油墨 (φ=1)

        Args:
            x, y, z: 空间坐标
            eta: 开口率

        Returns:
            φ ∈ [0, 1]
        """
        interface_width = 3e-6  # 3 μm

        if eta < 0.01:
            # 无开口：初始状态
            phi = 0.5 * (1 - np.tanh((z - self.h_ink) / (interface_width / 3)))
            return np.clip(phi, 0, 1)

        if eta > 0.99:
            return 0.0

        r = np.sqrt((x - self.cx) ** 2 + (y - self.cy) ** 2)

        # 开口半径
        r_open = np.sqrt(eta * self.Lx * self.Ly / np.pi)

        # 油墨堆高（体积守恒）
        ink_area = self.Lx * self.Ly - np.pi * r_open**2
        h_ink_edge = self.Lx * self.Ly * self.h_ink / max(ink_area, 1e-12)
        h_ink_edge = min(h_ink_edge, self.Lz * 0.8)

        # 径向分布
        radial_factor = 0.5 * (1 + np.tanh((r - r_open) / interface_width))

        if r < r_open - interface_width:
            phi = 0.0  # 中心透明
        elif r > r_open + interface_width:
            phi = 0.5 * (1 - np.tanh((z - h_ink_edge) / (interface_width / 2)))
        else:
            phi_center = 0.0
            phi_edge = 0.5 * (1 - np.tanh((z - h_ink_edge) / (interface_width / 2)))
            phi = phi_center * (1 - radial_factor) + phi_edge * radial_factor

        return np.clip(phi, 0, 1)

    def get_eta_triad(self, V_from: float, V_to: float, t_since: float) -> float:
        """
        基于 6D Triad 获取开口率

        Args:
            V_from: 起始电压 (V)
            V_to: 目标电压 (V)
            t_since: 电压变化后经过的时间 (s)

        Returns:
            开口率 η ∈ [0, aperture_max]
        """
        if V_to >= V_from:
            # 升压过程：先计算接触角，再转换为开口率
            theta = self.physics.contact_angle_rise(V_to, t_since, V_from)
            eta = self.physics.contact_angle_to_aperture(theta)
        else:
            # 降压过程：直接计算开口率
            eta = self.physics.aperture_fall(V_from, t_since, V_to)

        return eta

    def _find_t_step(self, v_seq: np.ndarray, t_seq: np.ndarray) -> float:
        """
        查找电压序列中电压变化的时刻

        Args:
            v_seq: 电压序列
            t_seq: 时间序列

        Returns:
            电压变化时刻 (s)
        """
        # 找到电压变化的索引
        diff = np.diff(v_seq)
        change_indices = np.where(diff != 0)[0]

        if len(change_indices) == 0:
            return 0.0

        # 返回第一个变化点的时间
        return float(t_seq[change_indices[0] + 1])

    def generate_training_data(
        self, n_samples: int = 100000
    ) -> Dict[str, torch.Tensor]:
        """
        生成完整训练数据集

        Returns:
            {
                'spatial_coords': (N, 3),
                'voltage_sequences': (N, seq_len, 1),
                'time_sequences': (N, seq_len, 1),
                'phi_targets': (N, 1)
            }
        """
        logger.info(f"生成 LSTM-PINN 训练数据: {n_samples} 样本")

        data_cfg = self.config.get("data", {})
        transitions = data_cfg.get("voltage_transitions", [])

        spatial_coords = []
        voltage_sequences = []
        time_sequences = []
        phi_targets = []

        # 1. 稳态数据 (30%)
        n_steady = int(n_samples * 0.3)
        voltages = [0, 10, 20, 30]
        n_per_v = n_steady // len(voltages)

        logger.info(f"  稳态数据: {n_steady} 样本")
        for V in voltages:
            v_seq, t_seq = self.generate_steady_state_sequence(V)
            eta = self.physics.get_steady_state_aperture(V)

            # 填充到固定长度
            v_seq_padded = np.zeros(self.seq_len)
            t_seq_padded = np.zeros(self.seq_len)
            n = min(len(v_seq), self.seq_len)
            v_seq_padded[-n:] = v_seq[-n:]
            t_seq_padded[-n:] = t_seq[-n:]

            for _ in range(n_per_v):
                x, y, z = self.sample_spatial_point(eta)
                phi = self.compute_target_phi(x, y, z, eta)

                spatial_coords.append(
                    [x / self.x_scale, y / self.y_scale, z / self.z_scale]
                )
                voltage_sequences.append(v_seq_padded / self.V_max)
                time_sequences.append(t_seq_padded / self.t_scale)
                phi_targets.append(phi)

        # 2. 升压数据 (30%)
        n_rise = int(n_samples * 0.3)
        rise_pairs = [(0, 10), (0, 20), (0, 30), (10, 20), (10, 30), (20, 30)]
        n_per_pair = n_rise // len(rise_pairs)

        logger.info(f"  升压数据: {n_rise} 样本")
        for V_from, V_to in rise_pairs:
            v_seq, t_seq = self.generate_step_rise_sequence(V_from, V_to)

            for _ in range(n_per_pair):
                # 随机选择时间点
                t_idx = np.random.randint(0, len(t_seq))
                t = t_seq[t_idx]

                # 计算该时刻的开口率
                if t < 0.002:
                    eta = self.physics.get_steady_state_aperture(V_from)
                else:
                    t_since = t - 0.002
                    theta = self.physics.contact_angle_rise(V_to, t_since, V_from)
                    eta = self.physics.contact_angle_to_aperture(theta)

                x, y, z = self.sample_spatial_point(eta)
                phi = self.compute_target_phi(x, y, z, eta)

                # 截断序列到当前时刻
                v_seq_truncated = v_seq[: t_idx + 1]
                t_seq_truncated = t_seq[: t_idx + 1]

                # 填充到固定长度
                v_seq_padded = np.zeros(self.seq_len)
                t_seq_padded = np.zeros(self.seq_len)
                n = min(len(v_seq_truncated), self.seq_len)
                v_seq_padded[-n:] = v_seq_truncated[-n:]
                t_seq_padded[-n:] = t_seq_truncated[-n:]

                spatial_coords.append(
                    [x / self.x_scale, y / self.y_scale, z / self.z_scale]
                )
                voltage_sequences.append(v_seq_padded / self.V_max)
                time_sequences.append(t_seq_padded / self.t_scale)
                phi_targets.append(phi)

        # 3. 降压数据 (30%)
        n_fall = int(n_samples * 0.3)
        fall_pairs = [(30, 0), (20, 0), (10, 0), (30, 20), (30, 10), (20, 10)]
        n_per_pair = n_fall // len(fall_pairs)

        logger.info(f"  降压数据: {n_fall} 样本")
        for V_from, V_to in fall_pairs:
            v_seq, t_seq = self.generate_step_fall_sequence(V_from, V_to)

            for _ in range(n_per_pair):
                t_idx = np.random.randint(0, len(t_seq))
                t = t_seq[t_idx]

                if t < 0.015:
                    eta = self.physics.get_steady_state_aperture(V_from)
                else:
                    t_since = t - 0.015
                    eta = self.physics.aperture_fall(V_from, t_since, V_to)

                x, y, z = self.sample_spatial_point(eta)
                phi = self.compute_target_phi(x, y, z, eta)

                v_seq_truncated = v_seq[: t_idx + 1]
                t_seq_truncated = t_seq[: t_idx + 1]

                v_seq_padded = np.zeros(self.seq_len)
                t_seq_padded = np.zeros(self.seq_len)
                n = min(len(v_seq_truncated), self.seq_len)
                v_seq_padded[-n:] = v_seq_truncated[-n:]
                t_seq_padded[-n:] = t_seq_truncated[-n:]

                spatial_coords.append(
                    [x / self.x_scale, y / self.y_scale, z / self.z_scale]
                )
                voltage_sequences.append(v_seq_padded / self.V_max)
                time_sequences.append(t_seq_padded / self.t_scale)
                phi_targets.append(phi)

        # 4. 多步序列数据 (10%)
        n_multi = n_samples - len(spatial_coords)
        multi_sequences = [
            [0, 20, 30, 20, 0],
            [0, 10, 20, 30, 20, 10, 0],
            [0, 30, 0],
            [0, 20, 0],
        ]
        n_per_seq = n_multi // len(multi_sequences)

        logger.info(f"  多步序列数据: {n_multi} 样本")
        for voltage_steps in multi_sequences:
            v_seq, t_seq = self.generate_multi_step_sequence(voltage_steps)

            for _ in range(n_per_seq):
                t_idx = np.random.randint(0, len(t_seq))
                t = t_seq[t_idx]
                V = v_seq[t_idx]

                eta = self.physics.get_steady_state_aperture(V)
                x, y, z = self.sample_spatial_point(eta)
                phi = self.compute_target_phi(x, y, z, eta)

                v_seq_truncated = v_seq[: t_idx + 1]
                t_seq_truncated = t_seq[: t_idx + 1]

                v_seq_padded = np.zeros(self.seq_len)
                t_seq_padded = np.zeros(self.seq_len)
                n = min(len(v_seq_truncated), self.seq_len)
                v_seq_padded[-n:] = v_seq_truncated[-n:]
                t_seq_padded[-n:] = t_seq_truncated[-n:]

                spatial_coords.append(
                    [x / self.x_scale, y / self.y_scale, z / self.z_scale]
                )
                voltage_sequences.append(v_seq_padded / self.V_max)
                time_sequences.append(t_seq_padded / self.t_scale)
                phi_targets.append(phi)

        # 转换为 Tensor
        logger.info(f"  总样本数: {len(spatial_coords)}")

        # 转换为 numpy 数组再转 tensor（避免警告）
        spatial_coords = np.array(spatial_coords, dtype=np.float32)
        voltage_sequences = np.array(voltage_sequences, dtype=np.float32)
        time_sequences = np.array(time_sequences, dtype=np.float32)
        phi_targets = np.array(phi_targets, dtype=np.float32)

        return {
            "spatial_coords": torch.from_numpy(spatial_coords).to(self.device),
            "voltage_sequences": torch.from_numpy(voltage_sequences)
            .unsqueeze(-1)
            .to(self.device),
            "time_sequences": torch.from_numpy(time_sequences)
            .unsqueeze(-1)
            .to(self.device),
            "phi_targets": torch.from_numpy(phi_targets).unsqueeze(-1).to(self.device),
        }
