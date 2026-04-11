#!/usr/bin/env python3
"""
LSTM-Hybrid-PINN 预测器
=======================

基于训练好的 LSTM-Hybrid-PINN 模型进行预测。

用法：
    from src.predictors import get_lstm_hybrid_predictor

    LSTMHybridPredictor = get_lstm_hybrid_predictor()
    predictor = LSTMHybridPredictor(
        model_path='outputs/train/lstm_hybrid_xxx/best_model.pth',
        config_path='config/lstm_dynamic_response.json'
    )

    phi = predictor.predict_phi(voltage_seq=[(0, 30, 0.01)], time=0.005)
"""

import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import json
import logging

logger = logging.getLogger("LSTM-Hybrid-Predictor")


class LSTMHybridPredictor:
    """LSTM-Hybrid-PINN 预测器"""

    def __init__(
        self,
        model_path: str,
        config_path: str = "config/lstm_dynamic_response.json",
        pinn_path: str = "outputs/train/pinn_20260205_174333/best_model.pth",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = torch.device(device)
        self.model = None
        self.config = None

        self._load_config(config_path)
        self._load_model(model_path, pinn_path)

    def _load_config(self, config_path: str):
        config_path = Path(config_path)
        if config_path.exists():
            with open(config_path) as f:
                self.config = json.load(f)
            logger.info(f"加载配置: {config_path}")
        else:
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

    def _load_model(self, model_path: str, pinn_path: str):
        checkpoint = torch.load(
            model_path, map_location=self.device, weights_only=False
        )
        self.best_loss = checkpoint.get("loss", float("inf"))
        self.epoch = checkpoint.get("epoch", 0)

        from src.models.lstm_pinn import LSTMHybridPINN
        from src.models.pinn_two_phase import TwoPhasePINN

        pretrained_pinn = None
        if Path(pinn_path).exists():
            pinn_ckpt = torch.load(
                pinn_path, map_location=self.device, weights_only=False
            )
            pinn_config = pinn_ckpt.get("config", {})
            pretrained_pinn = TwoPhasePINN(pinn_config)
            pretrained_pinn.load_state_dict(pinn_ckpt["model_state_dict"])
            logger.info(f"加载预训练 PINN: {pinn_path}")

        self.model = LSTMHybridPINN(
            pretrained_pinn=pretrained_pinn,
            config=self.config,
            freeze_pinn=False,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        logger.info(f"模型加载成功: {model_path}")
        logger.info(f"  Epoch: {self.epoch}, Loss: {self.best_loss:.6e}")

    def _prepare_voltage_seq(
        self, voltage_seq: List[Tuple[float, float, float]]
    ) -> torch.Tensor:
        t_max = self.config.get("sequence", {}).get("t_max", 0.05)
        seq = []
        for V_from, V_to, t_since in voltage_seq:
            seq.append([V_from / 30.0, V_to / 30.0, t_since / t_max])
        return torch.tensor([seq], dtype=torch.float32, device=self.device)

    def _prepare_spatial_coords(
        self,
        coords: Optional[Tuple[float, float, float]] = None,
        grid_size: Optional[Tuple[int, int, int]] = None,
    ) -> torch.Tensor:
        Lx, Ly, Lz = 174e-6, 174e-6, 20e-6

        if coords is not None:
            return torch.tensor(
                [[coords[0], coords[1], coords[2]]],
                dtype=torch.float32,
                device=self.device,
            )

        if grid_size is not None:
            nx, ny, nz = grid_size
            x = torch.linspace(0, Lx, nx, device=self.device)
            y = torch.linspace(0, Ly, ny, device=self.device)
            z = torch.linspace(0, Lz, nz, device=self.device)
            grid = torch.meshgrid(x, y, z, indexing="ij")
            return torch.stack([g.flatten() for g in grid], dim=-1)

        return torch.tensor(
            [[Lx / 2, Ly / 2, Lz / 2]], dtype=torch.float32, device=self.device
        )

    @torch.no_grad()
    def predict(
        self,
        voltage_seq: List[Tuple[float, float, float]],
        time: float = 0.0,
        coords: Optional[Tuple[float, float, float]] = None,
        grid_size: Optional[Tuple[int, int, int]] = None,
    ) -> Dict[str, np.ndarray]:
        voltage_tensor = self._prepare_voltage_seq(voltage_seq)
        spatial_coords = self._prepare_spatial_coords(coords, grid_size)
        t_tensor = torch.tensor([[time]], dtype=torch.float32, device=self.device)

        n_points = spatial_coords.shape[0]
        voltage_tensor = voltage_tensor.expand(n_points, -1, -1)
        t_tensor = t_tensor.expand(n_points, -1)

        output = self.model(spatial_coords, t_tensor, voltage_tensor)

        return {
            "u": output[:, 0].cpu().numpy(),
            "v": output[:, 1].cpu().numpy(),
            "w": output[:, 2].cpu().numpy(),
            "p": output[:, 3].cpu().numpy(),
            "phi": output[:, 4].cpu().numpy(),
        }

    @torch.no_grad()
    def predict_phi(
        self,
        voltage_seq: List[Tuple[float, float, float]],
        time: float = 0.0,
        coords: Optional[Tuple[float, float, float]] = None,
    ) -> float:
        result = self.predict(voltage_seq, time, coords)
        return float(result["phi"][0])

    @torch.no_grad()
    def predict_aperture(
        self,
        voltage_seq: List[Tuple[float, float, float]],
        time: float = 0.0,
    ) -> float:
        return self.predict_phi(voltage_seq, time, coords=None)

    @torch.no_grad()
    def voltage_response(
        self,
        V_from: float = 0.0,
        V_to: float = 30.0,
        t_duration: float = 0.05,
        n_steps: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray]:
        times = np.linspace(0, t_duration, n_steps)
        apertures = []

        for t in times:
            voltage_seq = [(V_from, V_to, t)]
            phi = self.predict_aperture(voltage_seq, time=0)
            apertures.append(phi)

        return times, np.array(apertures)

    @torch.no_grad()
    def multi_step_response(
        self,
        voltage_steps: List[Tuple[float, float, float]],
        n_points_per_step: int = 20,
    ) -> Dict[str, np.ndarray]:
        all_times = []
        all_voltages = []
        all_apertures = []

        t_current = 0.0
        cumulative_seq = []

        for V_from, V_to, t_duration in voltage_steps:
            times = np.linspace(t_current, t_current + t_duration, n_points_per_step)

            for t in times:
                seq = cumulative_seq + [(V_from, V_to, t - t_current)]
                phi = self.predict_aperture(seq, time=0)

                all_times.append(t)
                all_voltages.append(
                    V_from + (V_to - V_from) * (t - t_current) / t_duration
                )
                all_apertures.append(phi)

            cumulative_seq.append((V_from, V_to, t_duration))
            t_current += t_duration

        return {
            "times": np.array(all_times),
            "voltages": np.array(all_voltages),
            "apertures": np.array(all_apertures),
        }
