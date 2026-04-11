"""
LSTM-PINN 训练流程

包含：
- 训练循环
- 学习率调度
- 训练日志和可视化
"""

import logging
import os
import time
from typing import Dict, Optional, Tuple, Any, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .model import LSTMPINNModel
from .physics_loss import LSTMPINNPhysicsLoss
from .data_generator import SequenceDataGenerator

# 配置日志
logging.basicConfig(
    format="[%(asctime)s] %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("LSTM-PINN-Trainer")


class LSTMPINNTrainer:
    """
    LSTM-PINN 训练器
    
    支持：
    - 渐进式训练策略
    - 学习率调度
    - 训练日志
    - Checkpoint 保存
    """
    
    def __init__(
        self,
        model: LSTMPINNModel,
        config: Dict[str, Any],
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        初始化训练器
        
        Args:
            model: LSTM-PINN 模型
            config: 配置字典
            device: 训练设备
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # 训练配置
        training_config = config.get("training", {})
        self.epochs = training_config.get("epochs", 30000)
        self.batch_size = training_config.get("batch_size", 1024)
        self.learning_rate = training_config.get("learning_rate", 1e-3)
        self.min_lr = training_config.get("min_lr", 1e-6)
        self.gradient_clip = training_config.get("gradient_clip", 1.0)
        
        # 渐进式训练阶段
        self.stage1_epochs = training_config.get("stage1_epochs", 5000)
        self.stage2_epochs = training_config.get("stage2_epochs", 15000)
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs,
            eta_min=self.min_lr
        )
        # 损失函数
        self.data_loss_fn = nn.MSELoss()
        self.physics_loss_fn = LSTMPINNPhysicsLoss(config)
        self.return_velocity = getattr(self.model, "velocity_decoder", None) is not None
        physics_cfg = config.get("physics_loss", {})
        self.physics_enabled = physics_cfg.get("enabled", bool(physics_cfg))
        
        # 训练历史
        self.history = {
            "epoch": [],
            "data_loss": [],
            "physics_loss": [],
            "total_loss": [],
            "lr": [],
            "time": []
        }
        
        # 最佳模型
        self.best_loss = float("inf")
        self.best_epoch = 0
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        训练一个 epoch
        
        Args:
            dataloader: 数据加载器
            epoch: 当前 epoch
        
        Returns:
            损失字典
        """
        self.model.train()
        
        total_data_loss = 0.0
        total_physics_loss = 0.0
        n_batches = 0
        
        if not self.physics_enabled:
            physics_weight = 0.0
        else:
            if epoch < self.stage1_epochs:
                physics_weight = 0.0
            elif epoch < self.stage2_epochs:
                progress = (epoch - self.stage1_epochs) / (self.stage2_epochs - self.stage1_epochs)
                physics_weight = progress * 0.5
            else:
                physics_weight = 0.5
        
        for batch in dataloader:
            spatial_coords, voltage_seq, time_seq, phi_target = [
                b.to(self.device) for b in batch
            ]
            spatial_coords = spatial_coords.clone().detach().requires_grad_(True)
            time_seq = time_seq.clone().detach().requires_grad_(True)
            
            self.optimizer.zero_grad()
            
            output = self.model(spatial_coords, voltage_seq, time_seq, return_velocity=self.return_velocity)
            phi_pred = output["phi"]
            velocity = output.get("velocity")
            
            data_loss = self.data_loss_fn(phi_pred, phi_target)
            
            if physics_weight > 0:
                time_scalar = time_seq[:, -1, :]
                physics_losses = self.physics_loss_fn(phi_pred, velocity, spatial_coords, time_scalar)
                physics_loss = physics_losses["total"]
            else:
                physics_loss = torch.tensor(0.0, device=self.device)
            
            # 总损失
            total_loss = data_loss + physics_weight * physics_loss
            
            # 反向传播
            total_loss.backward()
            
            # 梯度裁剪
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
                )
            
            self.optimizer.step()
            
            total_data_loss += data_loss.item()
            total_physics_loss += physics_loss.item()
            n_batches += 1
        
        # 更新学习率
        self.scheduler.step()
        
        return {
            "data_loss": total_data_loss / n_batches,
            "physics_loss": total_physics_loss / n_batches,
            "total_loss": (total_data_loss + physics_weight * total_physics_loss) / n_batches,
            "physics_weight": physics_weight
        }
    
    def train(
        self,
        train_data: Dict[str, torch.Tensor],
        val_data: Optional[Dict[str, torch.Tensor]] = None,
        save_dir: Optional[str] = None,
        log_interval: int = 100,
        save_interval: int = 1000,
        callback: Optional[Callable] = None
    ) -> Dict[str, list]:
        """
        完整训练流程
        
        Args:
            train_data: 训练数据字典
            val_data: 验证数据字典（可选）
            save_dir: 保存目录
            log_interval: 日志间隔
            save_interval: 保存间隔
            callback: 回调函数
        
        Returns:
            训练历史
        """
        # 创建数据加载器
        train_dataset = TensorDataset(
            train_data["spatial_coords"],
            train_data["voltage_sequences"],
            train_data["time_sequences"],
            train_data["phi_targets"]
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        # 创建保存目录
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        logger.info(f"开始训练 LSTM-PINN 模型")
        logger.info(f"  设备: {self.device}")
        logger.info(f"  训练样本: {len(train_dataset)}")
        logger.info(f"  批量大小: {self.batch_size}")
        logger.info(f"  总 epochs: {self.epochs}")
        logger.info(f"  阶段 1 (纯数据): 0 - {self.stage1_epochs}")
        logger.info(f"  阶段 2 (渐进物理): {self.stage1_epochs} - {self.stage2_epochs}")
        logger.info(f"  阶段 3 (完整物理): {self.stage2_epochs} - {self.epochs}")
        
        start_time = time.time()
        
        for epoch in range(self.epochs):
            epoch_start = time.time()
            
            # 训练一个 epoch
            losses = self.train_epoch(train_loader, epoch)
            
            epoch_time = time.time() - epoch_start
            
            # 记录历史
            self.history["epoch"].append(epoch)
            self.history["data_loss"].append(losses["data_loss"])
            self.history["physics_loss"].append(losses["physics_loss"])
            self.history["total_loss"].append(losses["total_loss"])
            self.history["lr"].append(self.scheduler.get_last_lr()[0])
            self.history["time"].append(epoch_time)
            
            # 更新最佳模型
            if losses["total_loss"] < self.best_loss:
                self.best_loss = losses["total_loss"]
                self.best_epoch = epoch
                if save_dir:
                    self.model.save_checkpoint(
                        os.path.join(save_dir, "best_model.pt"),
                        optimizer=self.optimizer,
                        epoch=epoch,
                        loss=losses["total_loss"]
                    )
            
            # 日志
            if epoch % log_interval == 0 or epoch == self.epochs - 1:
                logger.info(
                    f"Epoch {epoch:5d} | "
                    f"Data: {losses['data_loss']:.6f} | "
                    f"Physics: {losses['physics_loss']:.6f} | "
                    f"Total: {losses['total_loss']:.6f} | "
                    f"LR: {self.scheduler.get_last_lr()[0]:.2e} | "
                    f"Time: {epoch_time:.2f}s"
                )
            
            # 定期保存
            if save_dir and epoch % save_interval == 0 and epoch > 0:
                self.model.save_checkpoint(
                    os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pt"),
                    optimizer=self.optimizer,
                    epoch=epoch,
                    loss=losses["total_loss"]
                )
            
            # 回调
            if callback:
                callback(epoch, losses, self.model)
        
        total_time = time.time() - start_time
        logger.info(f"训练完成！总时间: {total_time:.1f}s")
        logger.info(f"最佳模型: Epoch {self.best_epoch}, Loss: {self.best_loss:.6f}")
        
        # 保存最终模型
        if save_dir:
            self.model.save_checkpoint(
                os.path.join(save_dir, "final_model.pt"),
                optimizer=self.optimizer,
                epoch=self.epochs - 1,
                loss=losses["total_loss"]
            )
        
        return self.history
    
    def evaluate(
        self,
        test_data: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        评估模型
        
        Args:
            test_data: 测试数据字典
        
        Returns:
            评估指标
        """
        self.model.eval()
        
        spatial_coords = test_data["spatial_coords"].to(self.device)
        voltage_seq = test_data["voltage_sequences"].to(self.device)
        time_seq = test_data["time_sequences"].to(self.device)
        phi_target = test_data["phi_targets"].to(self.device)
        
        with torch.no_grad():
            output = self.model(spatial_coords, voltage_seq, time_seq)
            phi_pred = output["phi"]
            
            # MSE
            mse = self.data_loss_fn(phi_pred, phi_target).item()
            
            # MAE
            mae = torch.abs(phi_pred - phi_target).mean().item()
            
            # 最大误差
            max_error = torch.abs(phi_pred - phi_target).max().item()
        
        return {
            "mse": mse,
            "mae": mae,
            "max_error": max_error,
            "rmse": mse ** 0.5
        }


def train_lstm_pinn(
    config_path: str,
    output_dir: str,
    n_samples: int = 100000,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Tuple[LSTMPINNModel, Dict]:
    """
    训练 LSTM-PINN 模型的便捷函数
    
    Args:
        config_path: 配置文件路径
        output_dir: 输出目录
        n_samples: 训练样本数
        device: 训练设备
    
    Returns:
        训练好的模型和训练历史
    """
    from .config import load_lstm_pinn_config
    
    # 加载配置
    config = load_lstm_pinn_config(config_path)
    raw_config = config.get_raw_config()
    
    # 创建模型
    model = LSTMPINNModel(raw_config)
    
    # 生成训练数据
    logger.info("生成训练数据...")
    data_generator = SequenceDataGenerator(raw_config)
    train_data = data_generator.generate_training_data(n_samples=n_samples)
    
    # 转换为 tensor
    for key in train_data:
        if not isinstance(train_data[key], torch.Tensor):
            train_data[key] = torch.tensor(train_data[key], dtype=torch.float32)
    
    # 创建训练器
    trainer = LSTMPINNTrainer(model, raw_config, device)
    
    # 训练
    history = trainer.train(
        train_data,
        save_dir=output_dir,
        log_interval=100,
        save_interval=1000
    )
    
    return model, history
