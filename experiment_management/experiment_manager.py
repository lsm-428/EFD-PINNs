"""
实验管理器模块 - 用于管理训练实验的配置、结果和版本控制

功能：
1. 实验配置版本化
2. 训练结果记录
3. 实验对比分析
4. 实验复现支持
"""

import json
import os
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class ExperimentManager:
    """实验管理器类 - 管理训练实验的完整生命周期"""
    
    def __init__(self, base_dir: str = "./experiments"):
        """
        初始化实验管理器
        
        参数:
            base_dir: 实验根目录路径
        """
        self.base_dir = base_dir
        self.experiments_dir = os.path.join(base_dir, "experiments")
        self.configs_dir = os.path.join(base_dir, "configs")
        self.templates_dir = os.path.join(base_dir, "templates")
        
        # 创建必要的目录结构
        os.makedirs(self.experiments_dir, exist_ok=True)
        os.makedirs(self.configs_dir, exist_ok=True)
        os.makedirs(self.templates_dir, exist_ok=True)
        
        logger.info(f"实验管理器已初始化，基础目录: {base_dir}")
    
    def create_experiment(self, config: Dict[str, Any], description: str = "") -> tuple[str, str]:
        """
        创建新的训练实验
        
        参数:
            config: 训练配置字典
            description: 实验描述
            
        返回:
            (experiment_id, experiment_dir)
        """
        # 生成实验ID（带时间戳）
        experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        experiment_dir = os.path.join(self.experiments_dir, experiment_id)
        
        # 创建实验目录结构
        os.makedirs(experiment_dir, exist_ok=True)
        os.makedirs(os.path.join(experiment_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(experiment_dir, "reports"), exist_ok=True)
        os.makedirs(os.path.join(experiment_dir, "visualizations"), exist_ok=True)
        os.makedirs(os.path.join(experiment_dir, "logs"), exist_ok=True)
        
        # 增强配置信息
        enhanced_config = {
            "metadata": {
                "experiment_id": experiment_id,
                "created_at": datetime.now().isoformat(),
                "description": description,
                "config_version": "1.0"
            },
            **config
        }
        
        # 保存配置到实验目录
        config_path = os.path.join(experiment_dir, "config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(enhanced_config, f, indent=2, ensure_ascii=False)
        
        # 保存配置副本到配置目录
        config_copy_path = os.path.join(self.configs_dir, f"{experiment_id}_config.json")
        with open(config_copy_path, 'w', encoding='utf-8') as f:
            json.dump(enhanced_config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ 创建实验: {experiment_id}")
        logger.info(f"   描述: {description}")
        logger.info(f"   目录: {experiment_dir}")
        
        return experiment_id, experiment_dir
    
    def log_training_metrics(self, experiment_id: str, metrics: Dict[str, Any]) -> str:
        """
        记录训练指标
        
        参数:
            experiment_id: 实验ID
            metrics: 指标字典
            
        返回:
            指标文件路径
        """
        # 调试路径构建问题
        logger.info(f"调试路径: base_dir={self.base_dir}, experiments_dir={self.experiments_dir}, experiment_id={experiment_id}")
        
        # 修复路径处理：确保使用正确的实验ID而不是路径
        # 从路径中提取实验ID（如果experiment_id是路径）
        if os.path.isdir(experiment_id) or '/' in experiment_id or '\\' in experiment_id:
            # 提取最后一个目录名作为可能的实验ID
            candidate_id = os.path.basename(experiment_id)
            # 检查是否是有效的实验ID格式（以exp_开头）
            if candidate_id.startswith('exp_'):
                logger.info(f"从路径中提取实验ID: {candidate_id}")
                experiment_id = candidate_id
            else:
                logger.warning(f"警告: 提供的experiment_id既不是有效路径也不是标准实验ID格式: {experiment_id}")
                # 生成一个新的有效实验ID，而不是使用无效值
                fallback_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                logger.info(f"使用生成的默认实验ID: {fallback_id}")
                experiment_id = fallback_id
        # 如果experiment_id不是以exp_开头，也生成一个有效的ID
        elif not experiment_id.startswith('exp_'):
            logger.warning(f"警告: experiment_id不是有效的格式(应以exp_开头): {experiment_id}")
            fallback_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            logger.info(f"使用生成的默认实验ID: {fallback_id}")
            experiment_id = fallback_id
        
        # 构建正确的实验目录路径
        experiment_dir = os.path.join(self.experiments_dir, experiment_id)
        logger.info(f"构建实验目录路径: {experiment_dir}")
        
        # 确保目录存在
        os.makedirs(experiment_dir, exist_ok=True)
        os.makedirs(os.path.join(experiment_dir, "reports"), exist_ok=True)
        
        metrics_path = os.path.join(experiment_dir, "reports", "training_metrics.json")
        
        # 读取现有指标或创建新文件
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r', encoding='utf-8') as f:
                existing_metrics = json.load(f)
        else:
            existing_metrics = {}
        
        # 添加时间戳作为键
        timestamp = datetime.now().isoformat()
        existing_metrics[timestamp] = metrics
        
        # 保存指标
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(existing_metrics, f, indent=2, ensure_ascii=False)
        
        logger.debug(f"📊 记录训练指标到: {metrics_path}")
        return metrics_path
    
    def save_model_checkpoint(self, experiment_id: str, model_state: Dict[str, Any], 
                             epoch: int, loss: float) -> str:
        """
        保存模型检查点
        
        参数:
            experiment_id: 实验ID
            model_state: 模型状态字典
            epoch: 当前轮次
            loss: 当前损失
            
        返回:
            检查点文件路径
        """
        # 调试路径构建问题
        logger.info(f"调试检查点路径: experiment_id={experiment_id}")
        
        # 修复路径处理：确保使用正确的实验ID而不是路径
        # 从路径中提取实验ID（如果experiment_id是路径）
        if os.path.isdir(experiment_id) or '/' in experiment_id or '\\' in experiment_id:
            # 提取最后一个目录名作为可能的实验ID
            candidate_id = os.path.basename(experiment_id)
            # 检查是否是有效的实验ID格式（以exp_开头）
            if candidate_id.startswith('exp_'):
                logger.info(f"从路径中提取实验ID: {candidate_id}")
                experiment_id = candidate_id
            else:
                logger.warning(f"警告: 提供的experiment_id既不是有效路径也不是标准实验ID格式: {experiment_id}")
                # 生成一个新的有效实验ID，而不是使用无效值
                fallback_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                logger.info(f"使用生成的默认实验ID: {fallback_id}")
                experiment_id = fallback_id
        # 如果experiment_id不是以exp_开头，也生成一个有效的ID
        elif not experiment_id.startswith('exp_'):
            logger.warning(f"警告: experiment_id不是有效的格式(应以exp_开头): {experiment_id}")
            fallback_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            logger.info(f"使用生成的默认实验ID: {fallback_id}")
            experiment_id = fallback_id
        
        # 构建正确的实验目录路径
        experiment_dir = os.path.join(self.experiments_dir, experiment_id)
        logger.info(f"构建检查点实验目录路径: {experiment_dir}")
        
        # 确保目录存在
        os.makedirs(experiment_dir, exist_ok=True)
        checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch:06d}.pth")
        
        # 保存检查点
        import torch
        checkpoint = {
            'epoch': epoch,
            'loss': loss,
            'model_state_dict': model_state,
            'timestamp': datetime.now().isoformat()
        }
        torch.save(checkpoint, checkpoint_path)
        
        logger.info(f"💾 保存检查点: {checkpoint_path} (epoch: {epoch}, loss: {loss:.6f})")
        return checkpoint_path
    
    def get_experiment_info(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """
        获取实验信息
        
        参数:
            experiment_id: 实验ID
            
        返回:
            实验信息字典，如果不存在则返回None
        """
        experiment_dir = os.path.join(self.experiments_dir, experiment_id)
        config_path = os.path.join(experiment_dir, "config.json")
        
        if not os.path.exists(config_path):
            return None
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 检查是否有训练指标
        metrics_path = os.path.join(experiment_dir, "reports", "training_metrics.json")
        has_metrics = os.path.exists(metrics_path)
        
        # 检查检查点数量
        checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
        if os.path.exists(checkpoint_dir):
            checkpoint_count = len([f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')])
        else:
            checkpoint_count = 0
        
        info = {
            "experiment_id": experiment_id,
            "directory": experiment_dir,
            "config": config,
            "has_metrics": has_metrics,
            "checkpoint_count": checkpoint_count,
            "created_at": config.get("metadata", {}).get("created_at", "未知")
        }
        
        return info
    
    def list_experiments(self) -> List[Dict[str, Any]]:
        """
        列出所有实验
        
        返回:
            实验信息列表
        """
        experiments = []
        
        if not os.path.exists(self.experiments_dir):
            return experiments
        
        for item in os.listdir(self.experiments_dir):
            item_path = os.path.join(self.experiments_dir, item)
            if os.path.isdir(item_path) and item.startswith("exp_"):
                info = self.get_experiment_info(item)
                if info:
                    experiments.append(info)
        
        # 按创建时间排序（最新的在前）
        experiments.sort(key=lambda x: x["created_at"], reverse=True)
        
        return experiments
    
    def compare_experiments(self, experiment_ids: List[str]) -> Dict[str, Any]:
        """
        比较多个实验的结果
        
        参数:
            experiment_ids: 实验ID列表
            
        返回:
            比较结果字典
        """
        comparisons = {}
        
        for exp_id in experiment_ids:
            info = self.get_experiment_info(exp_id)
            if not info:
                continue
            
            # 获取最终训练指标
            metrics_path = os.path.join(self.experiments_dir, exp_id, "reports", "training_metrics.json")
            final_metrics = {}
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r', encoding='utf-8') as f:
                    all_metrics = json.load(f)
                if all_metrics:
                    # 获取最新的指标
                    latest_timestamp = sorted(all_metrics.keys())[-1]
                    final_metrics = all_metrics[latest_timestamp]
            
            comparisons[exp_id] = {
                "config": info["config"],
                "final_metrics": final_metrics,
                "checkpoint_count": info["checkpoint_count"]
            }
        
        return comparisons


def save_config_with_timestamp(config: Dict[str, Any], description: str = "") -> str:
    """
    保存带时间戳的配置副本（独立函数，便于单独使用）
    
    参数:
        config: 配置字典
        description: 配置描述
        
    返回:
        保存的配置文件路径
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_filename = f"train_config_{timestamp}.json"
    configs_dir = "./experiments/configs"
    
    os.makedirs(configs_dir, exist_ok=True)
    config_path = os.path.join(configs_dir, config_filename)
    
    # 增强配置信息
    enhanced_config = {
        "metadata": {
            "config_id": config_filename.replace(".json", ""),
            "created_at": datetime.now().isoformat(),
            "description": description,
            "config_version": "1.0"
        },
        **config
    }
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(enhanced_config, f, indent=2, ensure_ascii=False)
    
    logger.info(f"📄 保存配置副本: {config_path}")
    return config_path


# 使用示例
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 示例配置
    sample_config = {
        "model": {
            "input_dim": 62,
            "output_dim": 24,
            "hidden_layers": [64, 32, 16]
        },
        "training": {
            "epochs": 100,
            "batch_size": 64,
            "learning_rate": 0.001
        }
    }
    
    # 创建实验管理器
    manager = ExperimentManager()
    
    # 创建实验
    exp_id, exp_dir = manager.create_experiment(sample_config, "示例训练实验")
    
    # 记录训练指标
    metrics = {
        "epoch": 1,
        "train_loss": 0.5,
        "val_loss": 0.3,
        "learning_rate": 0.001
    }
    manager.log_training_metrics(exp_id, metrics)
    
    # 列出所有实验
    experiments = manager.list_experiments()
    print(f"\n📋 实验列表 ({len(experiments)} 个实验):")
    for exp in experiments:
        print(f"  - {exp['experiment_id']}: {exp['config']['metadata']['description']}")
    
    print("\n✅ 实验管理器测试完成！")