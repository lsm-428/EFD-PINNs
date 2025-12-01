"""
Experiment Result Comparison Tool - For analyzing and comparing multiple training experiment results

Features:
1. Multi-experiment metric comparison analysis
2. Training curve visualization comparison
3. Configuration difference analysis
4. Performance ranking and recommendations
5. Experiment report generation
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ExperimentComparator:
    """Experiment comparison analyzer"""
    
    def __init__(self, experiments_dir: str = "./experiments/experiments"):
        """
        Initialize experiment comparator
        
        Args:
            experiments_dir: Experiment directory path
        """
        self.experiments_dir = experiments_dir
        # Use path relative to experiment directory
        base_dir = os.path.dirname(experiments_dir) if experiments_dir else "./experiments"
        self.figures_dir = os.path.join(base_dir, "comparison_figures")
        os.makedirs(self.figures_dir, exist_ok=True)
        
        logger.info(f"Experiment comparator initialized, experiment directory: {experiments_dir}")
    
    def load_experiment_data(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """
        Load experiment data with enhanced directory and file handling
        
        Args:
            experiment_id: 实验ID（可以是配置ID如EXP-xxx或实际实验ID如exp_xxx）
            
        Returns:
            实验数据字典，如果不存在则返回None
        """
        # 尝试直接使用实验ID作为目录名
        experiment_dir = os.path.join(self.experiments_dir, experiment_id)
        
        # 收集所有可能的实验目录候选
        candidate_dirs = [experiment_dir]
        
        # 获取所有可能的实验目录和子目录
        all_possible_dirs = []
        
        # 递归收集所有实验目录（支持深层嵌套）
        def collect_exp_dirs(root_dir, current_depth=0, max_depth=3):
            if current_depth >= max_depth:
                return
                
            try:
                items = os.listdir(root_dir)
                for item in items:
                    item_path = os.path.join(root_dir, item)
                    if os.path.isdir(item_path):
                        # 收集所有目录，不只是以exp_开头的
                        rel_path = os.path.relpath(item_path, self.experiments_dir)
                        all_possible_dirs.append((rel_path, item_path))
                        # 递归进入子目录
                        collect_exp_dirs(item_path, current_depth + 1)
            except Exception as e:
                logger.debug(f"扫描目录失败 {root_dir}: {str(e)}")
        
        # 开始递归收集
        collect_exp_dirs(self.experiments_dir)
        logger.info(f"已收集 {len(all_possible_dirs)} 个可能的实验目录")
        
        # 特殊处理EXP-xxx格式的配置ID
        if experiment_id.startswith('EXP-'):
            logger.info(f"检测到EXP-xxx格式的配置ID，尝试查找匹配的实际实验目录")
            # 检查每个目录的配置文件是否包含此EXP-xxx ID
            matching_dirs = []
            
            for rel_path, abs_path in all_possible_dirs:
                # 查找可能的配置文件
                config_files = []
                try:
                    config_files = [f for f in os.listdir(abs_path) 
                                  if f.endswith('.json') and ('config' in f or 'experiment' in f)]
                except Exception as e:
                    logger.debug(f"读取目录内容失败 {abs_path}: {str(e)}")
                    continue
                
                for config_file in config_files:
                    config_path = os.path.join(abs_path, config_file)
                    try:
                        with open(config_path, 'r', encoding='utf-8') as f:
                            config_data = json.load(f)
                            # 检查配置中是否包含此EXP ID
                            if (config_data.get('id') == experiment_id or 
                                config_data.get('experiment_id') == experiment_id or
                                str(config_data).find(experiment_id) != -1):
                                matching_dirs.append((rel_path, abs_path))
                                logger.info(f"在目录 {abs_path} 的配置文件中找到EXP ID: {experiment_id}")
                                break
                    except Exception as e:
                        logger.debug(f"读取配置文件失败 {config_path}: {str(e)}")
                        continue
            
            if matching_dirs:
                # 按目录名称排序，选择最新的
                matching_dirs.sort(key=lambda x: x[0], reverse=True)
                for rel_path, abs_path in matching_dirs:
                    candidate_dirs.append(abs_path)
                    logger.info(f"添加匹配的实验目录: {abs_path}")
        # 处理普通实验ID，特别是处理多时间戳格式
        else:
            # 针对多时间戳目录名的特殊处理
            # 如果ID是像 exp_20251126_205654 这样的格式，我们也要匹配包含它的更长目录名
            matching_dirs = []
            for rel_path, abs_path in all_possible_dirs:
                # 匹配规则：
                # 1. 目录名完全匹配
                # 2. 目录名以该ID开头并跟着下划线和其他字符（处理多时间戳情况）
                # 3. 目录路径中包含该ID作为部分名称
                dir_name = os.path.basename(abs_path)
                if (dir_name == experiment_id or 
                    dir_name.startswith(f"{experiment_id}_") or 
                    experiment_id in rel_path):
                    matching_dirs.append((rel_path, abs_path))
                    logger.debug(f"潜在匹配目录: {abs_path}")
            
            if matching_dirs:
                # 按目录名称排序，选择最新的
                matching_dirs.sort(key=lambda x: x[0], reverse=True)
                for rel_path, abs_path in matching_dirs:
                    candidate_dirs.append(abs_path)
                    logger.info(f"添加可能匹配的实验目录: {abs_path}")
        
        # 确保candidate_dirs中的目录路径是唯一的
        candidate_dirs = list(set(candidate_dirs))
        logger.info(f"最终候选目录数量: {len(candidate_dirs)}")
        
        # 尝试在所有候选目录中查找配置和指标文件
        found_config = None
        found_metrics = None
        final_dir = None
        
        for candidate_dir in candidate_dirs:
            # 首先检查嵌套子目录情况
            nested_dirs = []
            try:
                if os.path.exists(candidate_dir):
                    nested_dirs = [os.path.join(candidate_dir, d) 
                                 for d in os.listdir(candidate_dir) 
                                 if os.path.isdir(os.path.join(candidate_dir, d)) 
                                 and d.startswith('exp_')]
            except Exception as e:
                logger.debug(f"检查嵌套子目录失败 {candidate_dir}: {str(e)}")
            
            # 将嵌套目录添加到检查列表
            check_dirs = [candidate_dir] + nested_dirs
            
            for check_dir in check_dirs:
                # 查找配置文件
                if found_config is None:
                    config_path = os.path.join(check_dir, "config.json")
                    alt_config_path = os.path.join(check_dir, "experiment_config.json")
                    
                    if os.path.exists(config_path):
                        try:
                            with open(config_path, 'r', encoding='utf-8') as f:
                                found_config = json.load(f)
                            final_dir = check_dir
                            logger.info(f"找到配置文件: {config_path}")
                        except Exception as e:
                            logger.error(f"读取配置文件失败 {config_path}: {str(e)}")
                    elif os.path.exists(alt_config_path):
                        try:
                            with open(alt_config_path, 'r', encoding='utf-8') as f:
                                found_config = json.load(f)
                            final_dir = check_dir
                            logger.info(f"找到备选配置文件: {alt_config_path}")
                        except Exception as e:
                            logger.error(f"读取备选配置文件失败 {alt_config_path}: {str(e)}")
                
                # 查找指标文件
                if found_metrics is None:
                    # 检查多个可能的指标文件路径
                    metrics_candidates = [
                        os.path.join(check_dir, "logs", "reports", "training_metrics.json"),
                        os.path.join(check_dir, "logs", "training_metrics.json"),
                        os.path.join(check_dir, "reports", "training_metrics.json"),
                        os.path.join(check_dir, "training_metrics.json"),
                        os.path.join(check_dir, "training_history.json")  # 备选格式
                    ]
                    
                    for metrics_path in metrics_candidates:
                        if os.path.exists(metrics_path):
                            try:
                                with open(metrics_path, 'r', encoding='utf-8') as f:
                                    found_metrics = json.load(f)
                                logger.info(f"找到训练指标文件: {metrics_path}")
                                # 更新最终目录
                                final_dir = check_dir
                                break
                            except Exception as e:
                                logger.error(f"读取指标文件失败 {metrics_path}: {str(e)}")
                
                # 如果两者都找到，跳出循环
                if found_config and found_metrics:
                    break
            
            if found_config and found_metrics:
                break
        
        # 如果找不到配置，尝试使用一个简单的默认配置
        if not found_config:
            logger.warning(f"未找到有效的配置文件，创建默认配置")
            found_config = {"metadata": {"description": f"Experiment {experiment_id}"}}
        
        # 如果找不到指标，返回空数据
        if not found_metrics:
            logger.warning(f"未找到有效的训练指标文件: {experiment_id}")
            # 尝试直接创建一个最小的训练历史数据结构，避免显示inf
            training_history = {
                "epoch": [0],
                "train_loss": [0.0],
                "val_loss": [0.0],
                "physics_loss": [0.0],
                "learning_rate": [0.001],
                "physics_weight": [1.0],
                "timestamp": [datetime.now().isoformat()]
            }
        else:
            # 解析训练历史
            training_history = self._parse_training_history(found_metrics)
        
        return {
            "experiment_id": experiment_id,
            "config": found_config,
            "training_history": training_history,
            "final_metrics": self._get_final_metrics(training_history),
            "metadata": found_config.get("metadata", {})
        }
    
    def _parse_training_history(self, metrics_data: Dict[str, Any]) -> Dict[str, List]:
        """解析训练历史数据，支持字典和列表两种格式"""
        if not metrics_data:
            return {}
        
        history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "physics_loss": [],
            "learning_rate": [],
            "physics_weight": [],
            "timestamp": []
        }
        
        sorted_timestamps = sorted(metrics_data.keys()) if isinstance(metrics_data, dict) else []
        
        for timestamp in sorted_timestamps:
            metrics = metrics_data[timestamp]
            
            # 处理字典格式
            if isinstance(metrics, dict):
                history["epoch"].append(metrics.get("epoch", 0))
                history["train_loss"].append(metrics.get("train_loss", float('inf')))
                history["val_loss"].append(metrics.get("val_loss", float('inf')))
                history["physics_loss"].append(metrics.get("physics_loss", float('inf')))
                history["learning_rate"].append(metrics.get("learning_rate", 0))
                history["physics_weight"].append(metrics.get("physics_weight", 0))
                history["timestamp"].append(timestamp)
            # 处理列表格式（列表中的每个元素是一个epoch的metrics）
            elif isinstance(metrics, list):
                for i, epoch_metrics in enumerate(metrics):
                    if isinstance(epoch_metrics, dict):
                        history["epoch"].append(epoch_metrics.get("epoch", i + 1))
                        history["train_loss"].append(epoch_metrics.get("train_loss", float('inf')))
                        history["val_loss"].append(epoch_metrics.get("val_loss", float('inf')))
                        history["physics_loss"].append(epoch_metrics.get("physics_loss", float('inf')))
                        history["learning_rate"].append(epoch_metrics.get("learning_rate", 0))
                        history["physics_weight"].append(epoch_metrics.get("physics_weight", 0))
                        history["timestamp"].append(f"{timestamp}_{i}")
                    else:
                        # 如果列表元素不是字典，使用默认值
                        history["epoch"].append(i + 1)
                        history["train_loss"].append(float('inf'))
                        history["val_loss"].append(float('inf'))
                        history["physics_loss"].append(float('inf'))
                        history["learning_rate"].append(0)
                        history["physics_weight"].append(0)
                        history["timestamp"].append(f"{timestamp}_{i}")
            # 其他格式，使用默认值
            else:
                history["epoch"].append(0)
                history["train_loss"].append(float('inf'))
                history["val_loss"].append(float('inf'))
                history["physics_loss"].append(float('inf'))
                history["learning_rate"].append(0)
                history["physics_weight"].append(0)
                history["timestamp"].append(timestamp)
        
        return history
    
    def _get_final_metrics(self, training_history: Dict[str, List]) -> Dict[str, float]:
        """获取最终训练指标"""
        if not training_history or not training_history["epoch"]:
            return {}
        
        final_epoch = len(training_history["epoch"]) - 1
        
        return {
            "final_train_loss": training_history["train_loss"][final_epoch],
            "final_val_loss": training_history["val_loss"][final_epoch],
            "final_physics_loss": training_history["physics_loss"][final_epoch],
            "final_learning_rate": training_history["learning_rate"][final_epoch],
            "final_physics_weight": training_history["physics_weight"][final_epoch],
            "total_epochs": len(training_history["epoch"])
        }
    
    def compare_experiments(self, experiment_ids: List[str]) -> Dict[str, Any]:
        """
        比较多个实验
        
        参数:
            experiment_ids: 实验ID列表
            
        返回:
            比较结果字典
        """
        if not experiment_ids:
            logger.warning("没有提供实验ID")
            return {}
        
        # 加载所有实验数据
        experiments_data = {}
        valid_experiments = []
        
        for exp_id in experiment_ids:
            data = self.load_experiment_data(exp_id)
            if data:
                experiments_data[exp_id] = data
                valid_experiments.append(exp_id)
            else:
                logger.warning(f"无法加载实验数据: {exp_id}")
        
        if not valid_experiments:
            logger.error("没有有效的实验数据")
            return {}
        
        # 生成对比分析
        comparison = {
            "experiments": experiments_data,
            "summary": self._generate_summary(experiments_data),
            "config_comparison": self._compare_configs(experiments_data),
            "performance_ranking": self._rank_experiments(experiments_data),
            "recommendations": self._generate_recommendations(experiments_data)
        }
        
        # 生成可视化图表
        self._generate_comparison_plots(experiments_data)
        
        return comparison
    
    def _generate_summary(self, experiments_data: Dict[str, Dict]) -> Dict[str, Any]:
        """生成实验摘要"""
        summary = {
            "total_experiments": len(experiments_data),
            "experiment_ids": list(experiments_data.keys()),
            "date_range": self._get_date_range(experiments_data),
            "total_training_epochs": sum(data["final_metrics"].get("total_epochs", 0) 
                                      for data in experiments_data.values()),
            "best_val_loss": float('inf'),
            "best_experiment": None
        }
        
        # 找到最佳验证损失
        for exp_id, data in experiments_data.items():
            val_loss = data["final_metrics"].get("final_val_loss", float('inf'))
            if val_loss < summary["best_val_loss"]:
                summary["best_val_loss"] = val_loss
                summary["best_experiment"] = exp_id
        
        return summary
    
    def _get_date_range(self, experiments_data: Dict[str, Dict]) -> Tuple[str, str]:
        """获取实验日期范围"""
        dates = []
        for data in experiments_data.values():
            created_at = data["metadata"].get("created_at", "")
            if created_at:
                try:
                    date_obj = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    dates.append(date_obj)
                except ValueError:
                    continue
        
        if dates:
            min_date = min(dates).strftime("%Y-%m-%d %H:%M")
            max_date = max(dates).strftime("%Y-%m-%d %H:%M")
            return min_date, max_date
        
        return "未知", "未知"
    
    def _compare_configs(self, experiments_data: Dict[str, Dict]) -> Dict[str, Any]:
        """比较实验配置"""
        config_comparison = {
            "model_configs": {},
            "training_configs": {},
            "differences": []
        }
        
        # 收集所有配置
        all_configs = {}
        for exp_id, data in experiments_data.items():
            all_configs[exp_id] = data["config"]
        
        # 比较模型配置
        model_keys = ["input_dim", "output_dim", "hidden_layers", "activation"]
        for key in model_keys:
            values = {}
            for exp_id, config in all_configs.items():
                model_config = config.get("model", {})
                values[exp_id] = model_config.get(key, "未设置")
            config_comparison["model_configs"][key] = values
        
        # 比较训练配置
        training_keys = ["epochs", "batch_size", "learning_rate", "optimizer"]
        for key in training_keys:
            values = {}
            for exp_id, config in all_configs.items():
                training_config = config.get("training", {})
                values[exp_id] = training_config.get(key, "未设置")
            config_comparison["training_configs"][key] = values
        
        # 识别配置差异
        self._identify_config_differences(config_comparison, all_configs)
        
        return config_comparison
    
    def _identify_config_differences(self, config_comparison: Dict[str, Any], 
                                    all_configs: Dict[str, Dict]):
        """识别配置差异"""
        differences = []
        
        # 检查模型配置差异
        for key, values in config_comparison["model_configs"].items():
            # 处理不可哈希的值（如列表）
            unique_values = set()
            for value in values.values():
                if isinstance(value, (list, dict)):
                    unique_values.add(str(value))
                else:
                    unique_values.add(value)
            
            if len(unique_values) > 1:
                differences.append({
                    "category": "模型配置",
                    "parameter": key,
                    "values": values
                })
        
        # 检查训练配置差异
        for key, values in config_comparison["training_configs"].items():
            # 处理不可哈希的值（如列表）
            unique_values = set()
            for value in values.values():
                if isinstance(value, (list, dict)):
                    unique_values.add(str(value))
                else:
                    unique_values.add(value)
            
            if len(unique_values) > 1:
                differences.append({
                    "category": "训练配置",
                    "parameter": key,
                    "values": values
                })
        
        config_comparison["differences"] = differences
    
    def _rank_experiments(self, experiments_data: Dict[str, Dict]) -> List[Dict[str, Any]]:
        """实验性能排名"""
        rankings = []
        
        for exp_id, data in experiments_data.items():
            final_metrics = data["final_metrics"]
            
            # 计算综合得分（验证损失越低越好）
            val_loss = final_metrics.get("final_val_loss", float('inf'))
            train_loss = final_metrics.get("final_train_loss", float('inf'))
            physics_loss = final_metrics.get("final_physics_loss", float('inf'))
            
            # 综合得分计算（验证损失权重最高）
            score = 0.6 * (1 / (val_loss + 1e-8)) + 0.2 * (1 / (train_loss + 1e-8)) + 0.2 * (1 / (physics_loss + 1e-8))
            
            rankings.append({
                "experiment_id": exp_id,
                "description": data["metadata"].get("description", "无描述"),
                "final_val_loss": val_loss,
                "final_train_loss": train_loss,
                "final_physics_loss": physics_loss,
                "total_epochs": final_metrics.get("total_epochs", 0),
                "score": score
            })
        
        # 按得分排序（从高到低）
        rankings.sort(key=lambda x: x["score"], reverse=True)
        
        # 添加排名
        for i, ranking in enumerate(rankings):
            ranking["rank"] = i + 1
        
        return rankings
    
    def _generate_recommendations(self, experiments_data: Dict[str, Dict]) -> List[str]:
        """生成训练建议"""
        recommendations = []
        
        if len(experiments_data) < 2:
            recommendations.append("建议运行更多实验以获得有意义的对比分析")
            return recommendations
        
        # 分析最佳实验的特征
        rankings = self._rank_experiments(experiments_data)
        best_exp_id = rankings[0]["experiment_id"]
        best_config = experiments_data[best_exp_id]["config"]
        
        # 生成基于最佳实验的建议
        recommendations.append(f"最佳实验 {best_exp_id} 的配置值得参考")
        
        # 分析学习率模式
        lr_analysis = self._analyze_learning_rates(experiments_data)
        if lr_analysis:
            recommendations.append(lr_analysis)
        
        # 分析物理权重模式
        physics_weight_analysis = self._analyze_physics_weights(experiments_data)
        if physics_weight_analysis:
            recommendations.append(physics_weight_analysis)
        
        return recommendations
    
    def _analyze_learning_rates(self, experiments_data: Dict[str, Dict]) -> Optional[str]:
        """分析学习率模式"""
        final_lrs = []
        for data in experiments_data.values():
            final_lr = data["final_metrics"].get("final_learning_rate", 0)
            final_lrs.append(final_lr)
        
        if len(final_lrs) >= 2:
            avg_lr = np.mean(final_lrs)
            if avg_lr < 1e-5:
                return "学习率可能设置过低，考虑增加初始学习率"
            elif avg_lr > 1e-2:
                return "学习率可能设置过高，考虑减小初始学习率"
        
        return None
    
    def _analyze_physics_weights(self, experiments_data: Dict[str, Dict]) -> Optional[str]:
        """Analyze physics weight patterns"""
        final_weights = []
        for data in experiments_data.values():
            final_weight = data["final_metrics"].get("final_physics_weight", 0)
            final_weights.append(final_weight)
        
        if len(final_weights) >= 2:
            avg_weight = np.mean(final_weights)
            if avg_weight < 0.1:
                return "Physics constraint weight is low, may need to increase physics constraint weight"
            elif avg_weight > 10:
                return "Physics constraint weight is high, may need to decrease physics constraint weight"
        
        return None
    
    def _generate_comparison_plots(self, experiments_data: Dict[str, Dict]):
        """Generate comparison plots"""
        if not experiments_data:
            return
        
        plt.style.use('seaborn-v0_8')
        
        # 创建损失对比图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Experiment Comparison Analysis', fontsize=16, fontweight='bold')
        
        # 训练损失对比
        ax1 = axes[0, 0]
        has_train_data = False
        for exp_id, data in experiments_data.items():
            history = data["training_history"]
            if history and history["epoch"] and "train_loss" in history:
                if len(history["train_loss"]) == len(history["epoch"]):
                    ax1.plot(history["epoch"], history["train_loss"], 
                            label=f'{exp_id}', linewidth=2, alpha=0.8)
                    has_train_data = True
        ax1.set_title('Training Loss Comparison')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Train Loss')
        if has_train_data:
            ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Validation loss comparison
        ax2 = axes[0, 1]
        has_val_data = False
        for exp_id, data in experiments_data.items():
            history = data["training_history"]
            if history and history["epoch"] and "val_loss" in history:
                if len(history["val_loss"]) == len(history["epoch"]):
                    ax2.plot(history["epoch"], history["val_loss"], 
                            label=f'{exp_id}', linewidth=2, alpha=0.8)
                    has_val_data = True
        ax2.set_title('Validation Loss Comparison')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Val Loss')
        if has_val_data:
            ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Physics loss comparison
        ax3 = axes[1, 0]
        has_physics_data = False
        for exp_id, data in experiments_data.items():
            history = data["training_history"]
            if history and history["epoch"] and "physics_loss" in history:
                if len(history["physics_loss"]) == len(history["epoch"]):
                    ax3.plot(history["epoch"], history["physics_loss"], 
                            label=f'{exp_id}', linewidth=2, alpha=0.8)
                    has_physics_data = True
        ax3.set_title('Physics Loss Comparison')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Physics Loss')
        if has_physics_data:
            ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Learning rate comparison
        ax4 = axes[1, 1]
        has_lr_data = False
        for exp_id, data in experiments_data.items():
            history = data["training_history"]
            if history and history["epoch"] and "learning_rate" in history:
                if len(history["learning_rate"]) == len(history["epoch"]):
                    ax4.plot(history["epoch"], history["learning_rate"], 
                            label=f'{exp_id}', linewidth=2, alpha=0.8)
                    has_lr_data = True
        ax4.set_title('Learning Rate Comparison')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Learning Rate')
        if has_lr_data:
            ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')
        
        plt.tight_layout()
        
        # Save chart
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(self.figures_dir, f"comparison_{timestamp}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Comparison chart saved: {plot_path}")
    
    def generate_comparison_report(self, experiment_ids: List[str], 
                                 output_path: Optional[str] = None) -> str:
        """
        Generate detailed comparison report
        
        Args:
            experiment_ids: List of experiment IDs
            output_path: Output file path
            
        Returns:
            Report file path
        """
        comparison = self.compare_experiments(experiment_ids)
        
        if not comparison:
            logger.error("Failed to generate comparison report")
            return ""
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.figures_dir, f"comparison_report_{timestamp}.txt")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("Experiment Comparison Analysis Report\n")
            f.write("=" * 80 + "\n\n")
            
            # Summary information
            summary = comparison["summary"]
            f.write("📋 Experiment Summary\n")
            f.write(f"   Total experiments: {summary['total_experiments']}\n")
            f.write(f"   Experiment IDs: {', '.join(summary['experiment_ids'])}\n")
            f.write(f"   Date range: {summary['date_range'][0]} to {summary['date_range'][1]}\n")
            f.write(f"   Total training epochs: {summary['total_training_epochs']}\n")
            f.write(f"   Best validation loss: {summary['best_val_loss']:.6f} (Experiment: {summary['best_experiment']})\n\n")
            
            # Performance ranking
            f.write("🏆 Performance Ranking\n")
            rankings = comparison["performance_ranking"]
            for rank in rankings:
                f.write(f"   {rank['rank']}. {rank['experiment_id']}: ")
                f.write(f"Val loss={rank['final_val_loss']:.6f}, ")
                f.write(f"Train loss={rank['final_train_loss']:.6f}, ")
                f.write(f"Physics loss={rank['final_physics_loss']:.6f}\n")
            f.write("\n")
            
            # Configuration differences
            f.write("⚙️  Configuration Difference Analysis\n")
            config_comp = comparison["config_comparison"]
            differences = config_comp["differences"]
            
            if differences:
                for diff in differences:
                    f.write(f"   {diff['category']} - {diff['parameter']}:\n")
                    for exp_id, value in diff["values"].items():
                        f.write(f"      {exp_id}: {value}\n")
                    f.write("\n")
            else:
                f.write("   All experiments have identical configurations\n\n")
            
            # Recommendations
            f.write("💡 Training Recommendations\n")
            recommendations = comparison["recommendations"]
            for i, rec in enumerate(recommendations, 1):
                f.write(f"   {i}. {rec}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("Report generated at: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
            f.write("=" * 80 + "\n")
        
        logger.info(f"📄 Comparison report generated: {output_path}")
        return output_path


# 使用示例
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 创建实验对比器
    comparator = ExperimentComparator()
    
    # 获取所有实验ID（示例）
    experiments_dir = "./experiments/experiments"
    if os.path.exists(experiments_dir):
        experiment_ids = [d for d in os.listdir(experiments_dir) 
                         if os.path.isdir(os.path.join(experiments_dir, d)) and d.startswith("exp_")]
        
        if experiment_ids:
            # 比较实验
            comparison = comparator.compare_experiments(experiment_ids[:3])  # 比较前3个实验
            
            # 生成报告
            report_path = comparator.generate_comparison_report(experiment_ids[:3])
            
            print(f"✅ 实验对比完成！报告已保存到: {report_path}")
        else:
            print("⚠️  没有找到实验数据，请先运行训练实验")
    else:
        print("⚠️  实验目录不存在，请先运行训练实验")