"""
实验报告生成工具 - 用于生成详细的实验分析报告

功能：
1. 实验详细报告生成
2. 训练过程分析
3. 性能指标统计
4. HTML可视化报告
5. 实验复现指南
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging
import webbrowser

logger = logging.getLogger(__name__)


class ExperimentReporter:
    """实验报告生成器"""
    
    def __init__(self, experiments_dir: str = "./experiments/experiments"):
        """
        初始化实验报告生成器
        
        参数:
            experiments_dir: 实验目录路径
        """
        self.experiments_dir = experiments_dir
        # 使用相对于实验目录的路径
        base_dir = os.path.dirname(experiments_dir) if experiments_dir else "./experiments"
        self.reports_dir = os.path.join(base_dir, "reports")
        os.makedirs(self.reports_dir, exist_ok=True)
        
        logger.info(f"实验报告生成器已初始化，实验目录: {experiments_dir}")
    
    def generate_detailed_report(self, experiment_id: str, 
                              output_format: str = "html") -> str:
        """
        生成详细实验报告
        
        参数:
            experiment_id: 实验ID
            output_format: 输出格式 (html/txt)
            
        返回:
            报告文件路径
        """
        experiment_dir = os.path.join(self.experiments_dir, experiment_id)
        
        if not os.path.exists(experiment_dir):
            logger.error(f"实验目录不存在: {experiment_dir}")
            return ""
        
        # 加载实验数据
        experiment_data = self._load_experiment_data(experiment_id)
        if not experiment_data:
            logger.error(f"无法加载实验数据: {experiment_id}")
            return ""
        
        # 生成报告
        if output_format.lower() == "html":
            return self._generate_html_report(experiment_id, experiment_data)
        else:
            return self._generate_text_report(experiment_id, experiment_data)
    
    def _load_experiment_data(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """加载实验数据"""
        experiment_dir = os.path.join(self.experiments_dir, experiment_id)
        
        # 加载配置
        config_path = os.path.join(experiment_dir, "config.json")
        if not os.path.exists(config_path):
            return None
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 加载训练指标
        metrics_path = os.path.join(experiment_dir, "reports", "training_metrics.json")
        metrics_data = {}
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r', encoding='utf-8') as f:
                metrics_data = json.load(f)
        
        # 解析训练历史
        training_history = self._parse_training_history(metrics_data)
        
        # 分析训练过程
        training_analysis = self._analyze_training_process(training_history)
        
        return {
            "experiment_id": experiment_id,
            "config": config,
            "training_history": training_history,
            "training_analysis": training_analysis,
            "metadata": config.get("metadata", {})
        }
    
    def _parse_training_history(self, metrics_data: Dict[str, Any]) -> Dict[str, List]:
        """解析训练历史数据"""
        if not metrics_data:
            return {}
        
        sorted_timestamps = sorted(metrics_data.keys())
        
        history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "physics_loss": [],
            "learning_rate": [],
            "physics_weight": [],
            "timestamp": []
        }
        
        for timestamp in sorted_timestamps:
            metrics = metrics_data[timestamp]
            
            history["epoch"].append(metrics.get("epoch", 0))
            history["train_loss"].append(metrics.get("train_loss", float('inf')))
            history["val_loss"].append(metrics.get("val_loss", float('inf')))
            history["physics_loss"].append(metrics.get("physics_loss", float('inf')))
            history["learning_rate"].append(metrics.get("learning_rate", 0))
            history["physics_weight"].append(metrics.get("physics_weight", 0))
            history["timestamp"].append(timestamp)
        
        return history
    
    def _analyze_training_process(self, training_history: Dict[str, List]) -> Dict[str, Any]:
        """分析训练过程"""
        if not training_history or not training_history["epoch"]:
            return {
                "total_epochs": 0,
                "final_train_loss": float('inf'),
                "final_val_loss": float('inf'),
                "final_physics_loss": float('inf'),
                "best_val_loss": float('inf'),
                "best_val_epoch": 0,
                "convergence_analysis": {
                    "status": "训练数据不足",
                    "analysis": "没有可用的训练历史数据"
                },
                "training_stability": {
                    "status": "未知",
                    "analysis": "没有可用的训练历史数据进行稳定性分析"
                }
            }
        
        epochs = training_history["epoch"]
        train_loss = training_history["train_loss"]
        val_loss = training_history["val_loss"]
        physics_loss = training_history["physics_loss"]
        
        analysis = {
            "total_epochs": len(epochs),
            "final_train_loss": train_loss[-1] if train_loss else float('inf'),
            "final_val_loss": val_loss[-1] if val_loss else float('inf'),
            "final_physics_loss": physics_loss[-1] if physics_loss else float('inf'),
            "best_val_loss": min(val_loss) if val_loss else float('inf'),
            "best_val_epoch": epochs[val_loss.index(min(val_loss))] if val_loss else 0,
            "convergence_analysis": self._analyze_convergence(train_loss, val_loss),
            "training_stability": self._analyze_stability(train_loss, val_loss)
        }
        
        return analysis
    
    def _analyze_convergence(self, train_loss: List[float], val_loss: List[float]) -> Dict[str, Any]:
        """分析收敛性"""
        if len(train_loss) < 10:
            return {
                "status": "训练轮次不足，无法分析收敛性",
                "analysis": "训练轮次不足，无法进行收敛性分析"
            }
        
        # 分析最后100个epoch的损失变化
        window_size = min(100, len(train_loss))
        recent_train = train_loss[-window_size:]
        recent_val = val_loss[-window_size:]
        
        # 计算斜率（判断是否收敛）
        train_slope = np.polyfit(range(window_size), recent_train, 1)[0]
        val_slope = np.polyfit(range(window_size), recent_val, 1)[0]
        
        convergence_status = "良好"
        if train_slope > 0.001 or val_slope > 0.001:
            convergence_status = "可能发散"
        elif abs(train_slope) < 1e-5 and abs(val_slope) < 1e-5:
            convergence_status = "已收敛"
        elif train_slope < -0.001 or val_slope < -0.001:
            convergence_status = "仍在收敛"
        
        return {
            "status": convergence_status,
            "train_slope": train_slope,
            "val_slope": val_slope,
            "analysis": f"训练损失斜率: {train_slope:.6f}, 验证损失斜率: {val_slope:.6f}"
        }
    
    def _analyze_stability(self, train_loss: List[float], val_loss: List[float]) -> Dict[str, Any]:
        """分析训练稳定性"""
        if len(train_loss) < 10:
            return {
                "status": "训练轮次不足，无法分析稳定性",
                "analysis": "训练轮次不足，无法进行稳定性分析"
            }
        
        # 计算损失波动性
        train_std = np.std(train_loss)
        val_std = np.std(val_loss)
        
        stability_status = "稳定"
        if train_std > 0.1 or val_std > 0.1:
            stability_status = "波动较大"
        elif train_std < 0.01 and val_std < 0.01:
            stability_status = "非常稳定"
        
        return {
            "status": stability_status,
            "train_std": train_std,
            "val_std": val_std,
            "analysis": f"训练损失标准差: {train_std:.6f}, 验证损失标准差: {val_std:.6f}"
        }
    
    def _generate_html_report(self, experiment_id: str, 
                            experiment_data: Dict[str, Any]) -> str:
        """生成HTML格式报告"""
        # 简化文件名，只使用实验ID
        report_path = os.path.join(self.reports_dir, f"{experiment_id}_report.html")
        
        # 生成训练曲线图
        plot_path = self._generate_training_plots(experiment_id, experiment_data["training_history"])
        
        # 创建HTML报告
        html_content = self._create_html_content(experiment_id, experiment_data, plot_path)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"📊 HTML报告已生成: {report_path}")
        return report_path
    
    def _generate_text_report(self, experiment_id: str, 
                            experiment_data: Dict[str, Any]) -> str:
        """生成文本格式报告"""
        # 简化文件名，只使用实验ID
        report_path = os.path.join(self.reports_dir, f"{experiment_id}_report.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"实验详细报告 - {experiment_id}\n")
            f.write("=" * 80 + "\n\n")
            
            # 基本信息
            metadata = experiment_data["metadata"]
            f.write("📋 实验基本信息\n")
            f.write(f"   实验ID: {experiment_id}\n")
            f.write(f"   创建时间: {metadata.get('created_at', '未知')}\n")
            f.write(f"   描述: {metadata.get('description', '无描述')}\n")
            f.write(f"   配置版本: {metadata.get('config_version', '未知')}\n\n")
            
            # 训练结果
            analysis = experiment_data["training_analysis"]
            f.write("📊 训练结果分析\n")
            f.write(f"   总训练轮次: {analysis['total_epochs']}\n")
            f.write(f"   最终训练损失: {analysis['final_train_loss']:.6f}\n")
            f.write(f"   最终验证损失: {analysis['final_val_loss']:.6f}\n")
            f.write(f"   最终物理损失: {analysis['final_physics_loss']:.6f}\n")
            f.write(f"   最佳验证损失: {analysis['best_val_loss']:.6f} (第{analysis['best_val_epoch']}轮)\n\n")
            
            # 收敛性分析
            convergence = analysis["convergence_analysis"]
            f.write("📈 收敛性分析\n")
            f.write(f"   状态: {convergence['status']}\n")
            f.write(f"   {convergence['analysis']}\n\n")
            
            # 稳定性分析
            stability = analysis["training_stability"]
            f.write("⚖️  训练稳定性分析\n")
            f.write(f"   状态: {stability['status']}\n")
            f.write(f"   {stability['analysis']}\n\n")
            
            # 配置信息
            config = experiment_data["config"]
            f.write("⚙️  训练配置\n")
            model_config = config.get("model", {})
            training_config = config.get("training", {})
            
            f.write("   模型配置:\n")
            for key, value in model_config.items():
                f.write(f"     {key}: {value}\n")
            
            f.write("\n   训练配置:\n")
            for key, value in training_config.items():
                f.write(f"     {key}: {value}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("报告生成时间: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
            f.write("=" * 80 + "\n")
        
        logger.info(f"📄 文本报告已生成: {report_path}")
        return report_path
    
    def _generate_training_plots(self, experiment_id: str, 
                               training_history: Dict[str, List]) -> str:
        """生成训练曲线图"""
        if not training_history or not training_history["epoch"]:
            return ""
        
        plt.style.use('seaborn-v0_8')
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'训练过程分析 - {experiment_id}', fontsize=16, fontweight='bold')
        
        epochs = training_history["epoch"]
        
        # 损失曲线
        ax1 = axes[0, 0]
        ax1.plot(epochs, training_history["train_loss"], label='训练损失', linewidth=2)
        ax1.plot(epochs, training_history["val_loss"], label='验证损失', linewidth=2)
        ax1.plot(epochs, training_history["physics_loss"], label='物理损失', linewidth=2)
        ax1.set_title('损失曲线')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # 学习率曲线
        ax2 = axes[0, 1]
        ax2.plot(epochs, training_history["learning_rate"], color='red', linewidth=2)
        ax2.set_title('学习率变化')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # 物理权重曲线
        ax3 = axes[1, 0]
        ax3.plot(epochs, training_history["physics_weight"], color='green', linewidth=2)
        ax3.set_title('物理权重变化')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Physics Weight')
        ax3.grid(True, alpha=0.3)
        
        # 损失对比（线性尺度）
        ax4 = axes[1, 1]
        ax4.plot(epochs, training_history["train_loss"], label='训练损失', linewidth=2)
        ax4.plot(epochs, training_history["val_loss"], label='验证损失', linewidth=2)
        ax4.set_title('损失对比（线性尺度）')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        plot_path = os.path.join(self.reports_dir, f"{experiment_id}_training_plots.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def _create_html_content(self, experiment_id: str, 
                           experiment_data: Dict[str, Any], 
                           plot_path: str) -> str:
        """创建HTML内容"""
        metadata = experiment_data["metadata"]
        analysis = experiment_data["training_analysis"]
        config = experiment_data["config"]
        
        html = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>实验报告 - {experiment_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        .header {{ background: #f4f4f4; padding: 20px; border-radius: 10px; margin-bottom: 30px; }}
        .section {{ margin-bottom: 30px; padding: 20px; border: 1px solid #ddd; border-radius: 10px; }}
        .metric {{ display: inline-block; margin: 10px; padding: 15px; background: #e7f3ff; border-radius: 5px; }}
        .config-table {{ width: 100%; border-collapse: collapse; }}
        .config-table th, .config-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        .config-table th {{ background-color: #f2f2f2; }}
        .plot {{ text-align: center; margin: 20px 0; }}
        .plot img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🏆 实验报告 - {experiment_id}</h1>
        <p><strong>创建时间:</strong> {metadata.get('created_at', '未知')}</p>
        <p><strong>描述:</strong> {metadata.get('description', '无描述')}</p>
        <p><strong>报告生成时间:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>
    
    <div class="section">
        <h2>📊 训练结果概览</h2>
        <div class="metric">
            <h3>总训练轮次</h3>
            <p style="font-size: 24px; font-weight: bold; color: #007bff;">{analysis['total_epochs']}</p>
        </div>
        <div class="metric">
            <h3>最佳验证损失</h3>
            <p style="font-size: 24px; font-weight: bold; color: #28a745;">{analysis['best_val_loss']:.6f}</p>
            <p>第 {analysis['best_val_epoch']} 轮</p>
        </div>
        <div class="metric">
            <h3>最终验证损失</h3>
            <p style="font-size: 24px; font-weight: bold; color: #dc3545;">{analysis['final_val_loss']:.6f}</p>
        </div>
    </div>
    
    <div class="section">
        <h2>📈 训练过程分析</h2>
        <div class="plot">
            <img src="{plot_path}" alt="训练过程图表">
        </div>
        
        <h3>收敛性分析</h3>
        <p><strong>状态:</strong> {analysis['convergence_analysis']['status']}</p>
        <p><strong>分析:</strong> {analysis['convergence_analysis']['analysis']}</p>
        
        <h3>训练稳定性分析</h3>
        <p><strong>状态:</strong> {analysis['training_stability']['status']}</p>
        <p><strong>分析:</strong> {analysis['training_stability']['analysis']}</p>
    </div>
    
    <div class="section">
        <h2>⚙️ 训练配置</h2>
        <h3>模型配置</h3>
        <table class="config-table">
            <tr><th>参数</th><th>值</th></tr>
        """
        
        # 添加模型配置
        model_config = config.get("model", {})
        for key, value in model_config.items():
            html += f"<tr><td>{key}</td><td>{value}</td></tr>\n"
        
        html += """
        </table>
        
        <h3>训练配置</h3>
        <table class="config-table">
            <tr><th>参数</th><th>值</th></tr>
        """
        
        # 添加训练配置
        training_config = config.get("training", {})
        for key, value in training_config.items():
            html += f"<tr><td>{key}</td><td>{value}</td></tr>\n"
        
        html += """
        </table>
    </div>
    
    <div class="section">
        <h2>💡 实验建议</h2>
        <ul>
        """
        
        # 生成建议
        suggestions = self._generate_suggestions(analysis)
        for suggestion in suggestions:
            html += f"<li>{suggestion}</li>\n"
        
        html += """
        </ul>
    </div>
    
    <div class="section">
        <h2>🔧 实验复现指南</h2>
        <p>要复现此实验，请使用以下配置：</p>
        <pre><code>"""
        
        # 添加配置JSON
        html += json.dumps(config, indent=2, ensure_ascii=False)
        
        html += """
</code></pre>
    </div>
</body>
</html>
"""
        
        return html
    
    def _generate_suggestions(self, analysis: Dict[str, Any]) -> List[str]:
        """生成训练建议"""
        suggestions = []
        
        # 基于收敛性分析的建议
        convergence = analysis["convergence_analysis"]
        if convergence["status"] == "可能发散":
            suggestions.append("训练可能发散，建议减小学习率或检查数据预处理")
        elif convergence["status"] == "仍在收敛":
            suggestions.append("训练仍在收敛，可以考虑增加训练轮次")
        
        # 基于稳定性分析的建议
        stability = analysis["training_stability"]
        if stability["status"] == "波动较大":
            suggestions.append("训练过程波动较大，建议调整批次大小或学习率调度策略")
        
        # 基于最终损失的建议
        if analysis["final_val_loss"] > 0.1:
            suggestions.append("最终验证损失较高，建议检查模型架构或数据质量")
        
        # 通用建议
        suggestions.append("考虑使用早停机制来防止过拟合")
        suggestions.append("可以尝试不同的优化器或学习率调度策略")
        
        return suggestions
    
    def open_report_in_browser(self, report_path: str):
        """在浏览器中打开报告"""
        try:
            webbrowser.open(f"file://{os.path.abspath(report_path)}")
            logger.info(f"🌐 在浏览器中打开报告: {report_path}")
        except Exception as e:
            logger.warning(f"无法在浏览器中打开报告: {e}")


# 使用示例
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 创建报告生成器
    reporter = ExperimentReporter()
    
    # 获取实验ID（示例）
    experiments_dir = "./experiments/experiments"
    if os.path.exists(experiments_dir):
        experiment_ids = [d for d in os.listdir(experiments_dir) 
                         if os.path.isdir(os.path.join(experiments_dir, d)) and d.startswith("exp_")]
        
        if experiment_ids:
            # 生成HTML报告
            report_path = reporter.generate_detailed_report(experiment_ids[0], "html")
            
            # 在浏览器中打开
            reporter.open_report_in_browser(report_path)
            
            print(f"✅ 实验报告已生成: {report_path}")
        else:
            print("⚠️  没有找到实验数据，请先运行训练实验")
    else:
        print("⚠️  实验目录不存在，请先运行训练实验")