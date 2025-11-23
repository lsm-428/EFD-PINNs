import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime
import os
import json
import logging
from collections import defaultdict

class TrainingTracker:
    """
    增强的训练跟踪器，用于记录、分析和可视化模型训练过程中的各种指标
    """
    def __init__(self, log_dir=None, config=None):
        self.log_dir = log_dir or f'logs_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        self.config = config or {}
        
        # 创建日志目录
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 初始化指标容器
        self.metrics = defaultdict(list)
        self.stage_metrics = {}
        self.best_metrics = {}
        
        # 性能跟踪
        self.performance = {
            'epoch_times': [],
            'forward_times': [],
            'backward_times': [],
            'optimizer_times': []
        }
        
        # 初始化日志记录器
        self.logger = self._setup_logger()
        self.logger.info(f"初始化训练跟踪器，日志目录: {self.log_dir}")
        
        # 保存配置
        if config:
            self.save_config(config)
    
    def _setup_logger(self):
        """设置日志记录器"""
        logger = logging.getLogger('EWPINN_TrainingTracker')
        logger.setLevel(logging.INFO)
        
        # 避免重复添加处理器
        if not logger.handlers:
            # 控制台处理器
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            logger.addHandler(console_handler)
            
            # 文件处理器
            file_handler = logging.FileHandler(
                os.path.join(self.log_dir, 'training_tracker.log')
            )
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            logger.addHandler(file_handler)
        
        return logger
    
    def save_config(self, config):
        """保存训练配置"""
        try:
            with open(os.path.join(self.log_dir, 'training_config.json'), 'w') as f:
                json.dump(config, f, indent=4)
            self.logger.info(f"训练配置已保存到 {os.path.join(self.log_dir, 'training_config.json')}")
        except Exception as e:
            self.logger.error(f"保存配置失败: {str(e)}")
    
    def start_epoch_timer(self):
        """开始记录epoch时间"""
        self.epoch_start_time = time.time()
    
    def end_epoch_timer(self):
        """结束记录epoch时间并保存"""
        epoch_time = time.time() - self.epoch_start_time
        self.performance['epoch_times'].append(epoch_time)
        return epoch_time
    
    def record_batch_time(self, phase, batch_time):
        """记录批次处理时间"""
        if phase in self.performance:
            self.performance[phase].append(batch_time)
    
    def log_epoch_metrics(self, epoch, stage, train_metrics, val_metrics, lr):
        """记录并保存epoch指标"""
        # 添加到全局指标
        for key, value in train_metrics.items():
            self.metrics[f'train_{key}'].append(value)
        for key, value in val_metrics.items():
            self.metrics[f'val_{key}'].append(value)
        
        self.metrics['epoch'].append(epoch)
        self.metrics['stage'].append(stage)
        self.metrics['lr'].append(lr)
        
        # 更新阶段指标
        if stage not in self.stage_metrics:
            self.stage_metrics[stage] = defaultdict(list)
        
        for key, value in train_metrics.items():
            self.stage_metrics[stage][f'train_{key}'].append(value)
        for key, value in val_metrics.items():
            self.stage_metrics[stage][f'val_{key}'].append(value)
        
        # 构建日志消息
        log_msg = f"阶段 {stage} - Epoch {epoch} - "
        log_msg += " - ".join([f"训练{key}: {value:.6f}" for key, value in train_metrics.items()])
        log_msg += " - "
        log_msg += " - ".join([f"验证{key}: {value:.6f}" for key, value in val_metrics.items()])
        log_msg += f" - 学习率: {lr:.6e}"
        
        # 添加性能信息
        if self.performance['epoch_times']:
            avg_epoch_time = np.mean(self.performance['epoch_times'][-min(10, len(self.performance['epoch_times'])):])
            log_msg += f" - 平均Epoch时间: {avg_epoch_time:.2f}s"
        
        self.logger.info(log_msg)
        
        # 保存最新指标
        self.save_metrics()
    
    def log_gradient_stats(self, model, epoch, stage):
        """记录梯度统计信息"""
        grad_norms = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = torch.norm(param.grad).item()
                grad_norms.append(grad_norm)
        
        if grad_norms:
            avg_grad_norm = np.mean(grad_norms)
            max_grad_norm = np.max(grad_norms)
            self.metrics['avg_grad_norm'].append(avg_grad_norm)
            self.metrics['max_grad_norm'].append(max_grad_norm)
            
            if stage not in self.stage_metrics:
                self.stage_metrics[stage] = defaultdict(list)
            
            self.stage_metrics[stage]['avg_grad_norm'].append(avg_grad_norm)
            self.stage_metrics[stage]['max_grad_norm'].append(max_grad_norm)
            
            self.logger.info(f"阶段 {stage} - Epoch {epoch} - 平均梯度范数: {avg_grad_norm:.6f} - 最大梯度范数: {max_grad_norm:.6f}")
    
    def log_physics_consistency(self, stage, results):
        """记录物理一致性验证结果"""
        if 'physics_consistency' not in self.metrics:
            self.metrics['physics_consistency'] = []
        
        self.metrics['physics_consistency'].append(results)
        
        # 格式化物理一致性结果
        consistency_str = ", ".join([f"{k}: {float(v) if isinstance(v, torch.Tensor) else v:.6f}"
                                    for k, v in results.items()])
        
        self.logger.info(f"阶段 {stage} - 物理一致性验证: {consistency_str}")
        
        # 保存物理一致性结果
        self.save_physics_consistency()
    
    def update_best_metrics(self, metrics, stage):
        """更新最佳指标"""
        if stage not in self.best_metrics:
            self.best_metrics[stage] = {}
        
        improved = False
        for key, value in metrics.items():
            # 对于损失类指标，我们寻找最小值
            if key.endswith('loss'):
                current_best = self.best_metrics[stage].get(key, float('inf'))
                if value < current_best:
                    self.best_metrics[stage][key] = value
                    improved = True
            # 对于其他指标，可以根据具体情况调整逻辑
        
        if improved:
            self.logger.info(f"阶段 {stage} - 最佳指标已更新")
            self.save_best_metrics()
        
        return improved
    
    def save_metrics(self):
        """保存所有指标到文件"""
        try:
            # 转换numpy数组和tensor为可序列化的类型
            serializable_metrics = {}
            for key, values in self.metrics.items():
                serializable_values = []
                for v in values:
                    if isinstance(v, np.ndarray):
                        serializable_values.append(v.tolist())
                    elif isinstance(v, torch.Tensor):
                        serializable_values.append(v.item())
                    elif isinstance(v, dict):
                        # 处理字典类型，递归转换
                        serializable_values.append({
                            k: float(val) if isinstance(val, torch.Tensor) else val
                            for k, val in v.items()
                        })
                    else:
                        serializable_values.append(v)
                serializable_metrics[key] = serializable_values
            
            with open(os.path.join(self.log_dir, 'training_metrics.json'), 'w') as f:
                json.dump(serializable_metrics, f, indent=4)
        except Exception as e:
            self.logger.error(f"保存指标失败: {str(e)}")
    
    def save_physics_consistency(self):
        """保存物理一致性验证结果"""
        try:
            if 'physics_consistency' in self.metrics:
                # 转换结果为可序列化类型
                serializable_results = []
                for result in self.metrics['physics_consistency']:
                    serializable_result = {}
                    for key, value in result.items():
                        if isinstance(value, torch.Tensor):
                            serializable_result[key] = value.item()
                        else:
                            serializable_result[key] = value
                    serializable_results.append(serializable_result)
                
                with open(os.path.join(self.log_dir, 'physics_consistency_results.json'), 'w') as f:
                    json.dump(serializable_results, f, indent=4)
        except Exception as e:
            self.logger.error(f"保存物理一致性结果失败: {str(e)}")
    
    def save_best_metrics(self):
        """保存最佳指标"""
        try:
            # 转换numpy数组和tensor为可序列化的类型
            serializable_best = {}
            for stage, metrics in self.best_metrics.items():
                serializable_best[stage] = {}
                for key, value in metrics.items():
                    if isinstance(value, np.ndarray):
                        serializable_best[stage][key] = value.tolist()
                    elif isinstance(value, torch.Tensor):
                        serializable_best[stage][key] = value.item()
                    else:
                        serializable_best[stage][key] = value
            
            with open(os.path.join(self.log_dir, 'best_metrics.json'), 'w') as f:
                json.dump(serializable_best, f, indent=4)
        except Exception as e:
            self.logger.error(f"保存最佳指标失败: {str(e)}")
    
    def plot_training_history(self, save_path=None, show_plot=False):
        """生成增强的训练历史可视化"""
        try:
            save_path = save_path or os.path.join(self.log_dir, 'training_history.png')
            
            # 检查是否有指标可绘制
            if not self.metrics or not self.metrics.get('epoch'):
                self.logger.warning("没有足够的训练历史数据用于绘制")
                return
            
            # 创建大图，分割为多个子图
            fig = plt.figure(figsize=(20, 15))
            gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)
            
            # 1. 损失图 (左上)
            ax1 = fig.add_subplot(gs[0, 0])
            for key in self.metrics:
                if key.startswith('train_') and 'loss' in key:
                    ax1.plot(self.metrics['epoch'], self.metrics[key], label=key)
            for key in self.metrics:
                if key.startswith('val_') and 'loss' in key:
                    ax1.plot(self.metrics['epoch'], self.metrics[key], label=key, linestyle='--')
            ax1.set_title('损失变化')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)
            ax1.set_yscale('log')  # 使用对数刻度更好地显示损失变化
            
            # 2. 学习率图 (右上)
            ax2 = fig.add_subplot(gs[0, 1])
            if 'lr' in self.metrics:
                ax2.plot(self.metrics['epoch'], self.metrics['lr'], 'g-')
                ax2.set_title('学习率变化')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Learning Rate')
                ax2.grid(True)
                ax2.set_yscale('log')
            
            # 3. 梯度范数图 (左下)
            ax3 = fig.add_subplot(gs[1, 0])
            if 'avg_grad_norm' in self.metrics:
                ax3.plot(self.metrics['epoch'], self.metrics['avg_grad_norm'], 'b-', label='平均梯度范数')
            if 'max_grad_norm' in self.metrics:
                ax3.plot(self.metrics['epoch'], self.metrics['max_grad_norm'], 'r-', label='最大梯度范数')
            ax3.set_title('梯度范数')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Gradient Norm')
            ax3.legend()
            ax3.grid(True)
            ax3.set_yscale('log')
            
            # 4. 性能时间图 (右下)
            ax4 = fig.add_subplot(gs[1, 1])
            if self.performance['epoch_times']:
                # 计算累积平均时间
                cumulative_avg = np.cumsum(self.performance['epoch_times']) / np.arange(1, len(self.performance['epoch_times']) + 1)
                ax4.plot(range(1, len(cumulative_avg) + 1), cumulative_avg, 'm-')
                ax4.set_title('Epoch平均执行时间')
                ax4.set_xlabel('Epoch')
                ax4.set_ylabel('Time (seconds)')
                ax4.grid(True)
            
            # 5. 物理一致性图 (底部)
            ax5 = fig.add_subplot(gs[2, :])
            if 'physics_consistency' in self.metrics:
                # 提取一致性指标
                consistency_epochs = []
                consistency_metrics = defaultdict(list)
                
                for i, result in enumerate(self.metrics['physics_consistency']):
                    # 假设每个一致性结果对应一个阶段结束
                    if i < len(self.metrics['stage']):
                        # 找到该阶段的最后一个epoch
                        stage = self.metrics['stage'][i]
                        stage_epochs = [e for j, e in enumerate(self.metrics['epoch']) 
                                      if j < len(self.metrics['stage']) and self.metrics['stage'][j] == stage]
                        if stage_epochs:
                            consistency_epochs.append(max(stage_epochs))
                            for key, value in result.items():
                                consistency_metrics[key].append(float(value) if isinstance(value, torch.Tensor) else value)
                
                # 绘制物理一致性指标
                for key, values in consistency_metrics.items():
                    if len(consistency_epochs) == len(values):
                        ax5.plot(consistency_epochs, values, marker='o', label=key)
                
                ax5.set_title('物理一致性指标')
                ax5.set_xlabel('Epoch')
                ax5.set_ylabel('Error')
                ax5.legend()
                ax5.grid(True)
                ax5.set_yscale('log')
            
            # 添加标题和信息
            fig.suptitle(f'EWPINN训练历史 - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', fontsize=16)
            
            # 保存图表
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"训练历史图表已保存到 {save_path}")
            
            if show_plot:
                plt.show()
            
            plt.close(fig)
            
        except Exception as e:
            self.logger.error(f"绘制训练历史失败: {str(e)}")
    
    def generate_training_report(self):
        """生成训练报告"""
        try:
            report_path = os.path.join(self.log_dir, 'training_report.md')
            
            with open(report_path, 'w') as f:
                f.write(f"# EWPINN训练报告\n\n")
                f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # 1. 训练配置
                f.write("## 训练配置\n\n")
                if self.config:
                    for key, value in self.config.items():
                        if isinstance(value, dict):
                            f.write(f"### {key}\n\n")
                            for k, v in value.items():
                                f.write(f"- **{k}**: {v}\n")
                        else:
                            f.write(f"- **{key}**: {value}\n")
                
                # 2. 最佳指标
                f.write("\n## 最佳指标\n\n")
                if self.best_metrics:
                    for stage, metrics in self.best_metrics.items():
                        f.write(f"### 阶段 {stage}\n\n")
                        for key, value in metrics.items():
                            f.write(f"- **{key}**: {value:.6f}\n")
                
                # 3. 性能统计
                f.write("\n## 性能统计\n\n")
                if self.performance['epoch_times']:
                    total_epochs = len(self.performance['epoch_times'])
                    total_time = sum(self.performance['epoch_times'])
                    avg_time = np.mean(self.performance['epoch_times'])
                    
                    f.write(f"- **总Epochs**: {total_epochs}\n")
                    f.write(f"- **总训练时间**: {total_time:.2f}s ({total_time/60:.2f}分钟)\n")
                    f.write(f"- **平均Epoch时间**: {avg_time:.2f}s\n")
                
                # 4. 物理一致性验证结果
                f.write("\n## 物理一致性验证\n\n")
                if 'physics_consistency' in self.metrics:
                    for i, result in enumerate(self.metrics['physics_consistency']):
                        stage = self.metrics['stage'][i] if i < len(self.metrics['stage']) else '未知'
                        f.write(f"### 阶段 {stage}\n\n")
                        for key, value in result.items():
                            val = float(value) if isinstance(value, torch.Tensor) else value
                            f.write(f"- **{key}**: {val:.6f}\n")
                
                # 5. 建议
                f.write("\n## 改进建议\n\n")
                suggestions = []
                
                # 基于训练历史生成建议
                if 'train_loss' in self.metrics and len(self.metrics['train_loss']) > 10:
                    recent_losses = self.metrics['train_loss'][-10:]
                    # 检查训练是否稳定
                    loss_std = np.std(recent_losses)
                    if loss_std < 1e-6:
                        suggestions.append("训练损失已趋于稳定，考虑早停或减小学习率")
                    
                    # 检查过拟合
                    if 'val_loss' in self.metrics and len(self.metrics['val_loss']) > 10:
                        recent_val_losses = self.metrics['val_loss'][-10:]
                        if recent_val_losses[-1] > recent_val_losses[0] * 1.2:
                            suggestions.append("验证损失呈上升趋势，可能存在过拟合，建议增加正则化或提前停止")
                
                # 基于梯度范数生成建议
                if 'max_grad_norm' in self.metrics:
                    max_grad = max(self.metrics['max_grad_norm'])
                    if max_grad > 10.0:
                        suggestions.append("梯度范数过大，建议实施梯度裁剪")
                
                # 基于物理一致性生成建议
                if 'physics_consistency' in self.metrics and self.metrics['physics_consistency']:
                    latest_consistency = self.metrics['physics_consistency'][-1]
                    if 'continuity_error' in latest_consistency:
                        continuity_error = float(latest_consistency['continuity_error']) if isinstance(latest_consistency['continuity_error'], torch.Tensor) else latest_consistency['continuity_error']
                        if continuity_error > 1e-3:
                            suggestions.append("连续性误差较大，建议增加物理约束权重或调整网络架构")
                
                if suggestions:
                    for suggestion in suggestions:
                        f.write(f"- {suggestion}\n")
                else:
                    f.write("- 训练过程表现良好，建议继续使用当前配置")
            
            self.logger.info(f"训练报告已生成: {report_path}")
            return report_path
            
        except Exception as e:
            self.logger.error(f"生成训练报告失败: {str(e)}")
            return None

# 性能分析器
class PerformanceAnalyzer:
    """
    性能分析器，用于分析模型训练和推理过程中的性能瓶颈
    """
    def __init__(self, log_dir=None):
        self.log_dir = log_dir
        self.logger = logging.getLogger('EWPINN_PerformanceAnalyzer')
        
        # 时间记录
        self.timings = defaultdict(list)
        self.profiler_start_time = None
    
    def start_profiler(self):
        """开始性能分析"""
        self.profiler_start_time = time.time()
    
    def end_profiler(self):
        """结束性能分析并返回耗时"""
        if self.profiler_start_time is not None:
            elapsed = time.time() - self.profiler_start_time
            self.profiler_start_time = None
            return elapsed
        return 0
    
    def record_operation(self, operation_name, elapsed_time):
        """记录操作耗时"""
        self.timings[operation_name].append(elapsed_time)
    
    def analyze_performance(self):
        """分析性能数据并生成报告"""
        try:
            report = {}
            
            for operation, times in self.timings.items():
                report[operation] = {
                    'count': len(times),
                    'mean': np.mean(times),
                    'median': np.median(times),
                    'min': np.min(times),
                    'max': np.max(times),
                    'std': np.std(times)
                }
            
            # 计算总耗时和各操作占比
            total_time = sum(sum(times) for times in self.timings.values())
            for operation in report:
                operation_total = sum(self.timings[operation])
                report[operation]['total'] = operation_total
                report[operation]['percentage'] = (operation_total / total_time * 100) if total_time > 0 else 0
            
            # 生成性能报告
            if self.log_dir:
                report_path = os.path.join(self.log_dir, 'performance_analysis.json')
                with open(report_path, 'w') as f:
                    json.dump(report, f, indent=4)
                self.logger.info(f"性能分析报告已保存到 {report_path}")
            
            return report
            
        except Exception as e:
            self.logger.error(f"分析性能失败: {str(e)}")
            return {}
    
    def generate_performance_summary(self):
        """生成性能分析摘要"""
        report = self.analyze_performance()
        
        if not report:
            return "暂无性能数据"
        
        summary = "性能分析摘要:\n\n"
        
        # 按总耗时排序
        sorted_ops = sorted(report.items(), key=lambda x: x[1]['total'], reverse=True)
        
        for op_name, stats in sorted_ops[:5]:  # 只显示前5个最耗时的操作
            summary += f"{op_name}:\n"
            summary += f"  - 总耗时: {stats['total']:.3f}s ({stats['percentage']:.1f}%)\n"
            summary += f"  - 平均耗时: {stats['mean']*1000:.2f}ms\n"
            summary += f"  - 操作次数: {stats['count']}\n"
        
        return summary