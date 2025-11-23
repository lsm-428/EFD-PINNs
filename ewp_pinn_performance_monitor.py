import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os
from datetime import datetime
from collections import defaultdict

class ModelPerformanceMonitor:
    """
    EWP-PINN模型性能监控与诊断工具
    提供全面的模型性能分析、诊断和可视化功能
    """
    
    def __init__(self, device='cpu', save_dir='./performance_reports'):
        """
        初始化性能监控器
        
        Args:
            device: 计算设备 (cpu 或 cuda)
            save_dir: 报告保存目录
        """
        self.device = device
        self.save_dir = save_dir
        self.metrics_history = defaultdict(list)
        self.diagnostic_results = {}
        self.current_stage = 0
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        
        # 设置可视化样式
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
    
    def log_training_metrics(self, epoch, train_loss, val_loss, train_mae=None, val_mae=None,
                           physics_loss=None, data_loss=None, learning_rate=None):
        """
        记录训练过程中的各种指标
        
        Args:
            epoch: 当前轮次
            train_loss: 训练损失
            val_loss: 验证损失
            train_mae: 训练MAE (可选)
            val_mae: 验证MAE (可选)
            physics_loss: 物理约束损失 (可选)
            data_loss: 数据损失 (可选)
            learning_rate: 当前学习率 (可选)
        """
        self.metrics_history['epoch'].append(epoch)
        self.metrics_history['train_loss'].append(train_loss)
        self.metrics_history['val_loss'].append(val_loss)
        
        if train_mae is not None:
            self.metrics_history['train_mae'].append(train_mae)
        if val_mae is not None:
            self.metrics_history['val_mae'].append(val_mae)
        if physics_loss is not None:
            self.metrics_history['physics_loss'].append(physics_loss)
        if data_loss is not None:
            self.metrics_history['data_loss'].append(data_loss)
        if learning_rate is not None:
            self.metrics_history['learning_rate'].append(learning_rate)
    
    def start_training_stage(self, stage_name, stage_config):
        """
        开始新的训练阶段
        
        Args:
            stage_name: 阶段名称
            stage_config: 阶段配置参数
        """
        self.current_stage += 1
        print(f"🔄 开始训练阶段 {self.current_stage}: {stage_name}")
        self.diagnostic_results[f'stage_{self.current_stage}'] = {
            'name': stage_name,
            'config': stage_config,
            'start_epoch': len(self.metrics_history['epoch']),
            'metrics': defaultdict(list)
        }
    
    def end_training_stage(self):
        """
        结束当前训练阶段并记录结果
        """
        stage_key = f'stage_{self.current_stage}'
        if stage_key in self.diagnostic_results:
            self.diagnostic_results[stage_key]['end_epoch'] = len(self.metrics_history['epoch']) - 1
            print(f"✅ 完成训练阶段 {self.current_stage}: {self.diagnostic_results[stage_key]['name']}")
    
    def analyze_convergence(self, patience=10, min_improvement=1e-4):
        """
        分析模型收敛情况
        
        Args:
            patience: 早停耐心值
            min_improvement: 最小改进阈值
            
        Returns:
            dict: 收敛分析结果
        """
        if len(self.metrics_history['val_loss']) < patience:
            return {'status': 'incomplete', 'message': '训练轮次不足，无法分析收敛情况'}
        
        val_losses = self.metrics_history['val_loss']
        best_loss = min(val_losses)
        best_epoch = val_losses.index(best_loss)
        
        # 检查是否过拟合
        recent_epochs = len(val_losses) - 1
        recent_loss = val_losses[-1]
        
        # 检查最后patience轮是否有改进
        has_recent_improvement = False
        for i in range(1, patience + 1):
            if recent_epochs - i >= 0 and val_losses[recent_epochs - i] > recent_loss + min_improvement:
                has_recent_improvement = True
                break
        
        # 计算收敛率
        if len(val_losses) > 10:
            initial_loss = np.mean(val_losses[:10])
            convergence_rate = (initial_loss - best_loss) / initial_loss
        else:
            convergence_rate = None
        
        result = {
            'status': 'converged' if not has_recent_improvement else 'converging',
            'best_epoch': best_epoch,
            'best_loss': best_loss,
            'final_loss': recent_loss,
            'convergence_rate': convergence_rate,
            'overfitting': recent_loss > best_loss * 1.1,
            'suggestion': self._generate_convergence_suggestion(has_recent_improvement, recent_loss, best_loss, convergence_rate)
        }
        
        self.diagnostic_results['convergence_analysis'] = result
        return result
    
    def _generate_convergence_suggestion(self, has_improvement, recent_loss, best_loss, convergence_rate):
        """生成收敛建议"""
        if not has_improvement:
            if recent_loss < 0.01:
                return "模型已很好收敛，性能优秀"
            elif convergence_rate is not None and convergence_rate < 0.5:
                return "收敛率较低，建议增加训练轮次或调整学习率"
            else:
                return "模型已收敛，可以考虑早停或调整超参数以进一步提升"
        else:
            return "模型仍在收敛中，可以继续训练"
    
    def analyze_model_bias_variance(self, model, X_train, y_train, X_val, y_val):
        """
        分析模型的偏差-方差权衡
        
        Args:
            model: 训练好的模型
            X_train: 训练数据特征
            y_train: 训练数据标签
            X_val: 验证数据特征
            y_val: 验证数据标签
            
        Returns:
            dict: 偏差-方差分析结果
        """
        model.eval()
        
        with torch.no_grad():
            # 确保数据在正确的设备上
            X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
            y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device)
            X_val = torch.tensor(X_val, dtype=torch.float32).to(self.device)
            y_val = torch.tensor(y_val, dtype=torch.float32).to(self.device)
            
            # 获取预测结果
            train_pred = model(X_train)
            val_pred = model(X_val)
            
            # 计算各种误差指标
            train_mse = mean_squared_error(y_train.cpu().numpy(), train_pred.cpu().numpy())
            val_mse = mean_squared_error(y_val.cpu().numpy(), val_pred.cpu().numpy())
            train_mae = mean_absolute_error(y_train.cpu().numpy(), train_pred.cpu().numpy())
            val_mae = mean_absolute_error(y_val.cpu().numpy(), val_pred.cpu().numpy())
            train_r2 = r2_score(y_train.cpu().numpy(), train_pred.cpu().numpy())
            val_r2 = r2_score(y_val.cpu().numpy(), val_pred.cpu().numpy())
        
        # 计算过拟合度（训练误差和验证误差的差异）
        overfit_ratio = val_mse / train_mse if train_mse > 0 else float('inf')
        
        # 分析结果
        if train_mse > 0.1 and overfit_ratio < 1.5:
            status = "高偏差"  # 欠拟合
            suggestion = "模型可能欠拟合，建议增加模型复杂度或减少正则化"
        elif overfit_ratio > 2.0:
            status = "高方差"  # 过拟合
            suggestion = "模型可能过拟合，建议增加正则化、使用早停或增加数据增强"
        else:
            status = "良好平衡"  # 平衡状态
            suggestion = "模型偏差-方差平衡良好"
        
        result = {
            'status': status,
            'train_metrics': {'mse': train_mse, 'mae': train_mae, 'r2': train_r2},
            'val_metrics': {'mse': val_mse, 'mae': val_mae, 'r2': val_r2},
            'overfit_ratio': overfit_ratio,
            'suggestion': suggestion
        }
        
        self.diagnostic_results['bias_variance_analysis'] = result
        return result
    
    def analyze_physics_integration(self):
        """
        分析物理约束集成效果
        
        Returns:
            dict: 物理约束分析结果
        """
        if 'physics_loss' not in self.metrics_history or 'data_loss' not in self.metrics_history:
            return {'status': 'incomplete', 'message': '缺少物理损失或数据损失记录'}
        
        physics_losses = np.array(self.metrics_history['physics_loss'])
        data_losses = np.array(self.metrics_history['data_loss'])
        
        # 计算物理约束和数据约束的相对重要性变化
        if len(physics_losses) > 0 and len(data_losses) > 0:
            initial_physics_weight = physics_losses[0] / (data_losses[0] + 1e-10)
            final_physics_weight = physics_losses[-1] / (data_losses[-1] + 1e-10)
            physics_weight_change = (final_physics_weight - initial_physics_weight) / (initial_physics_weight + 1e-10)
            
            # 计算物理损失的下降率
            physics_reduction = (physics_losses[0] - physics_losses[-1]) / (physics_losses[0] + 1e-10)
            data_reduction = (data_losses[0] - data_losses[-1]) / (data_losses[0] + 1e-10)
            
            # 分析物理约束的有效性
            if physics_reduction > 0.5 and data_reduction > 0.3:
                effectiveness = "优秀"
                suggestion = "物理约束有效提升了模型性能"
            elif physics_reduction > 0.3:
                effectiveness = "良好"
                suggestion = "物理约束对模型训练有积极影响"
            else:
                effectiveness = "待改进"
                suggestion = "物理约束效果不明显，建议调整权重或改进物理模型"
            
            result = {
                'physics_reduction': physics_reduction,
                'data_reduction': data_reduction,
                'physics_weight_change': physics_weight_change,
                'effectiveness': effectiveness,
                'suggestion': suggestion
            }
            
            self.diagnostic_results['physics_integration_analysis'] = result
            return result
        
        return {'status': 'incomplete', 'message': '物理损失或数据损失记录不足'}
    
    def analyze_gradient_flow(self, model):
        """
        分析梯度流动情况，检测梯度消失或爆炸问题
        
        Args:
            model: 训练中的模型
            
        Returns:
            dict: 梯度流动分析结果
        """
        gradient_stats = {}
        gradient_norms = []
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                gradient_norms.append(grad_norm)
                
                # 检查梯度异常
                if grad_norm < 1e-6:
                    status = "梯度消失"
                elif grad_norm > 1e3:
                    status = "梯度爆炸"
                else:
                    status = "正常"
                
                gradient_stats[name] = {
                    'norm': grad_norm,
                    'status': status,
                    'parameter_norm': param.norm().item()
                }
        
        # 计算整体梯度统计
        if gradient_norms:
            avg_grad_norm = np.mean(gradient_norms)
            std_grad_norm = np.std(gradient_norms)
            
            # 分析梯度健康状况
            if avg_grad_norm < 1e-6:
                overall_status = "严重梯度消失"
                suggestion = "梯度消失严重，建议使用残差连接、BatchNorm或调整激活函数"
            elif avg_grad_norm > 1e3:
                overall_status = "严重梯度爆炸"
                suggestion = "梯度爆炸严重，建议使用梯度裁剪、权重初始化或学习率调整"
            elif std_grad_norm > avg_grad_norm * 5:
                overall_status = "梯度不平衡"
                suggestion = "不同层梯度差异大，建议使用梯度均衡技术"
            else:
                overall_status = "健康"
                suggestion = "梯度流动良好"
            
            result = {
                'overall_status': overall_status,
                'avg_gradient_norm': avg_grad_norm,
                'std_gradient_norm': std_grad_norm,
                'gradient_stats': gradient_stats,
                'suggestion': suggestion
            }
        else:
            result = {'status': 'incomplete', 'message': '没有可用的梯度信息，确保在反向传播后调用此函数'}
        
        self.diagnostic_results['gradient_analysis'] = result
        return result
    
    def plot_training_curves(self, save_fig=True):
        """
        绘制训练曲线图
        
        Args:
            save_fig: 是否保存图表
            
        Returns:
            str: 图表保存路径（如果保存）
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 确保epoch数组存在
        if 'epoch' not in self.metrics_history or len(self.metrics_history['epoch']) == 0:
            print("⚠️  警告：没有足够的训练数据来绘制曲线图")
            plt.close()
            return None
        
        epoch_array = np.array(self.metrics_history['epoch'])
        
        # 损失曲线
        if 'train_loss' in self.metrics_history and len(self.metrics_history['train_loss']) == len(epoch_array):
            axes[0, 0].plot(epoch_array, self.metrics_history['train_loss'], label='训练损失')
        if 'val_loss' in self.metrics_history and len(self.metrics_history['val_loss']) == len(epoch_array):
            axes[0, 0].plot(epoch_array, self.metrics_history['val_loss'], label='验证损失')
        axes[0, 0].set_title('训练与验证损失')
        axes[0, 0].set_xlabel('轮次')
        axes[0, 0].set_ylabel('损失值')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # MAE曲线
        if 'train_mae' in self.metrics_history and 'val_mae' in self.metrics_history:
            if len(self.metrics_history['train_mae']) == len(epoch_array):
                axes[0, 1].plot(epoch_array, self.metrics_history['train_mae'], label='训练MAE')
            if len(self.metrics_history['val_mae']) == len(epoch_array):
                axes[0, 1].plot(epoch_array, self.metrics_history['val_mae'], label='验证MAE')
            axes[0, 1].set_title('训练与验证MAE')
            axes[0, 1].set_xlabel('轮次')
            axes[0, 1].set_ylabel('MAE值')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 物理与数据损失
        if 'physics_loss' in self.metrics_history and 'data_loss' in self.metrics_history:
            if len(self.metrics_history['physics_loss']) == len(epoch_array):
                axes[1, 0].plot(epoch_array, self.metrics_history['physics_loss'], label='物理损失')
            if len(self.metrics_history['data_loss']) == len(epoch_array):
                axes[1, 0].plot(epoch_array, self.metrics_history['data_loss'], label='数据损失')
            axes[1, 0].set_title('物理约束损失与数据损失')
            axes[1, 0].set_xlabel('轮次')
            axes[1, 0].set_ylabel('损失值')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 学习率曲线
        if 'learning_rate' in self.metrics_history and len(self.metrics_history['learning_rate']) == len(epoch_array):
            axes[1, 1].plot(epoch_array, self.metrics_history['learning_rate'], label='学习率')
            axes[1, 1].set_title('学习率变化')
            axes[1, 1].set_xlabel('轮次')
            axes[1, 1].set_ylabel('学习率')
            axes[1, 1].set_yscale('log')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_fig:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.save_dir, f'training_curves_{timestamp}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 训练曲线图已保存: {save_path}")
            plt.close()
            return save_path
        
        return None
    
    def plot_error_distribution(self, model, X_test, y_test, save_fig=True):
        """
        绘制预测误差分布
        
        Args:
            model: 训练好的模型
            X_test: 测试数据特征
            y_test: 测试数据标签
            save_fig: 是否保存图表
            
        Returns:
            str: 图表保存路径（如果保存）
        """
        model.eval()
        
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)
            y_pred = model(X_test_tensor).cpu().numpy()
            
        # 计算误差
        y_true = y_test
        errors = y_pred.flatten() - y_true.flatten()
        
        # 绘制误差分布图
        plt.figure(figsize=(10, 6))
        sns.histplot(errors, kde=True, bins=50)
        plt.axvline(x=0, color='r', linestyle='--', label='零误差')
        plt.title('预测误差分布')
        plt.xlabel('预测误差')
        plt.ylabel('频率')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if save_fig:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.save_dir, f'error_distribution_{timestamp}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 误差分布图已保存: {save_path}")
            plt.close()
            return save_path
        
        return None
    
    def plot_feature_importance(self, model, feature_names=None, top_n=20, save_fig=True):
        """
        绘制特征重要性图（基于输入层权重）
        
        Args:
            model: 训练好的模型
            feature_names: 特征名称列表
            top_n: 显示前N个重要特征
            save_fig: 是否保存图表
            
        Returns:
            str: 图表保存路径（如果保存）
        """
        # 尝试获取输入层权重
        for name, param in model.named_parameters():
            if 'input' in name.lower() and 'weight' in name.lower():
                weights = param.data.abs().cpu().numpy()
                break
        else:
            print("⚠️  无法找到输入层权重，跳过特征重要性分析")
            return None
        
        # 计算每个特征的平均权重
        if len(weights.shape) > 1:
            feature_importance = np.mean(weights, axis=0)
        else:
            feature_importance = weights
        
        # 获取前N个重要特征
        top_indices = np.argsort(feature_importance)[::-1][:top_n]
        top_importance = feature_importance[top_indices]
        
        # 特征名称
        if feature_names is None:
            feature_names = [f'特征_{i}' for i in range(len(feature_importance))]
        top_features = [feature_names[i] for i in top_indices]
        
        # 绘制特征重要性
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top_features)), top_importance, tick_label=top_features)
        plt.gca().invert_yaxis()  # 最重要的特征在顶部
        plt.title(f'前{top_n}个重要特征')
        plt.xlabel('特征重要性（权重绝对值）')
        plt.grid(True, axis='x', alpha=0.3)
        
        if save_fig:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.save_dir, f'feature_importance_{timestamp}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 特征重要性图已保存: {save_path}")
            plt.close()
            return save_path
        
        return None
    
    def generate_performance_report(self):
        """
        生成完整的性能报告
        
        Returns:
            dict: 完整的性能报告
        """
        # 执行所有分析
        convergence_result = self.analyze_convergence()
        
        # 生成报告
        report = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'training_summary': {
                'total_epochs': len(self.metrics_history['epoch']),
                'best_train_loss': min(self.metrics_history['train_loss']),
                'best_val_loss': min(self.metrics_history['val_loss']),
                'final_train_loss': self.metrics_history['train_loss'][-1],
                'final_val_loss': self.metrics_history['val_loss'][-1],
                'training_stages': self.current_stage
            },
            'convergence_analysis': convergence_result,
            'diagnostic_results': self.diagnostic_results,
            'recommendations': self._generate_recommendations()
        }
        
        # 保存报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.save_dir, f'performance_report_{timestamp}.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"📋 性能报告已保存: {report_path}")
        
        # 打印关键发现
        self._print_key_findings(report)
        
        return report
    
    def _generate_recommendations(self):
        """
        根据分析结果生成建议
        
        Returns:
            list: 建议列表
        """
        recommendations = []
        
        # 基于收敛分析的建议
        if 'convergence_analysis' in self.diagnostic_results:
            conv = self.diagnostic_results['convergence_analysis']
            if 'suggestion' in conv:
                recommendations.append(conv['suggestion'])
        
        # 基于偏差-方差分析的建议
        if 'bias_variance_analysis' in self.diagnostic_results:
            bv = self.diagnostic_results['bias_variance_analysis']
            if 'suggestion' in bv:
                recommendations.append(bv['suggestion'])
        
        # 基于物理集成分析的建议
        if 'physics_integration_analysis' in self.diagnostic_results:
            pi = self.diagnostic_results['physics_integration_analysis']
            if 'suggestion' in pi:
                recommendations.append(pi['suggestion'])
        
        # 基于梯度分析的建议
        if 'gradient_analysis' in self.diagnostic_results:
            ga = self.diagnostic_results['gradient_analysis']
            if 'suggestion' in ga:
                recommendations.append(ga['suggestion'])
        
        return recommendations
    
    def _print_key_findings(self, report):
        """
        打印关键发现
        
        Args:
            report: 性能报告
        """
        print("\n🔍 模型性能关键发现:")
        print(f"📊 总训练轮次: {report['training_summary']['total_epochs']}")
        print(f"🏆 最佳验证损失: {report['training_summary']['best_val_loss']:.6f}")
        print(f"📈 最终验证损失: {report['training_summary']['final_val_loss']:.6f}")
        
        # 收敛状态
        if 'convergence_analysis' in report:
            conv = report['convergence_analysis']
            if 'status' in conv:
                status_map = {
                    'converged': '✅ 已收敛',
                    'converging': '⏳ 收敛中',
                    'incomplete': '❓ 无法判断'
                }
                print(f"📉 收敛状态: {status_map.get(conv['status'], conv['status'])}")
        
        # 偏差-方差状态
        if 'bias_variance_analysis' in report:
            bv = report['bias_variance_analysis']
            if 'status' in bv:
                print(f"⚖️  偏差-方差状态: {bv['status']}")
        
        # 物理约束效果
        if 'physics_integration_analysis' in report:
            pi = report['physics_integration_analysis']
            if 'effectiveness' in pi:
                print(f"🔧 物理约束效果: {pi['effectiveness']}")
        
        # 建议
        if 'recommendations' in report and report['recommendations']:
            print("\n💡 改进建议:")
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"   {i}. {rec}")
    
    def export_diagnostics(self):
        """
        导出所有诊断结果和可视化图表
        
        Returns:
            dict: 导出文件路径
        """
        export_paths = {}
        
        # 保存训练曲线图
        export_paths['training_curves'] = self.plot_training_curves(save_fig=True)
        
        # 生成并保存性能报告
        report = self.generate_performance_report()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_paths['performance_report'] = os.path.join(self.save_dir, f'performance_report_{timestamp}.json')
        
        print("\n📤 诊断结果导出完成!")
        for key, path in export_paths.items():
            if path:
                print(f"   - {key}: {path}")
        
        return export_paths

# 辅助函数
import json

def analyze_checkpoint(checkpoint_path, device='cpu'):
    """
    分析已保存的模型检查点
    
    Args:
        checkpoint_path: 检查点文件路径
        device: 计算设备
        
    Returns:
        dict: 检查点分析结果
    """
    try:
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 提取关键信息
        analysis = {
            'checkpoint_path': checkpoint_path,
            'model_architecture': 'OptimizedEWPINN',
            'has_state_dict': 'model_state_dict' in checkpoint,
            'has_normalizer': 'normalizer' in checkpoint,
            'has_config': 'config' in checkpoint,
            'has_history': 'train_history' in checkpoint and 'val_history' in checkpoint,
            'hyperparameter_optimization': 'hyperparameter_optimization_history' in checkpoint
        }
        
        # 提取训练历史（如果有）
        if analysis['has_history']:
            analysis['training_epochs'] = len(checkpoint['train_history'])
            analysis['best_train_loss'] = min(checkpoint['train_history'])
            analysis['best_val_loss'] = min(checkpoint['val_history'])
        
        # 提取超参数优化信息（如果有）
        if analysis['hyperparameter_optimization']:
            analysis['optimization_rounds'] = len(checkpoint['hyperparameter_optimization_history'])
            analysis['best_hyperparameters'] = checkpoint['best_hyperparameters']
        
        print(f"✅ 检查点分析完成: {checkpoint_path}")
        return analysis
        
    except Exception as e:
        print(f"❌ 检查点分析失败: {str(e)}")
        return {'error': str(e), 'checkpoint_path': checkpoint_path}

def compare_models(model_paths, device='cpu'):
    """
    比较多个模型的性能
    
    Args:
        model_paths: 模型文件路径列表
        device: 计算设备
        
    Returns:
        dict: 模型比较结果
    """
    comparisons = []
    
    for path in model_paths:
        try:
            analysis = analyze_checkpoint(path, device)
            comparisons.append({
                'model_path': path,
                'best_val_loss': analysis.get('best_val_loss', float('inf')),
                'training_epochs': analysis.get('training_epochs', 0),
                'has_hyperopt': analysis.get('hyperparameter_optimization', False),
                'optimization_rounds': analysis.get('optimization_rounds', 0)
            })
        except Exception as e:
            print(f"❌ 无法分析模型: {path}, 错误: {str(e)}")
    
    # 按最佳验证损失排序
    comparisons.sort(key=lambda x: x['best_val_loss'])
    
    print("\n🏆 模型性能比较:")
    for i, model in enumerate(comparisons, 1):
        print(f"   {i}. 模型: {os.path.basename(model['model_path'])}")
        print(f"      最佳验证损失: {model['best_val_loss']:.6f}")
        print(f"      训练轮次: {model['training_epochs']}")
        print(f"      超参数优化: {'✅ 是' if model['has_hyperopt'] else '❌ 否'} ({model['optimization_rounds']}轮)")
    
    return comparisons