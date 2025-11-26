# 故障排除与调试指南

## 常见问题分类

### 1. 安装与依赖问题
```python
class InstallationTroubleshooter:
    """安装问题排查器"""
    
    def diagnose_installation_issues(self):
        """诊断安装问题"""
        issues = []
        
        # 检查Python版本
        python_issue = self.check_python_version()
        if python_issue:
            issues.append(python_issue)
        
        # 检查PyTorch安装
        pytorch_issue = self.check_pytorch_installation()
        if pytorch_issue:
            issues.append(pytorch_issue)
        
        # 检查CUDA支持
        cuda_issue = self.check_cuda_support()
        if cuda_issue:
            issues.append(cuda_issue)
        
        # 检查依赖包
        dependency_issues = self.check_dependencies()
        issues.extend(dependency_issues)
        
        return issues
    
    def check_python_version(self):
        """检查Python版本兼容性"""
        import sys
        
        if sys.version_info < (3, 8):
            return {
                'type': 'python_version',
                'severity': 'critical',
                'description': 'Python版本过低，需要3.8或更高版本',
                'solution': '升级Python到3.8+版本',
                'command': 'conda install python=3.8 或 pyenv install 3.8.0'
            }
        return None
    
    def check_pytorch_installation(self):
        """检查PyTorch安装"""
        try:
            import torch
            version = torch.__version__
            
            if tuple(map(int, version.split('.')[:2])) < (1, 9):
                return {
                    'type': 'pytorch_version',
                    'severity': 'critical',
                    'description': f'PyTorch版本{version}过低，需要1.9或更高版本',
                    'solution': '升级PyTorch到最新版本',
                    'command': 'pip install torch torchvision --upgrade'
                }
        except ImportError:
            return {
                'type': 'pytorch_missing',
                'severity': 'critical',
                'description': 'PyTorch未安装',
                'solution': '安装PyTorch',
                'command': 'pip install torch torchvision'
            }
        
        return None
```

### 2. GPU相关问题
```python
class GPUTroubleshooter:
    """GPU问题排查器"""
    
    def diagnose_gpu_issues(self):
        """诊断GPU相关问题"""
        issues = []
        
        # 检查GPU可用性
        gpu_available = self.check_gpu_availability()
        if not gpu_available:
            issues.append(self.get_gpu_unavailable_issue())
        
        # 检查CUDA版本
        cuda_version_issue = self.check_cuda_version()
        if cuda_version_issue:
            issues.append(cuda_version_issue)
        
        # 检查显存问题
        memory_issues = self.check_gpu_memory()
        issues.extend(memory_issues)
        
        # 检查驱动问题
        driver_issues = self.check_gpu_driver()
        issues.extend(driver_issues)
        
        return issues
    
    def check_gpu_availability(self):
        """检查GPU是否可用"""
        import torch
        
        try:
            return torch.cuda.is_available()
        except Exception as e:
            print(f"GPU检查失败: {e}")
            return False
    
    def get_gpu_unavailable_issue(self):
        """获取GPU不可用的问题描述"""
        return {
            'type': 'gpu_unavailable',
            'severity': 'warning',
            'description': 'GPU不可用，将使用CPU进行计算',
            'solution': '检查CUDA安装和GPU驱动，或使用CPU模式',
            'command': '检查nvidia-smi命令是否正常工作'
        }
    
    def check_gpu_memory(self):
        """检查GPU显存问题"""
        issues = []
        import torch
        
        if torch.cuda.is_available():
            # 检查显存总量
            total_memory = torch.cuda.get_device_properties(0).total_memory
            
            if total_memory < 4 * 1024**3:  # 小于4GB
                issues.append({
                    'type': 'low_gpu_memory',
                    'severity': 'warning',
                    'description': f'GPU显存较低: {total_memory/1024**3:.1f}GB',
                    'solution': '减小批次大小或使用CPU模式',
                    'command': '设置较小的batch_size参数'
                })
            
            # 检查当前显存使用
            allocated = torch.cuda.memory_allocated()
            cached = torch.cuda.memory_reserved()
            
            if allocated > total_memory * 0.9:
                issues.append({
                    'type': 'high_gpu_usage',
                    'severity': 'warning',
                    'description': 'GPU显存使用率过高',
                    'solution': '清理显存缓存或重启Python内核',
                    'command': 'torch.cuda.empty_cache()'
                })
        
        return issues
```

## 训练问题排查

### 1. 收敛问题
```python
class ConvergenceTroubleshooter:
    """收敛问题排查器"""
    
    def diagnose_convergence_issues(self, training_history):
        """诊断训练收敛问题"""
        issues = []
        
        # 检查损失不下降
        no_decrease_issue = self.check_loss_not_decreasing(training_history)
        if no_decrease_issue:
            issues.append(no_decrease_issue)
        
        # 检查损失爆炸
        explosion_issue = self.check_loss_explosion(training_history)
        if explosion_issue:
            issues.append(explosion_issue)
        
        # 检查振荡
        oscillation_issue = self.check_loss_oscillation(training_history)
        if oscillation_issue:
            issues.append(oscillation_issue)
        
        # 检查过拟合
        overfitting_issue = self.check_overfitting(training_history)
        if overfitting_issue:
            issues.append(overfitting_issue)
        
        return issues
    
    def check_loss_not_decreasing(self, history):
        """检查损失不下降问题"""
        losses = history.get('train_loss', [])
        
        if len(losses) < 10:
            return None
        
        # 检查最后10个epoch的损失变化
        recent_losses = losses[-10:]
        avg_loss = sum(recent_losses) / len(recent_losses)
        
        # 如果损失基本不变
        if max(recent_losses) - min(recent_losses) < avg_loss * 0.01:
            return {
                'type': 'loss_not_decreasing',
                'severity': 'warning',
                'description': '训练损失在最近10个epoch中基本没有下降',
                'solution': '尝试调整学习率、优化器或模型架构',
                'suggestions': [
                    '减小学习率',
                    '尝试不同的优化器',
                    '检查数据预处理',
                    '增加模型复杂度'
                ]
            }
        
        return None
    
    def check_loss_explosion(self, history):
        """检查损失爆炸问题"""
        losses = history.get('train_loss', [])
        
        if len(losses) < 2:
            return None
        
        # 检查损失是否突然增大
        for i in range(1, len(losses)):
            if losses[i] > losses[i-1] * 10:  # 损失增加10倍以上
                return {
                    'type': 'loss_explosion',
                    'severity': 'critical',
                    'description': f'在第{i}个epoch损失突然增大',
                    'solution': '立即停止训练，检查梯度爆炸问题',
                    'immediate_actions': [
                        '停止当前训练',
                        '检查梯度裁剪设置',
                        '减小学习率',
                        '检查数据异常值'
                    ]
                }
        
        return None
```

### 2. 梯度问题
```python
class GradientTroubleshooter:
    """梯度问题排查器"""
    
    def monitor_gradients(self, model, loss):
        """监控梯度状态"""
        import torch
        
        # 计算梯度
        loss.backward()
        
        gradient_info = {}
        
        # 检查梯度范数
        total_norm = 0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        
        total_norm = total_norm ** 0.5
        gradient_info['total_gradient_norm'] = total_norm
        
        # 检查梯度消失/爆炸
        if total_norm < 1e-10:
            gradient_info['issue'] = 'gradient_vanishing'
        elif total_norm > 1e5:
            gradient_info['issue'] = 'gradient_explosion'
        
        # 检查各层梯度
        layer_gradients = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                layer_gradients[name] = {
                    'mean': param.grad.data.mean().item(),
                    'std': param.grad.data.std().item(),
                    'norm': param.grad.data.norm(2).item()
                }
        
        gradient_info['layer_gradients'] = layer_gradients
        
        return gradient_info
    
    def diagnose_gradient_issues(self, gradient_info):
        """诊断梯度问题"""
        issues = []
        
        total_norm = gradient_info.get('total_gradient_norm', 0)
        
        # 梯度消失
        if total_norm < 1e-10:
            issues.append({
                'type': 'gradient_vanishing',
                'severity': 'critical',
                'description': '梯度消失，模型无法学习',
                'solution': '使用梯度裁剪、调整激活函数或修改网络架构',
                'actions': [
                    '使用ReLU激活函数替代sigmoid/tanh',
                    '添加批归一化层',
                    '使用残差连接',
                    '调整初始化方法'
                ]
            })
        
        # 梯度爆炸
        elif total_norm > 1e5:
            issues.append({
                'type': 'gradient_explosion',
                'severity': 'critical',
                'description': '梯度爆炸，训练不稳定',
                'solution': '应用梯度裁剪、减小学习率或使用权重衰减',
                'actions': [
                    '启用梯度裁剪: torch.nn.utils.clip_grad_norm_',
                    '减小学习率',
                    '增加权重衰减',
                    '检查数据预处理'
                ]
            })
        
        # 检查各层梯度分布
        layer_issues = self.check_layer_gradients(gradient_info.get('layer_gradients', {}))
        issues.extend(layer_issues)
        
        return issues
```

## 调试工具与技术

### 1. 日志系统
```python
class DebugLogger:
    """调试日志系统"""
    
    def __init__(self, log_level='INFO'):
        self.log_level = log_level
        self.levels = {'DEBUG': 10, 'INFO': 20, 'WARNING': 30, 'ERROR': 40}
        
    def setup_logging(self, log_file='debug.log'):
        """设置日志配置"""
        import logging
        
        logging.basicConfig(
            level=self.levels[self.log_level],
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger('EFD3D')
    
    def log_training_progress(self, epoch, loss, metrics):
        """记录训练进度"""
        self.logger.info(f'Epoch {epoch}: Loss = {loss:.6f}')
        
        for metric_name, metric_value in metrics.items():
            self.logger.debug(f'{metric_name}: {metric_value}')
    
    def log_gradient_info(self, gradient_info):
        """记录梯度信息"""
        total_norm = gradient_info.get('total_gradient_norm', 0)
        
        if total_norm > 1e5:
            self.logger.warning(f'梯度爆炸: {total_norm:.2e}')
        elif total_norm < 1e-10:
            self.logger.warning(f'梯度消失: {total_norm:.2e}')
        
        # 记录各层梯度统计
        for layer_name, grad_stats in gradient_info.get('layer_gradients', {}).items():
            self.logger.debug(f'{layer_name}: mean={grad_stats["mean"]:.2e}, std={grad_stats["std"]:.2e}')
```

### 2. 可视化调试
```python
class VisualizationDebugger:
    """可视化调试工具"""
    
    def plot_training_history(self, history, save_path=None):
        """绘制训练历史图表"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 损失曲线
        if 'train_loss' in history:
            axes[0, 0].plot(history['train_loss'], label='训练损失')
        if 'val_loss' in history:
            axes[0, 0].plot(history['val_loss'], label='验证损失')
        axes[0, 0].set_title('损失曲线')
        axes[0, 0].legend()
        axes[0, 0].set_yscale('log')
        
        # 学习率
        if 'learning_rate' in history:
            axes[0, 1].plot(history['learning_rate'])
            axes[0, 1].set_title('学习率变化')
        
        # 梯度范数
        if 'gradient_norm' in history:
            axes[1, 0].plot(history['gradient_norm'])
            axes[1, 0].set_title('梯度范数')
            axes[1, 0].set_yscale('log')
        
        # 其他指标
        metric_keys = [k for k in history.keys() if k not in ['train_loss', 'val_loss', 'learning_rate', 'gradient_norm']]
        for i, metric in enumerate(metric_keys[:4]):  # 最多显示4个指标
            row, col = 1 + i//2, i%2
            if row < 2 and col < 2:
                axes[row, col].plot(history[metric])
                axes[row, col].set_title(metric)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_gradient_distribution(self, model):
        """绘制梯度分布图"""
        import matplotlib.pyplot as plt
        
        gradients = []
        layer_names = []
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradients.append(param.grad.data.cpu().numpy().flatten())
                layer_names.append(name)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 梯度分布直方图
        for i, (grad, name) in enumerate(zip(gradients[:4], layer_names[:4])):
            row, col = i//2, i%2
            axes[row, col].hist(grad, bins=50, alpha=0.7)
            axes[row, col].set_title(f'{name}梯度分布')
            axes[row, col].set_yscale('log')
        
        plt.tight_layout()
        plt.show()
```

## 高级调试技术

### 1. 断点调试
```python
class AdvancedDebugger:
    """高级调试工具"""
    
    def setup_debug_points(self, model, config):
        """设置调试断点"""
        
        # 注册前向传播钩子
        self.forward_hooks = []
        
        for name, module in model.named_modules():
            if hasattr(module, 'weight'):
                hook = module.register_forward_hook(
                    self._create_forward_hook(name)
                )
                self.forward_hooks.append(hook)
        
        # 注册反向传播钩子
        self.backward_hooks = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                hook = param.register_hook(
                    self._create_backward_hook(name)
                )
                self.backward_hooks.append(hook)
    
    def _create_forward_hook(self, layer_name):
        """创建前向传播钩子"""
        def hook(module, input, output):
            # 记录激活统计
            stats = {
                'input_mean': input[0].mean().item(),
                'input_std': input[0].std().item(),
                'output_mean': output.mean().item(),
                'output_std': output.std().item()
            }
            
            # 检查激活值异常
            if abs(stats['output_mean']) > 100 or stats['output_std'] > 100:
                print(f"警告: {layer_name}激活值异常")
                print(f"  输入: mean={stats['input_mean']:.4f}, std={stats['input_std']:.4f}")
                print(f"  输出: mean={stats['output_mean']:.4f}, std={stats['output_std']:.4f}")
        
        return hook
    
    def _create_backward_hook(self, param_name):
        """创建反向传播钩子"""
        def hook(grad):
            # 检查梯度异常
            if grad is not None:
                grad_norm = grad.norm().item()
                
                if grad_norm > 1e5:
                    print(f"警告: {param_name}梯度爆炸: {grad_norm:.2e}")
                elif grad_norm < 1e-10:
                    print(f"警告: {param_name}梯度消失: {grad_norm:.2e}")
            
            return grad
        
        return hook
```

### 2. 性能分析
```python
class PerformanceProfiler:
    """性能分析器"""
    
    def profile_training(self, model, data_loader, num_batches=10):
        """分析训练性能"""
        import torch
        import time
        
        # 预热
        for i, batch in enumerate(data_loader):
            if i >= 2:
                break
            _ = model(batch)
        
        # 性能分析
        start_time = time.time()
        
        for i, batch in enumerate(data_loader):
            if i >= num_batches:
                break
            
            # 前向传播
            with torch.autograd.profiler.profile(use_cuda=torch.cuda.is_available()) as prof:
                output = model(batch)
                loss = output.mean()
                loss.backward()
            
            # 记录性能数据
            self.record_performance_data(prof, i)
        
        total_time = time.time() - start_time
        
        return self.generate_performance_report(total_time, num_batches)
    
    def record_performance_data(self, prof, batch_idx):
        """记录性能数据"""
        # 这里可以记录CPU时间、GPU时间、内存使用等
        pass
    
    def generate_performance_report(self, total_time, num_batches):
        """生成性能报告"""
        avg_batch_time = total_time / num_batches
        
        report = {
            'total_time': total_time,
            'num_batches': num_batches,
            'avg_batch_time': avg_batch_time,
            'throughput': num_batches / total_time,
            'bottlenecks': self.identify_bottlenecks()
        }
        
        return report
```

## 故障排除流程

### 1. 系统化排查流程
```python
class SystematicTroubleshooter:
    """系统化故障排除器"""
    
    def run_systematic_check(self):
        """运行系统化检查"""
        checklist = [
            self.check_environment,
            self.check_dependencies,
            self.check_configuration,
            self.check_data_integrity,
            self.check_model_architecture,
            self.check_training_setup
        ]
        
        results = {}
        
        for check_function in checklist:
            check_name = check_function.__name__
            try:
                results[check_name] = check_function()
            except Exception as e:
                results[check_name] = {
                    'status': 'error',
                    'message': f'检查失败: {str(e)}'
                }
        
        return self.generate_diagnostic_report(results)
    
    def check_environment(self):
        """检查运行环境"""
        import sys
        import torch
        
        env_info = {
            'python_version': sys.version,
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else 'N/A',
            'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
        
        return {'status': 'ok', 'info': env_info}
    
    def check_data_integrity(self):
        """检查数据完整性"""
        # 检查数据文件存在性
        # 检查数据格式
        # 检查数据范围
        pass
    
    def generate_diagnostic_report(self, results):
        """生成诊断报告"""
        report = {
            'summary': {
                'total_checks': len(results),
                'passed_checks': sum(1 for r in results.values() if r.get('status') == 'ok'),
                'failed_checks': sum(1 for r in results.values() if r.get('status') != 'ok')
            },
            'detailed_results': results,
            'recommendations': self.generate_recommendations(results)
        }
        
        return report
```

这个详细的故障排除与调试指南为开发者提供了从基础问题排查到高级调试技术的完整解决方案。