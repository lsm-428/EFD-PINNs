# 部署与性能优化指南

## 部署架构概览

EFD3D支持多种部署模式，从本地开发到生产环境：

```python
class DeploymentManager:
    """部署管理器：支持多种部署模式"""
    
    def __init__(self):
        self.deployment_modes = {
            'local': LocalDeployment(),
            'docker': DockerDeployment(),
            'kubernetes': KubernetesDeployment(),
            'cloud': CloudDeployment()
        }
    
    def deploy(self, mode, config, model_path):
        """部署模型到指定环境"""
        deployment = self.deployment_modes.get(mode)
        if deployment:
            return deployment.deploy(config, model_path)
        else:
            raise ValueError(f"不支持的部署模式: {mode}")
    
    def validate_deployment(self, mode, config):
        """验证部署配置"""
        deployment = self.deployment_modes.get(mode)
        return deployment.validate_config(config)
```

## 本地部署

### 1. 环境准备
```python
class LocalDeployment:
    """本地部署管理器"""
    
    def setup_environment(self):
        """设置本地开发环境"""
        
        # 检查系统要求
        requirements = self.check_system_requirements()
        if not requirements['met']:
            self.install_missing_requirements(requirements['missing'])
        
        # 创建虚拟环境
        self.create_virtual_environment()
        
        # 安装依赖
        self.install_dependencies()
        
        # 配置环境变量
        self.setup_environment_variables()
    
    def check_system_requirements(self):
        """检查系统要求"""
        requirements = {
            'python': '3.8+',
            'pytorch': '1.9+',
            'cuda': '11.1+',
            'memory': '8GB+',
            'storage': '10GB+'
        }
        
        return self.verify_requirements(requirements)
    
    def create_virtual_environment(self):
        """创建Python虚拟环境"""
        import subprocess
        
        # 创建venv
        subprocess.run(['python', '-m', 'venv', 'efd3d_env'])
        
        # 激活脚本
        activation_script = '''
#!/bin/bash
source efd3d_env/bin/activate
export PYTHONPATH=$PWD:$PYTHONPATH
export EFD3D_HOME=$PWD
'''
        
        with open('activate_efd3d.sh', 'w') as f:
            f.write(activation_script)
```

### 2. 依赖管理
```python
class DependencyManager:
    """依赖管理器：处理复杂的依赖关系"""
    
    def generate_requirements(self, deployment_type='full'):
        """生成requirements.txt文件"""
        
        base_requirements = {
            'torch>=1.9.0': 'PyTorch深度学习框架',
            'numpy>=1.21.0': '数值计算库',
            'matplotlib>=3.5.0': '绘图库',
            'scikit-learn>=1.0.0': '机器学习工具',
            'pandas>=1.3.0': '数据分析库'
        }
        
        optional_requirements = {
            'gpu': {
                'torchvision>=0.10.0': '计算机视觉扩展',
                'cudatoolkit=11.1': 'CUDA工具包'
            },
            'visualization': {
                'plotly>=5.0.0': '交互式可视化',
                'seaborn>=0.11.0': '统计可视化'
            },
            'development': {
                'pytest>=6.0.0': '测试框架',
                'black>=21.0.0': '代码格式化',
                'flake8>=4.0.0': '代码检查'
            }
        }
        
        requirements = base_requirements.copy()
        
        if deployment_type == 'gpu':
            requirements.update(optional_requirements['gpu'])
        if deployment_type == 'full':
            for category in optional_requirements.values():
                requirements.update(category)
        
        return requirements
    
    def create_requirements_file(self, deployment_type='full'):
        """创建requirements.txt文件"""
        requirements = self.generate_requirements(deployment_type)
        
        content = "# EFD3D项目依赖\n"
        content += "# 生成时间: {}\n\n".format(datetime.now().isoformat())
        
        for package, description in requirements.items():
            content += "{}  # {}\n".format(package, description)
        
        with open('requirements.txt', 'w') as f:
            f.write(content)
```

## Docker部署

### 1. Dockerfile配置
```dockerfile
# 基础镜像
FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONPATH=/app
ENV EFD3D_HOME=/app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY . .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 创建数据卷
VOLUME ["/app/data", "/app/models", "/app/outputs"]

# 暴露端口（如果使用Web服务）
EXPOSE 8000

# 设置启动命令
CMD ["python", "efd_pinns_train.py", "--config", "configs/production_config.json"]
```

### 2. Docker Compose配置
```yaml
version: '3.8'

services:
  efd3d-training:
    build: .
    image: efd3d:latest
    container_name: efd3d_trainer
    
    # 资源限制
    deploy:
      resources:
        limits:
          memory: 16G
          cpus: '4.0'
    
    # 数据卷
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./outputs:/app/outputs
      - ./configs:/app/configs
    
    # 环境变量
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - PYTHONUNBUFFERED=1
    
    # 重启策略
    restart: unless-stopped
    
    # 健康检查
    healthcheck:
      test: ["CMD", "python", "-c", "import torch; print('GPU available:', torch.cuda.is_available())"]
      interval: 30s
      timeout: 10s
      retries: 3

  # 监控服务
  efd3d-monitor:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - ./monitoring/grafana:/var/lib/grafana
    depends_on:
      - efd3d-training
```

## 性能优化

### 1. 内存优化
```python
class MemoryOptimizer:
    """内存优化器：减少模型内存占用"""
    
    def optimize_memory_usage(self, model, config):
        """优化内存使用"""
        
        optimized_model = model
        
        # 梯度检查点
        if config.get('use_gradient_checkpointing', True):
            optimized_model = self.apply_gradient_checkpointing(optimized_model)
        
        # 模型量化
        if config.get('use_quantization', False):
            optimized_model = self.quantize_model(optimized_model)
        
        # 激活检查点
        if config.get('use_activation_checkpointing', True):
            optimized_model = self.apply_activation_checkpointing(optimized_model)
        
        return optimized_model
    
    def apply_gradient_checkpointing(self, model):
        """应用梯度检查点"""
        from torch.utils.checkpoint import checkpoint_sequential
        
        # 将模型分段
        segments = self.split_model_into_segments(model)
        
        def checkpointed_forward(x):
            return checkpoint_sequential(segments, len(segments), x)
        
        model.forward = checkpointed_forward
        return model
    
    def quantize_model(self, model):
        """量化模型"""
        # 动态量化
        quantized_model = torch.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8
        )
        return quantized_model
```

### 2. 计算优化
```python
class ComputationOptimizer:
    """计算优化器：提升计算效率"""
    
    def optimize_computation(self, model, config):
        """优化计算性能"""
        
        # 混合精度训练
        if config.get('use_mixed_precision', True):
            model, optimizer = self.setup_mixed_precision(model, config)
        
        # 内核融合
        if config.get('use_kernel_fusion', True):
            model = self.fuse_kernels(model)
        
        # 并行计算
        if config.get('use_parallel_computation', True):
            model = self.parallelize_model(model)
        
        return model
    
    def setup_mixed_precision(self, model, config):
        """设置混合精度训练"""
        from torch.cuda.amp import autocast, GradScaler
        
        scaler = GradScaler()
        
        def mixed_precision_forward(x):
            with autocast():
                return model.original_forward(x)
        
        model.forward = mixed_precision_forward
        return model, scaler
    
    def fuse_kernels(self, model):
        """融合计算内核"""
        # 融合卷积和批归一化
        model = torch.quantization.fuse_modules(model, [
            ['conv1', 'bn1', 'relu1'],
            ['conv2', 'bn2', 'relu2']
        ])
        
        return model
```

### 3. I/O优化
```python
class IOOptimizer:
    """I/O优化器：优化数据加载和存储"""
    
    def optimize_io(self, data_loader, config):
        """优化I/O性能"""
        
        # 数据预加载
        if config.get('use_data_prefetching', True):
            data_loader = self.enable_data_prefetching(data_loader)
        
        # 数据压缩
        if config.get('use_data_compression', True):
            data_loader = self.compress_data(data_loader)
        
        # 缓存策略
        if config.get('use_caching', True):
            data_loader = self.implement_caching(data_loader)
        
        return data_loader
    
    def enable_data_prefetching(self, data_loader):
        """启用数据预加载"""
        from torch.utils.data import DataLoader
        
        # 使用多进程数据加载
        optimized_loader = DataLoader(
            data_loader.dataset,
            batch_size=data_loader.batch_size,
            shuffle=data_loader.shuffle,
            num_workers=4,  # 根据CPU核心数调整
            pin_memory=True,  # 固定内存
            prefetch_factor=2  # 预加载因子
        )
        
        return optimized_loader
    
    def compress_data(self, data_loader):
        """数据压缩"""
        import zlib
        import pickle
        
        def compressed_collate_fn(batch):
            # 压缩批次数据
            compressed_batch = []
            for data in batch:
                serialized = pickle.dumps(data)
                compressed = zlib.compress(serialized)
                compressed_batch.append(compressed)
            
            return compressed_batch
        
        data_loader.collate_fn = compressed_collate_fn
        return data_loader
```

## 监控与诊断

### 1. 性能监控
```python
class PerformanceMonitor:
    """性能监控器：实时监控系统性能"""
    
    def __init__(self):
        self.metrics = {
            'gpu_usage': [],
            'memory_usage': [],
            'throughput': [],
            'latency': []
        }
    
    def start_monitoring(self):
        """开始性能监控"""
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def _monitor_loop(self):
        """监控循环"""
        while True:
            # GPU使用率
            if torch.cuda.is_available():
                gpu_usage = torch.cuda.utilization()
                self.metrics['gpu_usage'].append(gpu_usage)
            
            # 内存使用
            memory_used = torch.cuda.memory_allocated() / 1e9  # GB
            self.metrics['memory_usage'].append(memory_used)
            
            # 吞吐量
            throughput = self.calculate_throughput()
            self.metrics['throughput'].append(throughput)
            
            time.sleep(1)  # 每秒采样一次
    
    def generate_performance_report(self):
        """生成性能报告"""
        report = {
            'summary': {
                'avg_gpu_usage': np.mean(self.metrics['gpu_usage']),
                'max_memory_used': np.max(self.metrics['memory_usage']),
                'avg_throughput': np.mean(self.metrics['throughput'])
            },
            'recommendations': self.generate_recommendations()
        }
        
        return report
```

### 2. 资源调度
```python
class ResourceScheduler:
    """资源调度器：动态分配计算资源"""
    
    def schedule_resources(self, workload, available_resources):
        """调度计算资源"""
        
        # 分析工作负载
        workload_analysis = self.analyze_workload(workload)
        
        # 分配资源
        resource_allocation = self.allocate_resources(workload_analysis, available_resources)
        
        # 优化配置
        optimized_config = self.optimize_config_for_resources(resource_allocation)
        
        return optimized_config
    
    def analyze_workload(self, workload):
        """分析工作负载特征"""
        analysis = {
            'computational_intensity': self.estimate_computational_intensity(workload),
            'memory_requirements': self.estimate_memory_requirements(workload),
            'io_requirements': self.estimate_io_requirements(workload),
            'parallelizability': self.estimate_parallelizability(workload)
        }
        
        return analysis
    
    def allocate_resources(self, workload_analysis, available_resources):
        """分配资源"""
        allocation = {}
        
        # GPU分配
        if available_resources['gpu'] > 0:
            gpu_allocation = min(
                workload_analysis['computational_intensity'] / 100,  # 简化计算
                available_resources['gpu']
            )
            allocation['gpu'] = gpu_allocation
        
        # 内存分配
        memory_allocation = min(
            workload_analysis['memory_requirements'],
            available_resources['memory'] * 0.8  # 保留20%给系统
        )
        allocation['memory'] = memory_allocation
        
        return allocation
```

## 最佳实践

### 1. 部署检查清单
```python
class DeploymentChecklist:
    """部署检查清单：确保部署成功"""
    
    def run_pre_deployment_checks(self):
        """运行部署前检查"""
        checks = [
            self.check_system_requirements,
            self.check_dependencies,
            self.check_configuration,
            self.check_model_compatibility,
            self.check_data_availability,
            self.check_permissions
        ]
        
        results = {}
        for check in checks:
            check_name = check.__name__
            try:
                results[check_name] = check()
            except Exception as e:
                results[check_name] = {'status': 'failed', 'error': str(e)}
        
        return results
    
    def check_system_requirements(self):
        """检查系统要求"""
        requirements = {
            'python_version': (3, 8),
            'pytorch_version': (1, 9),
            'cuda_version': (11, 1),
            'available_memory': 8 * 1024**3,  # 8GB
            'available_storage': 10 * 1024**3  # 10GB
        }
        
        return self.verify_system_requirements(requirements)
```

### 2. 性能基准测试
```python
class PerformanceBenchmark:
    """性能基准测试：评估系统性能"""
    
    def run_benchmark(self, model, test_data, iterations=100):
        """运行性能基准测试"""
        
        benchmark_results = {}
        
        # 推理性能
        inference_times = self.benchmark_inference(model, test_data, iterations)
        benchmark_results['inference'] = inference_times
        
        # 训练性能
        training_times = self.benchmark_training(model, test_data, iterations)
        benchmark_results['training'] = training_times
        
        # 内存使用
        memory_usage = self.benchmark_memory(model, test_data)
        benchmark_results['memory'] = memory_usage
        
        # 生成报告
        report = self.generate_benchmark_report(benchmark_results)
        
        return report
    
    def benchmark_inference(self, model, test_data, iterations):
        """基准测试推理性能"""
        times = []
        
        # 预热
        for _ in range(10):
            _ = model(test_data)
        
        # 正式测试
        for _ in range(iterations):
            start_time = time.time()
            _ = model(test_data)
            end_time = time.time()
            times.append(end_time - start_time)
        
        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times)
        }
```

这个详细的部署与性能优化指南为开发者提供了从本地开发到生产环境的完整部署方案和性能优化策略。