# 训练策略详解

## 训练架构概览

EFD3D采用多阶段渐进式训练策略，结合动态权重调整和自适应优化：

```python
class MultiStageTrainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.stage_manager = TrainingStageManager(config)
        self.optimizer_manager = OptimizerManager(config)
        self.constraint_manager = ConstraintManager(config)
    
    def train(self, train_loader, val_loader):
        """多阶段训练主循环"""
        for stage in self.stage_manager.get_stages():
            self.train_stage(stage, train_loader, val_loader)
```

## 训练阶段设计

### 阶段1：数据拟合初始化
**目标**：建立基础的数据映射关系
```python
class InitializationStage:
    def configure_stage(self):
        return {
            'learning_rate': 1e-3,
            'constraint_weights': [0.1, 0.1, 0.1, 0.1, 0.1, 0.5],  # 数据拟合权重高
            'batch_size': 32,
            'epochs': 100,
            'optimizer': 'Adam',
            'scheduler': 'CosineAnnealing'
        }
    
    def compute_loss(self, predictions, targets):
        """初期以MSE损失为主"""
        data_loss = F.mse_loss(predictions['main'], targets)
        constraint_loss = self.compute_constraint_loss(predictions) * 0.1  # 低权重
        return data_loss + constraint_loss
```

### 阶段2：物理约束引入
**目标**：逐步引入物理约束，确保物理一致性
```python
class PhysicsIntroductionStage:
    def configure_stage(self):
        return {
            'learning_rate': 5e-4,
            'constraint_weights': [0.2, 0.2, 0.2, 0.2, 0.2, 0.0],  # 均衡权重
            'batch_size': 64,
            'epochs': 200,
            'optimizer': 'AdamW',
            'scheduler': 'ReduceLROnPlateau'
        }
    
    def compute_loss(self, predictions, targets):
        """平衡数据拟合和物理约束"""
        data_loss = F.mse_loss(predictions['main'], targets)
        constraint_loss = self.compute_constraint_loss(predictions)
        
        # 动态权重调整
        current_epoch = self.get_current_epoch()
        physics_weight = min(1.0, current_epoch / 100)  # 逐步增加
        
        return data_loss * (1 - physics_weight) + constraint_loss * physics_weight
```

### 阶段3：精炼优化
**目标**：微调模型参数，优化物理一致性
```python
class RefinementStage:
    def configure_stage(self):
        return {
            'learning_rate': 1e-4,
            'constraint_weights': [0.3, 0.3, 0.2, 0.1, 0.1, 0.0],  # 物理约束主导
            'batch_size': 128,
            'epochs': 300,
            'optimizer': 'LAMB',
            'scheduler': 'CosineAnnealingWarmRestarts'
        }
    
    def compute_loss(self, predictions, targets):
        """物理约束主导的损失函数"""
        data_loss = F.mse_loss(predictions['main'], targets)
        constraint_loss = self.compute_constraint_loss(predictions)
        
        # 物理一致性惩罚项
        physics_penalty = self.compute_physics_penalty(predictions)
        
        return 0.1 * data_loss + 0.7 * constraint_loss + 0.2 * physics_penalty
```

## 优化器策略

### 自适应优化器选择
```python
class AdaptiveOptimizer:
    def select_optimizer(self, stage_config, model_parameters):
        optimizer_type = stage_config['optimizer']
        lr = stage_config['learning_rate']
        
        if optimizer_type == 'Adam':
            return optim.Adam(model_parameters, lr=lr, betas=(0.9, 0.999))
        elif optimizer_type == 'AdamW':
            return optim.AdamW(model_parameters, lr=lr, weight_decay=1e-4)
        elif optimizer_type == 'LAMB':
            return LAMB(model_parameters, lr=lr)
        elif optimizer_type == 'RAdam':
            return optim.RAdam(model_parameters, lr=lr)
        else:
            return optim.Adam(model_parameters, lr=lr)
```

### 学习率调度策略
```python
class LearningRateScheduler:
    def get_scheduler(self, optimizer, scheduler_type, total_epochs):
        if scheduler_type == 'CosineAnnealing':
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
        elif scheduler_type == 'ReduceLROnPlateau':
            return optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        elif scheduler_type == 'CosineAnnealingWarmRestarts':
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50)
        elif scheduler_type == 'OneCycleLR':
            return optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, 
                                               total_steps=total_epochs)
```

## 数据增强策略

### 物理引导的数据增强
```python
class PhysicsGuidedAugmentation:
    def augment_training_data(self, original_data, augmentation_factor=5):
        """基于物理原理的数据增强"""
        augmented_data = []
        
        for sample in original_data:
            # 物理合理的扰动
            perturbations = self.generate_physics_perturbations(sample)
            augmented_data.extend(perturbations)
            
            # 边界条件扩展
            boundary_samples = self.generate_boundary_variations(sample)
            augmented_data.extend(boundary_samples)
            
            # 参数空间采样
            parameter_variations = self.sample_parameter_space(sample)
            augmented_data.extend(parameter_variations)
        
        return augmented_data
    
    def generate_physics_perturbations(self, sample):
        """生成物理合理的扰动样本"""
        perturbations = []
        
        # 小幅度物理参数扰动
        for perturbation_level in [0.01, 0.05, 0.1]:
            perturbed = self.perturb_physical_parameters(sample, perturbation_level)
            if self.validate_physics(perturbed):
                perturbations.append(perturbed)
        
        return perturbations
```

## 训练监控与诊断

### 实时训练监控
```python
class TrainingMonitor:
    def __init__(self):
        self.metrics_history = {
            'train_loss': [], 'val_loss': [],
            'physics_violation': [], 'gradient_norm': [],
            'learning_rate': [], 'constraint_weights': []
        }
    
    def update_metrics(self, epoch, metrics):
        """更新训练指标"""
        for key, value in metrics.items():
            self.metrics_history[key].append((epoch, value))
        
        # 实时诊断
        self.diagnose_training_issues(epoch, metrics)
    
    def diagnose_training_issues(self, epoch, metrics):
        """诊断训练问题"""
        # 梯度爆炸检测
        if metrics.get('gradient_norm', 0) > 1000:
            self.log_warning(f"梯度爆炸检测于epoch {epoch}")
        
        # 过拟合检测
        if metrics.get('train_loss', 0) < 0.1 * metrics.get('val_loss', 1):
            self.log_warning(f"可能过拟合于epoch {epoch}")
        
        # 物理一致性恶化
        if len(self.metrics_history['physics_violation']) > 10:
            recent_violations = self.metrics_history['physics_violation'][-10:]
            if all(v1 < v2 for v1, v2 in zip(recent_violations[:-1], recent_violations[1:])):
                self.log_warning(f"物理一致性持续恶化于epoch {epoch}")
```

### 早停策略
```python
class EarlyStopping:
    def __init__(self, patience=20, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
    
    def __call__(self, val_loss):
        """检查是否应该早停"""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience
```

## 高级训练技巧

### 课程学习（Curriculum Learning）
```python
class CurriculumLearning:
    def get_training_difficulty(self, epoch, total_epochs):
        """根据训练进度调整样本难度"""
        progress = epoch / total_epochs
        
        if progress < 0.3:
            return 'easy'  # 简单样本
        elif progress < 0.6:
            return 'medium'  # 中等难度
        else:
            return 'hard'  # 困难样本
    
    def sample_training_data(self, difficulty):
        """根据难度采样训练数据"""
        if difficulty == 'easy':
            return self.sample_easy_cases()
        elif difficulty == 'medium':
            return self.sample_medium_cases()
        else:
            return self.sample_hard_cases()
```

### 对抗训练
```python
class AdversarialTraining:
    def generate_adversarial_examples(self, model, original_data):
        """生成对抗样本增强训练鲁棒性"""
        adversarial_data = []
        
        for sample in original_data:
            # FGSM攻击生成对抗样本
            adversarial_sample = self.fgsm_attack(model, sample)
            adversarial_data.append(adversarial_sample)
            
            # PGD攻击生成更强对抗样本
            pgd_sample = self.pgd_attack(model, sample)
            adversarial_data.append(pgd_sample)
        
        return adversarial_data
    
    def fgsm_attack(self, model, sample, epsilon=0.01):
        """快速梯度符号方法"""
        sample.requires_grad = True
        output = model(sample)
        loss = F.mse_loss(output, sample.target)
        loss.backward()
        
        # 生成对抗扰动
        perturbation = epsilon * sample.grad.sign()
        adversarial_sample = sample + perturbation
        
        return adversarial_sample.detach()
```

## 性能优化

### 混合精度训练
```python
class MixedPrecisionTraining:
    def __init__(self):
        self.scaler = torch.cuda.amp.GradScaler()
    
    def train_step(self, model, data, targets):
        """混合精度训练步骤"""
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = self.compute_loss(predictions, targets)
        
        # 缩放损失并反向传播
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss.item()
```

### 分布式训练
```python
class DistributedTraining:
    def setup_distributed_training(self):
        """设置分布式训练环境"""
        torch.distributed.init_process_group(backend='nccl')
        self.local_rank = int(os.environ['LOCAL_RANK'])
        
        # 模型并行化
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[self.local_rank]
        )
        
        return model
```

这个详细的训练策略文档为开发者提供了完整的训练架构、优化方法和高级技巧。