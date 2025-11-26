# 配置系统详解

## 配置架构概览

EFD3D采用分层配置系统，支持从简单到复杂的各种应用场景：

```python
class HierarchicalConfigSystem:
    """分层配置系统：支持默认配置、用户配置、场景配置的优先级合并"""
    
    def __init__(self):
        self.default_config = self.load_default_config()
        self.user_config = {}
        self.scenario_config = {}
        self.runtime_config = {}
    
    def load_config(self, config_path=None, scenario=None):
        """加载和合并配置"""
        # 加载默认配置
        config = self.default_config.copy()
        
        # 合并用户配置
        if config_path and os.path.exists(config_path):
            user_config = self.load_json_config(config_path)
            config = self.deep_merge(config, user_config)
        
        # 合并场景配置
        if scenario:
            scenario_config = self.load_scenario_config(scenario)
            config = self.deep_merge(config, scenario_config)
        
        # 合并运行时配置
        config = self.deep_merge(config, self.runtime_config)
        
        return config
    
    def deep_merge(self, base, update):
        """深度合并字典"""
        result = base.copy()
        
        for key, value in update.items():
            if (key in result and isinstance(result[key], dict) 
                and isinstance(value, dict)):
                result[key] = self.deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
```

## 核心配置文件结构

### 1. 模型配置（model_config.json）
```json
{
    "model": {
        "architecture": {
            "input_dim": 62,
            "output_dim": 24,
            "hidden_layers": [256, 512, 256],
            "activation": "silu",
            "use_residual": true,
            "use_attention": true,
            "attention_heads": 8
        },
        "initialization": {
            "weight_init": "xavier_uniform",
            "bias_init": "zeros",
            "gain": 1.0
        },
        "regularization": {
            "dropout_rate": 0.1,
            "weight_decay": 1e-4,
            "batch_norm": true
        }
    },
    "physics": {
        "constraints": {
            "navier_stokes": {
                "enabled": true,
                "weight": 0.3,
                "tolerance": 1e-6
            },
            "young_lippmann": {
                "enabled": true,
                "weight": 0.3,
                "parameters": {
                    "epsilon_0": 8.854e-12,
                    "dielectric_thickness": 1e-6
                }
            }
        }
    }
}
```

### 2. 训练配置（training_config.json）
```json
{
    "training": {
        "stages": {
            "initialization": {
                "epochs": 100,
                "learning_rate": 1e-3,
                "batch_size": 32,
                "constraint_weights": [0.1, 0.1, 0.1, 0.1, 0.1, 0.5]
            },
            "physics_introduction": {
                "epochs": 200,
                "learning_rate": 5e-4,
                "batch_size": 64,
                "constraint_weights": [0.2, 0.2, 0.2, 0.2, 0.2, 0.0]
            },
            "refinement": {
                "epochs": 300,
                "learning_rate": 1e-4,
                "batch_size": 128,
                "constraint_weights": [0.3, 0.3, 0.2, 0.1, 0.1, 0.0]
            }
        },
        "optimizer": {
            "type": "adamw",
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 1e-4
        },
        "scheduler": {
            "type": "cosine_annealing",
            "T_max": 600,
            "eta_min": 1e-6
        },
        "early_stopping": {
            "enabled": true,
            "patience": 20,
            "min_delta": 1e-4
        }
    },
    "data": {
        "augmentation": {
            "enabled": true,
            "factor": 5,
            "methods": ["physics_perturbation", "boundary_variation", "parameter_sampling"]
        },
        "validation": {
            "split_ratio": 0.2,
            "shuffle": true,
            "random_seed": 42
        }
    }
}
```

### 3. 场景配置模板
```json
{
    "scenario": "dc_step",
    "description": "直流阶跃响应分析",
    "parameters": {
        "voltage_range": [0, 100],
        "frequency": 0,
        "temperature": 25,
        "fluid_properties": {
            "viscosity": 0.001,
            "density": 1000,
            "surface_tension": 0.072
        }
    },
    "boundary_conditions": {
        "initial": {
            "contact_angle": 110,
            "meniscus_position": 0.5
        },
        "dirichlet": ["voltage_boundary"],
        "neumann": ["pressure_gradient"]
    },
    "monitoring": {
        "metrics": ["response_time", "overshoot", "settling_time"],
        "visualization": ["voltage_response", "contact_angle_dynamics"]
    }
}
```

## 配置解析与验证

### 1. 配置验证器
```python
class ConfigValidator:
    """配置验证器：确保配置的完整性和合理性"""
    
    def __init__(self):
        self.schema = self.load_validation_schema()
    
    def validate_config(self, config):
        """验证配置是否符合模式"""
        errors = []
        
        # 必需字段检查
        required_fields = ['model', 'training', 'physics']
        for field in required_fields:
            if field not in config:
                errors.append(f"缺少必需字段: {field}")
        
        # 类型检查
        type_checks = [
            ('model.architecture.input_dim', int),
            ('model.architecture.output_dim', int),
            ('training.stages.initialization.epochs', int),
            ('training.optimizer.learning_rate', (int, float))
        ]
        
        for path, expected_type in type_checks:
            value = self.get_nested_value(config, path)
            if value is not None and not isinstance(value, expected_type):
                errors.append(f"{path} 类型错误: 期望 {expected_type}, 得到 {type(value)}")
        
        # 范围检查
        range_checks = [
            ('training.optimizer.learning_rate', 1e-6, 1e-1),
            ('model.regularization.dropout_rate', 0.0, 0.5)
        ]
        
        for path, min_val, max_val in range_checks:
            value = self.get_nested_value(config, path)
            if value is not None and not (min_val <= value <= max_val):
                errors.append(f"{path} 超出范围: [{min_val}, {max_val}]")
        
        return errors
    
    def get_nested_value(self, config, path):
        """获取嵌套配置值"""
        keys = path.split('.')
        current = config
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        
        return current
```

### 2. 动态配置生成
```python
class DynamicConfigGenerator:
    """动态配置生成器：根据任务需求自动生成优化配置"""
    
    def generate_config_based_on_task(self, task_type, complexity, available_resources):
        """根据任务类型和复杂度生成配置"""
        
        base_config = self.get_base_config()
        
        # 根据任务类型调整
        task_specific = self.get_task_specific_config(task_type)
        base_config = self.deep_merge(base_config, task_specific)
        
        # 根据复杂度调整
        complexity_config = self.get_complexity_config(complexity)
        base_config = self.deep_merge(base_config, complexity_config)
        
        # 根据资源调整
        resource_config = self.get_resource_config(available_resources)
        base_config = self.deep_merge(base_config, resource_config)
        
        return base_config
    
    def get_task_specific_config(self, task_type):
        """获取任务特定配置"""
        configs = {
            'dc_step': {
                'physics': {
                    'constraints': {
                        'young_lippmann': {'weight': 0.4},
                        'navier_stokes': {'weight': 0.3}
                    }
                }
            },
            'ac_sweep': {
                'physics': {
                    'constraints': {
                        'charge_conservation': {'weight': 0.4},
                        'energy_conservation': {'weight': 0.3}
                    }
                }
            }
        }
        
        return configs.get(task_type, {})
    
    def get_complexity_config(self, complexity):
        """根据复杂度调整配置"""
        if complexity == 'simple':
            return {
                'model': {
                    'architecture': {
                        'hidden_layers': [128, 256, 128],
                        'use_attention': False
                    }
                },
                'training': {
                    'stages': {
                        'initialization': {'epochs': 50},
                        'physics_introduction': {'epochs': 100},
                        'refinement': {'epochs': 150}
                    }
                }
            }
        elif complexity == 'complex':
            return {
                'model': {
                    'architecture': {
                        'hidden_layers': [512, 1024, 512, 256],
                        'use_attention': True,
                        'attention_heads': 12
                    }
                },
                'training': {
                    'stages': {
                        'initialization': {'epochs': 200},
                        'physics_introduction': {'epochs': 300},
                        'refinement': {'epochs': 500}
                    }
                }
            }
        
        return {}
```

## 配置管理工具

### 1. 配置比较工具
```python
class ConfigComparator:
    """配置比较工具：分析不同配置的性能差异"""
    
    def compare_configs(self, config1, config2, performance_metrics):
        """比较两个配置的性能"""
        
        differences = self.find_config_differences(config1, config2)
        performance_diff = self.calculate_performance_difference(performance_metrics)
        
        analysis = {
            'config_differences': differences,
            'performance_impact': performance_diff,
            'recommendations': self.generate_recommendations(differences, performance_diff)
        }
        
        return analysis
    
    def find_config_differences(self, config1, config2):
        """找出配置差异"""
        differences = []
        
        def compare_dicts(dict1, dict2, path=""):
            all_keys = set(dict1.keys()) | set(dict2.keys())
            
            for key in all_keys:
                current_path = f"{path}.{key}" if path else key
                
                if key in dict1 and key not in dict2:
                    differences.append({
                        'path': current_path,
                        'type': 'removed',
                        'value1': dict1[key],
                        'value2': None
                    })
                elif key not in dict1 and key in dict2:
                    differences.append({
                        'path': current_path,
                        'type': 'added',
                        'value1': None,
                        'value2': dict2[key]
                    })
                elif dict1[key] != dict2[key]:
                    if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                        compare_dicts(dict1[key], dict2[key], current_path)
                    else:
                        differences.append({
                            'path': current_path,
                            'type': 'modified',
                            'value1': dict1[key],
                            'value2': dict2[key]
                        })
        
        compare_dicts(config1, config2)
        return differences
```

### 2. 配置优化器
```python
class ConfigOptimizer:
    """配置优化器：基于性能数据自动优化配置"""
    
    def optimize_config(self, base_config, performance_data):
        """基于性能数据优化配置"""
        
        optimized_config = base_config.copy()
        
        # 分析性能瓶颈
        bottlenecks = self.identify_bottlenecks(performance_data)
        
        # 根据瓶颈调整配置
        for bottleneck in bottlenecks:
            if bottleneck['type'] == 'memory':
                optimized_config = self.optimize_for_memory(optimized_config)
            elif bottleneck['type'] == 'convergence':
                optimized_config = self.optimize_for_convergence(optimized_config)
            elif bottleneck['type'] == 'accuracy':
                optimized_config = self.optimize_for_accuracy(optimized_config)
        
        return optimized_config
    
    def optimize_for_memory(self, config):
        """内存优化"""
        optimized = config.copy()
        
        # 减小批大小
        optimized['training']['stages']['initialization']['batch_size'] = 16
        optimized['training']['stages']['physics_introduction']['batch_size'] = 32
        optimized['training']['stages']['refinement']['batch_size'] = 64
        
        # 简化模型架构
        optimized['model']['architecture']['hidden_layers'] = [128, 256, 128]
        optimized['model']['architecture']['use_attention'] = False
        
        return optimized
    
    def optimize_for_convergence(self, config):
        """收敛性优化"""
        optimized = config.copy()
        
        # 调整学习率策略
        optimized['training']['scheduler']['type'] = 'reduce_lr_on_plateau'
        optimized['training']['scheduler']['patience'] = 10
        optimized['training']['scheduler']['factor'] = 0.5
        
        # 增加训练轮数
        for stage in optimized['training']['stages']:
            optimized['training']['stages'][stage]['epochs'] *= 1.5
        
        return optimized
```

## 最佳实践

### 1. 配置版本控制
```python
def add_config_versioning(config):
    """为配置添加版本信息"""
    config['metadata'] = {
        'version': '1.0.0',
        'created_at': datetime.now().isoformat(),
        'author': os.getenv('USER', 'unknown'),
        'git_commit': self.get_git_commit_hash()
    }
    return config
```

### 2. 配置模板系统
```python
class ConfigTemplateSystem:
    """配置模板系统：快速生成标准配置"""
    
    def get_template(self, template_name):
        """获取配置模板"""
        templates = {
            'quick_start': self.quick_start_template(),
            'high_accuracy': self.high_accuracy_template(),
            'memory_efficient': self.memory_efficient_template(),
            'fast_training': self.fast_training_template()
        }
        
        return templates.get(template_name, self.default_template())
    
    def quick_start_template(self):
        """快速开始模板"""
        return {
            'model': {
                'architecture': {
                    'hidden_layers': [128, 256, 128],
                    'use_residual': True
                }
            },
            'training': {
                'stages': {
                    'initialization': {'epochs': 50},
                    'physics_introduction': {'epochs': 100}
                }
            }
        }
```

这个详细的配置系统文档为开发者提供了完整的配置管理、验证和优化工具。