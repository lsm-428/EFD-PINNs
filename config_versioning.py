"""
配置版本化工具 - 用于管理训练配置的版本控制和变更追踪

功能：
1. 配置版本管理
2. 变更差异检测
3. 配置模板生成
4. 配置验证
"""

import json
import os
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import difflib
import logging

logger = logging.getLogger(__name__)


class ConfigVersionManager:
    """配置版本管理器 - 管理配置文件的版本控制和变更追踪"""
    
    def __init__(self, configs_dir: str = "./experiments/configs"):
        """
        初始化配置版本管理器
        
        参数:
            configs_dir: 配置目录路径
        """
        self.configs_dir = configs_dir
        os.makedirs(configs_dir, exist_ok=True)
        
        # 版本历史文件
        self.version_history_file = os.path.join(configs_dir, "version_history.json")
        self._init_version_history()
        
        logger.info(f"配置版本管理器已初始化，配置目录: {configs_dir}")
    
    def _init_version_history(self):
        """初始化版本历史记录"""
        if not os.path.exists(self.version_history_file):
            version_history = {
                "metadata": {
                    "created_at": datetime.now().isoformat(),
                    "last_updated": datetime.now().isoformat(),
                    "version_count": 0
                },
                "versions": {}
            }
            with open(self.version_history_file, 'w', encoding='utf-8') as f:
                json.dump(version_history, f, indent=2, ensure_ascii=False)
    
    def save_config_version(self, config: Dict[str, Any], description: str = "") -> str:
        """
        保存配置版本
        
        参数:
            config: 配置字典
            description: 版本描述
            
        返回:
            版本ID
        """
        # 计算配置哈希值
        config_hash = self._calculate_config_hash(config)
        
        # 检查是否已存在相同配置
        existing_version = self._find_existing_version(config_hash)
        if existing_version:
            logger.info(f"配置已存在，版本ID: {existing_version}")
            return existing_version
        
        # 生成版本ID
        version_id = f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 增强配置信息
        enhanced_config = {
            "metadata": {
                "version_id": version_id,
                "config_hash": config_hash,
                "created_at": datetime.now().isoformat(),
                "description": description,
                "file_size": len(json.dumps(config, ensure_ascii=False).encode('utf-8'))
            },
            "config": config
        }
        
        # 保存配置版本
        version_file = os.path.join(self.configs_dir, f"{version_id}.json")
        with open(version_file, 'w', encoding='utf-8') as f:
            json.dump(enhanced_config, f, indent=2, ensure_ascii=False)
        
        # 更新版本历史
        self._update_version_history(version_id, enhanced_config["metadata"])
        
        logger.info(f"✅ 保存配置版本: {version_id}")
        logger.info(f"   描述: {description}")
        logger.info(f"   哈希: {config_hash[:8]}...")
        
        return version_id
    
    def compare_configs(self, config1: Dict[str, Any], config2: Dict[str, Any]) -> Dict[str, Any]:
        """
        比较两个配置的差异
        
        参数:
            config1: 第一个配置
            config2: 第二个配置
            
        返回:
            差异分析结果
        """
        # 转换为JSON字符串进行比较
        config1_str = json.dumps(config1, indent=2, sort_keys=True, ensure_ascii=False)
        config2_str = json.dumps(config2, indent=2, sort_keys=True, ensure_ascii=False)
        
        # 计算差异
        diff = list(difflib.unified_diff(
            config1_str.splitlines(keepends=True),
            config2_str.splitlines(keepends=True),
            fromfile='config1',
            tofile='config2',
            n=3
        ))
        
        # 分析差异
        differences = []
        for line in diff:
            if line.startswith('+') and not line.startswith('+++'):
                differences.append({"type": "added", "content": line[1:].strip()})
            elif line.startswith('-') and not line.startswith('---'):
                differences.append({"type": "removed", "content": line[1:].strip()})
        
        return {
            "config1_hash": self._calculate_config_hash(config1),
            "config2_hash": self._calculate_config_hash(config2),
            "identical": config1_str == config2_str,
            "diff_count": len(differences),
            "differences": differences,
            "unified_diff": ''.join(diff)
        }
    
    def get_config_template(self, config_type: str = "standard") -> Dict[str, Any]:
        """
        获取配置模板
        
        参数:
            config_type: 配置类型 (standard, minimal, advanced)
            
        返回:
            配置模板
        """
        templates = {
            "standard": {
                "metadata": {
                    "template_type": "standard",
                    "description": "标准训练配置模板",
                    "required_fields": ["model", "training", "data"]
                },
                "model": {
                    "input_dim": "int: 输入维度",
                    "output_dim": "int: 输出维度",
                    "hidden_layers": "list[int]: 隐藏层大小",
                    "dropout_rate": "float: Dropout率 (0.0-1.0)",
                    "batch_norm": "bool: 是否使用批归一化",
                    "activation": "str: 激活函数 (ReLU, Tanh, Sigmoid)",
                    "use_residual": "bool: 是否使用残差连接"
                },
                "training": {
                    "epochs": "int: 训练轮次",
                    "batch_size": "int: 批次大小",
                    "learning_rate": "float: 学习率",
                    "weight_decay": "float: 权重衰减",
                    "validation_split": "float: 验证集比例 (0.0-1.0)",
                    "early_stopping_patience": "int: 早停耐心值",
                    "gradient_clipping": "float: 梯度裁剪阈值"
                },
                "data": {
                    "num_samples": "int: 样本数量",
                    "num_val_samples": "int: 验证样本数量",
                    "num_test_samples": "int: 测试样本数量",
                    "noise_level": "float: 噪声水平",
                    "augmentation": "bool: 是否使用数据增强"
                }
            },
            "minimal": {
                "metadata": {
                    "template_type": "minimal",
                    "description": "最小化训练配置模板",
                    "required_fields": ["model", "training"]
                },
                "model": {
                    "input_dim": "int: 输入维度",
                    "output_dim": "int: 输出维度",
                    "hidden_layers": "list[int]: 隐藏层大小"
                },
                "training": {
                    "epochs": "int: 训练轮次",
                    "batch_size": "int: 批次大小",
                    "learning_rate": "float: 学习率"
                }
            },
            "advanced": {
                "metadata": {
                    "template_type": "advanced",
                    "description": "高级训练配置模板",
                    "required_fields": ["model", "training", "data", "physics", "optimization"]
                },
                "model": {
                    "input_dim": "int: 输入维度",
                    "output_dim": "int: 输出维度",
                    "hidden_layers": "list[int]: 隐藏层大小",
                    "dropout_rate": "float: Dropout率",
                    "batch_norm": "bool: 批归一化",
                    "activation": "str: 激活函数",
                    "use_residual": "bool: 残差连接",
                    "spectral_norm": "bool: 谱归一化"
                },
                "training": {
                    "epochs": "int: 训练轮次",
                    "batch_size": "int: 批次大小",
                    "learning_rate": "float: 学习率",
                    "weight_decay": "float: 权重衰减",
                    "validation_split": "float: 验证集比例",
                    "early_stopping_patience": "int: 早停耐心值",
                    "gradient_clipping": "float: 梯度裁剪",
                    "mixed_precision": "bool: 混合精度训练"
                },
                "physics": {
                    "physics_weight": "float: 物理损失权重",
                    "boundary_weight": "float: 边界条件权重",
                    "adaptive_physics_weight": "bool: 自适应物理权重",
                    "num_physics_points": "int: 物理点数量"
                },
                "optimization": {
                    "optimizer": "str: 优化器类型",
                    "scheduler": "str: 学习率调度器",
                    "warmup_epochs": "int: 预热轮次",
                    "min_lr": "float: 最小学习率"
                }
            }
        }
        
        return templates.get(config_type, templates["standard"])
    
    def validate_config(self, config: Dict[str, Any], template_type: str = "standard") -> Dict[str, Any]:
        """
        验证配置的完整性和正确性
        
        参数:
            config: 待验证的配置
            template_type: 模板类型
            
        返回:
            验证结果
        """
        template = self.get_config_template(template_type)
        required_fields = template["metadata"]["required_fields"]
        
        validation_result = {
            "is_valid": True,
            "missing_fields": [],
            "invalid_types": [],
            "warnings": [],
            "suggestions": []
        }
        
        # 检查必需字段
        for field in required_fields:
            if field not in config:
                validation_result["missing_fields"].append(field)
                validation_result["is_valid"] = False
        
        # 检查字段类型（基础验证）
        for field, value in config.items():
            if field in template and isinstance(template[field], dict):
                # 这里可以添加更详细的类型验证
                pass
        
        # 提供建议
        if "learning_rate" in config.get("training", {}) and config["training"]["learning_rate"] > 0.1:
            validation_result["warnings"].append("学习率可能过高，建议使用较小的学习率")
        
        if "batch_size" in config.get("training", {}) and config["training"]["batch_size"] > 256:
            validation_result["suggestions"].append("批次大小较大，可能需要更多内存")
        
        return validation_result
    
    def list_config_versions(self) -> List[Dict[str, Any]]:
        """
        列出所有配置版本
        
        返回:
            版本信息列表
        """
        versions = []
        
        # 从版本历史文件读取
        with open(self.version_history_file, 'r', encoding='utf-8') as f:
            history = json.load(f)
        
        for version_id, version_info in history["versions"].items():
            version_file = os.path.join(self.configs_dir, f"{version_id}.json")
            if os.path.exists(version_file):
                versions.append({
                    "version_id": version_id,
                    "metadata": version_info,
                    "file_path": version_file
                })
        
        # 按创建时间排序
        versions.sort(key=lambda x: x["metadata"]["created_at"], reverse=True)
        
        return versions
    
    def _calculate_config_hash(self, config: Dict[str, Any]) -> str:
        """计算配置的哈希值"""
        config_str = json.dumps(config, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(config_str.encode('utf-8')).hexdigest()
    
    def _find_existing_version(self, config_hash: str) -> Optional[str]:
        """查找已存在的配置版本"""
        with open(self.version_history_file, 'r', encoding='utf-8') as f:
            history = json.load(f)
        
        for version_id, version_info in history["versions"].items():
            if version_info.get("config_hash") == config_hash:
                return version_id
        
        return None
    
    def _update_version_history(self, version_id: str, metadata: Dict[str, Any]):
        """更新版本历史记录"""
        with open(self.version_history_file, 'r', encoding='utf-8') as f:
            history = json.load(f)
        
        history["versions"][version_id] = metadata
        history["metadata"]["last_updated"] = datetime.now().isoformat()
        history["metadata"]["version_count"] = len(history["versions"])
        
        with open(self.version_history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)


def create_config_from_template(template_type: str = "standard", **kwargs) -> Dict[str, Any]:
    """
    从模板创建配置（便捷函数）
    
    参数:
        template_type: 模板类型
        **kwargs: 配置参数
        
    返回:
        配置字典
    """
    manager = ConfigVersionManager()
    template = manager.get_config_template(template_type)
    
    # 创建基础配置结构
    config = {}
    
    # 应用提供的参数
    for key, value in kwargs.items():
        if '.' in key:
            # 处理嵌套字段 (如 "model.input_dim")
            parts = key.split('.')
            current = config
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        else:
            config[key] = value
    
    return config


# 使用示例
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 创建版本管理器
    version_manager = ConfigVersionManager()
    
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
    
    # 保存配置版本
    version_id = version_manager.save_config_version(sample_config, "示例配置")
    
    # 验证配置
    validation = version_manager.validate_config(sample_config)
    print(f"\n🔍 配置验证结果:")
    print(f"   有效性: {'✅ 有效' if validation['is_valid'] else '❌ 无效'}")
    if validation["warnings"]:
        print(f"   警告: {validation['warnings']}")
    
    # 列出所有版本
    versions = version_manager.list_config_versions()
    print(f"\n📋 配置版本列表 ({len(versions)} 个版本):")
    for version in versions:
        print(f"  - {version['version_id']}: {version['metadata']['description']}")
    
    print("\n✅ 配置版本化工具测试完成！")