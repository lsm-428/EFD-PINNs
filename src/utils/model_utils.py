"""
模型工具函数

包含模型输出处理、加载等通用工具
"""

import logging
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any, Optional

logger = logging.getLogger("EWP-ModelUtils")


def extract_predictions(raw_output) -> torch.Tensor:
    """
    统一从模型输出中提取主要预测张量
    
    Args:
        raw_output: 模型输出，可以是 Tensor 或 dict
        
    Returns:
        torch.Tensor: 预测张量
        
    Raises:
        ValueError: 如果无法提取预测张量
    """
    # 直接张量
    if isinstance(raw_output, torch.Tensor):
        return raw_output

    # 字典形式
    if isinstance(raw_output, dict):
        # 优先使用明确键
        if 'main_predictions' in raw_output:
            val = raw_output['main_predictions']
            if isinstance(val, torch.Tensor):
                return val
            else:
                raise ValueError("'main_predictions' 存在但不是 torch.Tensor")

        # 其次尝试 'prediction' 键（兼容旧名）
        if 'prediction' in raw_output and isinstance(raw_output['prediction'], torch.Tensor):
            return raw_output['prediction']

        # 再尝试任何第一个张量类型的值
        for k, v in raw_output.items():
            if isinstance(v, torch.Tensor):
                logger.warning(f"未找到 'main_predictions'，使用字典中第一个张量键: {k}")
                return v

    raise ValueError(f"无法从模型输出中提取预测张量，类型={type(raw_output)}")


def load_model_with_mismatch_handling(
    model: nn.Module,
    checkpoint_path: str,
    strict: bool = False
) -> Tuple[nn.Module, List[str], List[str]]:
    """
    加载模型权重，处理架构不匹配
    
    Args:
        model: 目标模型
        checkpoint_path: 检查点路径
        strict: 是否严格匹配
        
    Returns:
        (model, missing_keys, unexpected_keys)
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # 尝试加载
    if strict:
        model.load_state_dict(state_dict, strict=True)
        return model, [], []
    
    # 非严格模式：部分加载
    model_state = model.state_dict()
    
    # 找出匹配的键
    matched_keys = []
    missing_keys = []
    unexpected_keys = list(state_dict.keys())
    
    for key in model_state.keys():
        if key in state_dict:
            if model_state[key].shape == state_dict[key].shape:
                model_state[key] = state_dict[key]
                matched_keys.append(key)
                unexpected_keys.remove(key)
            else:
                logger.warning(
                    f"Shape mismatch for {key}: "
                    f"model={model_state[key].shape}, "
                    f"checkpoint={state_dict[key].shape}"
                )
                missing_keys.append(key)
        else:
            missing_keys.append(key)
    
    model.load_state_dict(model_state, strict=False)
    
    if missing_keys:
        logger.warning(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        logger.warning(f"Unexpected keys: {unexpected_keys}")
    
    logger.info(f"Loaded {len(matched_keys)} / {len(model_state)} parameters")
    
    return model, missing_keys, unexpected_keys
