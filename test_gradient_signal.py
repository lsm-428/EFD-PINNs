"""
测试物理约束残差的梯度信号
验证修复后的PhysicsConstraints类是否能产生有效的梯度流
"""

import torch
import torch.nn as nn
from ewp_pinn_physics import PhysicsConstraints
import numpy as np
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('GradientTest')

def create_test_model(input_dim=4, hidden_dim=64, output_dim=4):
    """创建一个简单的测试模型"""
    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim)
    )
    return model

def test_gradient_flow():
    """测试梯度流是否正常工作"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 创建模型和物理约束对象
    model = create_test_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    physics = PhysicsConstraints()
    physics.device = device  # 确保物理约束对象知道设备
    
    # 创建测试数据 - 使用非零的速度场
    batch_size = 1000
    # 创建在[0,1]范围内的坐标
    coords = torch.rand(batch_size, 4, device=device)
    coords.requires_grad = True
    
    # 训练几个迭代
    num_epochs = 3
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # 前向传播
        predictions = model(coords)
        predictions = predictions.clone().requires_grad_(True)  # 创建叶子节点并确保需要梯度
        
        # 计算物理残差
        logger.info(f"\n第 {epoch+1} 轮迭代:")
        logger.info(f"预测形状: {predictions.shape}")
        logger.info(f"预测均值: {predictions.mean().item()}")
        
        # 记录预测的requires_grad状态
        logger.info(f"predictions.requires_grad: {predictions.requires_grad}")
        logger.info(f"coords.requires_grad: {coords.requires_grad}")
        
        # 计算残差
        residual_dict = physics.compute_navier_stokes_residual(coords, predictions)
        
        # 打印残差统计信息
        for key, residual in residual_dict.items():
            logger.info(f"{key}残差 - 均值: {residual.mean().item():.8f}, 标准差: {residual.std().item():.8f}")
            logger.info(f"{key}残差样本: {residual[:3].tolist()}")
        
        # 计算物理损失
        physics_loss = 0.0
        for key, residual in residual_dict.items():
            physics_loss += torch.mean(residual**2)
        
        logger.info(f"物理损失: {physics_loss.item():.8f}")
        
        # 反向传播
        physics_loss.backward(retain_graph=True)
        
        # 检查梯度
        has_grads = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                logger.info(f"参数 {name} 的梯度范数: {grad_norm:.8f}")
                if grad_norm > 0:
                    has_grads = True
        
        if has_grads:
            logger.info("✓ 成功获取到非零梯度!")
        else:
            logger.warning("✗ 未获取到非零梯度!")
        
        # 更新参数
        optimizer.step()
        
        # 检查参数是否变化
        with torch.no_grad():
            first_param_value = next(model.parameters()).mean().item()
        logger.info(f"模型参数均值: {first_param_value:.8f}")
    
    logger.info("\n测试完成!")
    return has_grads

def test_non_trivial_flow():
    """测试使用非零的、有物理意义的流动场"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建非零的流动场 - 使用简单的正弦波模式
    batch_size = 1000
    coords = torch.rand(batch_size, 4, device=device)
    coords.requires_grad = True
    
    # 创建有非零梯度的预测结果
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]
    t = coords[:, 3]
    
    # 创建一个有物理意义的流动场
    u = torch.sin(2 * np.pi * x) * torch.cos(2 * np.pi * y) * torch.exp(-t)
    v = -torch.cos(2 * np.pi * x) * torch.sin(2 * np.pi * y) * torch.exp(-t)
    w = torch.zeros_like(u)
    p = torch.sin(2 * np.pi * x) * torch.sin(2 * np.pi * y) * torch.exp(-2*t)
    
    # 堆叠成预测格式
    predictions = torch.stack([u, v, w, p], dim=1).clone().requires_grad_(True)  # 创建叶子节点
    
    logger.info("\n=== 测试非零流动场 ===")
    logger.info(f"预测均值: {predictions.mean().item()}")
    logger.info(f"u分量均值: {u.mean().item()}")
    logger.info(f"v分量均值: {v.mean().item()}")
    logger.info(f"p分量均值: {p.mean().item()}")
    
    # 计算残差
    physics = PhysicsConstraints()
    physics.device = device
    residual_dict = physics.compute_navier_stokes_residual(coords, predictions)
    
    # 打印残差统计信息
    for key, residual in residual_dict.items():
        logger.info(f"{key}残差 - 均值: {residual.mean().item():.8f}, 标准差: {residual.std().item():.8f}")
        logger.info(f"{key}残差最大值: {residual.abs().max().item():.8f}")
    
    # 检查残差是否非零
    all_non_zero = True
    for key, residual in residual_dict.items():
        if torch.all(torch.abs(residual) < 1e-6):
            logger.warning(f"{key}残差仍然接近零!")
            all_non_zero = False
    
    if all_non_zero:
        logger.info("✓ 所有残差都有非零值!")
    
    return all_non_zero

if __name__ == "__main__":
    logger.info("开始测试物理约束的梯度信号...")
    
    # 测试1: 梯度流是否正常
    grad_success = test_gradient_flow()
    
    # 测试2: 非零流动场的残差
    residual_success = test_non_trivial_flow()
    
    # 总结
    logger.info("\n=== 测试总结 ===")
    if grad_success and residual_success:
        logger.info("✅ 所有测试通过! 物理约束残差计算已成功修复!")
    else:
        logger.warning("⚠️ 部分测试未通过，可能需要进一步调整。")
        if grad_success:
            logger.info("  - 梯度流测试通过")
        else:
            logger.warning("  - 梯度流测试未通过")
        if residual_success:
            logger.info("  - 非零残差测试通过")
        else:
            logger.warning("  - 非零残差测试未通过")
