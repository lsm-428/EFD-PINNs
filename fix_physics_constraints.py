import torch
import numpy as np
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, '/home/scnu/Gitee/EFD3D')

# 导入物理约束类
from ewp_pinn_physics import PhysicsConstraints

# 创建测试数据
def create_test_data():
    # 创建简单的测试点
    x = torch.linspace(0, 1, 10)
    y = torch.linspace(0, 1, 10)
    z = torch.linspace(0, 1, 10)
    t = torch.linspace(0, 1, 10)
    
    # 创建网格
    X, Y, Z, T = torch.meshgrid(x, y, z, t, indexing='ij')
    
    # 展平成批次
    coords = torch.stack([X.flatten(), Y.flatten(), Z.flatten(), T.flatten()], dim=1)
    
    # 创建简单的预测（速度场和压力场）
    u = torch.sin(coords[:, 0]) * torch.cos(coords[:, 1]) * torch.cos(coords[:, 3])
    v = torch.cos(coords[:, 0]) * torch.sin(coords[:, 1]) * torch.cos(coords[:, 3])
    w = torch.zeros_like(u)
    p = torch.sin(coords[:, 0]) * torch.sin(coords[:, 1]) * torch.exp(-coords[:, 3])
    
    # 堆叠预测结果为一个Tensor
    predictions = torch.stack([u, v, w, p], dim=1)
    
    return coords, predictions

# 修复PhysicsConstraints类中的问题
def fix_physics_constraints(physics):
    """
    修复PhysicsConstraints类中的问题，确保物理约束残差正确计算
    """
    print("\n应用修复到PhysicsConstraints类...")
    
    # 保存原始方法
    original_compute = physics.compute_navier_stokes_residual
    
    # 定义修复后的方法
    def fixed_compute(coords, predictions):
        try:
            # 检查输入是否有效
            if coords is None or predictions is None:
                print("警告: 输入coords或predictions为None")
                return physics._empty_residual(coords)
            
            # 检查梯度是否启用
            if not coords.requires_grad:
                print("警告: coords的requires_grad未启用，启用梯度")
                coords.requires_grad = True
            
            # 检查设备一致性
            if coords.device != predictions.device:
                print(f"警告: 设备不一致 - coords: {coords.device}, predictions: {predictions.device}")
                predictions = predictions.to(coords.device)
            
            # 确保material_params存在
            if not hasattr(physics, 'material_params'):
                print("警告: material_params属性不存在，添加默认值")
                physics.material_params = {
                    'dynamic_viscosity': 0.001,
                    'density': 1000.0
                }
            
            # 计算梯度
            try:
                # 手动计算速度梯度
                u = predictions[:, 0]
                v = predictions[:, 1]
                w = predictions[:, 2]
                p = predictions[:, 3]
                
                # 计算连续性方程残差
                # 使用torch.autograd.grad计算梯度
                u_x = torch.autograd.grad(
                    u, coords, 
                    grad_outputs=torch.ones_like(u),
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True
                )[0][:, 0]
                
                v_y = torch.autograd.grad(
                    v, coords, 
                    grad_outputs=torch.ones_like(v),
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True
                )[0][:, 1]
                
                w_z = torch.autograd.grad(
                    w, coords, 
                    grad_outputs=torch.ones_like(w),
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True
                )[0][:, 2]
                
                # 连续性方程残差
                continuity_residual = u_x + v_y + w_z
                
                # 计算动量方程残差（简化版）
                mu = physics.material_params.get('dynamic_viscosity', 0.001)
                rho = physics.material_params.get('density', 1000.0)
                
                # 计算压力梯度
                p_x = torch.autograd.grad(
                    p, coords, 
                    grad_outputs=torch.ones_like(p),
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True
                )[0][:, 0]
                
                p_y = torch.autograd.grad(
                    p, coords, 
                    grad_outputs=torch.ones_like(p),
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True
                )[0][:, 1]
                
                p_z = torch.autograd.grad(
                    p, coords, 
                    grad_outputs=torch.ones_like(p),
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True
                )[0][:, 2]
                
                # 简化的动量方程残差
                momentum_u_residual = -p_x / rho
                momentum_v_residual = -p_y / rho
                momentum_w_residual = -p_z / rho
                
                # 返回残差字典
                return {
                    'continuity': continuity_residual.unsqueeze(1),
                    'momentum_u': momentum_u_residual.unsqueeze(1),
                    'momentum_v': momentum_v_residual.unsqueeze(1),
                    'momentum_w': momentum_w_residual.unsqueeze(1)
                }
                
            except Exception as e:
                print(f"计算梯度时出错: {e}")
                import traceback
                traceback.print_exc()
                
                # 作为回退，尝试使用原始方法
                try:
                    result = original_compute(coords, predictions)
                    # 检查结果是否全为零
                    is_all_zero = True
                    for key, residual in result.items():
                        if residual is not None and torch.any(residual.abs() > 1e-6):
                            is_all_zero = False
                            break
                    
                    if is_all_zero:
                        print("警告: 原始方法返回全零残差")
                    
                    return result
                except Exception as e2:
                    print(f"原始方法也失败: {e2}")
                    return physics._empty_residual(coords)
        
        except Exception as e:
            print(f"fixed_compute方法总错误: {e}")
            import traceback
            traceback.print_exc()
            return physics._empty_residual(coords)
    
    # 应用修复
    physics.compute_navier_stokes_residual = fixed_compute
    print("修复应用完成")

# 主测试函数
def test_fixed_physics_constraints():
    print("开始测试修复后的物理约束计算...")
    
    # 确保使用CUDA（如果可用）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建测试数据
    coords, predictions = create_test_data()
    coords = coords.to(device)
    predictions = predictions.to(device)
    
    print(f"测试数据形状: {coords.shape}")
    
    # 初始化物理约束类
    physics = PhysicsConstraints()
    physics.device = device
    physics.boundary_conditions_weights = {
        'continuity': 0.5,
        'momentum_u': 0.05,
        'momentum_v': 0.05,
        'momentum_w': 0.05
    }
    physics.material_params = {
        'dynamic_viscosity': 0.001,
        'density': 1000.0
    }
    
    # 启用梯度
    coords.requires_grad = True
    
    # 打印更多调试信息
    print("\n调试信息:")
    print(f"coords形状: {coords.shape}, requires_grad: {coords.requires_grad}")
    print(f"predictions形状: {predictions.shape}")
    
    # 测试修复前的结果
    print("\n修复前的残差计算:")
    result_before = physics.compute_navier_stokes_residual(coords, predictions)
    print("修复前的残差:")
    for key, residual in result_before.items():
        if residual is not None:
            print(f"{key}: 均值={residual.mean().item():.6f}, 标准差={residual.std().item():.6f}, 最大值={residual.abs().max().item():.6f}")
            print(f"  部分残差样本: {residual.flatten()[:5].tolist()}")
    
    # 应用修复
    fix_physics_constraints(physics)
    
    # 测试修复后的结果
    print("\n修复后的残差计算:")
    result_after = physics.compute_navier_stokes_residual(coords, predictions)
    print("修复后的残差:")
    for key, residual in result_after.items():
        if residual is not None:
            print(f"{key}: 均值={residual.mean().item():.6f}, 标准差={residual.std().item():.6f}, 最大值={residual.abs().max().item():.6f}")
            print(f"  部分残差样本: {residual.flatten()[:5].tolist()}")
    
    print("\n测试完成!")

if __name__ == "__main__":
    test_fixed_physics_constraints()