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
    
    # 堆叠预测结果为一个Tensor（根据错误信息，函数期望Tensor格式）
    predictions = torch.stack([u, v, w, p], dim=1)
    
    return coords, predictions

# 主测试函数
def test_physics_constraints():
    print("开始测试物理约束计算...")
    
    # 确保使用CUDA（如果可用）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建测试数据
    coords, predictions = create_test_data()
    coords = coords.to(device)
    
    # 将预测结果移到正确的设备
    predictions = predictions.to(device)
    
    print(f"测试数据形状: {coords.shape}")
    
    # 初始化物理约束类（不传递参数）
    physics = PhysicsConstraints()
    # 手动设置device属性
    physics.device = device
    # 添加boundary_conditions_weights属性并设置权重
    physics.boundary_conditions_weights = {
        'continuity': 0.5,
        'momentum_u': 0.05,
        'momentum_v': 0.05,
        'momentum_w': 0.05
    }
    # 添加material_params属性
    physics.material_params = {
        'dynamic_viscosity': 0.001,  # 水的动力粘度
        'density': 1000.0,           # 水的密度
        'surface_tension': 0.072,    # 水的表面张力
        'permittivity': 80.0         # 水的介电常数
    }
    # 添加一些调试标志
    physics._debug = True
    
    # 启用梯度
    coords.requires_grad = True
    
    # 打印更多调试信息
    print("\n调试信息:")
    print(f"coords形状: {coords.shape}, requires_grad: {coords.requires_grad}")
    print(f"predictions形状: {predictions.shape}")
    print(f"coords设备: {coords.device}")
    print(f"predictions设备: {predictions.device}")
    
    # 检查physics对象的属性
    print("\nphysics对象的部分属性:")
    for attr in ['device', 'boundary_conditions_weights']:
        if hasattr(physics, attr):
            print(f"{attr}: {getattr(physics, attr)}")
        else:
            print(f"{attr}: 不存在")
    
    # 测试compute_navier_stokes_residual方法（公共接口）
    print("\n测试公共接口compute_navier_stokes_residual...")
    try:
        # 检查是否存在compute_navier_stokes_residual方法
        if hasattr(physics, 'compute_navier_stokes_residual'):
            # 保存原始方法
            original_method = physics.compute_navier_stokes_residual
            
            # 定义一个包装方法来打印更多信息
            def wrapped_compute(*args, **kwargs):
                print("调用compute_navier_stokes_residual方法")
                print(f"参数1形状: {args[0].shape}")
                print(f"参数2形状: {args[1].shape}")
                # 调用原始方法
                result = original_method(*args, **kwargs)
                print("方法返回结果")
                return result
            
            # 替换方法
            physics.compute_navier_stokes_residual = wrapped_compute
        
        result = physics.compute_navier_stokes_residual(coords, predictions)
        print(f"公共接口返回类型: {type(result)}")
        if isinstance(result, dict):
            print("公共接口返回的残差:")
            for key, residual in result.items():
                if residual is not None:
                    print(f"{key}: 均值={residual.mean().item():.6f}, 标准差={residual.std().item():.6f}, 最大值={residual.abs().max().item():.6f}")
                    # 打印一些实际的残差值样本
                    if residual.numel() > 0:
                        print(f"  部分残差样本: {residual.flatten()[:5].tolist()}")
                else:
                    print(f"{key}: None")
    except Exception as e:
        print(f"\n调用公共接口时出错: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n测试完成!")

if __name__ == "__main__":
    test_physics_constraints()