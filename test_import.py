#!/usr/bin/env python3
"""
测试脚本，用于验证循环导入问题是否已解决
"""

import os
import sys

# 确保仓库根目录在路径中
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

def test_import_ewp_pinn_physics():
    """测试导入 ewp_pinn_physics 模块"""
    try:
        from ewp_pinn_physics import PINNConstraintLayer
        print("✓ 成功导入 PINNConstraintLayer")
        
        # 测试创建实例
        layer = PINNConstraintLayer()
        print("✓ 成功创建 PINNConstraintLayer 实例")
        
        return True
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False
    except Exception as e:
        print(f"✗ 其他错误: {e}")
        return False

def test_import_ewp_pinn_dynamic_weight():
    """测试导入 ewp_pinn_dynamic_weight 模块"""
    try:
        from ewp_pinn_dynamic_weight import DynamicPhysicsWeightScheduler
        print("✓ 成功导入 DynamicPhysicsWeightScheduler")
        
        # 测试创建实例
        scheduler = DynamicPhysicsWeightScheduler()
        print("✓ 成功创建 DynamicPhysicsWeightScheduler 实例")
        
        return True
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False
    except Exception as e:
        print(f"✗ 其他错误: {e}")
        return False

def test_import_generate_constraint_report():
    """测试导入 generate_constraint_report 模块"""
    try:
        # 测试导入脚本中的函数
        from scripts.generate_constraint_report import load_config, build_model_for_checkpoint
        print("✓ 成功导入 generate_constraint_report 模块函数")
        
        return True
    except ImportError as e:
        print(f"✗ 导入失败: {e}")
        return False
    except Exception as e:
        print(f"✗ 其他错误: {e}")
        return False

def main():
    """主测试函数"""
    print("开始测试循环导入问题...")
    print("=" * 50)
    
    results = []
    
    # 测试1: 导入 ewp_pinn_physics
    print("\n1. 测试导入 ewp_pinn_physics 模块:")
    results.append(test_import_ewp_pinn_physics())
    
    # 测试2: 导入 ewp_pinn_dynamic_weight
    print("\n2. 测试导入 ewp_pinn_dynamic_weight 模块:")
    results.append(test_import_ewp_pinn_dynamic_weight())
    
    # 测试3: 导入 generate_constraint_report
    print("\n3. 测试导入 generate_constraint_report 模块:")
    results.append(test_import_generate_constraint_report())
    
    print("\n" + "=" * 50)
    print("测试结果汇总:")
    
    if all(results):
        print("✓ 所有导入测试通过！循环导入问题已解决。")
        return True
    else:
        print("✗ 部分导入测试失败。")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)