#!/usr/bin/env python3
"""
配置文件比较脚本
对比v4.5和v4.6配置的差异
"""

import json
import sys

def compare_configs():
    """比较两个配置文件"""

    # 加载配置文件
    with open('/home/scnu/Gitee/EFD3D/config/v4.5-standard.json', 'r') as f:
        config_v45 = json.load(f)

    with open('/home/scnu/Gitee/EFD3D/config/v4.6-momentum-optimized.json', 'r') as f:
        config_v46 = json.load(f)

    print("=" * 70)
    print("EFD3D 配置文件对比: v4.5-standard vs v4.6-momentum-optimized")
    print("=" * 70)

    # 基本信息对比
    print("\n📋 基本信息:")
    print(f"  v4.5: {config_v45['metadata']['description']}")
    print(f"  v4.6: {config_v46['metadata']['description']}")

    # 物理权重对比
    if 'physics' in config_v45 and 'physics' in config_v46:
        print("\n🔧 物理参数对比:")

        # 全局权重对比
        physics_v45 = config_v45['physics']
        physics_v46 = config_v46['physics']

        global_params = ['ns_weight', 'surface_tension_weight']
        for param in global_params:
            v45_val = physics_v45.get(param, 'N/A')
            v46_val = physics_v46.get(param, 'N/A')
            if v45_val != v46_val:
                ratio = v46_val / v45_val if isinstance(v45_val, (int, float)) and v45_val != 0 else 0
                print(f"  {param}: {v45_val} → {v46_val} ({ratio:.1f}×)")
            else:
                print(f"  {param}: {v45_val} (未变)")

        # 残差权重对比
        weights_v45 = physics_v45.get('residual_weights', {})
        weights_v46 = physics_v46.get('residual_weights', {})

        if weights_v45 or weights_v46:
            print("\n⚖️  残差权重对比:")

            # 检查关键权重变化
            key_weights = [
                'momentum_u', 'momentum_v', 'momentum_w',
                'surface_tension', 'continuity', 'vof'
            ]

            for key in key_weights:
                v45_val = weights_v45.get(key, 'N/A')
                v46_val = weights_v46.get(key, 'N/A')

                if isinstance(v45_val, (int, float)) and isinstance(v46_val, (int, float)):
                    if v45_val != v46_val:
                        ratio = v46_val / v45_val if v45_val != 0 else float('inf')
                        print(f"  {key}: {v45_val} → {v46_val} ({ratio:.1f}×)")
                    else:
                        print(f"  {key}: {v45_val} (未变)")

            # 计算复合权重变化
            print("\n📈 复合权重变化 (residual × global):")
            ns_weight_v45 = physics_v45.get('ns_weight', 1.0)
            ns_weight_v46 = physics_v46.get('ns_weight', 1.0)

            for key in ['momentum_u', 'momentum_v', 'momentum_w']:
                v45_residual = weights_v45.get(key, 1.0)
                v46_residual = weights_v46.get(key, 1.0)

                v45_combined = v45_residual * ns_weight_v45
                v46_combined = v46_residual * ns_weight_v46

                if v45_combined != 0:
                    ratio = v46_combined / v45_combined
                    print(f"  {key}: {v45_combined:.3f} → {v46_combined:.3f} ({ratio:.1f}×)")

    # 训练参数对比
    if 'training' in config_v45 and 'training' in config_v46:
        print("\n🏋️  训练参数对比:")
        train_v45 = config_v45['training']
        train_v46 = config_v46['training']

        train_params = ['epochs', 'stage2_epochs', 'stage3_epochs', 'learning_rate']
        for param in train_params:
            v45_val = train_v45.get(param, 'N/A')
            v46_val = train_v46.get(param, 'N/A')
            if v45_val != v46_val:
                print(f"  {param}: {v45_val} → {v46_val}")
            else:
                print(f"  {param}: {v45_val} (未变)")

    print("\n" + "=" * 70)
    print("💡 总结:")
    print("  v4.5: 基础配置，已验证收敛")
    print("  v4.6: 优化动量权重，预期改善流场学习")
    print("=" * 70)

if __name__ == "__main__":
    compare_configs()