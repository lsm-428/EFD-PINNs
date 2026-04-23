#!/usr/bin/env python3
"""
验证权重配置加载脚本
检查配置文件中的权重是否正确加载到模型中
"""

import json
import sys
sys.path.append('/home/scnu/Gitee/EFD3D')

from src.physics.constraints import PINNConstraintLayer

def verify_weights():
    """验证权重配置"""

    # 加载配置文件
    with open('/home/scnu/Gitee/EFD3D/config/v4.5-standard.json', 'r') as f:
        config = json.load(f)

    print("=" * 60)
    print("权重配置验证")
    print("=" * 60)

    # 检查配置文件中的权重
    if 'physics' in config and 'residual_weights' in config['physics']:
        print("✅ 配置文件包含 residual_weights")
        config_weights = config['physics']['residual_weights']

        # 检查关键权重
        key_weights = ['momentum_u', 'momentum_v', 'momentum_w', 'surface_tension']
        print("\n关键权重检查:")
        for key in key_weights:
            if key in config_weights:
                print(f"  {key}: {config_weights[key]}")
            else:
                print(f"  ❌ {key}: 未找到")

        # 创建约束层实例
        try:
            constraint_layer = PINNConstraintLayer(config=config)
            print(f"\n✅ PINNConstraintLayer 创建成功")

            # 检查加载的权重
            print("\n实际加载的权重:")
            for key in key_weights:
                if key in constraint_layer.residual_weights:
                    weight = constraint_layer.residual_weights[key]
                    config_weight = config_weights.get(key, 'N/A')
                    status = "✅" if abs(weight - config_weight) < 1e-6 else "❌"
                    print(f"  {key}: {weight} (配置: {config_weight}) {status}")

            # 检查ns_weight
            print(f"\n全局ns_weight: {config['physics'].get('ns_weight', 'N/A')}")
            print(f"全局surface_tension_weight: {config['physics'].get('surface_tension_weight', 'N/A')}")

            # 计算复合权重
            print("\n复合权重计算:")
            ns_weight = config['physics'].get('ns_weight', 1.0)
            for key in ['momentum_u', 'momentum_v', 'momentum_w']:
                residual_weight = constraint_layer.residual_weights[key]
                combined = residual_weight * ns_weight
                print(f"  {key}: {residual_weight} × {ns_weight} = {combined}")

        except Exception as e:
            print(f"❌ 创建 PINNConstraintLayer 失败: {e}")
            return False

    else:
        print("❌ 配置文件缺少 residual_weights")
        return False

    print("\n" + "=" * 60)
    return True

if __name__ == "__main__":
    success = verify_weights()
    sys.exit(0 if success else 1)