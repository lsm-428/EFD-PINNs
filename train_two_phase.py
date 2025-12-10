#!/usr/bin/env python3
"""
第三阶段：两相流 PINN 训练
==========================

完整流场求解：Navier-Stokes + VOF 界面追踪

使用方法:
    python train_two_phase.py --epochs 30000
    python train_two_phase.py --config config/two_phase_config.json

作者: EFD-PINNs Team
"""

import sys

if __name__ == "__main__":
    # 导入并运行两相流训练
    from src.models.pinn_two_phase import main
    
    if len(sys.argv) == 1:
        print("=" * 60)
        print("第三阶段：两相流 PINN 训练")
        print("=" * 60)
        print("\n物理方程:")
        print("  • 连续性：∇·u = 0")
        print("  • VOF：∂φ/∂t + u·∇φ = 0")
        print("  • N-S：ρ(∂u/∂t + u·∇u) = -∇p + μ∇²u + F_st")
        print("\n使用方法:")
        print("  python train_two_phase.py --epochs 30000")
        print("  python train_two_phase.py --config config/two_phase_config.json")
        print("=" * 60)
    
    main()
