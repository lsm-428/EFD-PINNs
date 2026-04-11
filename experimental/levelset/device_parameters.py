#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EWD 像素器件参数定义
====================

完整的电润湿显示像素单元几何和物理参数
基于 generate_pyvista_3d.py 和 PHYSICS 配置

用途:
1. 作为训练脚本的参数输入
2. 作为 COMSOL 多物理场仿真的参考
3. 确保所有物理模型使用一致的参数
"""

import numpy as np

# ============================================================================
# 器件几何参数
# ============================================================================

DEVICE_GEOMETRY = {
    # ========== 像素单元尺寸 ==========
    'pixel_inner_size': [174e-6, 174e-6],  # [Lx, Ly] 围堰内部尺寸 (m)
    'pixel_outer_size': [184e-6, 184e-6],  # 围堰外部尺寸 (m)
    'fluid_height': 20e-6,                 # Lz 流体区域高度 (m)

    # ========== 围堰 (Weir/墙壁) ==========
    'weir_height': 20e-6,                  # 围堰高度 (m)
    'weir_width': 5e-6,                    # 围堰厚度 (m)
    'weir_position': {
        'z_start': 0.0,                    # 围堰底部 z 坐标
        'z_end': 20e-6,                    # 围堰顶部 z 坐标 (与流体高度相同)
    },

    # ========== 分层结构 (从下到上) ==========
    'layers': {
        # Layer 1: 底部 ITO 电极
        'ITO_bottom': {
            'thickness': 27.5e-9,           # 27.5 nm
            'material': 'ITO',
            'function': 'bottom_electrode',
            'voltage': 'V_applied',          # 施加电压
        },

        # Layer 2: 介电层
        'dielectric': {
            'thickness': 0.4e-6,            # 0.4 μm = 400 nm
            'material': 'SU-8',              # SU-8 光刻胶
            'epsilon_r': 3.0,                # 介电常数
            'breakdown_voltage': None,      # 未知，但30V安全
        },

        # Layer 3: 疏水层
        'hydrophobic': {
            'thickness': 0.4e-6,            # 0.4 μm = 400 nm
            'material': 'Teflon AF',
            'epsilon_r': 1.9,
            'contact_angle_0': 120.0,       # 初始接触角 (度)
        },

        # Layer 4: 油墨 (流体相 1)
        'ink': {
            'initial_thickness': 3e-6,      # 3 μm
            'material': '癸烷 (Decane)',     # 主要成分癸烷
            'color': 'red/black',           # 不透明
        },

        # Layer 5: 极性液体 (流体相 2)
        'polar_liquid': {
            'initial_thickness': 17e-6,     # 17 μm
            'material': '去离子水 (1MΩ·cm纯水)',  # 1MΩ 纯水
            'color': 'transparent',
        },

        # Layer 6: 顶部 ITO 电极
        'ITO_top': {
            'thickness': 27.5e-9,           # 27.5 nm
            'material': 'ITO',
            'function': 'top_electrode',
            'voltage': 0.0,                 # 接地
        },
    },
}

# ============================================================================
# 初始状态参数
# ============================================================================

INITIAL_STATE = {
    # 电压
    'voltage_initial': 0.0,                # V (初始电压)
    'voltage_min': 0.0,
    'voltage_max': 30.0,
    'voltage_threshold': 3.0,              # V_T 阈值电压 (暂定)

    # 油墨分布
    'ink': {
        'thickness': 3e-6,                 # 3 μm
        'volume': 174e-6 * 174e-6 * 3e-6,  # V_ink = 9.0828e-15 m³ (9.08 pL)
        'distribution': 'full_bottom',     # 完全覆盖底部
        'aperture_ratio': 0.0,             # η = 0% (无开口)
    },

    # 极性液体分布
    'polar': {
        'thickness': 17e-6,                # 17 μm
        'volume': 174e-6 * 174e-6 * 17e-6, # V_polar = 5.1469e-14 m³ (51.47 pL)
        'distribution': 'above_ink',       # 在油墨之上
    },

    # 界面位置 (Level Set)
    'interface': {
        'initial_position': 'z = 3e-6',    # ψ = 0 的位置
        'psi_field': 'z - h_ink',          # 初始 ψ 场公式
    },

    # 接触角
    'contact_angle': {
        'initial': 120.0,                  # θ₀ (度)
        'hysteresis': 10.0,                # Δθ = 10° 接触角滞后范围
        'advancing': 125.0,                # θ_adv = 125° 前进接触角
        'receding': 115.0,                 # θ_rec = 115° 后退接触角
    },
}

# ============================================================================
# 材料物理属性
# ============================================================================

MATERIAL_PROPERTIES = {
    # ========== Phase 1: 油墨 (癸烷) ==========
    'ink': {
        'density': 730.0,                  # ρ₁ (kg/m³) 癸烷密度@20°C
        'viscosity': 0.00092,              # μ₁ (Pa·s) 癸烷粘度@20°C
        'epsilon_r': 2.0,                 # 相对介电常数 (癸烷)
        'conductivity': 1e-14,             # 电导率 (S/m) 绝缘
        'surface_tension': 0.023,         # 表面张力 (N/m) 癸烷-空气
        'refractive_index': 1.41,          # 折射率
    },

    # ========== Phase 2: 极性液体 (去离子水) ==========
    'polar': {
        'density': 997.0,                  # ρ₂ (kg/m³) 水@25°C
        'viscosity': 0.00089,              # μ₂ (Pa·s) 水@25°C
        'epsilon_r': 78.4,                 # 相对介电常数 (水@25°C)
        'conductivity': 1e-6,               # 电导率 (S/m) 1MΩ·cm纯水
        'surface_tension': 0.072,         # 表面张力 (N/m) 水-空气@25°C
        'refractive_index': 1.333,          # 折射率
    },

    # ========== 界面属性 ==========
    'interface': {
        'ink_polar_tension': 0.045,        # γ₁₂ (N/m) 油墨-极性液体界面张力
        'ink_air_tension': 0.023,          # γ₁₀ = 0.023 N/m (癸烷-空气)
        'polar_air_tension': 0.072,        # γ₂₀ = 0.072 N/m (水-空气)
    },

    # ========== 固体壁面 ==========
    'wall': {
        'hydrophobic_layer': {
            'material': 'Teflon AF',
            'contact_angle_ink': 120.0,    # θ₀ (度) 对油墨
            'contact_angle_polar': 110.0,  # θ_polar = 110° 对极性液体
            'surface_energy': 0.018,       # σ_s = 18 mN/m 表面能 (Teflon AF)
            'roughness': 10e-9,            # R_a = 10 nm 表面粗糙度
        },
        'weir_material': 'SU-8',           # 围堰材料: SU-8 光刻胶
        'electrode_material': 'ITO',       # ITO 透明电极
    },
}

# ============================================================================
# 电学参数
# ============================================================================

ELECTRICAL_PROPERTIES = {
    # ========== 基础常数 ==========
    'epsilon_0': 8.854e-12,               # 真空介电常数 (F/m)

    # ========== 介电层参数 ==========
    'dielectric': {
        'epsilon_r': 3.0,                   # εᵣ (SU-8)
        'thickness': 0.4e-6,               # d = 0.4 μm
        'breakdown_field': 1e8,            # E_breakdown ≈ 100 MV/m (SU-8 典型值)
    },

    # ========== 电压参数 ==========
    'voltage': {
        'min': 0.0,
        'max': 30.0,
        'threshold': 3.0,                  # V_T 阈值电压 (暂定)
        'frequency': 0.0,                  # DC = 直流
        'waveform': 'DC',                  # 波形 (直流)
    },

    # ========== 电极参数 ==========
    'electrodes': {
        'bottom': {
            'material': 'ITO',
            'resistance': 10.0,            # R ≈ 10 Ω/sq (方块电阻)
            'transparency': 0.85,          # 85% 透光率
            'thickness': 27.5e-9,          # 27.5 nm ITO 厚度
        },
        'top': {
            'material': 'ITO',
            'resistance': 10.0,            # R ≈ 10 Ω/sq (方块电阻)
            'transparency': 0.85,          # 85% 透光率
            'thickness': 27.5e-9,          # 27.5 nm ITO 厚度
        },
    },
}

# ============================================================================
# 边界条件
# ============================================================================

BOUNDARY_CONDITIONS = {
    # ========== 底部 (z = 0) ==========
    'bottom': {
        'location': 'z = 0',
        'type': 'wall',                    # 固体壁面
        'velocity': 'no_slip',             # u = v = w = 0
        'pressure': 'zero_gradient',       # ∂p/∂z = 0
        'level_set': None,                 # TODO: 接触角边界条件?
        'electric_potential': 'V_applied', # V = V_applied
        'temperature': None,               # TODO: 温度边界条件
    },

    # ========== 顶部 (z = Lz) ==========
    'top': {
        'location': 'z = 20e-6',
        'type': 'wall',                    # 固体顶板
        'velocity': 'no_slip',             # u = v = w = 0
        'pressure': 'fixed',               # p = 0 (参考压力)
        'level_set': 'zero_gradient',      # ∂ψ/∂z = 0
        'electric_potential': 'ground',    # V = 0
        'temperature': None,               # TODO: 温度边界条件
    },

    # ========== 四周 (围堰) ==========
    'sides': {
        'location': 'x=0, x=Lx, y=0, y=Ly',
        'type': 'wall',                    # 围堰壁面
        'velocity': 'no_slip',             # u = v = w = 0
        'pressure': 'zero_gradient',       # ∂p/∂n = 0
        'level_set': 'no_flux',            # n·∇ψ = 0
        'electric_potential': 'insulation', # ∂V/∂n = 0
        'symmetry': None,                  # TODO: 是否需要对称边界?
    },

    # ========== 初始条件 (t = 0) ==========
    'initial': {
        'voltage': 0.0,
        'velocity_field': 'zero',          # u = v = w = 0
        'pressure_field': 'hydrostatic',   # TODO: 静水压力?
        'level_set': 'z - h_ink',          # ψ = z - 3μm
        'interface_position': 'z = 3e-6',  # 界面位置
    },
}

# ============================================================================
# 电润湿参数 (Young-Lippmann 方程)
# ============================================================================

ELECTROWETTING = {
    # Young-Lippmann 方程参数
    'theta_0': 120.0,                     # 初始接触角 (度)
    'epsilon_0': 8.854e-12,               # F/m
    'epsilon_r': 3.0,                     # 介电层介电常数 (SU-8)
    'd': 0.4e-6,                          # 介电层厚度 (m)
    'gamma': 0.045,                       # 界面张力 γ₁₂ (N/m)
    'V_T': 3.0,                           # 阈值电压 (V) 暂定

    # 动态响应参数
    'response_time': 5e-3,                # τ = 5ms 特征响应时间
    'damping_ratio': 0.7,                 # ζ = 0.7 阻尼比 (欠阻尼系统)
    'time_constant': 5e-3,                # τ = 5ms 时间常数

    # 接触角滞后
    'hysteresis': {
        'enabled': True,                   # 启用接触角滞后模型
        'advancing_angle': 125.0,          # θ_adv = 125° 前进角 (油墨前进)
        'receding_angle': 115.0,            # θ_rec = 115° 后退角 (油墨后退)
    },
}

# ============================================================================
# 数值仿真参数
# ============================================================================

SIMULATION_PARAMETERS = {
    # 空间离散化
    'mesh': {
        'nx': None,                        # TODO: x 方向网格数
        'ny': None,                        # TODO: y 方向网格数
        'nz': None,                        # TODO: z 方向网格数
        'refinement_interface': None,      # TODO: 界面附近网格加密
    },

    # 时间离散化
    'time': {
        't_max': 0.05,                     # 最大时间 50 ms
        'dt': None,                        # TODO: 时间步长
        'time_steps': None,                # TODO: 时间步数
    },

    # Level Set 参数
    'level_set': {
        'epsilon': 0.01,                   # 界面厚度参数
        'reinitialization': None,          # TODO: 是否需要重新初始化?
        'mass_correction': None,           # TODO: 质量修正?
    },
}

# ============================================================================
# 性能指标
# ============================================================================

PERFORMANCE_TARGETS = {
    # 开口率 (实验值)
    'aperture_ratio': {
        '0V': 0.0,                         # 0%
        '5V': 0.15,                        # 15% (估算)
        '10V': 0.35,                       # 35% (实验值)
        '15V': 0.52,                       # 52% (实验值)
        '20V': 0.667,                      # 66.7% (实验值)
        '25V': 0.78,                       # 78% (实验值)
        '30V': 0.844,                      # 84.4% (实验值)
        'max_theoretical': 0.85,           # 85% (理论最大值)
    },

    # 响应时间
    'response_time': {
        'rise_time': 5.0,                  # t_rise ≈ 5ms 上升时间
        'fall_time': 8.0,                  # t_fall ≈ 8ms 下降时间 (较慢)
        'settle_time': 15.0,               # t_settle ≈ 15ms 稳定时间
    },

    # 能量消耗
    'power': {
        'capacitance': 1.0e-10,            # C ≈ 100 pF (像素电容)
        'energy_per_switch': 4.5e-8,       # E = 0.5*C*V² = 45 nJ @ 30V
        'power_per_pixel': 1.5e-3,         # P = 1.5 mW @ 30V (假设 1kHz 刷新)
    },
}

# ============================================================================
# 导出 PHYSICS 字典 (兼容性层)
# ============================================================================

PHYSICS = {
    'Lx': DEVICE_GEOMETRY['pixel_inner_size'][0],
    'Ly': DEVICE_GEOMETRY['pixel_inner_size'][1],
    'Lz': DEVICE_GEOMETRY['fluid_height'],
    'h_ink': INITIAL_STATE['ink']['thickness'],
    'rho_oil': MATERIAL_PROPERTIES['ink']['density'],
    'mu_oil': MATERIAL_PROPERTIES['ink']['viscosity'],
    'rho_polar': MATERIAL_PROPERTIES['polar']['density'],
    'mu_polar': MATERIAL_PROPERTIES['polar']['viscosity'],
    'epsilon_0': ELECTRICAL_PROPERTIES['epsilon_0'],
    'epsilon_r': ELECTRICAL_PROPERTIES['dielectric']['epsilon_r'],
    'd_dielectric': ELECTRICAL_PROPERTIES['dielectric']['thickness'],
    'V_threshold': ELECTRICAL_PROPERTIES['voltage']['threshold'],
    'V_max': ELECTRICAL_PROPERTIES['voltage']['max'],
    't_max': SIMULATION_PARAMETERS['time']['t_max'],
    'eta_max': PERFORMANCE_TARGETS['aperture_ratio']['max_theoretical']
}

# ============================================================================
# 导出函数
# ============================================================================

def get_geometry_summary():
    """返回几何参数摘要"""
    return {
        '计算域尺寸 (m)': [DEVICE_GEOMETRY['pixel_inner_size'][0],
                            DEVICE_GEOMETRY['pixel_inner_size'][1],
                            DEVICE_GEOMETRY['fluid_height']],
        '围堰高度 (m)': DEVICE_GEOMETRY['weir_height'],
        '围堰厚度 (m)': DEVICE_GEOMETRY['weir_width'],
        '油墨厚度 (m)': INITIAL_STATE['ink']['thickness'],
        '极性液体厚度 (m)': INITIAL_STATE['polar']['thickness'],
    }

def get_material_summary():
    """返回材料属性摘要"""
    return {
        '油墨': {
            '密度 (kg/m³)': MATERIAL_PROPERTIES['ink']['density'],
            '粘度 (Pa·s)': MATERIAL_PROPERTIES['ink']['viscosity'],
            '介电常数': MATERIAL_PROPERTIES['ink']['epsilon_r'],
        },
        '极性液体': {
            '密度 (kg/m³)': MATERIAL_PROPERTIES['polar']['density'],
            '粘度 (Pa·s)': MATERIAL_PROPERTIES['polar']['viscosity'],
            '介电常数': MATERIAL_PROPERTIES['polar']['epsilon_r'],
        },
    }

def validate_parameters():
    """验证参数完整性，返回缺失参数列表"""
    missing = []

    # 检查关键参数
    if ELECTRICAL_PROPERTIES['dielectric']['breakdown_field'] is None:
        missing.append('介电层击穿场强')
    if ELECTROWETTING['response_time'] is None:
        missing.append('电润湿响应时间 τ')
    if ELECTROWETTING['damping_ratio'] is None:
        missing.append('阻尼比 ζ')

    return missing

def print_missing_parameters():
    """打印缺失的参数"""
    missing = validate_parameters()
    if missing:
        print("⚠️  以下参数需要补充:")
        for i, param in enumerate(missing, 1):
            print(f"  {i}. {param}")
    else:
        print("✅ 所有参数已完整定义")

    return missing

# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("EWD 像素器件参数定义")
    print("=" * 70)

    print("\n📐 几何参数:")
    geo = get_geometry_summary()
    for key, val in geo.items():
        if isinstance(val, list):
            print(f"  {key}: {val}")
        else:
            print(f"  {key}: {val:.2e}")

    print("\n🧪 材料属性:")
    mat = get_material_summary()
    for phase, props in mat.items():
        print(f"  {phase}:")
        for key, val in props.items():
            print(f"    {key}: {val}")

    print("\n⚠️  参数完整性检查:")
    missing = print_missing_parameters()

    if not missing:
        print("\n✅ 所有参数已定义，可以直接用于训练！")
    else:
        print(f"\n❌ 缺少 {len(missing)} 个参数，请补充后使用")

