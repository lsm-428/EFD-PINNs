import numpy as np
import torch
import torch.nn as nn

class EWPINNOutputLayer:
    """
    电润湿显示像素PINN输出层
    实现24维物理输出特征的处理、归一化和约束检查
    """
    
    def __init__(self, device='cpu'):
        self.device = device
        self.feature_groups = {
            'basic_physics_fields': 8,     # 基础物理场变量
            'interface_properties': 6,     # 界面几何与特性
            'contact_line_dynamics': 6,    # 接触线动力学
            'performance_metrics': 4       # 性能与工程指标
        }
        self.total_features = sum(self.feature_groups.values())  # 24维
        
        # 阶段化实现标记
        self.stage = 1  # 1: 基础(12维), 2: 扩展(18维), 3: 完整(24维)
        
        # 特征物理范围配置
        self.physical_ranges = self._init_physical_ranges()
        
        # 特征名称列表
        self.feature_names = self._init_feature_names()
        
        # 基础物理常数 - 基于真实器件 (2025-11-29 修正)
        self.epsilon_0 = 8.854e-12  # 真空介电常数
        self.epsilon_r = 4.0        # SU-8相对介电常数 (修正)
        self.gamma = 0.072          # 油-水界面张力 (N/m) (修正)
        self.d = 0.4e-6             # 介电层厚度 (m) (修正)
        
        print(f"EWPINN输出层已初始化 - 支持{self.total_features}维输出")
    
    def _init_physical_ranges(self):
        """
        初始化各输出特征的物理范围
        """
        ranges = {}
        
        # 第1组：基础物理场变量 (8维)
        ranges.update({
            # 流体动力学场 (5维)
            'p': (-1000.0, 1000.0),        # 压力场 [Pa]
            'u': (-0.01, 0.01),            # X方向速度 [m/s]
            'v': (-0.01, 0.01),            # Y方向速度 [m/s]
            'w': (-0.001, 0.001),          # Z方向速度 [m/s]
            'vorticity_z': (-100.0, 100.0), # Z方向涡量 [1/s]
            
            # 电场变量 (3维)
            'phi': (0.0, 100.0),           # 电势分布 [V]
            'E_x': (-1e6, 1e6),            # X方向电场 [V/m]
            'E_z': (0.0, 2.5e8)            # Z方向电场 [V/m]
        })
        
        # 第2组：界面几何与特性 (6维)
        ranges.update({
            # 界面形状参数 (4维)
            'h': (1.8e-6, 7.05e-6),        # 界面高度场 [m]
            'kappa': (-10000.0, 10000.0),  # 界面曲率 [1/m]
            'interface_slope_x': (-0.5, 0.5), # 界面X方向斜率 [rad]
            'interface_slope_y': (-0.5, 0.5), # 界面Y方向斜率 [rad]
            
            # 接触角参数 (2维)
            'theta': (0.5, 2.5),           # 局部接触角 [rad]
            'theta_equilibrium': (0.5, 2.5) # 平衡接触角 [rad]
        })
        
        # 第3组：接触线动力学 (6维)
        ranges.update({
            # 接触线位置与形状 (3维)
            'r_cl': (5e-6, 45e-6),         # 接触线半径 [m]
            'cl_shape_param': (0.0, 1.0),  # 接触线形状参数
            'cl_curvature': (-5000.0, 5000.0), # 接触线曲率 [1/m]
            
            # 接触线运动学 (3维)
            'v_cl': (-0.001, 0.001),       # 接触线法向速度 [m/s]
            'theta_adv': (1.0, 2.5),       # 前进接触角 [rad]
            'theta_rec': (0.5, 2.0)        # 后退接触角 [rad]
        })
        
        # 第4组：性能与工程指标 (4维)
        ranges.update({
            # 动态响应指标 (2维)
            'tau_response': (0.001, 0.1),  # 响应时间常数 [s]
            'switching_speed': (0.0, 0.002), # 开关速度指标 [m/s]
            
            # 稳定性与效率指标 (2维)
            'stability_index': (0.0, 1.0), # 稳定性指标
            'energy_efficiency': (0.0, 1.0) # 能量效率
        })
        
        return ranges
    
    def _init_feature_names(self):
        """
        初始化各阶段的特征名称列表
        """
        # 按阶段组织的特征名称
        feature_names = {
            1: [
                # 阶段1: 基础输出层 (12维)
                # 基础物理场
                'p', 'u', 'v', 'w', 'phi', 'E_z',
                # 界面特性
                'h', 'kappa', 'theta',
                # 接触线动力学
                'r_cl', 'v_cl',
                # 性能指标
                'tau_response'
            ],
            2: [
                # 阶段2: 扩展输出层 (18维)
                # 基础物理场（增加2维）
                'p', 'u', 'v', 'w', 'phi', 'E_z', 'E_x', 'vorticity_z',
                # 界面特性（增加2维）
                'h', 'kappa', 'theta', 'interface_slope_x', 'interface_slope_y',
                # 接触线动力学（增加2维）
                'r_cl', 'v_cl', 'cl_shape_param', 'theta_equilibrium',
                # 性能指标
                'tau_response'
            ],
            3: [
                # 阶段3: 完整输出层 (24维)
                # 基础物理场
                'p', 'u', 'v', 'w', 'phi', 'E_z', 'E_x', 'vorticity_z',
                # 界面特性
                'h', 'kappa', 'theta', 'interface_slope_x', 'interface_slope_y', 'theta_equilibrium',
                # 接触线动力学（增加3维）
                'r_cl', 'v_cl', 'cl_shape_param', 'cl_curvature', 'theta_adv', 'theta_rec',
                # 性能指标（增加3维）
                'tau_response', 'switching_speed', 'stability_index', 'energy_efficiency'
            ]
        }
        
        return feature_names
    
    def set_implementation_stage(self, stage):
        """
        设置输出层实现阶段
        stage: 1(基础12维), 2(扩展18维), 3(完整24维)
        """
        if stage in [1, 2, 3]:
            self.stage = stage
            print(f"已设置为阶段{stage}实现，当前输出维度: {len(self.feature_names[stage])}")
        else:
            raise ValueError("阶段必须为1、2或3")
    
    def get_current_dim(self):
        """
        获取当前阶段的输出维度
        """
        return len(self.feature_names[self.stage])
    
    def get_feature_names(self, stage=None):
        """
        获取指定阶段的特征名称列表
        """
        if stage is None:
            stage = self.stage
        return self.feature_names[stage].copy()
    
    def normalize_output(self, output_tensor, stage=None):
        """
        归一化输出特征到[-1, 1]范围
        
        参数:
        - output_tensor: 原始输出张量，形状为[batch_size, output_dim]
        - stage: 输出阶段，默认为当前设置
        
        返回:
        - 归一化后的输出张量
        """
        if stage is None:
            stage = self.stage
        
        feature_names = self.get_feature_names(stage)
        normalized = torch.zeros_like(output_tensor)
        
        for i, feature_name in enumerate(feature_names):
            if feature_name in self.physical_ranges:
                min_val, max_val = self.physical_ranges[feature_name]
                # 线性归一化到[-1, 1]
                normalized[:, i] = 2 * (output_tensor[:, i] - min_val) / (max_val - min_val) - 1
                # 确保值在范围内
                normalized[:, i] = torch.clamp(normalized[:, i], -1.0, 1.0)
            else:
                # 未知特征保持不变
                normalized[:, i] = output_tensor[:, i]
        
        return normalized
    
    def denormalize_output(self, normalized_tensor, stage=None):
        """
        将归一化的输出反转换为物理量
        
        参数:
        - normalized_tensor: 归一化的输出张量，范围[-1, 1]
        - stage: 输出阶段，默认为当前设置
        
        返回:
        - 反归一化后的物理量张量
        """
        if stage is None:
            stage = self.stage
        
        feature_names = self.get_feature_names(stage)
        denormalized = torch.zeros_like(normalized_tensor)
        
        for i, feature_name in enumerate(feature_names):
            if feature_name in self.physical_ranges:
                min_val, max_val = self.physical_ranges[feature_name]
                # 从[-1, 1]映射回物理范围
                denormalized[:, i] = (normalized_tensor[:, i] + 1) * (max_val - min_val) / 2 + min_val
            else:
                # 未知特征保持不变
                denormalized[:, i] = normalized_tensor[:, i]
        
        return denormalized
    
    def calculate_derived_parameters(self, output_dict):
        """
        计算衍生参数
        
        参数:
        - output_dict: 包含基本输出参数的字典
        
        返回:
        - 包含衍生参数的字典
        """
        derived = {}
        
        # 接触角滞后
        if 'theta_adv' in output_dict and 'theta_rec' in output_dict:
            derived['contact_angle_hysteresis'] = output_dict['theta_adv'] - output_dict['theta_rec']
        
        # 接触线迁移率
        if 'v_cl' in output_dict and 'theta' in output_dict and 'theta_equilibrium' in output_dict:
            # 避免除零
            theta_diff = output_dict['theta'] - output_dict['theta_equilibrium']
            if isinstance(theta_diff, torch.Tensor):
                # 对于张量，使用where处理零值
                derived['contact_line_mobility'] = torch.where(
                    torch.abs(theta_diff) > 1e-10,
                    output_dict['v_cl'] / theta_diff,
                    torch.zeros_like(theta_diff)
                )
            else:
                # 对于标量
                if abs(theta_diff) > 1e-10:
                    derived['contact_line_mobility'] = output_dict['v_cl'] / theta_diff
                else:
                    derived['contact_line_mobility'] = 0.0
        
        # 电润湿数
        if 'phi' in output_dict or 'E_z' in output_dict:
            V = output_dict.get('phi', 0.0)  # 使用电势作为电压
            derived['electrowetting_number'] = (self.epsilon_0 * self.epsilon_r * V**2) / (2 * self.gamma * self.d)
        
        # 毛细压力
        if 'kappa' in output_dict:
            derived['capillary_pressure'] = self.gamma * output_dict['kappa']
        
        # 体积变化率
        if 'r_cl' in output_dict and 'v_cl' in output_dict and 'h' in output_dict:
            derived['volume_change_rate'] = 2 * np.pi * output_dict['r_cl'] * output_dict['v_cl'] * output_dict['h']
        
        return derived
    
    def check_physical_constraints(self, output_dict):
        """
        检查物理约束条件
        
        参数:
        - output_dict: 输出参数字典
        
        返回:
        - 约束满足情况字典
        - 违反约束的错误信息列表
        """
        constraints = {}
        errors = []
        
        # 硬约束检查
        
        # 1. 接触角范围约束
        if 'theta' in output_dict:
            theta = output_dict['theta']
            min_theta, max_theta = 0.3, 2.8  # 约17°到160°
            if isinstance(theta, torch.Tensor):
                is_valid = torch.all((theta > min_theta) & (theta < max_theta))
                if not is_valid:
                    errors.append(f"接触角超出有效范围: {theta.min().item():.2f} - {theta.max().item():.2f} rad")
            else:
                if theta <= min_theta or theta >= max_theta:
                    errors.append(f"接触角超出有效范围: {theta:.2f} rad")
            constraints['contact_angle'] = len(errors) == 0
        
        # 2. 电场边界约束 (避免击穿)
        E_breakdown = 3e8  # 空气击穿场强估计值 [V/m]
        if 'E_z' in output_dict:
            E_z = output_dict['E_z']
            if isinstance(E_z, torch.Tensor):
                is_valid = torch.all(E_z < E_breakdown)
                if not is_valid:
                    errors.append(f"电场强度超过击穿阈值: {E_z.max().item():.2e} V/m")
            else:
                if E_z >= E_breakdown:
                    errors.append(f"电场强度超过击穿阈值: {E_z:.2e} V/m")
            constraints['electric_field'] = len(errors) == 0
        
        # 3. 接触线位置约束
        if 'r_cl' in output_dict:
            r_cl = output_dict['r_cl']
            r_min, r_max = 1e-6, 50e-6  # 合理的接触线半径范围 [m]
            if isinstance(r_cl, torch.Tensor):
                is_valid = torch.all((r_cl > r_min) & (r_cl < r_max))
                if not is_valid:
                    errors.append(f"接触线半径超出有效范围: {r_cl.min().item():.2e} - {r_cl.max().item():.2e} m")
            else:
                if r_cl <= r_min or r_cl >= r_max:
                    errors.append(f"接触线半径超出有效范围: {r_cl:.2e} m")
            constraints['contact_line'] = len(errors) == 0
        
        # 4. 界面高度约束
        if 'h' in output_dict:
            h = output_dict['h']
            min_h, max_h = 1e-6, 10e-6  # 合理的界面高度范围 [m]
            if isinstance(h, torch.Tensor):
                is_valid = torch.all((h > min_h) & (h < max_h))
                if not is_valid:
                    errors.append(f"界面高度超出有效范围: {h.min().item():.2e} - {h.max().item():.2e} m")
            else:
                if h <= min_h or h >= max_h:
                    errors.append(f"界面高度超出有效范围: {h:.2e} m")
            constraints['interface_height'] = len(errors) == 0
        
        # 软约束检查 (优化目标)
        
        # 1. Young-Lippmann方程一致性检查
        if 'theta' in output_dict and 'phi' in output_dict:
            theta = output_dict['theta']
            V = output_dict['phi']
            theta_0 = 1.57  # 初始接触角 (默认90度)
            
            # Young-Lippmann方程: cos(theta) = cos(theta_0) + (ε₀εᵣ/(2γd))V²
            if isinstance(theta, torch.Tensor):
                cos_theta_pred = torch.cos(theta)
                cos_theta_theory = torch.cos(torch.tensor(theta_0, device=self.device)) + \
                                 (self.epsilon_0 * self.epsilon_r * V**2) / (2 * self.gamma * self.d)
                error = torch.mean((cos_theta_pred - cos_theta_theory)**2)
                constraints['young_lippmann_error'] = error.item()
            else:
                cos_theta_pred = np.cos(theta)
                cos_theta_theory = np.cos(theta_0) + (self.epsilon_0 * self.epsilon_r * V**2) / (2 * self.gamma * self.d)
                error = (cos_theta_pred - cos_theta_theory)**2
                constraints['young_lippmann_error'] = error
        
        return constraints, errors
    
    def create_output_dict(self, output_tensor, stage=None):
        """
        将输出张量转换为带名称的字典
        
        参数:
        - output_tensor: 输出张量
        - stage: 输出阶段
        
        返回:
        - 包含特征名称和值的字典
        """
        if stage is None:
            stage = self.stage
        
        feature_names = self.get_feature_names(stage)
        output_dict = {}
        
        # 确保张量在CPU上并转换为numpy
        if isinstance(output_tensor, torch.Tensor):
            output_tensor = output_tensor.detach().cpu().numpy()
        
        # 处理批量情况
        if output_tensor.ndim > 1:
            # 批量数据，返回列表字典
            batch_size = output_tensor.shape[0]
            batch_dicts = []
            
            for i in range(batch_size):
                sample_dict = {}
                for j, name in enumerate(feature_names):
                    sample_dict[name] = output_tensor[i, j]
                batch_dicts.append(sample_dict)
            
            return batch_dicts
        else:
            # 单个样本
            for i, name in enumerate(feature_names):
                output_dict[name] = output_tensor[i]
            
            return output_dict
    
    def get_physical_range(self, feature_name):
        """
        获取指定特征的物理范围
        """
        if feature_name in self.physical_ranges:
            return self.physical_ranges[feature_name]
        else:
            raise ValueError(f"未知特征名称: {feature_name}")
    
    def update_material_parameters(self, epsilon_r=None, gamma=None, d=None):
        """
        更新材料参数
        
        参数:
        - epsilon_r: 相对介电常数
        - gamma: 表面张力 [N/m]
        - d: 介电层厚度 [m]
        """
        if epsilon_r is not None:
            self.epsilon_r = epsilon_r
        if gamma is not None:
            self.gamma = gamma
        if d is not None:
            self.d = d
        
        print(f"材料参数已更新: εr={self.epsilon_r}, γ={self.gamma} N/m, d={self.d} m")
    
    def generate_random_output(self, batch_size=1, stage=None):
        """
        生成随机但物理合理的输出样本
        
        参数:
        - batch_size: 批量大小
        - stage: 输出阶段
        
        返回:
        - 随机输出张量
        """
        if stage is None:
            stage = self.stage
        
        feature_names = self.get_feature_names(stage)
        output_dim = len(feature_names)
        output = np.zeros((batch_size, output_dim))
        
        for i, name in enumerate(feature_names):
            if name in self.physical_ranges:
                min_val, max_val = self.physical_ranges[name]
                # 生成均匀分布的随机值
                output[:, i] = np.random.uniform(min_val, max_val, size=batch_size)
            else:
                # 默认生成[-1, 1]范围的随机值
                output[:, i] = np.random.uniform(-1.0, 1.0, size=batch_size)
        
        return torch.tensor(output, dtype=torch.float32, device=self.device)

# 演示代码
if __name__ == "__main__":
    # 创建输出层实例
    output_layer = EWPINNOutputLayer()
    
    # 测试不同阶段
    print("\n=== 阶段1: 基础输出层 (12维) ===")
    output_layer.set_implementation_stage(1)
    print(f"特征名称: {output_layer.get_feature_names()}")
    
    print("\n=== 阶段2: 扩展输出层 (18维) ===")
    output_layer.set_implementation_stage(2)
    print(f"特征名称: {output_layer.get_feature_names()}")
    
    print("\n=== 阶段3: 完整输出层 (24维) ===")
    output_layer.set_implementation_stage(3)
    print(f"特征名称: {output_layer.get_feature_names()}")
    
    # 生成随机输出样本
    batch_size = 2
    random_output = output_layer.generate_random_output(batch_size=batch_size)
    print(f"\n随机输出张量形状: {random_output.shape}")
    
    # 归一化测试
    normalized = output_layer.normalize_output(random_output)
    print(f"归一化后范围: [{normalized.min().item():.4f}, {normalized.max().item():.4f}]")
    
    # 反归一化测试
    denormalized = output_layer.denormalize_output(normalized)
    
    # 转换为字典格式
    output_dicts = output_layer.create_output_dict(denormalized)
    print(f"\n输出字典数量: {len(output_dicts)}")
    print(f"第一个样本的部分特征:")
    first_sample = output_dicts[0]
    for key in list(first_sample.keys())[:5]:  # 只显示前5个特征
        print(f"  {key}: {first_sample[key]:.6e}")
    
    # 计算衍生参数
    derived = output_layer.calculate_derived_parameters(first_sample)
    print(f"\n衍生参数:")
    for key, value in derived.items():
        print(f"  {key}: {value:.6e}")
    
    # 检查物理约束
    constraints, errors = output_layer.check_physical_constraints(first_sample)
    print(f"\n物理约束检查:")
    print(f"约束满足情况: {constraints}")
    if errors:
        print(f"违反约束: {errors}")
    else:
        print("所有硬约束均满足")
    
    print("\n输出层功能测试完成！")