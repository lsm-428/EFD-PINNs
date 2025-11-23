import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from ewp_pinn_input_layer import EWPINNInputLayer
from ewp_pinn_output_layer import EWPINNOutputLayer

class EWPINN(nn.Module):
    """
    电润湿显示像素PINN模型
    整合输入层、神经网络和输出层
    """
    
    def __init__(self, input_dim=62, output_dim=24, hidden_dims=[128, 128, 128], device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(EWPINN, self).__init__()
        
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 初始化输入层和输出层
        self.input_layer = EWPINNInputLayer(device=device)
        self.output_layer = EWPINNOutputLayer(device=device)
        
        # 构建神经网络
        layers = []
        prev_dim = input_dim
        
        # 隐藏层
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers).to(device)
        
        # 初始化权重
        self._initialize_weights()
        
        print(f"EWPINN模型已初始化 - 设备: {device}, 输入维度: {input_dim}, 输出维度: {output_dim}")
    
    def _initialize_weights(self):
        """
        初始化网络权重
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        前向传播
        """
        return self.model(x)
    
    def physical_loss(self, inputs, outputs, output_dict):
        """
        计算物理约束损失
        
        参数:
        - inputs: 输入张量
        - outputs: 输出张量
        - output_dict: 输出参数字典
        
        返回:
        - 物理损失值
        """
        losses = []
        
        # 1. Young-Lippmann方程约束
        if 'theta' in output_dict and 'phi' in output_dict:
            theta = torch.tensor(output_dict['theta'], device=self.device)
            phi = torch.tensor(output_dict['phi'], device=self.device)
            theta_0 = 1.57  # 90度
            
            cos_theta_pred = torch.cos(theta)
            cos_theta_theory = torch.cos(torch.tensor(theta_0, device=self.device)) + \
                             (self.output_layer.epsilon_0 * self.output_layer.epsilon_r * phi**2) / \
                             (2 * self.output_layer.gamma * self.output_layer.d)
            
            yl_loss = nn.MSELoss()(cos_theta_pred, cos_theta_theory)
            losses.append(yl_loss)
        
        # 2. 接触角范围约束
        if 'theta' in output_dict:
            theta = torch.tensor(output_dict['theta'], device=self.device)
            min_theta = torch.tensor(0.3, device=self.device)
            max_theta = torch.tensor(2.8, device=self.device)
            
            # 使用软约束处理范围
            theta_penalty = torch.mean(torch.relu(min_theta - theta) + torch.relu(theta - max_theta))
            losses.append(theta_penalty * 10.0)  # 权重惩罚
        
        # 3. 电场约束
        if 'E_z' in output_dict:
            E_z = torch.tensor(output_dict['E_z'], device=self.device)
            E_breakdown = torch.tensor(3e8, device=self.device)
            
            e_field_penalty = torch.mean(torch.relu(E_z - E_breakdown))
            losses.append(e_field_penalty * 0.1)  # 较小的权重
        
        # 4. 接触线动力学约束
        if 'v_cl' in output_dict and 'theta' in output_dict and 'theta_equilibrium' in output_dict:
            v_cl = torch.tensor(output_dict['v_cl'], device=self.device)
            theta = torch.tensor(output_dict['theta'], device=self.device)
            theta_eq = torch.tensor(output_dict['theta_equilibrium'], device=self.device)
            
            # 接触线速度应与接触角差异相关
            theta_diff = theta - theta_eq
            cl_dynamics_loss = nn.MSELoss()(v_cl, theta_diff * 1e-3)  # 比例因子
            losses.append(cl_dynamics_loss)
        
        if losses:
            return sum(losses)
        else:
            return torch.tensor(0.0, device=self.device)
    
    def train_model(self, dataloader, epochs=1000, lr=1e-4, lambda_phys=0.1):
        """
        训练模型
        
        参数:
        - dataloader: 数据加载器
        - epochs: 训练轮数
        - lr: 学习率
        - lambda_phys: 物理损失权重
        """
        optimizer = optim.Adam(self.parameters(), lr=lr)
        mse_loss = nn.MSELoss()
        
        self.train()
        
        for epoch in range(epochs):
            total_loss = 0.0
            total_data_loss = 0.0
            total_phys_loss = 0.0
            
            for batch_inputs, batch_targets in dataloader:
                batch_inputs = batch_inputs.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                optimizer.zero_grad()
                
                # 前向传播
                outputs = self(batch_inputs)
                
                # 数据损失
                data_loss = mse_loss(outputs, batch_targets)
                
                # 物理损失
                batch_output_dict = self.output_layer.create_output_dict(outputs)
                
                # 简化物理损失计算，仅使用输出字典
                phys_loss = 0.0
                for i in range(len(batch_output_dict)):
                    phys_loss += self.physical_loss(
                        batch_inputs[i],
                        outputs[i],
                        batch_output_dict[i]
                    )
                phys_loss = phys_loss / len(batch_input_dict)
                
                # 总损失
                loss = data_loss + lambda_phys * phys_loss
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_data_loss += data_loss.item()
                total_phys_loss += phys_loss.item()
            
            # 打印训练进度
            if (epoch + 1) % 100 == 0:
                avg_loss = total_loss / len(dataloader)
                avg_data_loss = total_data_loss / len(dataloader)
                avg_phys_loss = total_phys_loss / len(dataloader)
                print(f"Epoch {epoch+1}/{epochs}, 总损失: {avg_loss:.6f}, 数据损失: {avg_data_loss:.6f}, 物理损失: {avg_phys_loss:.6f}")
    
    def predict(self, inputs, stage=None):
        """
        预测函数
        
        参数:
        - inputs: 输入张量
        - stage: 输出阶段
        
        返回:
        - 预测输出张量
        - 输出字典
        """
        self.eval()
        with torch.no_grad():
            inputs_tensor = torch.tensor(inputs, dtype=torch.float32).to(self.device)
            
            # 确保输入形状正确
            if inputs_tensor.ndim == 1:
                inputs_tensor = inputs_tensor.unsqueeze(0)
            
            outputs = self(inputs_tensor)
            
            # 转换为字典
            output_dicts = self.output_layer.create_output_dict(outputs, stage=stage)
            
            return outputs, output_dicts
    
    def validate_physical_constraints(self, inputs, stage=None):
        """
        验证物理约束
        
        参数:
        - inputs: 输入数据
        - stage: 输出阶段
        
        返回:
        - 约束检查结果
        - 错误信息
        """
        outputs, output_dicts = self.predict(inputs, stage=stage)
        
        all_constraints = []
        all_errors = []
        
        for output_dict in output_dicts:
            constraints, errors = self.output_layer.check_physical_constraints(output_dict)
            all_constraints.append(constraints)
            all_errors.extend(errors)
        
        return all_constraints, all_errors

class EWPINNDataset(torch.utils.data.Dataset):
    """
    电润湿显示像素PINN数据集
    """
    
    def __init__(self, input_data, target_data):
        self.input_data = torch.tensor(input_data, dtype=torch.float32)
        self.target_data = torch.tensor(target_data, dtype=torch.float32)
    
    def __len__(self):
        return len(self.input_data)
    
    def __getitem__(self, idx):
        return self.input_data[idx], self.target_data[idx]

# 演示整合系统的使用
def demo_integrated_system():
    """
    演示整合的电润湿显示像素PINN系统
    """
    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 初始化整合模型
    pinn_model = EWPINN(
        input_dim=62,  # 完整输入层维度
        output_dim=24, # 完整输出层维度
        hidden_dims=[128, 128, 128],
        device=device
    )
    
    # 设置阶段
    pinn_model.input_layer.set_implementation_stage(3)  # 完整输入层
    pinn_model.output_layer.set_implementation_stage(3)  # 完整输出层
    
    print(f"\n输入层维度: {pinn_model.input_layer.get_current_dim()}")
    print(f"输出层维度: {pinn_model.output_layer.get_current_dim()}")
    
    # 生成示例数据
    batch_size = 2
    
    # 生成示例输入
    random_inputs = []
    for _ in range(batch_size):
        # 生成示例输入字典
        input_dict = pinn_model.input_layer.generate_example_input()
        # 转换为输入向量并转为张量
        input_vector = torch.tensor(pinn_model.input_layer.create_input_vector(input_dict), dtype=torch.float32, device=device)
        random_inputs.append(input_vector)
    random_inputs = torch.stack(random_inputs)
    print(f"\n示例输入形状: {random_inputs.shape}")
    
    # 生成随机目标输出
    random_targets = pinn_model.output_layer.generate_random_output(batch_size=batch_size)
    print(f"随机目标形状: {random_targets.shape}")
    
    # 创建数据集和数据加载器
    dataset = EWPINNDataset(random_inputs.cpu().numpy(), random_targets.cpu().numpy())
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    
    # 前向传播测试
    print("\n前向传播测试:")
    with torch.no_grad():
        outputs = pinn_model(random_inputs)
        print(f"模型输出形状: {outputs.shape}")
    
    # 转换输出为字典格式
    output_dicts = pinn_model.output_layer.create_output_dict(outputs)
    
    # 由于无法从张量重建输入字典，我们重新生成一个用于展示
    demo_input_dict = pinn_model.input_layer.generate_example_input()
    print(f"\n演示输入特征:")
    key_inputs = ['V_applied', 'frequency', 'r_pixel', 'gamma_ow']
    for key in key_inputs:
        if key in demo_input_dict:
            print(f"  {key}: {demo_input_dict[key]:.4e}")
    
    print(f"\n第一个样本的关键输出特征:")
    key_outputs = ['p', 'theta', 'r_cl', 'E_z']
    for key in key_outputs:
        if key in output_dicts[0]:
            print(f"  {key}: {output_dicts[0][key]:.4e}")
    
    # 计算衍生参数
    derived = pinn_model.output_layer.calculate_derived_parameters(output_dicts[0])
    print(f"\n衍生参数:")
    for key, value in derived.items():
        print(f"  {key}: {value:.4e}")
    
    # 检查物理约束
    constraints, errors = pinn_model.validate_physical_constraints(random_inputs)
    print(f"\n物理约束检查:")
    if errors:
        print(f"违反约束: {errors}")
    else:
        print("所有样本满足硬约束")
    
    # 简单的电压响应模拟演示
    print("\n=== 电压响应模拟演示 ===")
    
    # 创建不同电压的输入
    voltages = np.linspace(0, 80, 5)  # 0V到80V
    voltage_responses = []
    
    for v in voltages:
        # 创建基础输入
        input_dict = pinn_model.input_layer.generate_example_input()
        
        # 修改电压值
        input_dict['V_applied'] = v
        
        # 转换为张量
        modified_input = torch.tensor(pinn_model.input_layer.create_input_vector(input_dict), dtype=torch.float32, device=device)
        
        # 预测
        _, output_dict = pinn_model.predict(modified_input.unsqueeze(0).cpu().numpy())
        
        # 记录响应
        response = {
            'voltage': v,
            'contact_angle': output_dict[0].get('theta', 0),
            'contact_radius': output_dict[0].get('r_cl', 0),
            'electric_field': output_dict[0].get('E_z', 0)
        }
        voltage_responses.append(response)
    
    # 打印电压响应
    print("电压响应特性:")
    for resp in voltage_responses:
        print(f"  V={resp['voltage']:.0f}V: θ={resp['contact_angle']:.3f}rad, r_cl={resp['contact_radius']:.2e}m, E_z={resp['electric_field']:.2e}V/m")
    
    # 可视化电压-接触角关系
    try:
        plt.figure(figsize=(10, 6))
        
        # 绘制模拟数据
        plt.plot(
            [r['voltage'] for r in voltage_responses],
            [r['contact_angle'] for r in voltage_responses],
            'o-', label='PINN预测'
        )
        
        # 绘制理论Young-Lippmann曲线作为参考
        theta_0 = 1.57  # 90度
        theory_angles = []
        for r in voltage_responses:
            V = r['voltage']
            # Young-Lippmann方程
            cos_theta = np.cos(theta_0) + (pinn_model.output_layer.epsilon_0 * pinn_model.output_layer.epsilon_r * V**2) / \
                      (2 * pinn_model.output_layer.gamma * pinn_model.output_layer.d)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)  # 确保在有效范围内
            theory_angle = np.arccos(cos_theta)
            theory_angles.append(theory_angle)
        
        plt.plot(
            [r['voltage'] for r in voltage_responses],
            theory_angles,
            '--', label='Young-Lippmann理论'
        )
        
        plt.xlabel('施加电压 (V)')
        plt.ylabel('接触角 (rad)')
        plt.title('电润湿显示像素的电压-接触角响应')
        plt.legend()
        plt.grid(True)
        plt.savefig('voltage_response.png')
        plt.close()
        print("\n电压响应曲线已保存到 voltage_response.png")
    except Exception as e:
        print(f"\n可视化过程中出错: {e}")
    
    print("\n整合系统演示完成！")

# 主函数
if __name__ == "__main__":
    demo_integrated_system()