#!/usr/bin/env python3
"""
物理一致性验证数据生成器
用于创建更有效的物理一致性验证样本数据，确保模型能够正确学习物理约束
"""

import torch
import numpy as np
import os
import json
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'generate_consistency_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger('ConsistencyDataGenerator')

try:
    # 导入必要的模块
    from ewp_pinn_model import EWPINN
    from ewp_pinn_input_layer import EWPINNInputLayer
    from ewp_pinn_output_layer import EWPINNOutputLayer
    logger.info("成功导入所有必要的模块")
except ImportError as e:
    logger.error(f"导入模块失败: {e}")
    import traceback
    traceback.print_exc()
    raise

def generate_enhanced_consistency_data(
    num_samples=2000,  # 增加默认样本数量以提高验证效果
    stage=3,
    device='cpu',
    output_dir='./consistency_data',
    density_factor=1.5  # 点密度因子参数
):
    """
    生成增强版的物理一致性验证数据
    
    Args:
        num_samples: 生成的样本数量（增加默认值以提高验证效果）
        stage: 模型实现阶段
        device: 计算设备
        output_dir: 输出目录
        density_factor: 点密度因子，控制网格点的密度
    
    Returns:
        生成的数据文件路径
    """
    try:
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 初始化模型和层
        logger.info("初始化模型和层...")
        model = EWPINN(input_dim=62, output_dim=24, device=device)
        input_layer = EWPINNInputLayer(device=device)
        input_layer.set_implementation_stage(stage)
        output_layer = EWPINNOutputLayer(device=device)
        output_layer.set_implementation_stage(stage)
        
        # 生成多样化的特征字典
        logger.info(f"生成 {num_samples} 个多样化的特征样本...")
        feature_dicts_list = []
        
        # 1. 基础样本生成
        base_samples = int(num_samples * 0.5)
        logger.info(f"生成 {base_samples} 个基础样本...")
        for _ in range(base_samples):
            example_input = input_layer.generate_example_input()
            feature_dicts_list.append(example_input)
        
        # 2. 边界条件样本生成 - 确保模型能够正确处理边界情况
        boundary_samples = int(num_samples * 0.25)
        logger.info(f"生成 {boundary_samples} 个边界条件样本...")
        for _ in range(boundary_samples):
            # 生成边界条件样本 - 在特征空间的边界采样
            example_input = input_layer.generate_example_input()
            # 修改部分特征为边界值
            for key in example_input:
                if isinstance(example_input[key], (int, float)):
                    # 随机设置为最小值或最大值
                    if np.random.random() < 0.5:
                        example_input[key] = 0.0  # 最小值
                    else:
                        example_input[key] = 1.0  # 最大值
                elif isinstance(example_input[key], (list, np.ndarray)):
                    # 对于数组，修改部分元素为边界值
                    for i in range(len(example_input[key])):
                        if np.random.random() < 0.3:  # 30%的概率修改
                            example_input[key][i] = 0.0 if np.random.random() < 0.5 else 1.0
            feature_dicts_list.append(example_input)
        
        # 3. 极端条件样本生成 - 确保模型在极端条件下的稳定性
        extreme_samples = int(num_samples * 0.25)
        logger.info(f"生成 {extreme_samples} 个极端条件样本...")
        for _ in range(extreme_samples):
            # 生成极端条件样本
            example_input = input_layer.generate_example_input()
            # 为电润湿数和毛细管数设置极端值
            if 'electrowetting_number' in example_input:
                example_input['electrowetting_number'] = np.random.choice([0.0, 0.5, 1.0])
            if 'capillary_number' in example_input:
                example_input['capillary_number'] = np.random.choice([0.0, 5e-3, 1e-2])
            feature_dicts_list.append(example_input)
        
        # 确保达到目标样本数量
        while len(feature_dicts_list) < num_samples:
            example_input = input_layer.generate_example_input()
            feature_dicts_list.append(example_input)
        
        # 创建批量输入并转换为tensor
        logger.info("创建批量输入并转换为tensor...")
        features_array = input_layer.create_batch_input(feature_dicts_list)
        features = input_layer.to_tensor(features_array)
        
        # 使用输出层生成标签
        logger.info("生成标签数据...")
        labels = output_layer.generate_random_output(batch_size=num_samples)
        
        # 生成更具代表性的物理约束点
        logger.info("生成增强版物理约束点...")
        # 1. 规则网格点 - 确保空间覆盖，增加密度
        # 使用密度因子增加网格点数量
        base_grid_size = int(np.sqrt(num_samples // 2))
        grid_size = max(10, int(base_grid_size * density_factor))  # 确保最小网格大小
        
        # 生成更密集的坐标点，使用非线性分布以在边界处增加密度
        x_coords = np.linspace(0.0, 1.0, grid_size)
        y_coords = np.linspace(0.0, 1.0, grid_size)
        z_coords = np.linspace(0.0, 1.0, max(5, int(grid_size * 0.7)))  # 增加z方向的点数比例
        
        # 增强时间维度采样 - 非线性分布，在初期和末期增加采样密度
        t_grid_size = max(8, int(grid_size * 0.6))
        # 创建非线性时间分布（初期和末期更密集）
        t_lin = np.linspace(0, 1, t_grid_size)
        t_coords = 0.5 * (1 - np.cos(t_lin * np.pi))  # 余弦分布，边界更密集
    
        grid_points = []
        for x in x_coords:
            for y in y_coords:
                for z in z_coords:
                    # 为每个空间点选择多个时间点，提高时间覆盖
                    # 对70%的空间点分配多个时间点
                    if np.random.random() < 0.7:
                        # 选择2-3个时间点
                        num_times = 2 + int(np.random.random() * 2)
                        selected_times = np.random.choice(t_coords, num_times, replace=False)
                        for t in selected_times:
                            grid_points.append([x, y, z, t])
                    else:
                        # 其余点分配1个时间点
                        t = np.random.choice(t_coords)
                        grid_points.append([x, y, z, t])
        
        logger.info(f"规则网格点生成完成，数量: {len(grid_points)}")
    
        # 2. 随机点 - 增加随机性和分布质量
        # 计算需要生成的随机点数量，确保总数合理
        target_random_count = int(num_samples * 0.4)  # 增加随机点比例到40%
        random_points_count = max(0, target_random_count - len(grid_points))
        
        if random_points_count > 0:
            logger.info(f"生成 {random_points_count} 个增强版随机点...")
            # 使用分层采样策略，在不同区域生成随机点
            random_points = []
            
            # 基础均匀随机点 (60%)
            base_random_count = int(random_points_count * 0.6)
            if base_random_count > 0:
                base_random = np.random.rand(base_random_count, 4)
                random_points.append(base_random)
            
            # 边界附近的随机点 (25%) - 增加边界附近的采样密度
            boundary_random_count = int(random_points_count * 0.25)
            if boundary_random_count > 0:
                boundary_random = np.random.rand(boundary_random_count, 4)
                # 将部分坐标推向边界
                for i in range(boundary_random_count):
                    # 随机选择一个维度推向边界
                    dim = np.random.randint(0, 4)
                    # 推向0或1边界
                    if np.random.random() < 0.5:
                        boundary_random[i, dim] = 0.0 + np.random.random() * 0.1  # 靠近0边界
                    else:
                        boundary_random[i, dim] = 1.0 - np.random.random() * 0.1  # 靠近1边界
                random_points.append(boundary_random)
            
            # 时间维度密集采样点 (15%) - 特别关注时间维度
            time_focus_count = random_points_count - base_random_count - boundary_random_count
            if time_focus_count > 0:
                time_focus = np.random.rand(time_focus_count, 4)
                # 让时间维度更集中在初期和末期
                for i in range(time_focus_count):
                    # 使用平方变换让时间更集中在边界
                    if np.random.random() < 0.5:
                        time_focus[i, 3] = 0.2 * (np.random.random() ** 2)  # 靠近t=0
                    else:
                        time_focus[i, 3] = 1.0 - 0.2 * (np.random.random() ** 2)  # 靠近t=1
                random_points.append(time_focus)
            
            # 合并所有随机点
            if random_points:
                random_points = np.vstack(random_points)
            else:
                random_points = np.array([]).reshape(0, 4)
            
            # 确保随机点数量正确
            if len(random_points) > random_points_count:
                random_points = random_points[:random_points_count]
        else:
            random_points = np.array([]).reshape(0, 4)
            
        logger.info(f"随机点生成完成，数量: {len(random_points)}")
    
        # 3. 边界点 - 增强边界覆盖，增加数量和质量
        # 增加边界点比例到15%
        boundary_count = int(num_samples * 0.15)
        boundary_points = []
        
        logger.info(f"生成 {boundary_count} 个增强版边界点...")
        
        # 空间边界点生成函数
        def generate_boundary_points(dim, value, count):
            """生成指定维度上的边界点"""
            points = []
            for _ in range(count):
                point = np.random.rand(4)
                point[dim] = value  # 设置边界值
                # 对时间维度使用特殊处理，增加边界附近采样
                if np.random.random() < 0.7:
                    # 70%的概率使用时间边界附近的值
                    if np.random.random() < 0.5:
                        point[3] = 0.0 + np.random.random() * 0.1
                    else:
                        point[3] = 1.0 - np.random.random() * 0.1
                points.append(point.tolist())
            return points
        
        # 增加每个空间边界的点数量
        spatial_boundary_count = max(1, boundary_count // 8)  # 每个边界面
        
        # x边界
        boundary_points.extend(generate_boundary_points(0, 0.0, spatial_boundary_count))
        boundary_points.extend(generate_boundary_points(0, 1.0, spatial_boundary_count))
        
        # y边界
        boundary_points.extend(generate_boundary_points(1, 0.0, spatial_boundary_count))
        boundary_points.extend(generate_boundary_points(1, 1.0, spatial_boundary_count))
        
        # z边界
        boundary_points.extend(generate_boundary_points(2, 0.0, spatial_boundary_count))
        boundary_points.extend(generate_boundary_points(2, 1.0, spatial_boundary_count))
    
        # 4. 时间边界点 - 大幅增强时间边界条件覆盖
        time_boundary_count = max(1, boundary_count // 2)
        # t=0边界 - 初始条件非常重要
        for _ in range(time_boundary_count):
            # 生成更密集的初始条件点
            for _ in range(2):  # 每个循环生成2个点
                point = np.random.rand(4)
                point[3] = 0.0  # 精确的t=0
                boundary_points.append(point.tolist())
        
        # t=1边界 - 最终状态
        for _ in range(time_boundary_count):
            point = np.random.rand(4)
            point[3] = 1.0  # 精确的t=1
            boundary_points.append(point.tolist())
        
        # 5. 角落点 - 增加4维空间中的角落点采样
        corner_count = max(1, boundary_count // 16)
        corners = [
            [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0],  # 对角角落
            [0.0, 0.0, 1.0, 0.0], [1.0, 1.0, 0.0, 1.0],  # 其他重要角落
            [0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 0.0], [1.0, 0.0, 0.0, 1.0]
        ]
        
        for corner in corners:
            for _ in range(corner_count):
                # 添加角落点的小幅扰动版本，增加局部覆盖
                perturbed = corner.copy()
                for i in range(4):
                    perturbed[i] += np.random.uniform(-0.05, 0.05)  # 5%的扰动
                    perturbed[i] = max(0.0, min(1.0, perturbed[i]))  # 确保在定义域内
                boundary_points.append(perturbed)
        
        logger.info(f"边界点生成完成，数量: {len(boundary_points)}")
        
        # 合并所有点
        all_points = []
        all_points.extend(grid_points)
        if len(random_points) > 0:
            all_points.extend(random_points.tolist())
        all_points.extend(boundary_points)
        
        # 转换为numpy数组
        if all_points:
            physics_points = np.array(all_points)
            
            # 去重和打乱顺序，避免过拟合特定区域
            logger.info(f"合并后的物理点总数: {len(physics_points)}")
            physics_points = np.unique(physics_points, axis=0)
            logger.info(f"去重后的物理点数量: {len(physics_points)}")
            np.random.shuffle(physics_points)
            
            # 确保最终点数量合理
            if len(physics_points) > num_samples * 1.2:  # 允许适度超出目标数量
                physics_points = physics_points[:int(num_samples * 1.2)]
                logger.info(f"调整后的物理点数量: {len(physics_points)}")
        else:
            physics_points = np.array([]).reshape(0, 4)
            logger.warning("未生成任何物理点！")
        
        logger.info(f"最终生成物理约束点数量: {len(physics_points)}, 维度: {physics_points.shape[1]}")
        
        # 保存数据
        data_to_save = {
            'physics_points': physics_points,  # 4维物理约束点 (x, y, z, t)
            'metadata': {
                'generated_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'num_points': len(physics_points),
                'dimensions': physics_points.shape[1],
                'version': 'enhanced_v1.1',
                'description': '4维物理约束点，包含时间维度'
            }
        }
        
        # 生成保存路径
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"physics_points_{num_samples}.npz")
        
        # 保存数据到npz文件
        np.savez_compressed(output_path, **data_to_save)
        logger.info(f"物理约束点数据已保存到: {output_path}")
        
        return physics_points, output_path
        
        # 保存生成的数据
        logger.info("保存生成的数据...")
        data_file = os.path.join(output_dir, f'consistency_data_{timestamp}.npz')
        np.savez_compressed(
            data_file,
            features=features.cpu().numpy(),
            labels=labels.cpu().numpy(),
            physics_points=physics_points.cpu().numpy()
        )
        
        # 保存元数据
        metadata_file = os.path.join(output_dir, f'consistency_data_{timestamp}_metadata.json')
        metadata = {
            'timestamp': timestamp,
            'num_samples': num_samples,
            'stage': stage,
            'features_shape': features.shape,
            'labels_shape': labels.shape,
            'physics_points_shape': physics_points.shape,
            'generation_details': {
                'base_samples': base_samples,
                'boundary_samples': boundary_samples,
                'extreme_samples': extreme_samples,
                'grid_points_count': len(grid_points),
                'random_points_count': random_points_count,
                'boundary_points_count': len(boundary_points)
            }
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        logger.info(f"成功生成并保存物理一致性验证数据：{data_file}")
        logger.info(f"元数据保存到：{metadata_file}")
        
        # 创建一个简单的加载示例脚本
        example_script = os.path.join(output_dir, f'load_consistency_data_example.py')
        with open(example_script, 'w') as f:
            f.write(f'''
#!/usr/bin/env python3
"""
加载物理一致性验证数据的示例脚本
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from ewp_pinn_model import EWPINN
from ewp_data_interface import create_dataset
from ewp_pinn_input_layer import EWPINNInputLayer

def load_and_validate_consistency_data():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载数据
    data_file = '{data_file}'
    print(f"加载数据文件: {{data_file}}")
    data = np.load(data_file)
    
    # 恢复tensor
    features = torch.tensor(data['features'], dtype=torch.float32, device=device)
    labels = torch.tensor(data['labels'], dtype=torch.float32, device=device)
    physics_points = torch.tensor(data['physics_points'], dtype=torch.float32, device=device)
    
    print(f"特征形状: {{features.shape}}")
    print(f"标签形状: {{labels.shape}}")
    print(f"物理点形状: {{physics_points.shape}}")
    
    # 创建输入层
    input_layer = EWPINNInputLayer(device=device)
    input_layer.set_implementation_stage({stage})
    
    # 创建数据集和数据加载器
    dataset = create_dataset(features, labels, input_layer=input_layer, stage={stage}, device=device)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # 初始化模型（如果需要进行验证）
    # model = EWPINN(input_dim=62, output_dim=24, device=device)
    # consistency_results = model._validate_physics_consistency(dataloader)
    # print(f"物理一致性验证结果: {{consistency_results}}")
    
    return features, labels, physics_points, dataloader

if __name__ == "__main__":
    load_and_validate_consistency_data()
''')
        
        logger.info(f"创建加载示例脚本：{example_script}")
        
        return data_file, metadata_file, example_script
        
    except Exception as e:
        logger.error(f"生成物理一致性验证数据时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def main():
    """
    主函数：生成不同规模的物理约束点数据集
    """
    # 设置日志
    setup_logging()
    
    # 创建输出目录
    output_dir = "./physics_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成不同规模的数据集 - 增加默认点数量以提高物理约束效果
    datasets = {
        "standard": 2000,    # 标准规模：2000个物理约束点（原1000）
        "large": 8000,       # 大规模：8000个物理约束点（原5000）
        "extra_large": 15000  # 超大规模：15000个物理约束点（原10000）
    }
    
    for dataset_name, num_samples in datasets.items():
        try:
            logger.info(f"\n开始生成 {dataset_name} 规模数据集 ({num_samples} 个点)...")
            # 调用增强版本的物理点生成函数
            physics_points, save_path = generate_enhanced_consistency_data(
                num_samples=num_samples,
                output_dir=output_dir,
                device="cpu",  # 在CPU上生成，节省GPU资源
                density_factor=1.7  # 增加点密度因子
            )
            logger.info(f"{dataset_name} 数据集生成完成！")
            logger.info(f"生成点数: {len(physics_points)}")
            logger.info(f"数据维度: {physics_points.shape[1]}")
            logger.info(f"保存路径: {save_path}")
            
        except Exception as e:
            logger.error(f"生成 {dataset_name} 数据集时出错: {str(e)}", exc_info=True)
    
    logger.info("\n所有数据集生成任务完成！")
    

if __name__ == "__main__":
    main()