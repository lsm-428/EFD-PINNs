import torch
import torch.nn as nn
import numpy as np
import os
import json
import copy
from datetime import datetime
import time
from collections import defaultdict
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Union, Callable

# 导入现有模型和工具
from ewp_pinn_model import OptimizedEWPINN, extract_predictions
from ewp_pinn_optimized_train import load_model, compare_model_performance

class EWPINNEnsembleModel:
    """
    EWPINN集成模型类，用于管理多个EWPINN模型并执行集成预测
    
    支持多种集成策略：
    - 简单平均 (Simple Average)
    - 加权平均 (Weighted Average)
    - 投票机制 (Voting)
    - 堆叠集成 (Stacking)
    """
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        初始化集成模型
        
        Args:
            device: 运行设备
        """
        self.device = device
        self.models = []  # 存储模型列表
        self.model_weights = []  # 存储模型权重
        self.normalizers = []  # 存储每个模型的数据标准化器
        self.model_metadatas = []  # 存储每个模型的元数据
        self.ensemble_strategy = 'weighted_average'  # 默认使用加权平均策略
        self.ensemble_info = {
            'version': '1.0',
            'creation_time': datetime.now().isoformat(),
            'models_count': 0,
            'strategy': self.ensemble_strategy
        }
    
    def add_model(self, model_path: str, weight: float = 1.0) -> bool:
        """
        添加单个模型到集成
        
        Args:
            model_path: 模型文件路径
            weight: 模型权重（用于加权平均）
            
        Returns:
            bool: 是否添加成功
        """
        try:
            print(f"📂 加载模型: {model_path}")
            # 使用现有的load_model函数加载模型
            model, normalizer, metadata = load_model(model_path, device=self.device)
            
            if model is not None:
                model.eval()  # 设置为评估模式
                self.models.append(model)
                self.model_weights.append(weight)
                self.normalizers.append(normalizer)
                self.model_metadatas.append(metadata)
                self.ensemble_info['models_count'] += 1
                
                print(f"✅ 成功添加模型 #{self.ensemble_info['models_count']}")
                print(f"   模型信息: 版本={metadata.get('model_info', {}).get('version', 'unknown')}")
                print(f"   模型权重: {weight}")
                
                return True
            else:
                print(f"❌ 模型加载失败: {model_path}")
                return False
        except Exception as e:
            print(f"❌ 添加模型时出错: {str(e)}")
            return False
    
    def add_models_from_directory(self, directory: str, pattern: str = "*.pth", 
                                 weight_strategy: str = "uniform") -> int:
        """
        从目录中批量添加模型
        
        Args:
            directory: 包含模型文件的目录
            pattern: 文件匹配模式
            weight_strategy: 权重策略 ('uniform', 'performance_based', 'custom')
            
        Returns:
            int: 成功添加的模型数量
        """
        import glob
        model_paths = glob.glob(os.path.join(directory, pattern))
        added_count = 0
        
        print(f"🔍 发现 {len(model_paths)} 个潜在模型文件")
        
        for i, model_path in enumerate(model_paths):
            print(f"\n[{i+1}/{len(model_paths)}] 处理: {model_path}")
            
            # 根据策略确定权重
            if weight_strategy == "uniform":
                weight = 1.0
            else:
                # 暂时默认权重
                weight = 1.0
            
            if self.add_model(model_path, weight):
                added_count += 1
        
        print(f"\n✅ 批量添加完成: 成功添加 {added_count}/{len(model_paths)} 个模型")
        
        # 如果是均匀权重，归一化
        if weight_strategy == "uniform" and added_count > 0:
            total_weight = sum(self.model_weights[-added_count:])
            for i in range(len(self.model_weights) - added_count, len(self.model_weights)):
                self.model_weights[i] = self.model_weights[i] / total_weight
            
            print(f"🔄 已归一化模型权重")
        
        return added_count
    
    def set_ensemble_strategy(self, strategy: str) -> None:
        """
        设置集成策略
        
        Args:
            strategy: 集成策略，可选值: 'simple_average', 'weighted_average', 'voting'
        """
        valid_strategies = ['simple_average', 'weighted_average', 'voting']
        if strategy not in valid_strategies:
            raise ValueError(f"无效的集成策略: {strategy}。有效选项: {valid_strategies}")
        
        self.ensemble_strategy = strategy
        self.ensemble_info['strategy'] = strategy
        print(f"✅ 已设置集成策略: {strategy}")
    
    def set_model_weights(self, weights: List[float]) -> bool:
        """
        设置模型权重（用于加权平均）
        
        Args:
            weights: 权重列表，长度必须与模型数量一致
            
        Returns:
            bool: 是否设置成功
        """
        if len(weights) != len(self.models):
            print(f"❌ 权重数量 ({len(weights)}) 与模型数量 ({len(self.models)}) 不匹配")
            return False
        
        # 验证权重是否为正数
        if any(w <= 0 for w in weights):
            print("❌ 权重必须为正数")
            return False
        
        self.model_weights = copy.copy(weights)
        
        # 归一化权重
        total_weight = sum(weights)
        self.model_weights = [w / total_weight for w in self.model_weights]
        
        print(f"✅ 已设置并归一化模型权重: {self.model_weights}")
        return True
    
    def predict(self, inputs: torch.Tensor, use_normalization: bool = True) -> torch.Tensor:
        """
        使用集成模型进行预测
        
        Args:
            inputs: 输入数据张量
            use_normalization: 是否使用标准化器
            
        Returns:
            torch.Tensor: 集成预测结果
        """
        if not self.models:
            raise ValueError("集成模型为空，请先添加模型")
        
        # 确保输入在正确的设备上
        inputs = inputs.to(self.device)
        
        with torch.no_grad():
            # 应用标准化（如果需要）
            normalized_inputs = []
            for i, normalizer in enumerate(self.normalizers):
                if use_normalization and normalizer is not None:
                    normalized_inputs.append(normalizer.transform_features(inputs.clone()))
                else:
                    normalized_inputs.append(inputs.clone())
            
            # 获取所有模型的预测（确保提取为张量）
            all_predictions = []
            for i, (model, norm_input) in enumerate(zip(self.models, normalized_inputs)):
                raw_pred = model(norm_input)
                try:
                    pred = extract_predictions(raw_pred)
                except Exception:
                    # 兼容 fallback：尝试将 numpy/list 转为 tensor
                    if isinstance(raw_pred, (list, tuple, np.ndarray)):
                        pred = torch.tensor(raw_pred, device=self.device)
                    else:
                        pred = torch.as_tensor(raw_pred).to(self.device)

                all_predictions.append(pred)
            
            # 根据策略进行集成
            if self.ensemble_strategy == 'simple_average':
                # 简单平均
                ensemble_pred = torch.stack(all_predictions).mean(dim=0)
            
            elif self.ensemble_strategy == 'weighted_average':
                # 加权平均
                weighted_preds = [pred * weight for pred, weight in zip(all_predictions, self.model_weights)]
                ensemble_pred = torch.stack(weighted_preds).sum(dim=0)
            
            elif self.ensemble_strategy == 'voting':
                # 投票机制（对于回归问题，我们使用中位数）
                ensemble_pred = torch.stack(all_predictions).median(dim=0)[0]
            
            return ensemble_pred
    
    def evaluate(self, test_data: torch.Tensor, test_labels: torch.Tensor, 
                use_normalization: bool = True) -> Dict[str, float]:
        """
        评估集成模型性能
        
        Args:
            test_data: 测试数据
            test_labels: 测试标签
            use_normalization: 是否使用标准化器
            
        Returns:
            Dict: 评估指标字典
        """
        # 确保数据在正确的设备上
        test_data = test_data.to(self.device)
        test_labels = test_labels.to(self.device)
        
        # 应用标签标准化（如果需要）
        if use_normalization and self.normalizers[0] is not None:
            original_test_labels = test_labels.clone()  # 保存原始标签用于评估
            normalized_test_labels = self.normalizers[0].transform_labels(test_labels)
        else:
            original_test_labels = test_labels
            normalized_test_labels = test_labels
        
        # 获取预测结果
        predictions = self.predict(test_data, use_normalization)
        
        # 反向标准化预测结果（如果需要）
        if use_normalization and self.normalizers[0] is not None:
            predictions = self.normalizers[0].inverse_transform_labels(predictions)
        
        # 计算评估指标
        metrics = {
            'ensemble_mse': nn.MSELoss()(predictions, original_test_labels).item(),
            'ensemble_mae': nn.L1Loss()(predictions, original_test_labels).item(),
            'ensemble_rmse': torch.sqrt(nn.MSELoss()(predictions, original_test_labels)).item()
        }
        
        # 计算每个单独模型的性能
        individual_metrics = []
        for i, (model, normalizer) in enumerate(zip(self.models, self.normalizers)):
            with torch.no_grad():
                # 对每个模型应用其特定的标准化
                model_inputs = normalizer.transform_features(test_data.clone()) if use_normalization and normalizer is not None else test_data.clone()
                raw_model_pred = model(model_inputs)
                try:
                    model_pred = extract_predictions(raw_model_pred)
                except Exception:
                    if isinstance(raw_model_pred, (list, tuple, np.ndarray)):
                        model_pred = torch.tensor(raw_model_pred, device=self.device)
                    else:
                        model_pred = torch.as_tensor(raw_model_pred).to(self.device)

                # 反向标准化
                if use_normalization and normalizer is not None:
                    model_pred = normalizer.inverse_transform_labels(model_pred)

                model_mse = nn.MSELoss()(model_pred, original_test_labels).item()
                model_mae = nn.L1Loss()(model_pred, original_test_labels).item()
                
                individual_metrics.append({
                    'model_index': i,
                    'mse': model_mse,
                    'mae': model_mae,
                    'rmse': np.sqrt(model_mse)
                })
        
        metrics['individual_models'] = individual_metrics
        
        # 计算集成增益
        avg_individual_mse = np.mean([m['mse'] for m in individual_metrics])
        best_individual_mse = min([m['mse'] for m in individual_metrics])
        
        metrics['avg_individual_mse'] = avg_individual_mse
        metrics['best_individual_mse'] = best_individual_mse
        metrics['ensemble_gain_from_avg'] = ((avg_individual_mse - metrics['ensemble_mse']) / avg_individual_mse) * 100
        metrics['ensemble_gain_from_best'] = ((best_individual_mse - metrics['ensemble_mse']) / best_individual_mse) * 100
        
        return metrics
    
    def optimize_weights(self, validation_data: torch.Tensor, validation_labels: torch.Tensor,
                        use_normalization: bool = True, iterations: int = 100) -> List[float]:
        """
        优化集成权重以最大化性能
        
        Args:
            validation_data: 验证数据
            validation_labels: 验证标签
            use_normalization: 是否使用标准化器
            iterations: 优化迭代次数
            
        Returns:
            List[float]: 优化后的权重
        """
        if not self.models:
            raise ValueError("集成模型为空，请先添加模型")
        
        print("🔧 开始优化集成权重...")
        
        # 确保数据在正确的设备上
        val_data = validation_data.to(self.device)
        val_labels = validation_labels.to(self.device)
        
        # 预先计算所有模型在验证集上的预测
        all_predictions = []
        with torch.no_grad():
            for i, (model, normalizer) in enumerate(zip(self.models, self.normalizers)):
                # 应用标准化
                if use_normalization and normalizer is not None:
                    model_inputs = normalizer.transform_features(val_data.clone())
                else:
                    model_inputs = val_data.clone()
                raw_pred = model(model_inputs)
                try:
                    pred = extract_predictions(raw_pred)
                except Exception:
                    if isinstance(raw_pred, (list, tuple, np.ndarray)):
                        pred = torch.tensor(raw_pred, device=self.device)
                    else:
                        pred = torch.as_tensor(raw_pred).to(self.device)

                # 反向标准化
                if use_normalization and normalizer is not None:
                    pred = normalizer.inverse_transform_labels(pred)

                all_predictions.append(pred)
        
        # 初始化权重
        weights = torch.ones(len(self.models), requires_grad=True, device=self.device)
        optimizer = torch.optim.Adam([weights], lr=0.1)
        
        best_loss = float('inf')
        best_weights = weights.clone().detach().cpu().numpy()
        
        # 优化循环
        for iteration in range(iterations):
            optimizer.zero_grad()
            
            # 确保权重为正并归一化
            normalized_weights = nn.functional.softmax(weights, dim=0)
            
            # 计算加权预测
            weighted_preds = [pred * w for pred, w in zip(all_predictions, normalized_weights)]
            ensemble_pred = torch.stack(weighted_preds).sum(dim=0)
            
            # 计算损失
            loss = nn.MSELoss()(ensemble_pred, val_labels)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 记录最佳权重
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_weights = normalized_weights.clone().detach().cpu().numpy()
            
            # 打印进度
            if (iteration + 1) % 10 == 0:
                print(f"  迭代 {iteration+1}/{iterations}, 损失: {loss.item():.6f}")
        
        # 转换为Python列表并确保归一化
        best_weights = best_weights.tolist()
        total = sum(best_weights)
        best_weights = [w / total for w in best_weights]
        
        print(f"✅ 权重优化完成")
        print(f"   最佳损失: {best_loss:.6f}")
        print(f"   优化权重: {best_weights}")
        
        # 更新模型权重
        self.model_weights = best_weights
        
        return best_weights
    
    def save_ensemble(self, save_path: str) -> bool:
        """
        保存集成模型
        
        Args:
            save_path: 保存路径
            
        Returns:
            bool: 是否保存成功
        """
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # 保存集成配置
            ensemble_data = {
                'ensemble_info': self.ensemble_info,
                'model_weights': self.model_weights,
                'save_time': datetime.now().isoformat(),
                'torch_version': torch.__version__
            }
            
            # 保存到JSON文件
            config_path = save_path.replace('.pth', '.json')
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(ensemble_data, f, ensure_ascii=False, indent=2)
            
            print(f"✅ 集成配置已保存至: {config_path}")
            
            # 这里我们不保存完整的模型，只保存配置
            # 在加载时，会根据路径重新加载各个模型
            
            return True
        except Exception as e:
            print(f"❌ 保存集成模型失败: {str(e)}")
            return False
    
    def load_ensemble(self, config_path: str, model_dir: str = None) -> bool:
        """
        加载集成模型配置
        
        Args:
            config_path: 配置文件路径
            model_dir: 模型文件目录（如果与配置文件不同）
            
        Returns:
            bool: 是否加载成功
        """
        try:
            # 读取配置文件
            with open(config_path, 'r', encoding='utf-8') as f:
                ensemble_data = json.load(f)
            
            # 重置当前集成
            self.models = []
            self.model_weights = []
            self.normalizers = []
            self.model_metadatas = []
            
            # 加载集成信息
            self.ensemble_info = ensemble_data.get('ensemble_info', {})
            self.ensemble_strategy = self.ensemble_info.get('strategy', 'weighted_average')
            
            print(f"✅ 加载集成配置成功")
            print(f"   集成策略: {self.ensemble_strategy}")
            print(f"   模型数量: {self.ensemble_info.get('models_count', 0)}")
            
            # 注意：这里需要额外的模型路径信息
            # 在实际使用时，需要确保模型文件在正确的位置
            
            return True
        except Exception as e:
            print(f"❌ 加载集成模型失败: {str(e)}")
            return False
    
    def generate_ensemble_report(self, test_data: torch.Tensor = None, 
                               test_labels: torch.Tensor = None, 
                               use_normalization: bool = True,
                               save_path: str = None) -> Dict[str, Any]:
        """
        生成集成模型性能报告
        
        Args:
            test_data: 测试数据
            test_labels: 测试标签
            use_normalization: 是否使用标准化器
            save_path: 保存报告的路径
            
        Returns:
            Dict: 报告数据
        """
        report = {
            'ensemble_info': self.ensemble_info,
            'model_details': []
        }
        
        # 收集每个模型的详细信息
        for i, (model, weight, metadata) in enumerate(zip(self.models, self.model_weights, self.model_metadatas)):
            model_report = {
                'index': i,
                'weight': weight,
                'metadata': metadata,
                'model_info': metadata.get('model_info', {})
            }
            report['model_details'].append(model_report)
        
        # 如果提供了测试数据，进行评估
        if test_data is not None and test_labels is not None:
            metrics = self.evaluate(test_data, test_labels, use_normalization)
            report['evaluation_metrics'] = metrics
        
        # 保存报告
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f"✅ 集成报告已保存至: {save_path}")
        
        return report
    
    def plot_performance_comparison(self, metrics: Dict[str, Any], save_path: str = None) -> None:
        """
        绘制集成模型与单个模型的性能比较图
        
        Args:
            metrics: 评估指标
            save_path: 保存图表的路径
        """
        # 提取数据
        individual_mses = [m['mse'] for m in metrics['individual_models']]
        ensemble_mse = metrics['ensemble_mse']
        
        # 创建图表
        plt.figure(figsize=(12, 6))
        
        # 绘制单个模型性能
        plt.bar(range(len(individual_mses)), individual_mses, 
                color='skyblue', label='单个模型')
        
        # 绘制集成模型性能
        plt.bar(len(individual_mses), ensemble_mse, 
                color='salmon', label='集成模型')
        
        # 添加标签和标题
        plt.xlabel('模型')
        plt.ylabel('MSE损失')
        plt.title('集成模型与单个模型性能比较')
        plt.xticks(range(len(individual_mses) + 1), 
                  [f'模型{i}' for i in range(len(individual_mses))] + ['集成'])
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 显示增益信息
        plt.figtext(0.5, 0.01, 
                   f"集成增益（相对于平均）: {metrics['ensemble_gain_from_avg']:.2f}% | "
                   f"集成增益（相对于最佳）: {metrics['ensemble_gain_from_best']:.2f}%",
                   ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.3, "pad":5})
        
        # 保存图表
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 性能比较图已保存至: {save_path}")
        
        plt.close()


def create_ensemble_from_directory(model_dir: str, output_dir: str = None, 
                                  strategy: str = 'weighted_average',
                                  optimize_weights: bool = True,
                                  val_data: Optional[torch.Tensor] = None,
                                  val_labels: Optional[torch.Tensor] = None) -> EWPINNEnsembleModel:
    """
    从目录中的模型创建集成模型
    
    Args:
        model_dir: 包含模型文件的目录
        output_dir: 输出目录
        strategy: 集成策略
        optimize_weights: 是否优化权重
        val_data: 用于优化权重的验证数据
        val_labels: 用于优化权重的验证标签
        
    Returns:
        EWPINNEnsembleModel: 创建的集成模型
    """
    # 创建输出目录
    if output_dir is None:
        output_dir = os.path.join(model_dir, 'ensemble')
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建集成模型
    ensemble = EWPINNEnsembleModel()
    
    # 设置集成策略
    ensemble.set_ensemble_strategy(strategy)
    
    # 添加模型
    added_count = ensemble.add_models_from_directory(model_dir)
    
    if added_count == 0:
        print("❌ 未添加任何模型，集成创建失败")
        return None
    
    # 优化权重（如果需要）
    if optimize_weights and val_data is not None and val_labels is not None:
        ensemble.optimize_weights(val_data, val_labels)
    
    # 保存集成配置
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ensemble_path = os.path.join(output_dir, f'ewp_pinn_ensemble_{timestamp}.json')
    ensemble.save_ensemble(ensemble_path)
    
    print(f"\n✅ 集成模型创建完成")
    print(f"   模型数量: {added_count}")
    print(f"   集成策略: {strategy}")
    print(f"   配置保存: {ensemble_path}")
    
    return ensemble


def main():
    """
    示例：创建和评估EWPINN集成模型
    """
    # 示例用法
    print("🚀 EWPINN集成学习演示")
    
    # 创建集成模型
    ensemble = EWPINNEnsembleModel()
    
    # 设置集成策略
    ensemble.set_ensemble_strategy('weighted_average')
    
    print("\n📋 集成学习框架已准备就绪")
    print("   使用方法:")
    print("   1. 通过 add_model() 或 add_models_from_directory() 添加模型")
    print("   2. 使用 set_ensemble_strategy() 选择集成策略")
    print("   3. 通过 optimize_weights() 优化集成权重")
    print("   4. 使用 evaluate() 评估集成性能")
    print("   5. 通过 generate_ensemble_report() 生成报告")
    

if __name__ == "__main__":
    main()