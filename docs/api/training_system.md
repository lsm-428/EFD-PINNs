# 训练系统 API

**最后更新**: 2026-02-04
**版本**: v4.5

## Trainer

**类路径**: `src.models.pinn_two_phase.Trainer`

管理两相流 PINN 的全生命周期，包括数据采样、权重调度、损失计算和模型持久化。

### 构造函数
```python
def __init__(self, config: Dict[str, Any] = None):
    """
    初始化训练器。
    
    参数:
    - config: 包含网络架构、物理常数、权重调度的配置字典
    """
```

### 主要方法

#### `train`
```python
def train(self):
    """
    执行主训练循环 (默认 60,000 Epochs)。
    - 阶段 1 (0-1.5k): 纯数据学习
    - 阶段 2 (1.5k-5.5k): 引入 Continuity + VOF
    - 阶段 3 (5.5k+): 完整物理 (NS + Surface Tension)
    """
```

#### `get_physics_weights`
```python
def get_physics_weights(self, epoch: int) -> Dict[str, float]:
    """
    根据当前 Epoch 返回物理损失权重。
    实现三阶段渐进式训练策略。
    """
```

---

## DataGenerator

**类路径**: `src.models.pinn_two_phase.DataGenerator`

负责生成 6D Triad 训练数据，采用界面加密和 0.5V 步进采样策略。

### 主要方法

#### `generate_training_data`
```python
def generate_training_data(self, n_points: int) -> Dict[str, torch.Tensor]:
    """
    生成一批训练数据。
    
    返回字典:
    - 'interface': 界面附近的高密度采样点
    - 'domain': 全域随机采样点
    - 'initial': t=0 初始条件点
    - 'boundary': 壁面边界点
    """
```

### 采样策略
1. **0.5V 步进**: 电压采样覆盖 [0, 30V] 区间，步长 0.5V（61个点）。
2. **界面加密**: 在 Stage 1 预测的界面附近增加采样密度。
3. **Triad 生成**: 自动生成 (V_from, V_to, t_since) 组合，覆盖升压、降压和稳态工况。
4. **时间采样**: 三段策略 (early/mid/late) 覆盖 0-50ms 范围。

---

## 训练配置 (Config)

标准训练配置结构 (示例):

```python
DEFAULT_CONFIG = {
    "model": {
        "hidden_phi": [64, 64, 64, 32],
        "hidden_vel": [64, 64, 32],
    },
    "training": {
        "epochs": 60000,
        "batch_size": 4096,
        "learning_rate": 0.0003,
        "stage1_epochs": 1500,
        "stage2_epochs": 4000,
    },
    "physics": {
        "interface_weight": 500.0,
        "ic_weight": 100.0,
        "bc_weight": 50.0,
        "continuity_weight": 0.5,
        "vof_weight": 0.5,
        "ns_weight": 0.1,
        "surface_tension_weight": 0.01,
        "sharpening_weight": 1.0,
    },
    "data": {
        "n_interface": 100000,
        "n_domain": 20000,
        "voltages": [0.5 * i for i in range(0, 61)],  # 0.5V 步进 (61点)
        "times": 50,
    }
}
```
