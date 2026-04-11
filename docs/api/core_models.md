# 核心模型 API

**最后更新**: 2026-02-04
**版本**: v4.5

## TwoPhasePINN

**类路径**: `src.models.pinn_two_phase.TwoPhasePINN`

两相流物理信息神经网络，采用 6D Triad 输入架构和分离式网络设计，用于求解电润湿微流体动力学。

### v4.5 架构特点
- **分离网络 (Separated Networks)**:
  - `phi_net`: 预测相场 φ (6D input -> 1D output)
  - `vel_net`: 预测速度与压力 (7D input: 6D coords + phi -> 4D output)
- **输入维度**: 6D Triad `(x, y, z, V_from, V_to, t_since)`
  - `V_from`: 跳变前电压
  - `V_to`: 跳变后电压（当前电压）
  - `t_since`: 跳变后经过的时间
- **输出维度**: 5D `(u, v, w, p, phi)`
- **激活函数**: GELU (隐藏层), Sigmoid (Phi 输出), Linear (Velocity 输出)

### v4.5 核心指标
| 指标 | 数值 | 说明 |
|------|------|------|
| 30V 开口率 | **83.4%** | 基于校准物理参数 |
| 体积误差 | **<1%** | VOF 输运方程体积守恒 |
| 训练总轮次 | **60,000** | 三阶段渐进式训练 |

### 构造函数
```python
def __init__(self, config: Dict[str, Any] = None):
    """
    Args:
        config: 配置字典，包含网络层数定义
            - model.hidden_phi: Phi 网络隐藏层列表 (默认 [64, 64, 64, 32])
            - model.hidden_vel: Velocity 网络隐藏层列表 (默认 [64, 64, 32])
    """
```

### 前向传播
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x: (batch_size, 6) - [x, y, z, V_from, V_to, t_since]
    
    Returns:
        (batch_size, 5) - [u, v, w, p, phi]
    """

def forward_triplet(
    self,
    spatial_coords: torch.Tensor,
    voltage_triplet: torch.Tensor
) -> torch.Tensor:
    """
    三元组格式的前向传播（与 LSTM-Hybrid-PINN 接口一致）
    
    Args:
        spatial_coords: (batch, 3) - (x, y, z)
        voltage_triplet: (batch, 3) - (V_from, V_to, t_since)
    
    Returns:
        (batch, 5) - [u, v, w, p, phi]
    """
```

---

## LSTMPINNModel

**类路径**: `src.models.lstm_pinn.model.LSTMPINNModel`

基于 LSTM 的时空序列混合模型，组合 LSTM 编码器和 MLP 解码器，用于捕捉长时程动态效应（如迟滞和恢复过程）。

### 架构特点
- **编码器**: `VoltageEncoder` (LSTM)，处理电压时间序列 `(V_0, V_1, ..., V_t)`。
- **解码器**:
  - `PhiDecoder`: 将 LSTM 隐状态和空间坐标解码为 φ 场。
  - `VelocityDecoder` (可选): 将 LSTM 隐状态和空间坐标解码为速度场。
- **输入**: 空间坐标 + 电压序列 + 时间序列。

### 构造函数
```python
def __init__(self, config: Dict[str, Any]):
    """
    Args:
        config: 配置字典
            - lstm: LSTM 编码器配置 (input_dim, hidden_dim, num_layers)
            - phi_decoder: φ 解码器配置 (hidden_layers, activation)
            - velocity_decoder: 速度解码器配置 (enabled, hidden_layers)
    """
```

### 主要方法
```python
def forward(
    self,
    spatial_coords: torch.Tensor,
    voltage_sequence: torch.Tensor,
    time_sequence: Optional[torch.Tensor] = None,
    return_velocity: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Returns:
        字典包含:
        - "phi": (batch, 1)
        - "velocity": (batch, 3) (如果请求)
        - "hidden": (batch, hidden_dim)
    """

def predict_phi(
    self,
    spatial_coords: torch.Tensor,
    voltage_sequence: torch.Tensor,
    time_sequence: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """预测 φ 值的简化接口"""
```

---

## EnhancedApertureModel

**类路径**: `src.models.aperture_model.EnhancedApertureModel`

物理基准模型，负责接触角到开口率的映射、油墨分布计算以及电容反馈效应模拟。继承自 `ApertureModel`。

### v4.5 核心功能
- **电容反馈**: 模拟极性液体覆盖面积增加导致的电容增加，进而增强电润湿力的正反馈效应。
- **体积守恒**: 计算满足油墨体积守恒的液滴高度分布 `h(r)`。
- **动态响应**: 结合电容器充电模型 ($V_{eff}$) 和 HybridPredictor 的接触角动力学。
- **30V 校准**: 接触角与开口率已校准至 30V 83.4% 目标。

### 主要方法
```python
def predict_enhanced(
    self, 
    voltage: float, 
    time: float = None
) -> Dict:
    """
    增强预测，包含电容器充电效应和体积守恒计算。
    
    Returns:
        包含以下键的字典:
        - effective_voltage: 有效电压 (考虑 RC 充电)
        - theta: 接触角
        - aperture_ratio: 开口率 [0, 1]
        - r_open: 透明区域半径
        - r, h: 径向坐标和油墨高度分布数组
        - volume_error: 体积误差百分比
    """

def theta_eta_from_triad(
    self,
    V_from: float,
    V_to: float,
    t_since: float,
) -> Tuple[float, float]:
    """从三元组输入计算接触角和开口率 (Stage 1 Tutor 接口)"""

def aperture_step_response(
    self,
    V_start: float = 0.0,
    V_end: float = 30.0,
    duration: float = 0.10,
    t_step: float = 0.002,
    num_points: int = 500
) -> Tuple[np.ndarray, np.ndarray]:
    """计算开口率阶跃响应曲线"""
```

---

## PINNAperturePredictor

**类路径**: `src.predictors.pinn_aperture.PINNAperturePredictor`

面向用户的推理接口，封装了训练好的 `TwoPhasePINN` 模型，提供从物理场预测到指标计算的完整流程。

### v4.5 特性
- 支持 6D Triad 任意电压序列输入
- 自动加载 v4.5 训练好的模型
- 集成界面锐化后处理

### 主要方法

#### `predict`
预测标量开口率。
```python
def predict(
    self,
    voltage: float,
    time: float,
    n_points: int = 100
) -> float:
    """
    Args:
        voltage: 电压 (V)
        time: 跳变后经过的时间 (s)
        n_points: 采样点数
    
    Returns:
        开口率 (0.0 - 1.0)
    """
```

#### `predict_full_field`
预测完整的三维物理场。
```python
def predict_full_field(
    self,
    voltage: float,
    time: float,
    n_points: Tuple[int, int, int] = (50, 50, 20)
) -> Dict[str, np.ndarray]:
    """
    Args:
        voltage: 电压 (V)
        time: 跳变后经过的时间 (s)
        n_points: 空间采样点数 (nx, ny, nz)
    
    Returns:
        字典包含:
        - 'u', 'v', 'w': 速度分量 (m/s)
        - 'p': 压力 (Pa)
        - 'phi': 相场 [0,1]
        - 'X', 'Y', 'Z': 坐标网格
    """
```

#### `compute_aperture_contour`
计算接触线轮廓和相关几何指标。
```python
def compute_aperture_contour(
    self,
    voltage: float,
    time: float,
    n_points: int = 100,
    method: str = "contour"
) -> dict:
    """
    Args:
        voltage: 电压 (V)
        time: 跳变后经过的时间 (s)
        n_points: 采样点数
        method: 轮廓计算方法 ("contour" 或 "threshold")
    
    Returns:
        - aperture: 开口率
        - contour_x, contour_y: 接触线坐标
        - contour_r: 等效半径
        - contour_area: 开口面积
    """
```

---

## v4.5 推荐配置

### 训练配置 (`v4.5-standard.json`)

```json
{
  "training": {
    "epochs": 60000,
    "stage1_epochs": 1500,
    "stage2_epochs": 4000,
    "learning_rate": 0.0003,
    "batch_size": 4096
  },
  "model": {
    "hidden_phi": [128, 128, 64, 32],
    "hidden_vel": [64, 64, 32]
  },
  "physics": {
    "continuity_weight": 0.5,
    "vof_weight": 0.5,
    "ns_weight": 0.1,
    "surface_tension_weight": 0.01,
    "sharpening_weight": 1.0
  }
}
```
