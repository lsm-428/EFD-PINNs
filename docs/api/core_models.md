# 核心模型 API

**最后更新**: 2025-12-10

## EnhancedApertureModel 类

**文件**: `src/models/aperture_model.py`

增强版开口率模型，已校准参数与实验数据一致。

### 类定义
```python
class EnhancedApertureModel(ApertureModel):
    """增强版开口率模型（已校准）"""
```

### 构造函数
```python
def __init__(
    self,
    config_path: str = 'config/stage6_wall_effect.json',
    predictor: Optional['HybridPredictor'] = None,
    tau_rc: float = 0.1e-3
):
```

**参数**:
- `config_path`: 配置文件路径（推荐使用校准后的配置）
- `predictor`: HybridPredictor 实例（可选）
- `tau_rc`: 电容器充电时间常数

### 主要方法

#### get_contact_angle
```python
def get_contact_angle(
    self, 
    voltage: float, 
    time: float = None,
    V_initial: float = 0.0,
    t_step: float = 0.0
) -> float:
    """获取接触角"""
```

#### contact_angle_to_aperture_ratio
```python
def contact_angle_to_aperture_ratio(self, theta: float) -> float:
    """接触角 → 开口率（已校准）"""
```

### 使用示例
```python
from src.models.aperture_model import EnhancedApertureModel

# 使用校准后的配置
model = EnhancedApertureModel(config_path='config/stage6_wall_effect.json')

# 预测开口率
for V in [0, 10, 20, 30]:
    theta = model.get_contact_angle(V)
    eta = model.contact_angle_to_aperture_ratio(theta)
    print(f"V={V}V: θ={theta:.1f}°, η={eta*100:.1f}%")

# 输出:
# V=0V: θ=120.0°, η=0.0%
# V=10V: θ=119.2°, η=10.3%
# V=20V: θ=115.2°, η=66.7%  ← 实验值 67%
# V=30V: θ=108.2°, η=84.4%
```

---

## TwoPhasePINN 类

**文件**: `src/models/pinn_two_phase.py`

两相流物理信息神经网络，用于预测电润湿显示中的体积分数场。

### 类定义
```python
class TwoPhasePINN(nn.Module):
    """两相流 PINN 模型"""
```

### 构造函数
```python
def __init__(self, config: Dict[str, Any] = None):
```

**参数**:
- `config`: 配置字典，包含网络结构和物理参数

### 主要方法

#### forward
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    前向传播
    
    Args:
        x: (batch, 5) - (x, y, z, t, V)
    
    Returns:
        (batch, 5) - (u, v, w, p, phi)
    """
```

### 使用示例
```python
from src.models.pinn_two_phase import TwoPhasePINN

model = TwoPhasePINN()
inputs = torch.randn(100, 5)  # (x, y, z, t, V)
outputs = model(inputs)  # (u, v, w, p, phi)
```

---

## HybridPredictor 类

**文件**: `src/predictors/hybrid_predictor.py`

混合预测器，结合解析公式预测接触角动态响应。

### 类定义
```python
class HybridPredictor:
    """混合预测器：Young-Lippmann + 二阶欠阻尼响应"""
```

### 构造函数
```python
def __init__(
    self,
    model_path: str = None,
    config_path: str = None,
    use_model_for_steady_state: bool = False,
    device: str = 'cpu'
):
```

### 主要方法

#### predict
```python
def predict(
    self, 
    voltage: float, 
    time: float, 
    V_initial: float = 0.0,
    t_step: float = 0.0
) -> float:
    """预测接触角"""
```

#### young_lippmann
```python
def young_lippmann(self, V: float) -> float:
    """Young-Lippmann 方程计算平衡接触角"""
```

#### step_response
```python
def step_response(
    self,
    V_start: float = 0.0,
    V_end: float = 30.0,
    duration: float = 0.02,
    t_step: float = 0.002,
    num_points: int = 500
) -> Tuple[np.ndarray, np.ndarray]:
    """计算阶跃响应"""
```

### 使用示例
```python
from src.predictors import HybridPredictor

predictor = HybridPredictor(config_path='config/stage6_wall_effect.json')

# 单点预测
theta = predictor.predict(voltage=20, time=0.01)

# 阶跃响应
t, theta = predictor.step_response(V_start=0, V_end=20)
```

---

## PINNAperturePredictor 类

**文件**: `src/predictors/pinn_aperture.py`

基于 PINN 的开口率预测器。

### 类定义
```python
class PINNAperturePredictor:
    """PINN 开口率预测器"""
```

### 构造函数
```python
def __init__(
    self, 
    checkpoint_path: Optional[str] = None, 
    device: Optional[str] = None
):
```

### 主要方法

#### predict
```python
def predict(
    self, 
    voltage: float, 
    time: float, 
    n_points: int = 100
) -> float:
    """预测开口率"""
```

#### predict_phi_field
```python
def predict_phi_field(
    self, 
    voltage: float, 
    time: float, 
    n_points: int = 100,
    z: float = 0.0
) -> np.ndarray:
    """预测 φ 场"""
```

### 使用示例
```python
from src.predictors.pinn_aperture import PINNAperturePredictor

predictor = PINNAperturePredictor()
eta = predictor.predict(voltage=20, time=0.02)
phi_field = predictor.predict_phi_field(voltage=20, time=0.02)
```

---

## 物理常量（已校准）

```python
PHYSICS = {
    # 几何参数
    "Lx": 174e-6,           # 像素宽度 (m)
    "Ly": 174e-6,           # 像素高度 (m)
    "Lz": 20e-6,            # 围堰高度 (m)
    "h_ink": 3e-6,          # 油墨层厚度 (m)
    
    # 电润湿参数（已校准）
    "theta0": 120.0,        # 初始接触角 (度)
    "gamma": 0.050,         # 极性液体表面张力 (N/m)
    "epsilon_r": 3.0,       # SU-8 相对介电常数
    "epsilon_h": 1.9,       # Teflon 相对介电常数
    "d_dielectric": 4e-7,   # SU-8 厚度 (m) = 400nm
    "d_hydrophobic": 4e-7,  # Teflon 厚度 (m) = 400nm
    "V_threshold": 3.0,     # 阈值电压 (V)
    
    # 动力学参数
    "tau": 0.005,           # 响应时间常数 (s)
    "zeta": 0.8,            # 阻尼比
    "t_max": 0.02,          # 最大时间 (s)
}
```

## 开口率映射参数（已校准）

```python
APERTURE_MAPPING = {
    "k": 0.8,               # 陡度参数
    "theta_scale": 6.0,     # 角度缩放因子
    "alpha": 0.05,          # 电容反馈强度
    "aperture_max": 0.85,   # 最大开口率
}
```
