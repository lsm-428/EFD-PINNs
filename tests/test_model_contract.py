import os
import sys
import torch

# 添加项目根目录到Python路径，以便直接导入模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# 直接导入所需的模块和函数
from ewp_pinn_model import EWPINN, extract_predictions


def test_forward_returns_main_predictions():
    """测试模型前向传播返回包含main_predictions的字典"""
    device = 'cpu'
    model = EWPINN(input_dim=62, output_dim=24, device=device)
    model.eval()

    # 创建随机输入
    x = torch.randn(4, model.input_dim)
    out = model.forward(x)

    assert isinstance(out, dict), "forward 应返回字典"
    assert 'main_predictions' in out, "返回字典必须包含 'main_predictions' 键"
    main = out['main_predictions']
    assert isinstance(main, torch.Tensor)
    assert main.shape[0] == 4
    assert main.shape[1] == model.output_dim


def test_extract_predictions_tensor_and_dict():
    """测试extract_predictions函数处理张量和字典输入"""
    device = 'cpu'
    model = EWPINN(input_dim=62, output_dim=24, device=device)
    model.eval()

    x = torch.randn(3, model.input_dim)
    out = model.forward(x)

    # test dict input
    pred = extract_predictions(out)
    assert isinstance(pred, torch.Tensor)
    assert pred.shape[0] == 3

    # test direct tensor input
    t = torch.randn(2, model.output_dim)
    pred2 = extract_predictions(t)
    assert isinstance(pred2, torch.Tensor)
    assert pred2.shape[0] == 2


def test_model_initialization():
    """测试模型初始化参数"""
    model = EWPINN(input_dim=62, output_dim=24, device='cpu')
    
    assert model.input_dim == 62
    assert model.output_dim == 24
    assert model.device == 'cpu'
    
    # 测试模型参数数量
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count > 0, "模型应该包含参数"


def test_model_eval_mode():
    """测试模型评估模式"""
    model = EWPINN(input_dim=62, output_dim=24, device='cpu')
    model.eval()
    
    # 检查模型是否在评估模式
    assert not model.training, "模型应该在评估模式"
    
    # 测试前向传播在评估模式下正常工作
    x = torch.randn(1, model.input_dim)
    out = model.forward(x)
    assert isinstance(out, dict), "评估模式下前向传播应返回字典"
