import os
import importlib.util
import torch


def _load_model_module():
    """Load ewp_pinn_model.py as a module without importing package __init__."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    module_path = os.path.join(repo_root, 'ewp_pinn_model.py')
    spec = importlib.util.spec_from_file_location('ewp_pinn_model', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_forward_returns_main_predictions():
    mod = _load_model_module()
    EWPINN = mod.EWPINN

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
    mod = _load_model_module()
    EWPINN = mod.EWPINN
    extract_predictions = mod.extract_predictions

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
