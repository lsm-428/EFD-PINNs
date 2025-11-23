import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

UNIT_RULES = {
    'dielectric_thickness_d': ('um', 0.1, 10.0),
    'epsilon_r': ('', 1.5, 10.0),
    'surface_tension_gamma': ('mN/m', 10.0, 80.0),
    'theta0': ('deg', 60.0, 140.0),
    'frequency_hz': ('Hz', 1.0, 1e6),
    'viscosity': ('mPa·s', 0.5, 500.0),
    'density': ('kg/m3', 600.0, 2000.0)
}

def validate_units(feature_array):
    meta = {'unit_checks': []}
    # 可在此绑定具体列与规则；示例检查：
    checks = [
        ('dielectric_thickness_d', 18),
        ('epsilon_r', 19),
        ('surface_tension_gamma', 21),
        ('theta0', 23),
        ('frequency_hz', 32),
        ('viscosity', 36),
        ('density', 37)
    ]
    for name, col in checks:
        unit, lo, hi = UNIT_RULES[name]
        vals = feature_array[:, col]
        lo_cnt = int((vals < lo).sum())
        hi_cnt = int((vals > hi).sum())
        meta['unit_checks'].append({'field': name, 'unit': unit, 'low_violations': lo_cnt, 'high_violations': hi_cnt})
    return meta

class EWPINNDataset(Dataset):
    def __init__(self, features, labels, input_layer=None, stage=None, device=None):
        if isinstance(features, np.ndarray):
            features = torch.tensor(features, dtype=torch.float32)
        if isinstance(labels, np.ndarray):
            labels = torch.tensor(labels, dtype=torch.float32)
        if not torch.is_tensor(features) or not torch.is_tensor(labels):
            raise TypeError("features and labels must be tensors or numpy arrays")
        self.device = device if device is not None else (features.device if hasattr(features, "device") else torch.device("cpu"))
        self.features = features.to(self.device)
        self.labels = labels.to(self.device)
        self.input_layer = input_layer
        self.stage = stage
        try:
            if self.features.dim() == 2 and self.features.shape[1] >= 62:
                self.meta = validate_units(self.features.cpu().numpy())
            else:
                self.meta = {}
        except Exception:
            self.meta = {}
    def __len__(self):
        return self.features.shape[0]
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def create_dataset(features, labels, input_layer=None, stage=None, device=None):
    return EWPINNDataset(features, labels, input_layer=input_layer, stage=stage, device=device)

def create_dataloader(features, labels, batch_size, shuffle, input_layer=None, stage=None, device=None, num_workers=0, drop_last=False, pin_memory=False):
    dataset = EWPINNDataset(features, labels, input_layer=input_layer, stage=stage, device=device)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)

def load_npz(path, device=None):
    data = np.load(path)
    features = torch.tensor(data['features'], dtype=torch.float32)
    labels = torch.tensor(data['labels'], dtype=torch.float32)
    if device is not None:
        features = features.to(device)
        labels = labels.to(device)
    return features, labels
