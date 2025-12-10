#!/usr/bin/env python3
"""
绘制训练历史曲线

用法:
    python plot_training_history_dir.py /path/to/training_history.json
    python plot_training_history_dir.py /path/to/outputs_pinn_xxx/  (自动查找)
"""
import json
import sys
import os
import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

def load_history_from_json(json_path):
    """从 JSON 文件加载训练历史"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return {
        'train_losses': data.get('train_losses', []),
        'val_losses': data.get('val_losses', []),
        'physics_losses': data.get('physics_losses', []),
        'epochs': data.get('epochs', [])
    }

def load_history_from_checkpoint(checkpoint_path):
    """从模型检查点加载训练历史"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    history = checkpoint.get('history', {})
    return {
        'train_losses': history.get('loss', history.get('train_loss', history.get('total_loss', []))),
        'val_losses': history.get('val_loss', []),
        'physics_losses': history.get('physics', history.get('physics_loss', [])),
        'interface_losses': history.get('interface', []),
        'epochs': history.get('epoch', list(range(1, len(history.get('loss', [])) + 1)))
    }

def main():
    if len(sys.argv) < 2:
        print('Usage: python plot_training_history_dir.py /path/to/training_history.json')
        print('       python plot_training_history_dir.py /path/to/outputs_pinn_xxx/')
        sys.exit(1)

    path = sys.argv[1]
    
    # 确定输入类型
    if os.path.isdir(path):
        # 目录：查找 JSON 或模型文件
        json_files = glob.glob(os.path.join(path, '*.json'))
        history_json = os.path.join(path, 'training_history.json')
        final_model = os.path.join(path, 'final_model.pth')
        best_model = os.path.join(path, 'best_model.pth')
        
        if os.path.exists(history_json):
            data = load_history_from_json(history_json)
            out_dir = path
        elif os.path.exists(final_model):
            data = load_history_from_checkpoint(final_model)
            out_dir = path
        elif os.path.exists(best_model):
            data = load_history_from_checkpoint(best_model)
            out_dir = path
        else:
            print(f'No training history found in: {path}')
            sys.exit(2)
    elif path.endswith('.json'):
        if not os.path.exists(path):
            print('File not found:', path)
            sys.exit(2)
        data = load_history_from_json(path)
        out_dir = os.path.dirname(path) or '.'
    elif path.endswith('.pth'):
        if not os.path.exists(path):
            print('File not found:', path)
            sys.exit(2)
        data = load_history_from_checkpoint(path)
        out_dir = os.path.dirname(path) or '.'
    else:
        print(f'Unknown file type: {path}')
        sys.exit(2)

    train = data.get('train_losses', [])
    val = data.get('val_losses', [])
    phys = data.get('physics_losses', [])
    interface = data.get('interface_losses', [])
    epochs = data.get('epochs', [])

    if not train and not val and not phys and not interface:
        print('No loss data found')
        sys.exit(3)

    plt.figure(figsize=(10, 6))
    
    x = epochs if epochs else range(1, max(len(train), len(val), len(phys), len(interface)) + 1)
    
    if train:
        plt.plot(x[:len(train)], train, label='total_loss', linewidth=2)
    if val:
        plt.plot(x[:len(val)], val, label='val_loss', linewidth=1)
    if phys:
        plt.plot(x[:len(phys)], phys, label='physics_loss', linestyle='--')
    if interface:
        plt.plot(x[:len(interface)], interface, label='interface_loss', linestyle=':')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    plt.tight_layout()
    
    out_path = os.path.join(out_dir, 'loss_curves.png')
    plt.savefig(out_path, dpi=150)
    print(f'Saved: {out_path}')

if __name__ == '__main__':
    main()
