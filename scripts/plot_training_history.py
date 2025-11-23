import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

json_path = '/home/scnu/Gitee/EFD3D/output/long_run_debug/training_history.json'
out_dir = '/home/scnu/Gitee/EFD3D/output/long_run_debug'
out_path = os.path.join(out_dir, 'loss_curves.png')

with open(json_path, 'r') as f:
    data = json.load(f)

train = data.get('train_losses', [])
val = data.get('val_losses', [])
phys = data.get('physics_losses', [])

plt.figure(figsize=(10,6))
if train:
    plt.plot(range(1, len(train)+1), train, label='train_loss', linewidth=1)
if val:
    plt.plot(range(1, len(val)+1), val, label='val_loss', linewidth=2)
if phys:
    # physics loss may be constant or same length as val; align to its length
    plt.plot(range(1, len(phys)+1), phys, label='physics_loss', linestyle='--')

plt.xlabel('Step / Epoch')
plt.ylabel('Loss')
plt.title('Training / Validation / Physics Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
os.makedirs(out_dir, exist_ok=True)
plt.savefig(out_path)
print('Saved plot to', out_path)
