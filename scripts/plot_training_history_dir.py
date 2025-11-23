import json,sys,os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print('Usage: python plot_training_history_dir.py /path/to/training_history.json')
    sys.exit(1)

json_path = sys.argv[1]
if not os.path.exists(json_path):
    print('File not found:', json_path)
    sys.exit(2)

out_dir = os.path.dirname(json_path)
with open(json_path,'r') as f:
    data = json.load(f)
train = data.get('train_losses', [])
val = data.get('val_losses', [])
phys = data.get('physics_losses', [])

plt.figure(figsize=(10,6))
if train:
    plt.plot(range(1,len(train)+1), train, label='train_loss', linewidth=1)
if val:
    plt.plot(range(1,len(val)+1), val, label='val_loss', linewidth=2)
if phys:
    plt.plot(range(1,len(phys)+1), phys, label='physics_loss', linestyle='--')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Loss curves')
plt.legend()
plt.grid(True)
plt.tight_layout()
out_path = os.path.join(out_dir, 'loss_curves.png')
plt.savefig(out_path)
print('Saved', out_path)
