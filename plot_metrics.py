import matplotlib.pyplot as plt
import re

log_file = "training.log"

metrics_dict = {}
current_epoch = None

with open(log_file, "r") as f:
    for line in f:
        epoch_match = re.search(r"Epoch (\d+)/\d+", line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            if current_epoch not in metrics_dict:
                metrics_dict[current_epoch] = {}
        
        if current_epoch is not None:
            train_loss_match = re.search(r"Train Loss: ([\d.]+)", line)
            if train_loss_match:
                metrics_dict[current_epoch]['train_loss'] = float(train_loss_match.group(1))
            
            val_match = re.search(r"Val Loss: ([\d.]+) \| Val IoU: ([\d.]+) \| Val Dice: ([\d.]+)", line)
            if val_match:
                metrics_dict[current_epoch]['val_loss'] = float(val_match.group(1))
                metrics_dict[current_epoch]['val_iou'] = float(val_match.group(2))
                metrics_dict[current_epoch]['val_dice'] = float(val_match.group(3))

# Extract the final values
sorted_epochs = sorted([e for e in metrics_dict.keys() if 'val_loss' in metrics_dict[e]])
e_list = []
tl_list = []
vl_list = []
viou_list = []
vdice_list = []

for e in sorted_epochs:
    m = metrics_dict[e]
    if 'train_loss' in m and 'val_loss' in m:
        e_list.append(e)
        tl_list.append(m['train_loss'])
        vl_list.append(m['val_loss'])
        viou_list.append(m['val_iou'])
        vdice_list.append(m['val_dice'])

plt.figure(figsize=(12, 5))

# Plot Losses
plt.subplot(1, 2, 1)
plt.plot(e_list, tl_list, label='Train Loss', marker='o')
plt.plot(e_list, vl_list, label='Val Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)

# Plot Metrics
plt.subplot(1, 2, 2)
plt.plot(e_list, viou_list, label='Val IoU', marker='s')
plt.plot(e_list, vdice_list, label='Val Dice', marker='s')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.title('Validation IoU and Dice')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('metrics_plot.png', dpi=300)
print(f"Plot saved to metrics_plot.png with {len(e_list)} epochs.")
