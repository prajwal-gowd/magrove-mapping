import torch
import os

checkpoint_path = 'checkpoints/best_model.pth'
if not os.path.exists(checkpoint_path):
    print("Checkpoint not found!")
    exit(1)

print("Loading checkpoint...")
checkpoint = torch.load(checkpoint_path, map_location='cpu')

state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint

new_state_dict = {}

# Mapping from old decoder indices to new decoder indices
# Old: 0, 1 (conv1, bn1), 3, 4 (conv2, bn2), 6, 7 (conv3, bn3), 9, 10 (conv4, bn4), 12, 13 (conv5, bn5), 15 (conv_out)
# New: 0, 1 (conv1, bn1), 4, 5 (conv2, bn2), 8, 9 (conv3, bn3), 12, 13 (conv4, bn4), 15, 16 (conv5, bn5), 18 (conv_out)
mapping = {
    '0': '0', '1': '1',
    '3': '4', '4': '5',
    '6': '8', '7': '9',
    '9': '12', '10': '13',
    '12': '15', '13': '16',
    '15': '18'
}

for k, v in state_dict.items():
    if k.startswith('decoder.'):
        parts = k.split('.')
        old_idx = parts[1]
        if old_idx in mapping:
            new_idx = mapping[old_idx]
            new_k = f"decoder.{new_idx}.{parts[2]}"
            new_state_dict[new_k] = v
        else:
            print(f"Warning: unmapped decoder key {k}")
            new_state_dict[k] = v
    else:
        new_state_dict[k] = v

if 'model_state_dict' in checkpoint:
    checkpoint['model_state_dict'] = new_state_dict
else:
    checkpoint = new_state_dict

print("Saving patched checkpoint...")
# Save a backup just in case
import shutil
shutil.copy(checkpoint_path, checkpoint_path + '.bak')
torch.save(checkpoint, checkpoint_path)
print("Done! Checkpoint patched for new Dropout2d architecture.")
