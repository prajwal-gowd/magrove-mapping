import os
import torch

# Dataset Config
DATA_DIR = "dataset_patches"
IMAGE_SIZE = 224
BATCH_SIZE = 8
NUM_CLASSES = 1  # 1 for binary segmentation

# Training Config
LEARNING_RATE = 1e-4
BACKBONE_LR = 5e-5           # Lower LR for pretrained backbone
WEIGHT_DECAY = 1e-4           # AdamW weight decay for regularization
NUM_EPOCHS = 50               # More epochs for convergence
WARMUP_EPOCHS = 3             # Linear warmup before cosine decay
GRAD_CLIP_MAX_NORM = 1.0      # Gradient clipping
BACKBONE_FREEZE_EPOCHS = 5    # Freeze backbone for first N epochs
SAVE_PATH = "checkpoints/best_model.pth"

# Loss Config
BCE_WEIGHT = 0.3              # Lower BCE weight
DICE_WEIGHT = 0.7             # Higher Dice weight (directly optimizes IoU proxy)
FOCAL_ALPHA = 0.75            # Focal loss alpha for class balance
FOCAL_GAMMA = 2.0             # Focal loss gamma for hard example mining

# Model Config
BACKBONE_NAME = "resnet50"
TRANSFORMER_LAYERS = 2
TRANSFORMER_HEADS = 8
TRANSFORMER_DIM_FEEDFORWARD = 512
DROPOUT = 0.1
DECODER_DROPOUT = 0.2         # Dropout in decoder to reduce overfitting

# Device Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
